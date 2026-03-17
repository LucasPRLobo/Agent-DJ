"""Background Fetcher — continuously searches and pre-downloads tracks matching the vibe.

Runs as an async task alongside the DJ loop. Maintains a pool of analyzed,
ready-to-play tracks so the DJ never has to wait for a download.
"""

import asyncio
import random
from pathlib import Path

import numpy as np

from ..analyzer.audio import TrackProfile, analyze_track
from ..analyzer.track_store import TrackStore
from ..planner.vibe_profile import VibeProfile
from .auto_fetch import generate_search_queries, quick_analyze
from .youtube import download_track, search_tracks


class BackgroundFetcher:
    """Continuously fetches and analyzes tracks matching the session vibe."""

    def __init__(
        self,
        store: TrackStore,
        audio_dir: str = "samples",
        pool_target: int = 10,
        fetch_interval: float = 30.0,
    ):
        self.store = store
        self.audio_dir = audio_dir
        self.pool_target = pool_target  # target number of unplayed analyzed tracks
        self.fetch_interval = fetch_interval
        self.vibe: VibeProfile | None = None
        self.played_paths: set[str] = set()
        self.fetching_queries: set[str] = set()  # avoid duplicate fetches
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self, vibe: VibeProfile):
        """Start background fetching for a vibe."""
        self.vibe = vibe
        self._running = True
        self._task = asyncio.create_task(self._run())

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    def mark_played(self, file_path: str):
        """Mark a track as played so we don't count it in the pool."""
        self.played_paths.add(file_path)

    def update_vibe(self, vibe: VibeProfile):
        """Update the target vibe (e.g., after user feedback)."""
        self.vibe = vibe

    @property
    def pool_size(self) -> int:
        """Number of unplayed, analyzed tracks available."""
        all_tracks = self.store.get_all()
        return len([t for t in all_tracks if t.file_path not in self.played_paths])

    async def _run(self):
        """Main fetch loop."""
        while self._running:
            try:
                pool = self.pool_size
                if pool < self.pool_target and self.vibe:
                    await self._fetch_batch()

                # Wait before checking again
                await asyncio.sleep(self.fetch_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Background fetcher error: {e}")
                await asyncio.sleep(self.fetch_interval)

    async def _fetch_batch(self):
        """Fetch a batch of tracks."""
        if not self.vibe:
            return

        # Generate diverse search queries
        queries = self._generate_diverse_queries()
        if not queries:
            return

        loop = asyncio.get_event_loop()
        downloaded_this_batch = 0
        max_per_batch = 2

        for query in queries:
            if not self._running:
                break
            if self.pool_size >= self.pool_target:
                break
            if downloaded_this_batch >= max_per_batch:
                break
            if query in self.fetching_queries:
                continue

            self.fetching_queries.add(query)
            print(f"[BG Fetch] Searching: {query}")

            try:
                # Search with more results for diversity, then pick randomly
                results = await loop.run_in_executor(
                    None, lambda q=query: search_tracks(q, max_results=8)
                )

                # Shuffle to avoid always picking the most popular version
                if len(results) > 2:
                    # Keep first result as an option but shuffle the rest
                    top = results[0]
                    rest = results[1:]
                    random.shuffle(rest)
                    results = [top] + rest

                for result in results:
                    if not self._running:
                        break

                    # Skip if already in library (by video ID)
                    existing = self.store.get_all()
                    existing_ids = {Path(t.file_path).stem for t in existing}
                    if result.video_id in existing_ids:
                        continue

                    # Extract core song name and check for duplicates
                    # This catches "Girl from Ipanema" by different artists
                    if _is_duplicate_song(result.title, existing):
                        print(f"[BG Fetch] Skipped duplicate: {result.title}")
                        continue

                    # Skip very short or very long tracks
                    if result.duration < 90 or result.duration > 480:
                        continue

                    # Download
                    print(f"[BG Fetch] Downloading: {result.title}")
                    dl = await loop.run_in_executor(
                        None, lambda r=result: download_track(r.url, self.audio_dir)
                    )

                    if not dl:
                        continue

                    # Quick analyze to check BPM/key compatibility
                    profile = await loop.run_in_executor(
                        None, lambda: quick_analyze(dl.file_path)
                    )

                    if not profile:
                        continue

                    # Set proper title
                    title = f"{dl.title} - {dl.artist}" if dl.artist != "Unknown" else dl.title
                    profile.title = title

                    # Check if BPM is in an acceptable range
                    bpm_low, bpm_high = self.vibe.bpm_range
                    if not (bpm_low - 15 <= profile.bpm <= bpm_high + 15):
                        print(f"[BG Fetch] Skipped {title} — BPM {profile.bpm} out of range")
                        continue

                    # Save quick profile
                    self.store.save(profile)
                    print(f"[BG Fetch] Added: {title} ({profile.bpm} BPM, {profile.key})")
                    downloaded_this_batch += 1

                    # Full analysis in background
                    asyncio.create_task(self._full_analyze(dl.file_path, title))

                    # Delay between downloads to avoid hammering YouTube
                    await asyncio.sleep(8)

            except Exception as e:
                print(f"[BG Fetch] Error for '{query}': {e}")
            finally:
                self.fetching_queries.discard(query)

    def _generate_diverse_queries(self) -> list[str]:
        """Generate specific search queries — actual song names, not generic genre searches."""
        if not self.vibe:
            return []

        queries = []
        genres = list(self.vibe.genres.keys())
        mood_tags = self.vibe.mood_tags or []
        examples = self.vibe.example_tracks or []

        # Track what we already have to avoid searching for the same stuff
        existing = self.store.get_all()
        existing_titles_lower = {t.title.lower() for t in existing}

        # 1. Example tracks from onboarding (highest priority — actual song names)
        for ex in examples:
            if ex.lower() not in existing_titles_lower:
                queries.append(ex)

        # 2. Artist-based searches from what's already in the library
        artists_seen = set()
        for t in existing:
            if " - " in t.title:
                artist = t.title.split(" - ")[-1].strip()
                if artist.lower() not in artists_seen:
                    artists_seen.add(artist.lower())
                    queries.append(f"{artist} best songs")

        # 3. Specific song searches per genre (more targeted than generic genre queries)
        genre_song_prompts = {
            "RnB / Neo-Soul": ["Erykah Badu", "D'Angelo", "Frank Ocean", "SZA", "Lauryn Hill",
                               "Jhené Aiko", "Daniel Caesar", "H.E.R.", "Solange", "Summer Walker"],
            "Jazz": ["Miles Davis", "John Coltrane", "Robert Glasper", "Kamasi Washington",
                     "Herbie Hancock", "Thelonious Monk", "Tom Misch", "Nujabes"],
            "House": ["Disclosure", "Kaytranada", "Kerri Chandler", "Frankie Knuckles",
                      "Fred again..", "Ross From Friends", "Mall Grab"],
            "Funk": ["Vulfpeck", "Thundercat", "Anderson .Paak", "Daft Punk", "Jamiroquai"],
            "Soul": ["Marvin Gaye", "Stevie Wonder", "Aretha Franklin", "Leon Bridges", "Alicia Keys"],
            "Bossa Nova / Latin Jazz": ["Antonio Carlos Jobim", "Astrud Gilberto", "Stan Getz",
                                         "Bebel Gilberto", "Buena Vista Social Club"],
            "Hip-Hop": ["J Dilla", "Madlib", "A Tribe Called Quest", "Kendrick Lamar", "Tyler the Creator"],
            "Ambient / Downtempo": ["Bonobo", "Tycho", "Boards of Canada", "Khruangbin"],
            "Smooth / Contemporary Jazz": ["Norah Jones", "Diana Krall", "Gregory Porter", "José James"],
        }

        for genre in genres:
            artists = genre_song_prompts.get(genre, [])
            if artists:
                available = [a for a in artists if a.lower() not in artists_seen]
                for artist in random.sample(available, min(2, len(available))):
                    # Vary the query to get different results each time
                    query_styles = [
                        f"{artist} {genre}",
                        f"{artist} deep cuts",
                        f"{artist} underrated songs",
                        f"{artist} album tracks",
                        f"{artist} B sides",
                    ]
                    queries.append(random.choice(query_styles))

        # 4. Discovery queries — find less obvious tracks
        if genres:
            discovery_templates = [
                "underrated {genre} artists",
                "{genre} hidden gems",
                "{genre} underground",
                "lesser known {genre} songs",
                "{genre} deep cuts playlist",
            ]
            genre = random.choice(genres)
            queries.append(random.choice(discovery_templates).format(genre=genre))

        # Shuffle and limit
        random.shuffle(queries)
        return queries[:6]

    async def _full_analyze(self, file_path: str, title: str):
        """Run full analysis in background."""
        loop = asyncio.get_event_loop()
        try:
            profile = await loop.run_in_executor(
                None,
                lambda: analyze_track(file_path, run_classifier=True, title=title),
            )
            self.store.save(profile)
            print(f"[BG Fetch] Full analysis done: {title}")
        except Exception as e:
            print(f"[BG Fetch] Full analysis failed for {title}: {e}")


def _clean_song_name(title: str) -> str:
    """Extract the core song name from a YouTube title.

    "Daft Punk - Around The World (Official Video Remastered)" → "around the world"
    "The Girl From Ipanema - Stan Getz & Astrud Gilberto" → "the girl from ipanema"
    """
    name = title.lower()

    # Remove parenthetical/bracket content
    import re
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\[[^\]]*\]", "", name)

    # Split on common separators and take the song name part
    for sep in [" - ", " | ", " – ", " — "]:
        if sep in name:
            parts = name.split(sep)
            # Usually "Artist - Song" or "Song - Artist"
            # Take the shorter part as artist, longer as song
            # But prefer the second part for "Artist - Song" format
            name = parts[1] if len(parts) > 1 else parts[0]
            break

    # Remove common suffixes
    for suffix in ["official video", "official audio", "official music video",
                   "lyrics", "lyric video", "audio", "visualizer", "remastered",
                   "hq", "hd", "live", "ft.", "feat.", "music video"]:
        name = name.replace(suffix, "")

    # Clean up
    name = re.sub(r"[^\w\s]", " ", name)  # remove punctuation
    name = re.sub(r"\s+", " ", name).strip()

    return name


def _is_duplicate_song(new_title: str, existing_tracks: list) -> bool:
    """Check if a song is a duplicate of something already in the library.

    Compares core song names to catch different versions/recordings
    of the same song (e.g., multiple "Girl from Ipanema" versions).
    """
    new_name = _clean_song_name(new_title)
    if len(new_name) < 3:
        return False

    for track in existing_tracks:
        existing_name = _clean_song_name(track.title)
        if len(existing_name) < 3:
            continue

        # Exact match
        if new_name == existing_name:
            return True

        # One contains the other (catches "girl from ipanema" vs "the girl from ipanema")
        if len(new_name) > 5 and len(existing_name) > 5:
            if new_name in existing_name or existing_name in new_name:
                return True

        # Word overlap — if 80%+ of words match, it's likely the same song
        new_words = set(new_name.split())
        existing_words = set(existing_name.split())
        if len(new_words) >= 3 and len(existing_words) >= 3:
            overlap = len(new_words & existing_words)
            shorter = min(len(new_words), len(existing_words))
            if overlap / shorter >= 0.8:
                return True

    return False
