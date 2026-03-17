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
                # Search for candidates
                results = await loop.run_in_executor(
                    None, lambda q=query: search_tracks(q, max_results=3)
                )

                for result in results:
                    if not self._running:
                        break

                    # Skip if already in library (by video ID or similar title)
                    existing = self.store.get_all()
                    existing_ids = {Path(t.file_path).stem for t in existing}
                    existing_titles = {t.title.lower().split(" - ")[0].strip() for t in existing}
                    if result.video_id in existing_ids:
                        continue

                    # Check title similarity to avoid duplicates
                    result_title_clean = result.title.lower()
                    # Remove common suffixes
                    for suffix in ["official video", "official audio", "lyrics", "hq", "hd",
                                   "music video", "audio", "visualizer", "remastered"]:
                        result_title_clean = result_title_clean.replace(suffix, "")
                    result_title_clean = result_title_clean.strip(" ()-[]|")
                    if any(result_title_clean in et or et in result_title_clean
                           for et in existing_titles if len(et) > 5):
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
                # Pick 2 random artists we don't already have
                available = [a for a in artists if a.lower() not in artists_seen]
                for artist in random.sample(available, min(2, len(available))):
                    queries.append(f"{artist} {genre}")

        # 4. Mood-specific (but with actual song context, not just "chill music")
        if mood_tags and genres:
            genre = genres[0]
            tags = " ".join(random.sample(mood_tags, min(2, len(mood_tags))))
            queries.append(f"{tags} {genre} songs")

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
