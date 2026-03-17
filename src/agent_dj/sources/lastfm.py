"""Last.fm discovery — find diverse artists and tracks using Last.fm's deep music graph.

Uses artist similarity (20 years of scrobbling data), tag-based browsing,
and recursive exploration to find tracks that match a vibe without
repeating the same popular songs.
"""

import os
import random
from dataclasses import dataclass

import pylast


@dataclass
class DiscoveredTrack:
    """A track discovered via Last.fm."""
    title: str
    artist: str
    search_query: str  # what to search on YouTube


def get_lastfm_client() -> pylast.LastFMNetwork | None:
    """Get a Last.fm API client. Returns None if no API key configured."""
    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        return None
    return pylast.LastFMNetwork(api_key=api_key)


def discover_by_similar_artists(
    seed_artists: list[str],
    n_tracks: int = 10,
    depth: int = 2,
) -> list[DiscoveredTrack]:
    """Find tracks by exploring similar artists recursively.

    Goes depth levels deep into the similar artist graph,
    getting progressively more obscure artists.

    Args:
        seed_artists: Starting artist names.
        n_tracks: Number of tracks to return.
        depth: How many levels deep to explore (1 = direct similar, 2 = similar-of-similar).

    Returns:
        List of discovered tracks.
    """
    client = get_lastfm_client()
    if not client:
        return []

    discovered_artists: set[str] = set()
    current_level = list(seed_artists)
    all_artists_by_depth: list[list[str]] = []

    for d in range(depth):
        next_level = []
        for artist_name in current_level:
            try:
                artist = client.get_artist(artist_name)
                similar = artist.get_similar(limit=15)
                for sim_artist, _match in similar:
                    name = str(sim_artist)
                    if name.lower() not in discovered_artists:
                        discovered_artists.add(name.lower())
                        next_level.append(name)
            except Exception as e:
                print(f"[Last.fm] Error getting similar to {artist_name}: {e}")
                continue

        all_artists_by_depth.append(next_level)
        current_level = next_level

    # Weight toward deeper (more obscure) artists
    weighted_artists = []
    for d, artists in enumerate(all_artists_by_depth):
        # Deeper = more likely to be picked
        weight = d + 1
        weighted_artists.extend([(a, weight) for a in artists])

    if not weighted_artists:
        return []

    # Weighted random selection
    total_weight = sum(w for _, w in weighted_artists)
    selected_artists = set()
    tracks = []

    attempts = 0
    while len(tracks) < n_tracks and attempts < n_tracks * 3:
        attempts += 1
        # Weighted random pick
        r = random.uniform(0, total_weight)
        cumulative = 0
        picked = weighted_artists[0][0]
        for name, w in weighted_artists:
            cumulative += w
            if cumulative >= r:
                picked = name
                break

        if picked in selected_artists:
            continue
        selected_artists.add(picked)

        # Get top tracks for this artist
        try:
            artist = client.get_artist(picked)
            top = artist.get_top_tracks(limit=5)
            if top:
                # Pick a random track, not just the #1
                track_item = random.choice(top[:min(5, len(top))])
                track_name = str(track_item.item)
                tracks.append(DiscoveredTrack(
                    title=track_name,
                    artist=picked,
                    search_query=f"{picked} - {track_name}",
                ))
        except Exception:
            continue

    return tracks


def discover_by_tags(
    tags: list[str],
    n_tracks: int = 10,
    avoid_artists: set[str] | None = None,
) -> list[DiscoveredTrack]:
    """Find tracks by Last.fm tags (genre, mood, style).

    Last.fm tags are user-generated and cover niche genres
    like "acid jazz", "neo-soul", "dark ambient" etc.

    Args:
        tags: Tag names to search (e.g., ["neo-soul", "smooth jazz"]).
        n_tracks: Number of tracks to return.
        avoid_artists: Artists to skip.

    Returns:
        List of discovered tracks.
    """
    client = get_lastfm_client()
    if not client:
        return []

    avoid = {a.lower() for a in (avoid_artists or set())}
    tracks = []

    for tag_name in tags:
        try:
            tag = client.get_tag(tag_name)

            # Get top artists for this tag, then sample from deeper in the list
            top_artists = tag.get_top_artists(limit=50)
            if not top_artists:
                continue

            # Skip the top 5 most popular — they're too obvious
            # Sample from position 5-50
            deeper_artists = top_artists[5:] if len(top_artists) > 10 else top_artists
            random.shuffle(deeper_artists)

            for artist_item in deeper_artists[:5]:
                artist_name = str(artist_item.item)
                if artist_name.lower() in avoid:
                    continue

                try:
                    artist = client.get_artist(artist_name)
                    top = artist.get_top_tracks(limit=5)
                    if top:
                        track_item = random.choice(top[:min(5, len(top))])
                        track_name = str(track_item.item)
                        tracks.append(DiscoveredTrack(
                            title=track_name,
                            artist=artist_name,
                            search_query=f"{artist_name} - {track_name}",
                        ))
                        avoid.add(artist_name.lower())
                except Exception:
                    continue

                if len(tracks) >= n_tracks:
                    break

        except Exception as e:
            print(f"[Last.fm] Error with tag '{tag_name}': {e}")
            continue

        if len(tracks) >= n_tracks:
            break

    return tracks


def discover_tracks(
    seed_artists: list[str] | None = None,
    tags: list[str] | None = None,
    n_tracks: int = 10,
    avoid_artists: set[str] | None = None,
) -> list[DiscoveredTrack]:
    """Main discovery function — combines similar artists + tag browsing.

    Args:
        seed_artists: Artists to seed similarity search.
        tags: Genre/mood tags for tag-based search.
        n_tracks: Total tracks to return.
        avoid_artists: Artists to skip.

    Returns:
        Mixed list of discovered tracks from both methods.
    """
    tracks = []
    avoid = set(avoid_artists or set())

    # Half from similar artists, half from tags
    if seed_artists:
        similar = discover_by_similar_artists(seed_artists, n_tracks=n_tracks // 2 + 1, depth=2)
        for t in similar:
            if t.artist.lower() not in avoid:
                tracks.append(t)
                avoid.add(t.artist.lower())

    if tags:
        tag_tracks = discover_by_tags(tags, n_tracks=n_tracks // 2 + 1, avoid_artists=avoid)
        tracks.extend(tag_tracks)

    random.shuffle(tracks)
    return tracks[:n_tracks]


# Map DJ genres to Last.fm tags
DJ_GENRE_TO_LASTFM_TAGS: dict[str, list[str]] = {
    "RnB / Neo-Soul": ["neo-soul", "rnb", "contemporary r&b", "new soul"],
    "Jazz": ["jazz", "modern jazz", "post-bop", "hard bop"],
    "Smooth / Contemporary Jazz": ["smooth jazz", "contemporary jazz", "jazz vocal"],
    "Bossa Nova / Latin Jazz": ["bossa nova", "latin jazz", "mpb", "brazilian jazz"],
    "House": ["deep house", "house", "tech house", "soulful house"],
    "Techno": ["techno", "minimal techno", "deep techno"],
    "Disco / Nu-Disco": ["nu-disco", "disco", "boogie"],
    "Funk": ["funk", "p-funk", "boogie funk"],
    "Soul": ["soul", "northern soul", "southern soul", "motown"],
    "Hip-Hop": ["hip-hop", "boom bap", "conscious hip-hop", "jazzy hip-hop"],
    "Trap": ["trap", "southern hip-hop"],
    "Afrobeat": ["afrobeat", "afrobeats", "highlife"],
    "Ambient / Downtempo": ["downtempo", "ambient", "chillout", "trip-hop"],
    "Drum & Bass / Jungle": ["drum and bass", "jungle", "liquid funk"],
    "Reggae / Roots": ["reggae", "roots reggae", "dub"],
    "Dancehall / Ragga": ["dancehall", "ragga"],
    "Pop": ["pop", "indie pop", "synth-pop"],
    "Rock": ["rock", "indie rock", "alternative"],
    "Blues": ["blues", "electric blues", "delta blues"],
}
