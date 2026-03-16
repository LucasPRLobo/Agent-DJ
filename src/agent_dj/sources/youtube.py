"""YouTube music source — search and download audio via yt-dlp.

Provides song search, metadata extraction, and audio download
for the demo version of Agent DJ.
"""

import json
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SearchResult:
    """A search result from YouTube."""
    video_id: str
    title: str
    artist: str
    duration: int  # seconds
    url: str
    thumbnail: str


@dataclass
class DownloadResult:
    """Result of downloading a track."""
    file_path: str
    title: str
    artist: str
    duration: int
    video_id: str


def search_tracks(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search YouTube for music tracks.

    Args:
        query: Search query (song name, artist, etc.)
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult objects.
    """
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--no-download",
        "--flat-playlist",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"Search failed: {result.stderr[:200]}")
            return []

        results = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                results.append(SearchResult(
                    video_id=data.get("id", ""),
                    title=data.get("title", "Unknown"),
                    artist=data.get("channel", data.get("uploader", "Unknown")),
                    duration=int(data.get("duration", 0) or 0),
                    url=data.get("url", f"https://www.youtube.com/watch?v={data.get('id', '')}"),
                    thumbnail=data.get("thumbnail", ""),
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        return results

    except subprocess.TimeoutExpired:
        print("Search timed out")
        return []


def download_track(
    query_or_url: str,
    output_dir: str | Path = "samples",
    audio_format: str = "wav",
) -> DownloadResult | None:
    """Download a track from YouTube as an audio file.

    Args:
        query_or_url: YouTube URL or search query.
        output_dir: Directory to save the audio file.
        audio_format: Output format (wav, mp3, flac).

    Returns:
        DownloadResult or None if download failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If it's a search query (not a URL), prefix with ytsearch
    if not query_or_url.startswith("http"):
        source = f"ytsearch1:{query_or_url}"
    else:
        source = query_or_url

    # Output template — use video ID to avoid duplicates
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        source,
        "--extract-audio",
        f"--audio-format={audio_format}",
        "--audio-quality=0",  # best quality
        "--output", output_template,
        "--print-json",
        "--no-playlist",
        # Limit to reasonable length (skip hour-long mixes)
        "--match-filter", "duration < 600",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            print(f"Download failed: {result.stderr[:300]}")
            return None

        # Parse the JSON output for metadata
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                video_id = data.get("id", "unknown")
                file_path = str(output_dir / f"{video_id}.{audio_format}")

                # Extract artist/title from the video title
                title, artist = _parse_title(data.get("title", "Unknown"))

                return DownloadResult(
                    file_path=file_path,
                    title=title,
                    artist=artist,
                    duration=int(data.get("duration", 0) or 0),
                    video_id=video_id,
                )
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    except subprocess.TimeoutExpired:
        print("Download timed out")
        return None


def download_multiple(
    queries: list[str],
    output_dir: str | Path = "samples",
    audio_format: str = "wav",
) -> list[DownloadResult]:
    """Download multiple tracks.

    Args:
        queries: List of search queries or URLs.
        output_dir: Directory to save audio files.
        audio_format: Output format.

    Returns:
        List of successful DownloadResult objects.
    """
    results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Downloading: {query}")
        result = download_track(query, output_dir, audio_format)
        if result:
            print(f"  -> {result.title} by {result.artist} ({result.duration}s)")
            results.append(result)
        else:
            print(f"  -> FAILED")
    return results


def _parse_title(raw_title: str) -> tuple[str, str]:
    """Try to extract artist and song title from a YouTube video title.

    Common formats:
    - "Artist - Song Title"
    - "Artist - Song Title (Official Video)"
    - "Song Title | Artist"

    Returns:
        (title, artist) tuple.
    """
    # Remove common suffixes
    cleaned = raw_title
    for suffix in [
        "(Official Video)", "(Official Music Video)", "(Official Audio)",
        "(Lyrics)", "(Lyric Video)", "[Official Video]", "[Official Audio]",
        "(Audio)", "(Visualizer)", "(HQ)", "(HD)",
        "(Official Visualizer)", "(Music Video)",
    ]:
        cleaned = cleaned.replace(suffix, "").strip()

    # Try "Artist - Title" format
    if " - " in cleaned:
        parts = cleaned.split(" - ", 1)
        return parts[1].strip(), parts[0].strip()

    # Try "Title | Artist" format
    if " | " in cleaned:
        parts = cleaned.split(" | ", 1)
        return parts[0].strip(), parts[1].strip()

    # Can't parse — return full title as title
    return cleaned.strip(), "Unknown"
