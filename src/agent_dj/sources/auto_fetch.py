"""Auto-fetch — search and download tracks on demand when the library doesn't have a good match."""

import asyncio
from pathlib import Path

import numpy as np

from ..analyzer.audio import TrackProfile
from ..analyzer.track_store import TrackStore
from .youtube import download_track, search_tracks


def generate_search_queries(
    vibe_genres: dict[str, float],
    current_track: TrackProfile | None = None,
    target_bpm: float | None = None,
    example_tracks: list[str] | None = None,
    played_titles: set[str] | None = None,
) -> list[str]:
    """Generate ranked search queries for finding the next track.

    Args:
        vibe_genres: Genre preferences from VibeProfile.
        current_track: Currently playing track (for similarity context).
        target_bpm: Target BPM for the next track.
        example_tracks: User-provided example tracks from onboarding.
        played_titles: Titles of already-played tracks to avoid.

    Returns:
        List of search queries, most specific first.
    """
    played_titles = played_titles or set()
    queries = []

    # 1. Example tracks from onboarding that haven't been played
    if example_tracks:
        for example in example_tracks:
            if example not in played_titles:
                queries.append(example)

    # 2. Similar to current track
    if current_track and current_track.title:
        clean_title = current_track.title.split(" - ")[0] if " - " in current_track.title else current_track.title
        queries.append(f"songs similar to {clean_title}")

    # 3. Genre + BPM hint
    top_genres = sorted(vibe_genres.items(), key=lambda x: -x[1])[:2]
    genre_str = " ".join(g for g, _ in top_genres)
    if target_bpm:
        queries.append(f"{genre_str} {target_bpm:.0f} BPM")
    else:
        queries.append(f"{genre_str} music")

    # 4. Genre only
    if top_genres:
        queries.append(f"best {top_genres[0][0]} tracks")

    return queries


def quick_analyze(file_path: str) -> TrackProfile | None:
    """Fast analysis — BPM, key, and basic energy only. ~3 seconds.

    Used for on-demand fetched tracks where we need to verify
    compatibility before queuing. Full analysis runs async later.
    """
    import librosa
    import essentia.standard as es
    from ..analyzer.camelot import key_to_camelot

    try:
        # Load only first 30 seconds at lower sample rate
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
        duration = librosa.get_duration(filename=file_path)

        # BPM
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # Key
        audio_full = es.MonoLoader(filename=file_path, sampleRate=44100)()
        key, scale, _ = es.KeyExtractor()(audio_full)
        camelot = key_to_camelot(key, scale)

        # Basic energy
        rms = float(np.sqrt(np.mean(y**2)))
        energy_curve = [min(rms * 5, 1.0)]

        from ..analyzer.structure import MixPoint, Section

        return TrackProfile(
            file_path=file_path,
            title=Path(file_path).stem,
            duration=round(duration, 2),
            bpm=round(bpm, 1),
            beats=beat_times[:50],  # keep first 50 beats
            downbeats=beat_times[::4][:12],
            key=camelot,
            key_name=f"{key} {scale}",
            energy_curve=energy_curve,
            loudness_lufs=-14.0,  # estimate
            spectral_centroid=0.0,
            spectral_bandwidth=0.0,
            mfcc=[0.0] * 13,
            onset_strength_mean=0.0,
            segments=[Section(start=0.0, end=duration, label="body", energy=energy_curve[0])],
            mix_in_points=[MixPoint(time=0.0, type="intro", confidence=0.5, duration=min(16.0, duration * 0.1))],
            mix_out_points=[MixPoint(time=max(0, duration - 20), type="outro", confidence=0.5, duration=20.0)],
            loop_candidates=[],
            classification=None,
        )
    except Exception as e:
        print(f"Quick analysis failed: {e}")
        return None


async def fetch_and_analyze(
    query: str,
    output_dir: str = "samples",
    timeout: float = 30.0,
) -> TrackProfile | None:
    """Download a track from YouTube and run quick analysis.

    Runs in a thread pool to avoid blocking the async loop.

    Args:
        query: Search query or YouTube URL.
        output_dir: Directory to save audio.
        timeout: Maximum time to wait.

    Returns:
        TrackProfile or None if failed/timeout.
    """
    loop = asyncio.get_event_loop()

    try:
        # Download in thread pool
        dl = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: download_track(query, output_dir)),
            timeout=timeout,
        )

        if not dl:
            return None

        # Quick analyze in thread pool
        profile = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: quick_analyze(dl.file_path)),
            timeout=15.0,
        )

        if profile:
            # Set proper title from download metadata
            title = f"{dl.title} - {dl.artist}" if dl.artist != "Unknown" else dl.title
            profile.title = title

        return profile

    except asyncio.TimeoutError:
        print(f"Auto-fetch timed out for: {query}")
        return None
    except Exception as e:
        print(f"Auto-fetch failed for {query}: {e}")
        return None


async def full_analyze_background(
    file_path: str,
    title: str,
    store: TrackStore,
):
    """Run full analysis in background and update the store.

    Called after quick_analyze to fill in ML classification,
    structure detection, and detailed features.
    """
    loop = asyncio.get_event_loop()

    try:
        from ..analyzer.audio import analyze_track
        profile = await loop.run_in_executor(
            None,
            lambda: analyze_track(file_path, run_classifier=True, title=title),
        )
        store.save(profile)
        print(f"Background analysis complete: {title}")
    except Exception as e:
        print(f"Background analysis failed for {title}: {e}")
