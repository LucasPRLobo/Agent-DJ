"""CLI for Agent DJ."""

import json
from pathlib import Path

import click

from .analyzer.audio import TrackProfile, analyze_track
from .analyzer.track_store import TrackStore

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


@click.group()
def cli():
    """Agent DJ — AI-powered live DJ agent."""
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--db", default="tracks.db", help="Path to track database.")
@click.option("--no-classify", is_flag=True, help="Skip ML classification (faster).")
@click.option("--model-dir", default=None, help="Path to ML model files.")
def analyze(path: str, db: str, no_classify: bool, model_dir: str | None):
    """Analyze audio files and store their profiles.

    PATH can be a single audio file or a directory of audio files.
    """
    store = TrackStore(db)
    path = Path(path)
    model_path = Path(model_dir) if model_dir else None

    if path.is_file():
        files = [path]
    else:
        files = sorted(
            f for f in path.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS
        )

    if not files:
        click.echo(f"No audio files found in {path}")
        return

    click.echo(f"Found {len(files)} audio file(s) to analyze.\n")

    for i, f in enumerate(files, 1):
        click.echo(f"[{i}/{len(files)}] Analyzing: {f.name}")
        try:
            profile = analyze_track(f, run_classifier=not no_classify, model_dir=model_path)
            store.save(profile)
            _print_profile_summary(profile)
            click.echo()
        except Exception as e:
            click.echo(f"  ERROR: {e}\n")

    click.echo(f"Done. {store.count()} tracks in database ({db}).")


@cli.command()
@click.option("--db", default="tracks.db", help="Path to track database.")
@click.option("--json-output", is_flag=True, help="Output as JSON.")
def library(db: str, json_output: bool):
    """Show all analyzed tracks in the database."""
    store = TrackStore(db)
    tracks = store.get_all()

    if not tracks:
        click.echo("No tracks in database. Run 'agent-dj analyze' first.")
        return

    if json_output:
        click.echo(json.dumps([t.to_dict() for t in tracks], indent=2))
    else:
        click.echo(f"Library: {len(tracks)} tracks\n")
        click.echo(f"  {'Title':<35} {'BPM':>6} {'Key':>4} {'Dur':>5} {'LUFS':>6} {'Dance':>6} {'Genre'}")
        click.echo(f"  {'─' * 35} {'─' * 6} {'─' * 4} {'─' * 5} {'─' * 6} {'─' * 6} {'─' * 20}")
        for t in sorted(tracks, key=lambda t: t.bpm):
            dance = f"{t.classification.danceability:.2f}" if t.classification else "  -  "
            genre = t.classification.genres[0][0] if t.classification and t.classification.genres else "-"
            # Truncate genre to 20 chars
            genre = genre[:20]
            click.echo(
                f"  {t.title:<35} {t.bpm:>6.1f} {t.key:>4} {t.duration:>5.0f}s {t.loudness_lufs:>5.1f} {dance:>6} {genre}"
            )


@cli.command()
@click.argument("track_path", type=click.Path(exists=True))
@click.option("--db", default="tracks.db", help="Path to track database.")
def inspect(track_path: str, db: str):
    """Show detailed analysis for a specific track."""
    store = TrackStore(db)
    profile = store.get(track_path)

    if not profile:
        # Try matching by filename
        all_tracks = store.get_all()
        matches = [t for t in all_tracks if Path(t.file_path).name == Path(track_path).name]
        if matches:
            profile = matches[0]

    if not profile:
        click.echo(f"Track not found in database. Run 'agent-dj analyze {track_path}' first.")
        return

    _print_profile_detail(profile)


@cli.command()
@click.argument("track_path", type=click.Path(exists=True))
@click.option("--db", default="tracks.db", help="Path to track database.")
@click.option("--top-n", default=5, help="Number of similar tracks to show.")
def similar(track_path: str, db: str, top_n: int):
    """Find tracks most similar to a given track (by embedding)."""
    store = TrackStore(db)
    profile = store.get(track_path)

    if not profile:
        click.echo(f"Track not found in database.")
        return

    if not profile.classification or not profile.classification.embedding:
        click.echo(f"No embedding available for this track. Re-analyze with ML classification.")
        return

    results = store.find_similar_by_embedding(
        profile.classification.embedding,
        top_n=top_n + 1,  # +1 because the track itself will be in results
        exclude_paths={profile.file_path},
    )

    if not results:
        click.echo("No similar tracks found.")
        return

    click.echo(f"Tracks similar to: {profile.title}\n")
    for t, score in results[:top_n]:
        click.echo(f"  {score:.3f}  {t.title:<35} {t.bpm:>6.1f} BPM | {t.key:>3}")


@cli.command()
@click.option("--db", default="tracks.db", help="Path to track database.")
@click.option("--preset", type=click.Choice(["chill_jazz", "house_party", "dinner"]),
              default=None, help="Use a preset vibe profile.")
@click.option("--strategy", type=click.Choice(["greedy_scoring", "graph_greedy", "graph_beam", "clustering"]),
              default="greedy_scoring", help="Track selection strategy.")
@click.option("--duration", default=None, type=int, help="Set duration in minutes.")
@click.option("--n-tracks", default=None, type=int, help="Number of tracks to select.")
@click.option("--json-output", is_flag=True, help="Output as JSON.")
def plan(db: str, preset: str | None, strategy: str, duration: int | None,
         n_tracks: int | None, json_output: bool):
    """Plan a DJ set from analyzed tracks."""
    from .planner.set_planner import SelectionStrategy, plan_set
    from .planner.vibe_profile import chill_jazz_party, dinner_ambient, house_party

    store = TrackStore(db)
    tracks = store.get_all()

    if not tracks:
        click.echo("No tracks in database. Run 'agent-dj analyze' first.")
        return

    # Get vibe profile
    if preset == "chill_jazz":
        vibe = chill_jazz_party(duration or 45)
    elif preset == "house_party":
        vibe = house_party(duration or 120)
    elif preset == "dinner":
        vibe = dinner_ambient(duration or 90)
    else:
        # Default
        vibe = chill_jazz_party(duration or 45)
        click.echo("Using default 'chill_jazz' preset. Use --preset to change.\n")

    if duration:
        vibe.duration_minutes = duration

    strat = SelectionStrategy(strategy)
    set_plan = plan_set(tracks, vibe, strat, n_tracks)

    if json_output:
        click.echo(set_plan.to_json())
    else:
        click.echo(set_plan.summary())


@cli.command(name="download-models")
@click.option("--model-dir", default=None, help="Path to store model files.")
def download_models(model_dir: str | None):
    """Download all required ML models."""
    from .analyzer.classifier import download_all_models
    model_path = Path(model_dir) if model_dir else None
    click.echo("Downloading all models...")
    paths = download_all_models(model_path)
    click.echo(f"\nDownloaded {len(paths)} models:")
    for key, path in paths.items():
        click.echo(f"  {key}: {path}")


def _print_profile_summary(profile: TrackProfile):
    """Print a one-line summary of a track profile."""
    click.echo(
        f"  BPM: {profile.bpm} | Key: {profile.key} ({profile.key_name}) | "
        f"Duration: {profile.duration:.0f}s | Loudness: {profile.loudness_lufs} LUFS"
    )
    click.echo(f"  Segments: {' → '.join(s.label for s in profile.segments)}")
    click.echo(
        f"  Mix-in: {len(profile.mix_in_points)} | "
        f"Mix-out: {len(profile.mix_out_points)} | "
        f"Loops: {len(profile.loop_candidates)}"
    )
    if profile.classification and profile.classification.genres:
        top3 = profile.classification.genres[:3]
        genres_str = ", ".join(f"{g[0]} ({g[1]:.2f})" for g in top3)
        click.echo(f"  Genres: {genres_str}")
        click.echo(
            f"  Mood: happy={profile.classification.mood_happy:.2f} "
            f"relaxed={profile.classification.mood_relaxed:.2f} "
            f"aggressive={profile.classification.mood_aggressive:.2f} "
            f"sad={profile.classification.mood_sad:.2f}"
        )
        click.echo(f"  Danceability: {profile.classification.danceability:.2f}")


def _print_profile_detail(profile: TrackProfile):
    """Print full detail for a track profile."""
    click.echo(f"═══ {profile.title} ═══\n")

    click.echo("  BASICS")
    click.echo(f"  File:     {profile.file_path}")
    click.echo(f"  Duration: {profile.duration:.1f}s ({profile.duration / 60:.1f} min)")
    click.echo(f"  BPM:      {profile.bpm}")
    click.echo(f"  Key:      {profile.key} ({profile.key_name})")
    click.echo(f"  Loudness: {profile.loudness_lufs} LUFS")
    click.echo()

    click.echo("  SPECTRAL")
    click.echo(f"  Centroid:   {profile.spectral_centroid:.1f} Hz")
    click.echo(f"  Bandwidth:  {profile.spectral_bandwidth:.1f} Hz")
    click.echo(f"  Onset str:  {profile.onset_strength_mean:.4f}")
    click.echo()

    click.echo("  STRUCTURE")
    for s in profile.segments:
        bar = "█" * int(s.energy * 20)
        click.echo(f"  {s.start:>6.1f}s - {s.end:>6.1f}s  {s.label:<12} energy: {bar} {s.energy:.2f}")
    click.echo()

    click.echo("  ENERGY CURVE")
    curve = profile.energy_curve
    max_width = 40
    for i, e in enumerate(curve):
        bar = "█" * int(e * max_width)
        click.echo(f"  {i:>3}: {bar} {e:.2f}")
    click.echo()

    click.echo("  MIX POINTS")
    click.echo("  In:")
    for p in profile.mix_in_points:
        click.echo(f"    {p.time:>6.1f}s  {p.type:<12} conf={p.confidence:.1f}  dur={p.duration:.1f}s")
    click.echo("  Out:")
    for p in profile.mix_out_points:
        click.echo(f"    {p.time:>6.1f}s  {p.type:<12} conf={p.confidence:.1f}  dur={p.duration:.1f}s")
    click.echo("  Loops:")
    for p in profile.loop_candidates:
        click.echo(f"    {p.time:>6.1f}s  {p.type:<12} conf={p.confidence:.1f}  dur={p.duration:.1f}s")
    click.echo()

    if profile.classification:
        clf = profile.classification
        click.echo("  ML CLASSIFICATION")
        click.echo("  Genres:")
        for genre, conf in clf.genres[:10]:
            bar = "█" * int(conf * 40)
            click.echo(f"    {genre:<40} {bar} {conf:.3f}")
        click.echo()
        click.echo(f"  Mood happy:      {clf.mood_happy:.3f}")
        click.echo(f"  Mood relaxed:    {clf.mood_relaxed:.3f}")
        click.echo(f"  Mood aggressive: {clf.mood_aggressive:.3f}")
        click.echo(f"  Mood sad:        {clf.mood_sad:.3f}")
        click.echo(f"  Danceability:    {clf.danceability:.3f}")
        click.echo(f"  Voice/Instr:     {clf.voice_instrumental:.3f}")
        click.echo(f"  Embedding dim:   {len(clf.embedding)}")


@cli.command()
@click.argument("query")
@click.option("--max-results", default=5, type=int, help="Number of results.")
def search(query: str, max_results: int):
    """Search for tracks on YouTube."""
    from .sources.youtube import search_tracks
    click.echo(f"Searching: {query}\n")
    results = search_tracks(query, max_results)
    if not results:
        click.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        dur = f"{r.duration // 60}:{r.duration % 60:02d}" if r.duration else "?"
        click.echo(f"  {i}. {r.title}")
        click.echo(f"     {r.artist} | {dur} | {r.url}")


@cli.command(name="add")
@click.argument("queries", nargs=-1, required=True)
@click.option("--db", default="tracks.db", help="Path to track database.")
@click.option("--output-dir", default="samples", help="Audio output directory.")
@click.option("--no-classify", is_flag=True, help="Skip ML classification.")
def add_tracks(queries: tuple[str, ...], db: str, output_dir: str, no_classify: bool):
    """Download tracks from YouTube, analyze, and add to library.

    Pass song names or YouTube URLs as arguments.

    Examples:
        agent-dj add "Daft Punk - Around the World"
        agent-dj add "Norah Jones Don't Know Why" "Erykah Badu Bag Lady"
    """
    from .sources.youtube import download_track

    store = TrackStore(db)

    for i, query in enumerate(queries, 1):
        click.echo(f"\n[{i}/{len(queries)}] {query}")

        # Download
        click.echo("  Downloading...")
        dl = download_track(query, output_dir)
        if not dl:
            click.echo("  FAILED to download.")
            continue
        click.echo(f"  -> {dl.title} by {dl.artist} ({dl.duration}s)")

        # Analyze
        click.echo("  Analyzing...")
        try:
            profile = analyze_track(dl.file_path, run_classifier=not no_classify)
            store.save(profile)
            _print_profile_summary(profile)
        except Exception as e:
            click.echo(f"  ERROR analyzing: {e}")
            continue

    click.echo(f"\nLibrary now has {store.count()} tracks.")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host.")
@click.option("--port", default=8000, type=int, help="Server port.")
def serve(host: str, port: int):
    """Start the Agent DJ backend server."""
    from .api.app import run_server
    click.echo(f"Starting Agent DJ server on {host}:{port}")
    click.echo(f"API docs: http://localhost:{port}/docs")
    run_server(host=host, port=port)


if __name__ == "__main__":
    cli()
