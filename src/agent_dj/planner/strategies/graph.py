"""Strategy B: Similarity Graph + Path Optimization.

Build a weighted graph where tracks are nodes and edge weights represent
transition quality. Find the best path through the graph that follows
the target energy arc.
"""

from dataclasses import dataclass

import numpy as np

from ...analyzer.audio import TrackProfile
from ...analyzer.camelot import camelot_distance
from ..vibe_profile import VibeProfile


@dataclass
class Edge:
    """A directed edge between two tracks in the transition graph."""
    from_track: TrackProfile
    to_track: TrackProfile
    weight: float       # 0-1, higher = better transition
    bpm_diff: float
    key_distance: int
    embedding_similarity: float
    energy_diff: float


def compute_edge_weight(
    track_a: TrackProfile,
    track_b: TrackProfile,
) -> Edge:
    """Compute transition quality between two tracks.

    Higher weight = better transition.
    """
    # BPM proximity (penalize >8 BPM difference)
    bpm_diff = abs(track_a.bpm - track_b.bpm)
    bpm_score = max(0, 1.0 - bpm_diff / 12.0)

    # Key compatibility
    key_dist = camelot_distance(track_a.key, track_b.key)
    if key_dist == 0:
        key_score = 1.0
    elif key_dist == 1:
        key_score = 0.85
    elif key_dist == 2:
        key_score = 0.5
    else:
        key_score = max(0, 1.0 - key_dist * 0.15)

    # Embedding similarity (timbral compatibility)
    emb_sim = 0.5
    if (track_a.classification and track_b.classification and
            track_a.classification.embedding and track_b.classification.embedding):
        emb_a = np.array(track_a.classification.embedding)
        emb_b = np.array(track_b.classification.embedding)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a > 0 and norm_b > 0:
            emb_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
            emb_sim = (emb_sim + 1) / 2  # normalize from [-1,1] to [0,1]

    # Energy continuity (smooth changes preferred)
    energy_a = float(np.mean(track_a.energy_curve)) if track_a.energy_curve else 0.5
    energy_b = float(np.mean(track_b.energy_curve)) if track_b.energy_curve else 0.5
    energy_diff = abs(energy_a - energy_b)
    energy_score = 1.0 - energy_diff

    # Weighted combination
    weight = (
        0.30 * bpm_score +
        0.25 * key_score +
        0.25 * emb_sim +
        0.20 * energy_score
    )

    return Edge(
        from_track=track_a,
        to_track=track_b,
        weight=round(weight, 4),
        bpm_diff=round(bpm_diff, 1),
        key_distance=key_dist,
        embedding_similarity=round(emb_sim, 4),
        energy_diff=round(energy_diff, 4),
    )


def build_transition_graph(tracks: list[TrackProfile]) -> dict[str, dict[str, Edge]]:
    """Build a complete directed graph of transition quality between all track pairs.

    Returns:
        Nested dict: graph[track_a.file_path][track_b.file_path] = Edge
    """
    graph: dict[str, dict[str, Edge]] = {}

    for a in tracks:
        graph[a.file_path] = {}
        for b in tracks:
            if a.file_path == b.file_path:
                continue
            graph[a.file_path][b.file_path] = compute_edge_weight(a, b)

    return graph


def select_tracks_graph_greedy(
    tracks: list[TrackProfile],
    vibe: VibeProfile,
    n_tracks: int | None = None,
) -> list[TrackProfile]:
    """Select tracks using greedy path through the transition graph.

    At each step, pick the neighbor with best combined score of:
    - Edge weight (transition quality)
    - Energy match to target arc

    Args:
        tracks: Available tracks.
        vibe: Target vibe profile.
        n_tracks: Number of tracks to select.

    Returns:
        Ordered list of selected tracks.
    """
    if not tracks:
        return []

    if n_tracks is None:
        avg_duration = np.mean([t.duration for t in tracks])
        n_tracks = max(1, int((vibe.duration_minutes * 60) / avg_duration))
    n_tracks = min(n_tracks, len(tracks))

    graph = build_transition_graph(tracks)

    # Pick starting track: best energy match to arc start + genre match
    target_start_energy = vibe.get_target_energy(0.0)
    start_scores = []
    for t in tracks:
        t_energy = float(np.mean(t.energy_curve)) if t.energy_curve else 0.5
        energy_match = 1.0 - abs(t_energy - target_start_energy)
        start_scores.append((t, energy_match))
    start_scores.sort(key=lambda x: -x[1])
    start_track = start_scores[0][0]

    selected = [start_track]
    visited = {start_track.file_path}

    for i in range(1, n_tracks):
        position = i / max(n_tracks - 1, 1)
        target_energy = vibe.get_target_energy(position)
        current = selected[-1]

        # Score unvisited neighbors
        best_score = -1
        best_track = None

        for t in tracks:
            if t.file_path in visited:
                continue

            edge = graph[current.file_path][t.file_path]
            t_energy = float(np.mean(t.energy_curve)) if t.energy_curve else 0.5
            energy_match = 1.0 - abs(t_energy - target_energy)

            combined = 0.6 * edge.weight + 0.4 * energy_match

            if combined > best_score:
                best_score = combined
                best_track = t

        if best_track is None:
            break

        selected.append(best_track)
        visited.add(best_track.file_path)

    return selected


def select_tracks_beam_search(
    tracks: list[TrackProfile],
    vibe: VibeProfile,
    n_tracks: int | None = None,
    beam_width: int = 5,
) -> list[TrackProfile]:
    """Select tracks using beam search through the transition graph.

    Maintains top-k paths at each step instead of just the single best.

    Args:
        tracks: Available tracks.
        vibe: Target vibe profile.
        n_tracks: Number of tracks.
        beam_width: Number of paths to maintain.

    Returns:
        Ordered list of selected tracks (best path).
    """
    if not tracks:
        return []

    if n_tracks is None:
        avg_duration = np.mean([t.duration for t in tracks])
        n_tracks = max(1, int((vibe.duration_minutes * 60) / avg_duration))
    n_tracks = min(n_tracks, len(tracks))

    graph = build_transition_graph(tracks)

    # Initialize beams with each possible starting track
    target_start_energy = vibe.get_target_energy(0.0)
    beams: list[tuple[float, list[TrackProfile]]] = []
    for t in tracks:
        t_energy = float(np.mean(t.energy_curve)) if t.energy_curve else 0.5
        score = 1.0 - abs(t_energy - target_start_energy)
        beams.append((score, [t]))

    beams.sort(key=lambda x: -x[0])
    beams = beams[:beam_width]

    for step in range(1, n_tracks):
        position = step / max(n_tracks - 1, 1)
        target_energy = vibe.get_target_energy(position)
        new_beams: list[tuple[float, list[TrackProfile]]] = []

        for beam_score, path in beams:
            current = path[-1]
            visited = {t.file_path for t in path}

            for t in tracks:
                if t.file_path in visited:
                    continue

                edge = graph[current.file_path][t.file_path]
                t_energy = float(np.mean(t.energy_curve)) if t.energy_curve else 0.5
                energy_match = 1.0 - abs(t_energy - target_energy)
                step_score = 0.6 * edge.weight + 0.4 * energy_match

                new_beams.append((beam_score + step_score, path + [t]))

        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:beam_width]

        if not beams:
            break

    if beams:
        return beams[0][1]
    return []
