"""Strategy A: Feature-Distance Scoring (Baseline).

Greedy approach — for each position in the set, score all candidate tracks
against the target vibe and previous track, then pick the best.
"""

from dataclasses import dataclass

import numpy as np

from ...analyzer.audio import TrackProfile
from ...analyzer.camelot import camelot_distance
from ...analyzer.genre_taxonomy import DJGenreProfile, map_to_dj_genres
from ..vibe_profile import VibeProfile


@dataclass
class TrackScore:
    """Detailed score breakdown for a candidate track."""
    track: TrackProfile
    total: float
    energy_score: float
    bpm_score: float
    key_score: float
    genre_score: float
    mood_score: float
    transition_score: float


def score_track(
    candidate: TrackProfile,
    vibe: VibeProfile,
    target_energy: float,
    prev_track: TrackProfile | None = None,
    weights: dict[str, float] | None = None,
) -> TrackScore:
    """Score a candidate track for a position in the set.

    Args:
        candidate: Track to evaluate.
        vibe: Target vibe profile.
        target_energy: Target energy level at this position (0-1).
        prev_track: Previous track in the set (None for first track).
        weights: Custom scoring weights. Defaults to balanced weights.

    Returns:
        TrackScore with total and per-dimension breakdown.
    """
    w = weights or {
        "energy": 0.25,
        "bpm": 0.20,
        "key": 0.15,
        "genre": 0.20,
        "mood": 0.10,
        "transition": 0.10,
    }

    # -- Energy match --
    track_energy = float(np.mean(candidate.energy_curve)) if candidate.energy_curve else 0.5
    energy_score = 1.0 - abs(track_energy - target_energy)

    # -- BPM fit --
    bpm_low, bpm_high = vibe.bpm_range
    bpm_range_width = max(bpm_high - bpm_low, 1)
    if bpm_low <= candidate.bpm <= bpm_high:
        bpm_score = 1.0
    else:
        distance = min(abs(candidate.bpm - bpm_low), abs(candidate.bpm - bpm_high))
        bpm_score = max(0, 1.0 - distance / bpm_range_width)

    # If we have a previous track, also penalize large BPM jumps
    if prev_track:
        bpm_diff = abs(candidate.bpm - prev_track.bpm)
        bpm_continuity = max(0, 1.0 - bpm_diff / 10.0)  # Penalize >10 BPM jump
        bpm_score = 0.5 * bpm_score + 0.5 * bpm_continuity

    # -- Key compatibility --
    key_score = 0.5  # default if no previous track
    if prev_track:
        dist = camelot_distance(prev_track.key, candidate.key)
        if dist == 0:
            key_score = 1.0
        elif dist == 1:
            key_score = 0.8
        elif dist == 2:
            key_score = 0.5
        else:
            key_score = max(0, 1.0 - dist * 0.15)

    # -- Genre match --
    genre_score = 0.0
    if candidate.classification and candidate.classification.genres:
        dj_profile = map_to_dj_genres(candidate.classification.genres)
        # Score = sum of (track genre weight * vibe genre weight) for matching genres
        for genre, track_weight in dj_profile.genres.items():
            if genre in vibe.genres:
                genre_score += track_weight * vibe.genres[genre]
        # Normalize to 0-1
        max_possible = sum(vibe.genres.values())
        if max_possible > 0:
            genre_score = min(1.0, genre_score / max_possible)

    # -- Mood match --
    mood_score = 0.5
    if candidate.classification:
        clf = candidate.classification
        mood_diffs = [
            abs(clf.mood_happy - vibe.mood_happy),
            abs(clf.mood_aggressive - vibe.mood_aggressive),
            abs(clf.mood_relaxed - vibe.mood_relaxed),
            abs(clf.mood_sad - vibe.mood_sad),
        ]
        mood_score = 1.0 - (sum(mood_diffs) / len(mood_diffs))

    # -- Transition feasibility --
    transition_score = 0.5
    if prev_track and candidate.mix_in_points:
        best_in = max(candidate.mix_in_points, key=lambda p: p.confidence)
        transition_score = best_in.confidence
        if prev_track.mix_out_points:
            best_out = max(prev_track.mix_out_points, key=lambda p: p.confidence)
            transition_score = (transition_score + best_out.confidence) / 2

    # -- Total --
    total = (
        w["energy"] * energy_score +
        w["bpm"] * bpm_score +
        w["key"] * key_score +
        w["genre"] * genre_score +
        w["mood"] * mood_score +
        w["transition"] * transition_score
    )

    return TrackScore(
        track=candidate,
        total=round(total, 4),
        energy_score=round(energy_score, 4),
        bpm_score=round(bpm_score, 4),
        key_score=round(key_score, 4),
        genre_score=round(genre_score, 4),
        mood_score=round(mood_score, 4),
        transition_score=round(transition_score, 4),
    )


def select_tracks_greedy(
    tracks: list[TrackProfile],
    vibe: VibeProfile,
    n_tracks: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[TrackScore]:
    """Select and order tracks using greedy scoring.

    For each position, picks the highest-scoring remaining track.

    Args:
        tracks: Available tracks to choose from.
        vibe: Target vibe profile.
        n_tracks: Number of tracks to select. If None, estimated from duration.
        weights: Custom scoring weights.

    Returns:
        Ordered list of TrackScore objects.
    """
    if not tracks:
        return []

    # Estimate number of tracks needed
    if n_tracks is None:
        avg_duration = np.mean([t.duration for t in tracks])
        n_tracks = max(1, int((vibe.duration_minutes * 60) / avg_duration))
    n_tracks = min(n_tracks, len(tracks))

    remaining = list(tracks)
    selected: list[TrackScore] = []

    for i in range(n_tracks):
        position = i / max(n_tracks - 1, 1)
        target_energy = vibe.get_target_energy(position)
        prev = selected[-1].track if selected else None

        # Score all remaining tracks
        scores = [
            score_track(t, vibe, target_energy, prev, weights)
            for t in remaining
        ]
        scores.sort(key=lambda s: -s.total)

        best = scores[0]
        selected.append(best)
        remaining.remove(best.track)

    return selected
