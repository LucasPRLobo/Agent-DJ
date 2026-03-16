"""Set Planner — orchestrates track selection and transition planning.

Takes a VibeProfile + track library → complete SetPlan with ordered tracks
and transition details. Supports multiple selection strategies and can
operate in full-set or rolling (2-3 ahead) mode.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from ..analyzer.audio import TrackProfile
from ..analyzer.track_store import TrackStore
from .transition_planner import TransitionPlan, plan_transition
from .vibe_profile import VibeProfile


class SelectionStrategy(str, Enum):
    GREEDY_SCORING = "greedy_scoring"
    GRAPH_GREEDY = "graph_greedy"
    GRAPH_BEAM = "graph_beam"
    CLUSTERING = "clustering"


@dataclass
class SetPlan:
    """Complete plan for a DJ set."""
    tracks: list[TrackProfile]
    transitions: list[TransitionPlan]
    vibe: VibeProfile
    strategy_used: str
    total_duration_seconds: float
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tracks": [
                {"file_path": t.file_path, "title": t.title, "bpm": t.bpm,
                 "key": t.key, "duration": t.duration}
                for t in self.tracks
            ],
            "transitions": [t.to_dict() for t in self.transitions],
            "vibe": self.vibe.to_dict(),
            "strategy_used": self.strategy_used,
            "total_duration_seconds": self.total_duration_seconds,
            "metrics": self.metrics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        """Human-readable summary of the set plan."""
        lines = [
            f"Set Plan ({self.strategy_used})",
            f"Duration: {self.total_duration_seconds / 60:.1f} min | Tracks: {len(self.tracks)}",
            f"BPM range: {min(t.bpm for t in self.tracks):.0f} - {max(t.bpm for t in self.tracks):.0f}",
            "",
        ]
        for i, track in enumerate(self.tracks):
            prefix = f"  {i + 1}."
            line = f"{prefix} {track.title:<35} {track.bpm:>6.1f} BPM | {track.key:>3}"
            if i < len(self.transitions):
                tr = self.transitions[i]
                line += f"  → [{tr.type.value}]"
            lines.append(line)

        if self.metrics:
            lines.append("")
            lines.append("Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)


def plan_set(
    tracks: list[TrackProfile],
    vibe: VibeProfile,
    strategy: SelectionStrategy = SelectionStrategy.GREEDY_SCORING,
    n_tracks: int | None = None,
) -> SetPlan:
    """Plan a complete DJ set.

    Args:
        tracks: Available tracks in the library.
        vibe: Target vibe profile.
        strategy: Which track selection strategy to use.
        n_tracks: Number of tracks. If None, estimated from duration.

    Returns:
        SetPlan with ordered tracks and transition plans.
    """
    # Filter tracks that are within a reasonable BPM range
    bpm_low, bpm_high = vibe.bpm_range
    bpm_margin = 15  # allow some flexibility beyond the range
    filtered = [
        t for t in tracks
        if bpm_low - bpm_margin <= t.bpm <= bpm_high + bpm_margin
    ]

    # If filtering is too aggressive, use all tracks
    if len(filtered) < 3:
        filtered = tracks

    # Select and order tracks using chosen strategy
    ordered = _select_tracks(filtered, vibe, strategy, n_tracks)

    if not ordered:
        return SetPlan(
            tracks=[], transitions=[], vibe=vibe,
            strategy_used=strategy.value,
            total_duration_seconds=0, metrics={},
        )

    # Plan transitions between each consecutive pair
    transitions = []
    for i in range(len(ordered) - 1):
        track_a = ordered[i]
        track_b = ordered[i + 1]

        # Determine energy direction
        position_a = i / max(len(ordered) - 1, 1)
        position_b = (i + 1) / max(len(ordered) - 1, 1)
        energy_a = vibe.get_target_energy(position_a)
        energy_b = vibe.get_target_energy(position_b)

        if energy_b > energy_a + 0.1:
            direction = "rising"
        elif energy_b < energy_a - 0.1:
            direction = "falling"
        else:
            direction = "sustain"

        transition = plan_transition(track_a, track_b, direction)
        transitions.append(transition)

    # Calculate total duration (approximate — doesn't account for overlaps)
    total_duration = sum(t.duration for t in ordered)

    # Calculate quality metrics
    metrics = _calculate_metrics(ordered, transitions, vibe)

    return SetPlan(
        tracks=ordered,
        transitions=transitions,
        vibe=vibe,
        strategy_used=strategy.value,
        total_duration_seconds=round(total_duration, 1),
        metrics=metrics,
    )


def plan_next_tracks(
    current_track: TrackProfile,
    played_tracks: list[TrackProfile],
    available_tracks: list[TrackProfile],
    vibe: VibeProfile,
    position: float,
    n_ahead: int = 3,
) -> SetPlan:
    """Rolling planner — plan the next N tracks from current position.

    Used during live playback. Plans ahead without committing to a full set.

    Args:
        current_track: Currently playing track.
        played_tracks: Tracks already played (excluded from selection).
        available_tracks: All available tracks.
        vibe: Current vibe profile (may have been updated mid-session).
        position: Current position in the set (0-1).
        n_ahead: Number of tracks to plan ahead.

    Returns:
        SetPlan with the next N tracks and their transitions.
    """
    played_paths = {t.file_path for t in played_tracks} | {current_track.file_path}
    remaining = [t for t in available_tracks if t.file_path not in played_paths]

    if not remaining:
        return SetPlan(
            tracks=[], transitions=[], vibe=vibe,
            strategy_used="rolling", total_duration_seconds=0, metrics={},
        )

    # Use greedy scoring from current position
    from .strategies.scoring import score_track

    selected = []
    prev = current_track
    candidates = list(remaining)

    for i in range(min(n_ahead, len(candidates))):
        step_position = min(1.0, position + (i + 1) * 0.05)
        target_energy = vibe.get_target_energy(step_position)

        scores = [
            score_track(t, vibe, target_energy, prev)
            for t in candidates
        ]
        scores.sort(key=lambda s: -s.total)

        best = scores[0]
        selected.append(best.track)
        candidates.remove(best.track)
        prev = best.track

    # Plan transitions
    transition_chain = [current_track] + selected
    transitions = []
    for i in range(len(transition_chain) - 1):
        pos = min(1.0, position + i * 0.05)
        next_pos = min(1.0, position + (i + 1) * 0.05)
        e_now = vibe.get_target_energy(pos)
        e_next = vibe.get_target_energy(next_pos)

        if e_next > e_now + 0.1:
            direction = "rising"
        elif e_next < e_now - 0.1:
            direction = "falling"
        else:
            direction = "sustain"

        transitions.append(plan_transition(
            transition_chain[i], transition_chain[i + 1], direction
        ))

    total_dur = sum(t.duration for t in selected)

    return SetPlan(
        tracks=selected,
        transitions=transitions,
        vibe=vibe,
        strategy_used="rolling",
        total_duration_seconds=round(total_dur, 1),
        metrics={},
    )


def _select_tracks(
    tracks: list[TrackProfile],
    vibe: VibeProfile,
    strategy: SelectionStrategy,
    n_tracks: int | None,
) -> list[TrackProfile]:
    """Dispatch to the selected track selection strategy."""
    if strategy == SelectionStrategy.GREEDY_SCORING:
        from .strategies.scoring import select_tracks_greedy
        results = select_tracks_greedy(tracks, vibe, n_tracks)
        return [r.track for r in results]

    elif strategy == SelectionStrategy.GRAPH_GREEDY:
        from .strategies.graph import select_tracks_graph_greedy
        return select_tracks_graph_greedy(tracks, vibe, n_tracks)

    elif strategy == SelectionStrategy.GRAPH_BEAM:
        from .strategies.graph import select_tracks_beam_search
        return select_tracks_beam_search(tracks, vibe, n_tracks)

    elif strategy == SelectionStrategy.CLUSTERING:
        from .strategies.clustering import select_tracks_clustering
        return select_tracks_clustering(tracks, vibe, n_tracks)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _calculate_metrics(
    tracks: list[TrackProfile],
    transitions: list[TransitionPlan],
    vibe: VibeProfile,
) -> dict:
    """Calculate quality metrics for a set plan."""
    if not tracks:
        return {}

    from ..analyzer.camelot import camelot_compatible

    # BPM smoothness
    bpm_diffs = [abs(tracks[i].bpm - tracks[i + 1].bpm) for i in range(len(tracks) - 1)]
    max_bpm_jump = max(bpm_diffs) if bpm_diffs else 0
    avg_bpm_diff = float(np.mean(bpm_diffs)) if bpm_diffs else 0

    # Key compatibility rate
    key_compatible = sum(
        1 for i in range(len(tracks) - 1)
        if camelot_compatible(tracks[i].key, tracks[i + 1].key)
    )
    key_compat_rate = key_compatible / max(len(tracks) - 1, 1)

    # Energy arc adherence
    target_energies = [
        vibe.get_target_energy(i / max(len(tracks) - 1, 1))
        for i in range(len(tracks))
    ]
    actual_energies = [
        float(np.mean(t.energy_curve)) if t.energy_curve else 0.5
        for t in tracks
    ]
    energy_mse = float(np.mean([(a - t) ** 2 for a, t in zip(actual_energies, target_energies)]))

    # Transition type distribution
    type_counts: dict[str, int] = {}
    for tr in transitions:
        type_counts[tr.type.value] = type_counts.get(tr.type.value, 0) + 1

    return {
        "max_bpm_jump": round(max_bpm_jump, 1),
        "avg_bpm_diff": round(avg_bpm_diff, 1),
        "key_compatibility_rate": round(key_compat_rate, 2),
        "energy_arc_mse": round(energy_mse, 4),
        "transition_types": type_counts,
    }
