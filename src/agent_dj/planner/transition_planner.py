"""Transition planning between tracks.

Given two adjacent tracks in a set, determine the optimal transition:
type, timing, BPM adjustment, and parameters.
"""

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from ..analyzer.audio import TrackProfile
from ..analyzer.camelot import camelot_distance


class TransitionType(str, Enum):
    """Available transition types."""
    CROSSFADE = "crossfade"          # Beat-aligned volume crossfade
    BASS_SWAP = "bass_swap"          # EQ transition — swap bass frequencies
    LOOP_BLEND = "loop_blend"        # Loop outgoing track, blend incoming
    HARD_CUT = "hard_cut"            # Instant switch on downbeat
    ECHO_OUT = "echo_out"            # Reverb/echo tail on outgoing, clean entry
    BREAKDOWN_BLEND = "breakdown"    # Use a breakdown as the transition point


@dataclass
class TransitionPlan:
    """Complete plan for transitioning between two tracks."""
    from_track: str          # file path
    to_track: str            # file path
    type: TransitionType
    duration_beats: int      # transition duration in beats
    duration_seconds: float

    # Timing
    mix_out_time: float      # seconds into track A to start mixing out
    mix_in_time: float       # seconds into track B to start mixing in

    # BPM
    from_bpm: float
    to_bpm: float
    target_bpm: float        # the BPM both tracks should match at
    bpm_adjust_track: str    # "from", "to", or "both"

    # Gain
    from_gain_db: float      # loudness normalization for track A
    to_gain_db: float        # loudness normalization for track B

    # Loop (if applicable)
    loop_start: float | None = None
    loop_end: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


def plan_transition(
    track_a: TrackProfile,
    track_b: TrackProfile,
    energy_direction: str = "sustain",  # "rising", "falling", "sustain"
) -> TransitionPlan:
    """Plan the optimal transition between two tracks.

    Considers BPM difference, key compatibility, energy context, and
    available mix points to choose the best transition type and timing.

    Args:
        track_a: Outgoing track.
        track_b: Incoming track.
        energy_direction: Whether energy should be rising, falling, or sustaining.

    Returns:
        TransitionPlan with all details needed to execute the transition.
    """
    bpm_diff = abs(track_a.bpm - track_b.bpm)
    key_dist = camelot_distance(track_a.key, track_b.key)

    # -- Choose transition type --
    trans_type = _choose_transition_type(bpm_diff, key_dist, energy_direction, track_a, track_b)

    # -- Duration --
    duration_beats = _get_transition_duration(trans_type, track_a.bpm)

    # -- Mix points --
    mix_out_time = _get_mix_out_time(track_a)
    mix_in_time = _get_mix_in_time(track_b)

    # -- BPM matching --
    target_bpm, adjust_track = _plan_bpm_match(track_a.bpm, track_b.bpm)

    # -- Gain normalization --
    target_lufs = -14.0  # standard loudness target
    from_gain = target_lufs - track_a.loudness_lufs
    to_gain = target_lufs - track_b.loudness_lufs

    # -- Duration in seconds --
    duration_seconds = round(duration_beats * (60.0 / target_bpm), 2)

    # -- Loop points (for loop_blend) --
    loop_start = None
    loop_end = None
    if trans_type == TransitionType.LOOP_BLEND and track_a.loop_candidates:
        best_loop = max(track_a.loop_candidates, key=lambda p: p.confidence)
        loop_start = best_loop.time
        loop_end = best_loop.time + best_loop.duration

    return TransitionPlan(
        from_track=track_a.file_path,
        to_track=track_b.file_path,
        type=trans_type,
        duration_beats=duration_beats,
        duration_seconds=duration_seconds,
        mix_out_time=mix_out_time,
        mix_in_time=mix_in_time,
        from_bpm=track_a.bpm,
        to_bpm=track_b.bpm,
        target_bpm=target_bpm,
        bpm_adjust_track=adjust_track,
        from_gain_db=round(from_gain, 1),
        to_gain_db=round(to_gain, 1),
        loop_start=loop_start,
        loop_end=loop_end,
    )


def _choose_transition_type(
    bpm_diff: float,
    key_dist: int,
    energy_direction: str,
    track_a: TrackProfile,
    track_b: TrackProfile,
) -> TransitionType:
    """Choose the best transition type based on context."""
    # Hard cut: large BPM difference or energy direction calls for it
    if bpm_diff > 10:
        return TransitionType.HARD_CUT

    # Echo out: energy is dropping significantly
    if energy_direction == "falling":
        return TransitionType.ECHO_OUT

    # Bass swap: small BPM diff, compatible keys, both tracks have good energy
    if bpm_diff < 5 and key_dist <= 1:
        a_energy = float(np.mean(track_a.energy_curve)) if track_a.energy_curve else 0.5
        b_energy = float(np.mean(track_b.energy_curve)) if track_b.energy_curve else 0.5
        if a_energy > 0.5 and b_energy > 0.5:
            return TransitionType.BASS_SWAP

    # Loop blend: outgoing track has good loop candidates
    if track_a.loop_candidates and bpm_diff < 8:
        best_loop = max(track_a.loop_candidates, key=lambda p: p.confidence)
        if best_loop.confidence > 0.5:
            return TransitionType.LOOP_BLEND

    # Breakdown blend: incoming track starts with a breakdown
    if track_b.segments:
        first_seg = track_b.segments[0]
        if first_seg.label in ("intro", "breakdown") and first_seg.energy < 0.3:
            return TransitionType.BREAKDOWN_BLEND

    # Default: beatmatched crossfade
    return TransitionType.CROSSFADE


def _get_transition_duration(trans_type: TransitionType, bpm: float) -> int:
    """Get transition duration in beats based on type."""
    if trans_type == TransitionType.HARD_CUT:
        return 1
    elif trans_type == TransitionType.ECHO_OUT:
        return 16
    elif trans_type == TransitionType.BASS_SWAP:
        return 32
    elif trans_type == TransitionType.LOOP_BLEND:
        return 32
    elif trans_type == TransitionType.BREAKDOWN_BLEND:
        return 16
    else:  # CROSSFADE
        return 16


def _get_mix_out_time(track: TrackProfile) -> float:
    """Get the best time to start mixing out of a track."""
    if track.mix_out_points:
        best = max(track.mix_out_points, key=lambda p: p.confidence)
        return best.time
    # Fallback: 30 seconds before the end
    return max(0, track.duration - 30)


def _get_mix_in_time(track: TrackProfile) -> float:
    """Get the best time to start mixing into a track."""
    if track.mix_in_points:
        best = max(track.mix_in_points, key=lambda p: p.confidence)
        return best.time
    return 0.0


def _plan_bpm_match(bpm_a: float, bpm_b: float) -> tuple[float, str]:
    """Decide target BPM and which track to adjust.

    Strategy: adjust the track that needs less change.
    If diff is very small (<2 BPM), match to the incoming track.
    """
    diff = abs(bpm_a - bpm_b)

    if diff < 2:
        # Minor difference — match to incoming
        return bpm_b, "from"
    elif diff < 5:
        # Meet in the middle
        return round((bpm_a + bpm_b) / 2, 1), "both"
    else:
        # Large diff — adjust the one with less absolute change needed
        mid = (bpm_a + bpm_b) / 2
        if abs(bpm_a - mid) < abs(bpm_b - mid):
            return round(mid, 1), "both"
        else:
            return bpm_b, "from"
