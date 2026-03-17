"""Energy tracking — monitor actual vs target energy and compensate for drift."""

import numpy as np

from ..analyzer.audio import TrackProfile
from .vibe_profile import VibeProfile


class EnergyTracker:
    """Tracks actual energy vs target energy arc and adjusts targets to compensate."""

    def __init__(self, vibe: VibeProfile):
        self.vibe = vibe
        self.played_energies: list[float] = []
        self.target_energies: list[float] = []

    def record(self, track: TrackProfile, position_in_set: float):
        """Record the energy of a played track."""
        actual = float(np.mean(track.energy_curve)) if track.energy_curve else 0.5
        target = self.vibe.get_target_energy(position_in_set)
        self.played_energies.append(actual)
        self.target_energies.append(target)

    @property
    def energy_deficit(self) -> float:
        """Positive = we're behind the arc (too chill). Negative = ahead (too intense)."""
        if not self.played_energies:
            return 0.0
        window = min(3, len(self.played_energies))
        recent_actual = float(np.mean(self.played_energies[-window:]))
        recent_target = float(np.mean(self.target_energies[-window:]))
        return recent_target - recent_actual

    def adjust_next_target(self, base_target: float) -> float:
        """Adjust the next track's target energy to compensate for drift."""
        deficit = self.energy_deficit
        adjusted = base_target + deficit * 0.5
        return float(np.clip(adjusted, 0.1, 1.0))

    @property
    def tracks_played(self) -> int:
        return len(self.played_energies)


def apply_energy_feedback(vibe: VibeProfile, direction: str, current_position: float):
    """Shift the remaining energy arc based on crowd feedback.

    Args:
        vibe: Current vibe profile (modified in place).
        direction: "up" or "down".
        current_position: Current position in the set (0-1).
    """
    shift = 0.15 if direction == "up" else -0.15

    for i in range(len(vibe.energy_arc)):
        pos = i / max(len(vibe.energy_arc) - 1, 1)
        if pos > current_position:
            vibe.energy_arc[i] = float(np.clip(vibe.energy_arc[i] + shift, 0.1, 1.0))
