"""VibeProfile — structured representation of the desired DJ set vibe.

Created from the onboarding chat (via LLM) or manually for testing.
Used by the track selector and set planner to drive decisions.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np


class EnergyArcType(str, Enum):
    """Preset energy arc shapes."""
    STEADY = "steady"                # Constant energy throughout
    SLOW_BUILD = "slow_build"        # Gradually increase, peak at end
    BUILD_AND_DROP = "build_and_drop" # Build to peak in middle, wind down
    WAVES = "waves"                  # Oscillate between high and low
    PEAK_START = "peak_start"        # Start hot, gradually cool
    CUSTOM = "custom"                # User-defined curve


@dataclass
class VibeProfile:
    """Describes the desired vibe for a DJ set."""
    # Genre preferences (DJ taxonomy genre names → weight 0-1)
    genres: dict[str, float] = field(default_factory=dict)

    # Tempo range
    bpm_range: tuple[float, float] = (90, 130)

    # Energy arc: list of floats 0-1 representing target energy over time
    # Length determines granularity (e.g., 10 values = 10 checkpoints)
    energy_arc: list[float] = field(default_factory=lambda: [0.5])
    energy_arc_type: EnergyArcType = EnergyArcType.BUILD_AND_DROP

    # Mood targets (0-1)
    mood_happy: float = 0.5
    mood_aggressive: float = 0.3
    mood_relaxed: float = 0.5
    mood_sad: float = 0.1

    # Danceability preference (0-1)
    danceability_min: float = 0.3

    # Mood tags (free-form descriptors)
    mood_tags: list[str] = field(default_factory=list)

    # Duration in minutes
    duration_minutes: int = 60

    # Transition style preference
    transition_style: str = "smooth"  # "smooth", "energetic", "mixed"

    # Genres/artists to avoid
    avoid_genres: list[str] = field(default_factory=list)

    # Example tracks the user mentioned (titles or file paths)
    example_tracks: list[str] = field(default_factory=list)

    def get_target_energy(self, position: float) -> float:
        """Get target energy at a given position in the set.

        Args:
            position: 0.0 (start) to 1.0 (end)

        Returns:
            Target energy level (0-1).
        """
        if not self.energy_arc or len(self.energy_arc) == 1:
            return self.energy_arc[0] if self.energy_arc else 0.5

        # Interpolate between arc points
        idx_float = position * (len(self.energy_arc) - 1)
        idx_low = int(idx_float)
        idx_high = min(idx_low + 1, len(self.energy_arc) - 1)
        frac = idx_float - idx_low

        return self.energy_arc[idx_low] * (1 - frac) + self.energy_arc[idx_high] * frac

    def to_dict(self) -> dict:
        d = asdict(self)
        d["energy_arc_type"] = self.energy_arc_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "VibeProfile":
        data["energy_arc_type"] = EnergyArcType(data.get("energy_arc_type", "build_and_drop"))
        data["bpm_range"] = tuple(data.get("bpm_range", (90, 130)))
        return cls(**data)


def generate_energy_arc(arc_type: EnergyArcType, n_points: int = 20) -> list[float]:
    """Generate an energy arc curve from a preset type.

    Args:
        arc_type: The shape of the energy curve.
        n_points: Number of points in the curve.

    Returns:
        List of energy values (0-1).
    """
    t = np.linspace(0, 1, n_points)

    if arc_type == EnergyArcType.STEADY:
        curve = np.ones(n_points) * 0.6

    elif arc_type == EnergyArcType.SLOW_BUILD:
        curve = 0.3 + 0.6 * t**1.5

    elif arc_type == EnergyArcType.BUILD_AND_DROP:
        # Bell curve peaking at ~60% through
        peak_pos = 0.6
        curve = np.exp(-((t - peak_pos) ** 2) / (2 * 0.15**2))
        curve = 0.2 + 0.7 * curve / curve.max()

    elif arc_type == EnergyArcType.WAVES:
        curve = 0.5 + 0.3 * np.sin(2 * np.pi * 2.5 * t)

    elif arc_type == EnergyArcType.PEAK_START:
        curve = 0.9 - 0.5 * t**1.2

    elif arc_type == EnergyArcType.CUSTOM:
        curve = np.ones(n_points) * 0.5

    else:
        curve = np.ones(n_points) * 0.5

    return [round(float(v), 3) for v in np.clip(curve, 0.1, 1.0)]


# ── Preset vibe profiles for testing ──

def chill_jazz_party(duration: int = 45) -> VibeProfile:
    """Chill jazz/RnB house party preset."""
    return VibeProfile(
        genres={"Jazz": 0.3, "Bossa Nova / Latin Jazz": 0.2, "RnB / Neo-Soul": 0.3,
                "Soul": 0.1, "Smooth / Contemporary Jazz": 0.1},
        bpm_range=(80, 115),
        energy_arc=generate_energy_arc(EnergyArcType.BUILD_AND_DROP),
        energy_arc_type=EnergyArcType.BUILD_AND_DROP,
        mood_happy=0.6, mood_relaxed=0.7, mood_aggressive=0.1, mood_sad=0.1,
        danceability_min=0.2,
        mood_tags=["smooth", "groovy", "warm", "sophisticated"],
        duration_minutes=duration,
        transition_style="smooth",
    )


def house_party(duration: int = 120) -> VibeProfile:
    """High-energy house/electronic party preset."""
    return VibeProfile(
        genres={"House": 0.4, "Disco / Nu-Disco": 0.2, "Techno": 0.15,
                "Funk": 0.1, "RnB / Neo-Soul": 0.1, "Pop": 0.05},
        bpm_range=(118, 130),
        energy_arc=generate_energy_arc(EnergyArcType.BUILD_AND_DROP),
        energy_arc_type=EnergyArcType.BUILD_AND_DROP,
        mood_happy=0.8, mood_relaxed=0.2, mood_aggressive=0.3, mood_sad=0.0,
        danceability_min=0.5,
        mood_tags=["energetic", "danceable", "uplifting", "fun"],
        duration_minutes=duration,
        transition_style="energetic",
    )


def dinner_ambient(duration: int = 90) -> VibeProfile:
    """Quiet dinner background music preset."""
    return VibeProfile(
        genres={"Bossa Nova / Latin Jazz": 0.3, "Smooth / Contemporary Jazz": 0.3,
                "Ambient / Downtempo": 0.2, "Soul": 0.1, "Jazz": 0.1},
        bpm_range=(70, 110),
        energy_arc=generate_energy_arc(EnergyArcType.STEADY),
        energy_arc_type=EnergyArcType.STEADY,
        mood_happy=0.4, mood_relaxed=0.9, mood_aggressive=0.0, mood_sad=0.1,
        danceability_min=0.0,
        mood_tags=["calm", "elegant", "background", "warm"],
        duration_minutes=duration,
        transition_style="smooth",
    )
