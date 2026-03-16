"""Track structure detection using self-similarity and novelty.

Detects structural sections (intro, verse, chorus, breakdown, outro)
by analyzing the self-similarity matrix and detecting novelty peaks
at structural boundaries.
"""

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass
class Section:
    """A structural section of a track."""
    start: float         # seconds
    end: float           # seconds
    label: str           # 'intro', 'verse', 'chorus', 'breakdown', 'drop', 'outro'
    energy: float        # average energy of this section (0-1)


@dataclass
class MixPoint:
    """A candidate point for mixing in or out of a track."""
    time: float          # seconds
    type: str            # 'intro', 'outro', 'breakdown', 'drop', 'loop'
    confidence: float    # 0-1, how good this point is for mixing
    duration: float      # how long this section lasts


def detect_structure(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    min_section_bars: int = 4,
) -> list[Section]:
    """Detect track structure using self-similarity matrix + novelty detection.

    1. Compute chroma features synced to beats
    2. Build a self-similarity (recurrence) matrix
    3. Apply a checkerboard kernel to find novelty peaks (structural boundaries)
    4. Label sections based on energy, repetition, and position

    Args:
        y: Audio signal (mono)
        sr: Sample rate
        beat_times: Beat positions in seconds (from beat tracking)
        min_section_bars: Minimum section length in bars (4/4 assumed)

    Returns:
        List of Section objects ordered by start time.
    """
    if len(beat_times) < 16:
        # Too short for meaningful structure detection
        duration = librosa.get_duration(y=y, sr=sr)
        rms = float(np.sqrt(np.mean(y**2)))
        return [Section(start=0.0, end=duration, label="body", energy=min(rms * 5, 1.0))]

    # --- Step 1: Beat-synchronous chroma features ---
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

    # --- Step 2: Self-similarity matrix ---
    # Use cosine similarity on chroma for harmonic structure
    chroma_norm = librosa.util.normalize(chroma_sync, axis=0)
    sim_matrix = chroma_norm.T @ chroma_norm

    # --- Step 3: Novelty detection with checkerboard kernel ---
    novelty = _checkerboard_novelty(sim_matrix, kernel_size=16)

    # Smooth novelty curve to reduce noise
    novelty = uniform_filter1d(novelty, size=4)

    # --- Step 4: Find boundary peaks ---
    min_beats_between = min_section_bars * 4  # 4 beats per bar
    boundaries = _find_boundaries(novelty, min_distance=min_beats_between)

    # Add start and end
    boundaries = [0] + boundaries + [len(beat_times) - 1]

    # --- Step 5: Compute energy per section ---
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_sync = librosa.util.sync(rms.reshape(1, -1), beat_frames, aggregate=np.mean)[0]

    # --- Step 6: Label sections ---
    sections = _label_sections(boundaries, beat_times, rms_sync, y, sr)

    return sections


def _checkerboard_novelty(sim_matrix: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """Compute novelty curve using a checkerboard kernel on the similarity matrix.

    A checkerboard kernel highlights transitions where the local structure
    changes (e.g., from verse to chorus).
    """
    n = sim_matrix.shape[0]
    half = kernel_size // 2
    novelty = np.zeros(n)

    # Build checkerboard kernel
    kernel = np.ones((kernel_size, kernel_size))
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1

    for i in range(half, n - half):
        patch = sim_matrix[i - half:i + half, i - half:i + half]
        if patch.shape == kernel.shape:
            novelty[i] = np.sum(patch * kernel)

    # Normalize
    max_val = np.max(np.abs(novelty))
    if max_val > 0:
        novelty = novelty / max_val

    # Only positive values (boundaries show as positive peaks)
    novelty = np.maximum(novelty, 0)

    return novelty


def _find_boundaries(
    novelty: np.ndarray,
    min_distance: int = 16,
    threshold_ratio: float = 0.3,
) -> list[int]:
    """Find boundary positions as peaks in the novelty curve.

    Args:
        novelty: Novelty curve (one value per beat)
        min_distance: Minimum beats between boundaries
        threshold_ratio: Only keep peaks above this fraction of max novelty

    Returns:
        List of beat indices where boundaries occur.
    """
    threshold = np.max(novelty) * threshold_ratio
    peaks = []

    for i in range(1, len(novelty) - 1):
        if novelty[i] < threshold:
            continue
        # Local maximum check
        if novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
            peaks.append((i, novelty[i]))

    # Enforce minimum distance: keep highest peaks first
    peaks.sort(key=lambda x: -x[1])
    selected = []
    for idx, score in peaks:
        if all(abs(idx - s) >= min_distance for s in selected):
            selected.append(idx)

    return sorted(selected)


def _label_sections(
    boundaries: list[int],
    beat_times: np.ndarray,
    rms_sync: np.ndarray,
    y: np.ndarray,
    sr: int,
) -> list[Section]:
    """Label sections based on energy, position, and repetition patterns.

    Labeling heuristics:
    - First section with low energy → intro
    - Last section with low energy → outro
    - Lowest energy sections → breakdown
    - Highest energy sections → chorus/peak
    - Everything else → verse
    """
    duration = librosa.get_duration(y=y, sr=sr)
    n_sections = len(boundaries) - 1

    if n_sections <= 0:
        return [Section(start=0.0, end=duration, label="body", energy=0.5)]

    # Compute per-section energy
    section_energies = []
    for i in range(n_sections):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        seg_rms = rms_sync[start_idx:end_idx]
        energy = float(np.mean(seg_rms)) if len(seg_rms) > 0 else 0.0
        section_energies.append(energy)

    # Normalize energies to 0-1
    max_energy = max(section_energies) if section_energies else 1.0
    if max_energy > 0:
        section_energies = [e / max_energy for e in section_energies]

    avg_energy = sum(section_energies) / len(section_energies) if section_energies else 0.5

    sections = []
    for i in range(n_sections):
        start_beat = boundaries[i]
        end_beat = boundaries[i + 1]
        start_time = float(beat_times[start_beat]) if start_beat < len(beat_times) else 0.0
        end_time = float(beat_times[end_beat]) if end_beat < len(beat_times) else duration
        energy = section_energies[i]

        # Label based on position and energy
        if i == 0 and energy < avg_energy * 0.7:
            label = "intro"
        elif i == n_sections - 1 and energy < avg_energy * 0.7:
            label = "outro"
        elif energy < avg_energy * 0.4:
            label = "breakdown"
        elif energy > avg_energy * 1.3:
            label = "chorus"
        elif energy > avg_energy:
            label = "peak"
        else:
            label = "verse"

        sections.append(Section(
            start=round(start_time, 2),
            end=round(end_time, 2),
            label=label,
            energy=round(energy, 3),
        ))

    # Merge consecutive same-label sections
    merged = [sections[0]]
    for sec in sections[1:]:
        if sec.label == merged[-1].label:
            merged[-1] = Section(
                start=merged[-1].start,
                end=sec.end,
                label=sec.label,
                energy=round((merged[-1].energy + sec.energy) / 2, 3),
            )
        else:
            merged.append(sec)

    return merged


def find_mix_in_points(sections: list[Section], beat_times: np.ndarray) -> list[MixPoint]:
    """Find the best points to mix INTO this track."""
    points = []

    for sec in sections:
        if sec.label == "intro":
            points.append(MixPoint(
                time=sec.start,
                type="intro",
                confidence=0.9,
                duration=round(sec.end - sec.start, 2),
            ))
        elif sec.label == "breakdown":
            points.append(MixPoint(
                time=sec.start,
                type="breakdown",
                confidence=0.7,
                duration=round(sec.end - sec.start, 2),
            ))

    # Fallback: track start
    if not points:
        fallback_dur = float(beat_times[min(32, len(beat_times) - 1)]) if len(beat_times) > 1 else 8.0
        points.append(MixPoint(time=0.0, type="intro", confidence=0.5, duration=min(16.0, fallback_dur)))

    return sorted(points, key=lambda p: -p.confidence)


def find_mix_out_points(
    sections: list[Section],
    beat_times: np.ndarray,
    duration: float,
) -> list[MixPoint]:
    """Find the best points to mix OUT of this track."""
    points = []

    for sec in sections:
        if sec.label == "outro":
            points.append(MixPoint(
                time=sec.start,
                type="outro",
                confidence=0.9,
                duration=round(sec.end - sec.start, 2),
            ))
        elif sec.label == "breakdown" and sec.start > duration * 0.5:
            points.append(MixPoint(
                time=sec.start,
                type="breakdown",
                confidence=0.7,
                duration=round(sec.end - sec.start, 2),
            ))

    # Fallback: last 16 beats
    if not points:
        fallback_time = float(beat_times[-min(16, len(beat_times))]) if len(beat_times) > 1 else max(0, duration - 16)
        points.append(MixPoint(
            time=fallback_time,
            type="outro",
            confidence=0.5,
            duration=round(duration - fallback_time, 2),
        ))

    return sorted(points, key=lambda p: -p.confidence)


def find_loop_candidates(
    sections: list[Section],
    beat_times: np.ndarray,
) -> list[MixPoint]:
    """Find sections suitable for looping during transitions.

    Good candidates: breakdowns, intros, outros, verses with
    consistent energy and at least 4 bars duration.
    """
    candidates = []

    for sec in sections:
        if sec.label in ("breakdown", "intro", "outro", "verse"):
            seg_duration = sec.end - sec.start
            if seg_duration >= 4.0:  # at least ~4 bars
                conf = 0.7 if sec.label == "breakdown" else 0.5 if sec.label in ("intro", "outro") else 0.3
                candidates.append(MixPoint(
                    time=sec.start,
                    type="loop",
                    confidence=conf,
                    duration=round(min(seg_duration, 16.0), 2),
                ))

    return sorted(candidates, key=lambda p: -p.confidence)
