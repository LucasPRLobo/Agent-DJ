"""Audio feature extraction pipeline for track analysis.

Extracts DJ-relevant features from audio files: BPM, key, energy curve,
beat grid, structural segments, mix points, loop candidates, spectral
features, and ML classification (genre, mood, embeddings).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import essentia.standard as es
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .camelot import key_to_camelot
from .classifier import ClassificationResult, classify_track
from .structure import (
    MixPoint,
    Section,
    detect_structure,
    find_loop_candidates,
    find_mix_in_points,
    find_mix_out_points,
)


@dataclass
class TrackProfile:
    """Complete DJ-relevant analysis of a track."""
    file_path: str
    title: str
    duration: float                     # seconds

    # Rhythm
    bpm: float
    beats: list[float]                  # beat times in seconds
    downbeats: list[float]              # downbeat (bar start) times

    # Harmony
    key: str                            # Camelot notation (e.g., '8A')
    key_name: str                       # Human-readable (e.g., 'A minor')

    # Energy & Loudness
    energy_curve: list[float]           # normalized energy per 2-bar segment
    loudness_lufs: float

    # Spectral features (mean values)
    spectral_centroid: float            # brightness
    spectral_bandwidth: float           # frequency spread
    mfcc: list[float]                   # 13-dim timbral fingerprint
    onset_strength_mean: float          # rhythmic density

    # Structure
    segments: list[Section] = field(default_factory=list)
    mix_in_points: list[MixPoint] = field(default_factory=list)
    mix_out_points: list[MixPoint] = field(default_factory=list)
    loop_candidates: list[MixPoint] = field(default_factory=list)

    # ML Classification
    classification: ClassificationResult | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.classification:
            d["classification"] = self.classification.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "TrackProfile":
        data["segments"] = [Section(**s) for s in data.get("segments", [])]
        data["mix_in_points"] = [MixPoint(**p) for p in data.get("mix_in_points", [])]
        data["mix_out_points"] = [MixPoint(**p) for p in data.get("mix_out_points", [])]
        data["loop_candidates"] = [MixPoint(**p) for p in data.get("loop_candidates", [])]
        clf_data = data.pop("classification", None)
        if clf_data:
            data["classification"] = ClassificationResult.from_dict(clf_data)
        else:
            data["classification"] = None
        return cls(**data)


def analyze_track(
    file_path: str | Path,
    run_classifier: bool = True,
    model_dir: Path | None = None,
    title: str | None = None,
) -> TrackProfile:
    """Run full analysis pipeline on an audio file.

    Args:
        file_path: Path to audio file (MP3, WAV, FLAC, etc.)
        run_classifier: Whether to run ML classification (genre, mood, embeddings)
        model_dir: Directory containing ML model files

    Returns:
        TrackProfile with all extracted features.
    """
    file_path = Path(file_path)
    if not title:
        title = file_path.stem

    # Load audio (mono, 44.1kHz for librosa)
    y, sr = librosa.load(str(file_path), sr=44100, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # -- BPM & Beat Grid --
    bpm, beats, downbeats = _extract_beats(y, sr)

    # -- Key Detection --
    key, key_name = _extract_key(file_path)

    # -- Energy Curve --
    energy_curve = _extract_energy_curve(y, sr, beats)

    # -- Loudness --
    loudness_lufs = _extract_loudness(file_path)

    # -- Spectral Features --
    spectral_centroid, spectral_bandwidth, mfcc, onset_strength_mean = _extract_spectral(y, sr)

    # -- Structure Detection --
    beat_times_arr = np.array(beats)
    segments = detect_structure(y, sr, beat_times_arr)

    # -- Mix Points --
    mix_in_points = find_mix_in_points(segments, beat_times_arr)
    mix_out_points = find_mix_out_points(segments, beat_times_arr, duration)
    loop_cands = find_loop_candidates(segments, beat_times_arr)

    # -- ML Classification --
    classification = None
    if run_classifier:
        try:
            classification = classify_track(file_path, model_dir=model_dir)
        except Exception as e:
            print(f"  Warning: ML classification failed: {e}")

    return TrackProfile(
        file_path=str(file_path),
        title=title,
        duration=round(duration, 2),
        bpm=bpm,
        beats=[round(float(b), 4) for b in beats],
        downbeats=[round(float(d), 4) for d in downbeats],
        key=key,
        key_name=key_name,
        energy_curve=energy_curve,
        loudness_lufs=loudness_lufs,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        mfcc=mfcc,
        onset_strength_mean=onset_strength_mean,
        segments=segments,
        mix_in_points=mix_in_points,
        mix_out_points=mix_out_points,
        loop_candidates=loop_cands,
        classification=classification,
    )


def _extract_beats(y: np.ndarray, sr: int) -> tuple[float, list[float], list[float]]:
    """Extract BPM, beat positions, and downbeat positions."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # Estimate downbeats (every 4 beats, assuming 4/4 time)
    downbeats = beat_times[::4].tolist() if len(beat_times) >= 4 else beat_times.tolist()

    return round(bpm, 1), beat_times.tolist(), downbeats


def _extract_key(file_path: Path) -> tuple[str, str]:
    """Detect musical key using essentia and map to Camelot notation."""
    audio = es.MonoLoader(filename=str(file_path), sampleRate=44100)()
    key_extractor = es.KeyExtractor()
    key, scale, _strength = key_extractor(audio)

    camelot = key_to_camelot(key, scale)
    key_name = f"{key} {scale}"

    return camelot, key_name


def _extract_energy_curve(
    y: np.ndarray, sr: int, beats: list[float], bars_per_segment: int = 2
) -> list[float]:
    """Calculate energy curve averaged over multi-bar segments."""
    beats_per_segment = bars_per_segment * 4
    segment_energies = []

    for i in range(0, len(beats) - beats_per_segment, beats_per_segment):
        start_sample = librosa.time_to_samples(beats[i], sr=sr)
        end_idx = min(i + beats_per_segment, len(beats) - 1)
        end_sample = librosa.time_to_samples(beats[end_idx], sr=sr)

        segment = y[start_sample:end_sample]
        if len(segment) > 0:
            rms = float(np.sqrt(np.mean(segment**2)))
            segment_energies.append(rms)

    if not segment_energies:
        return [0.5]

    max_e = max(segment_energies)
    if max_e > 0:
        segment_energies = [e / max_e for e in segment_energies]

    return [round(e, 3) for e in segment_energies]


def _extract_loudness(file_path: Path) -> float:
    """Measure integrated loudness in LUFS."""
    data, rate = sf.read(str(file_path))
    meter = pyln.Meter(rate)

    if data.ndim == 1:
        data = np.column_stack([data, data])

    loudness = meter.integrated_loudness(data)
    return round(float(loudness), 1)


def _extract_spectral(
    y: np.ndarray, sr: int
) -> tuple[float, float, list[float], float]:
    """Extract spectral features: centroid, bandwidth, MFCCs, onset strength."""
    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = round(float(np.mean(centroid)), 2)

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = round(float(np.mean(bandwidth)), 2)

    # MFCCs (13 coefficients, mean over time)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = [round(float(x), 4) for x in np.mean(mfcc, axis=1)]

    # Onset strength (rhythmic density)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = round(float(np.mean(onset_env)), 4)

    return centroid_mean, bandwidth_mean, mfcc_mean, onset_mean
