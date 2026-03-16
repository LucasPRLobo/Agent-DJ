"""Strategy C: Embedding-Based Clustering.

Cluster tracks by audio similarity, then plan the set as a journey
through clusters following the energy arc.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from ...analyzer.audio import TrackProfile
from ...analyzer.camelot import camelot_distance
from ..vibe_profile import VibeProfile


@dataclass
class Cluster:
    """A group of musically similar tracks."""
    id: int
    tracks: list[TrackProfile]
    centroid: np.ndarray
    avg_energy: float
    avg_bpm: float
    primary_genres: list[str]


def cluster_tracks(
    tracks: list[TrackProfile],
    n_clusters: int | None = None,
) -> list[Cluster]:
    """Cluster tracks by audio embedding similarity.

    Args:
        tracks: Tracks with embeddings.
        n_clusters: Number of clusters. If None, auto-determined.

    Returns:
        List of Cluster objects.
    """
    # Filter tracks with embeddings
    tracks_with_emb = [t for t in tracks if t.classification and t.classification.embedding]
    if len(tracks_with_emb) < 3:
        # Not enough for clustering, return single cluster
        return [Cluster(
            id=0, tracks=tracks_with_emb,
            centroid=np.zeros(1),
            avg_energy=0.5, avg_bpm=120, primary_genres=[],
        )]

    embeddings = np.array([t.classification.embedding for t in tracks_with_emb])

    # Auto-determine number of clusters
    if n_clusters is None:
        n_clusters = max(2, min(len(tracks_with_emb) // 4, 8))

    n_clusters = min(n_clusters, len(tracks_with_emb))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters = []
    for c_id in range(n_clusters):
        mask = labels == c_id
        cluster_tracks_list = [tracks_with_emb[i] for i in range(len(tracks_with_emb)) if mask[i]]

        if not cluster_tracks_list:
            continue

        avg_energy = float(np.mean([
            np.mean(t.energy_curve) if t.energy_curve else 0.5
            for t in cluster_tracks_list
        ]))
        avg_bpm = float(np.mean([t.bpm for t in cluster_tracks_list]))

        # Get most common genres in cluster
        from ...analyzer.genre_taxonomy import map_to_dj_genres
        genre_counts: dict[str, float] = {}
        for t in cluster_tracks_list:
            if t.classification and t.classification.genres:
                profile = map_to_dj_genres(t.classification.genres)
                for g, s in profile.genres.items():
                    genre_counts[g] = genre_counts.get(g, 0) + s
        primary = sorted(genre_counts, key=genre_counts.get, reverse=True)[:3]

        clusters.append(Cluster(
            id=c_id,
            tracks=cluster_tracks_list,
            centroid=kmeans.cluster_centers_[c_id],
            avg_energy=round(avg_energy, 3),
            avg_bpm=round(avg_bpm, 1),
            primary_genres=primary,
        ))

    return clusters


def select_tracks_clustering(
    tracks: list[TrackProfile],
    vibe: VibeProfile,
    n_tracks: int | None = None,
) -> list[TrackProfile]:
    """Select tracks by navigating through clusters following the energy arc.

    1. Cluster all tracks
    2. For each position in the set, pick the cluster whose avg energy
       best matches the target
    3. Within that cluster, pick the track with best BPM/key compatibility

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

    clusters = cluster_tracks(tracks)
    if not clusters:
        return []

    selected: list[TrackProfile] = []
    used_paths: set[str] = set()

    for i in range(n_tracks):
        position = i / max(n_tracks - 1, 1)
        target_energy = vibe.get_target_energy(position)
        prev = selected[-1] if selected else None

        # Pick best cluster for this position
        # Score each cluster by energy match and available tracks
        best_track = None
        best_score = -1

        for cluster in clusters:
            available = [t for t in cluster.tracks if t.file_path not in used_paths]
            if not available:
                continue

            cluster_energy_match = 1.0 - abs(cluster.avg_energy - target_energy)

            for t in available:
                t_energy = float(np.mean(t.energy_curve)) if t.energy_curve else 0.5
                energy_match = 1.0 - abs(t_energy - target_energy)

                # BPM/key compatibility with previous
                transition_score = 0.5
                if prev:
                    bpm_diff = abs(t.bpm - prev.bpm)
                    bpm_compat = max(0, 1.0 - bpm_diff / 10.0)
                    key_dist = camelot_distance(prev.key, t.key)
                    key_compat = max(0, 1.0 - key_dist * 0.2)
                    transition_score = 0.5 * bpm_compat + 0.5 * key_compat

                score = (
                    0.3 * cluster_energy_match +
                    0.3 * energy_match +
                    0.4 * transition_score
                )

                if score > best_score:
                    best_score = score
                    best_track = t

        if best_track is None:
            break

        selected.append(best_track)
        used_paths.add(best_track.file_path)

    return selected
