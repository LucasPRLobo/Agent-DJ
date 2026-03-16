"""Store and retrieve analyzed track profiles."""

import json
import sqlite3
from pathlib import Path

import numpy as np

from .audio import TrackProfile


class TrackStore:
    """SQLite-backed store for analyzed track profiles."""

    def __init__(self, db_path: str | Path = "tracks.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    file_path TEXT PRIMARY KEY,
                    title TEXT,
                    bpm REAL,
                    key_camelot TEXT,
                    loudness_lufs REAL,
                    duration REAL,
                    danceability REAL,
                    embedding BLOB,
                    profile_json TEXT
                )
            """)
            # Add indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bpm ON tracks(bpm)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON tracks(key_camelot)")

    def save(self, profile: TrackProfile):
        """Save or update a track profile."""
        # Store embedding as binary blob for efficient similarity search
        embedding_blob = None
        danceability = 0.0
        if profile.classification:
            if profile.classification.embedding:
                embedding_blob = np.array(profile.classification.embedding, dtype=np.float32).tobytes()
            danceability = profile.classification.danceability

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tracks
                (file_path, title, bpm, key_camelot, loudness_lufs, duration,
                 danceability, embedding, profile_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.file_path,
                    profile.title,
                    profile.bpm,
                    profile.key,
                    profile.loudness_lufs,
                    profile.duration,
                    danceability,
                    embedding_blob,
                    profile.to_json(),
                ),
            )

    def get(self, file_path: str) -> TrackProfile | None:
        """Load a track profile by file path."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT profile_json FROM tracks WHERE file_path = ?",
                (file_path,),
            ).fetchone()

        if row is None:
            return None
        return TrackProfile.from_dict(json.loads(row[0]))

    def get_all(self) -> list[TrackProfile]:
        """Load all track profiles."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT profile_json FROM tracks").fetchall()
        return [TrackProfile.from_dict(json.loads(row[0])) for row in rows]

    def find_compatible(
        self,
        bpm: float,
        key: str,
        bpm_tolerance: float = 8.0,
    ) -> list[TrackProfile]:
        """Find tracks compatible with given BPM and key."""
        from .camelot import camelot_compatible

        all_tracks = self.get_all()
        return [
            t for t in all_tracks
            if abs(t.bpm - bpm) <= bpm_tolerance or camelot_compatible(t.key, key)
        ]

    def find_similar_by_embedding(
        self,
        embedding: list[float],
        top_n: int = 10,
        exclude_paths: set[str] | None = None,
    ) -> list[tuple[TrackProfile, float]]:
        """Find tracks most similar to a given embedding using cosine similarity.

        Args:
            embedding: Reference embedding vector
            top_n: Number of results to return
            exclude_paths: File paths to exclude from results

        Returns:
            List of (TrackProfile, similarity_score) tuples, sorted by similarity desc.
        """
        exclude_paths = exclude_paths or set()
        ref = np.array(embedding, dtype=np.float32)
        ref_norm = np.linalg.norm(ref)
        if ref_norm == 0:
            return []

        results = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT file_path, embedding, profile_json FROM tracks WHERE embedding IS NOT NULL"
            ).fetchall()

        for file_path, emb_blob, profile_json in rows:
            if file_path in exclude_paths:
                continue
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            similarity = float(np.dot(ref, emb) / (ref_norm * emb_norm))
            profile = TrackProfile.from_dict(json.loads(profile_json))
            results.append((profile, similarity))

        results.sort(key=lambda x: -x[1])
        return results[:top_n]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()
        return row[0]
