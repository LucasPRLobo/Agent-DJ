"""DJ Brain Loop — the core async loop that drives the rolling planner.

Runs continuously while a session is in the "playing" state.
Plans 1 track ahead, auto-fetches from YouTube when needed,
tracks energy, and responds to vibe changes.
"""

import asyncio
import os

import numpy as np

from ..analyzer.audio import TrackProfile
from ..analyzer.track_store import TrackStore
from ..planner.energy_tracker import EnergyTracker
from ..sources.background_fetcher import BackgroundFetcher
from ..planner.strategies.scoring import score_track
from ..planner.transition_planner import plan_transition
from ..planner.vibe_profile import VibeProfile
from ..sources.auto_fetch import (
    fetch_and_analyze,
    full_analyze_background,
    generate_search_queries,
)
from .session import Session, SessionState
from .websocket import ConnectionManager

# Minimum score to accept a local track (below this, try auto-fetch)
QUALITY_THRESHOLD = 0.35


class DJLoop:
    """Manages the rolling DJ brain for a session."""

    def __init__(
        self,
        session: Session,
        track_store: TrackStore,
        ws_mgr: ConnectionManager,
        audio_dir: str = "samples",
    ):
        self.session = session
        self.store = track_store
        self.ws_mgr = ws_mgr
        self.audio_dir = audio_dir
        self.energy_tracker: EnergyTracker | None = None
        self.played_tracks: list[TrackProfile] = []
        self.current_track: TrackProfile | None = None
        self.next_track: TrackProfile | None = None
        self.next_transition = None
        self.needs_replan = False
        self.bg_fetcher = BackgroundFetcher(track_store, audio_dir)
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self):
        """Start the DJ loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    def stop(self):
        """Stop the DJ loop."""
        self._running = False
        self.bg_fetcher.stop()
        if self._task:
            self._task.cancel()

    def request_replan(self):
        """Signal that the vibe changed and next track should be re-planned."""
        self.needs_replan = True
        self.next_track = None
        # Update background fetcher with new vibe
        if self.session.vibe_profile:
            self.bg_fetcher.update_vibe(self.session.vibe_profile)

    async def _run(self):
        """Main DJ loop."""
        vibe = self.session.vibe_profile
        if not vibe:
            return

        self.energy_tracker = EnergyTracker(vibe)

        # Start background fetcher — continuously builds the track pool
        self.bg_fetcher.start(vibe)

        # Play first track
        first_track = await self._pick_first_track(vibe)
        if not first_track:
            await self._send_chat("I don't have any tracks that match this vibe. Try adding some music first!")
            return

        self.current_track = first_track
        self.played_tracks.append(first_track)
        self.energy_tracker.record(first_track, 0.0)

        await self._send_play_track(first_track)

        # Main loop — plan ahead while current track plays
        while self._running and self.session.state == SessionState.PLAYING:
            try:
                await self._plan_next_if_needed(vibe)
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"DJ loop error: {e}")
                await asyncio.sleep(5)

    async def _pick_first_track(self, vibe: VibeProfile) -> TrackProfile | None:
        """Pick the best starting track."""
        tracks = self.store.get_all()
        if not tracks:
            # Try auto-fetch
            track = await self._auto_fetch(vibe, target_energy=vibe.get_target_energy(0.0))
            return track

        target_energy = vibe.get_target_energy(0.0)
        scored = [
            score_track(t, vibe, target_energy, prev_track=None)
            for t in tracks
        ]
        scored.sort(key=lambda s: -s.total)
        return scored[0].track if scored else None

    async def _plan_next_if_needed(self, vibe: VibeProfile):
        """Plan the next track if we don't have one queued."""
        if self.next_track and not self.needs_replan:
            return

        self.needs_replan = False
        n_played = len(self.played_tracks)
        set_position = min(1.0, n_played * 0.05)  # rough position estimate

        # Get adjusted energy target
        raw_target = vibe.get_target_energy(set_position)
        target_energy = self.energy_tracker.adjust_next_target(raw_target) if self.energy_tracker else raw_target

        # Score local library — exclude played tracks and BPM outliers
        all_tracks = self.store.get_all()
        played_paths = {t.file_path for t in self.played_tracks}
        played_titles = {t.title.lower() for t in self.played_tracks}
        bpm_low, bpm_high = vibe.bpm_range
        candidates = [
            t for t in all_tracks
            if t.file_path not in played_paths
            and t.title.lower() not in played_titles
            and bpm_low - 20 <= t.bpm <= bpm_high + 20  # filter obvious BPM mismatches
        ]

        best_track = None
        best_score = 0.0

        if candidates:
            scored = [
                score_track(t, vibe, target_energy, prev_track=self.current_track)
                for t in candidates
            ]
            scored.sort(key=lambda s: -s.total)
            best_track = scored[0].track
            best_score = scored[0].total

        # Auto-fetch if local quality is too low or no candidates
        if best_score < QUALITY_THRESHOLD or not best_track:
            fetched = await self._auto_fetch(vibe, target_energy)
            if fetched:
                # Re-score the fetched track
                fetched_score = score_track(fetched, vibe, target_energy, self.current_track)
                if not best_track or fetched_score.total > best_score:
                    best_track = fetched
                    best_score = fetched_score.total

        if not best_track:
            return

        # Plan transition
        energy_a = float(np.mean(self.current_track.energy_curve)) if self.current_track and self.current_track.energy_curve else 0.5
        energy_b = float(np.mean(best_track.energy_curve)) if best_track.energy_curve else 0.5

        if energy_b > energy_a + 0.1:
            direction = "rising"
        elif energy_b < energy_a - 0.1:
            direction = "falling"
        else:
            direction = "sustain"

        transition = plan_transition(self.current_track, best_track, direction) if self.current_track else None

        self.next_track = best_track

        # Send to client
        await self._send_queue_next(best_track, transition)

    async def _auto_fetch(self, vibe: VibeProfile, target_energy: float) -> TrackProfile | None:
        """Try to fetch a new track from YouTube."""
        queries = generate_search_queries(
            vibe_genres=vibe.genres,
            current_track=self.current_track,
            target_bpm=(vibe.bpm_range[0] + vibe.bpm_range[1]) / 2,
            example_tracks=vibe.example_tracks,
            played_titles={t.title for t in self.played_tracks},
        )

        for query in queries[:3]:  # try up to 3 queries
            print(f"DJ auto-fetch: searching '{query}'")
            track = await fetch_and_analyze(query, self.audio_dir, timeout=25)
            if track:
                # Save to store (quick profile)
                self.store.save(track)
                # Start full analysis in background
                asyncio.create_task(
                    full_analyze_background(track.file_path, track.title, self.store)
                )
                return track

        return None

    def on_transition_complete(self):
        """Called by the frontend (via WebSocket) when a transition finishes."""
        if self.next_track:
            self.played_tracks.append(self.next_track)
            self.bg_fetcher.mark_played(self.next_track.file_path)
            set_pos = min(1.0, len(self.played_tracks) * 0.05)
            if self.energy_tracker:
                self.energy_tracker.record(self.next_track, set_pos)
            self.current_track = self.next_track
            self.next_track = None

    async def _send_play_track(self, track: TrackProfile):
        """Tell the client to start playing a track."""
        filename = track.file_path.split("/")[-1]
        await self.ws_mgr.send_to_session(self.session.id, {
            "type": "play_track",
            "track": {
                "title": track.title,
                "bpm": track.bpm,
                "key": track.key,
                "duration": track.duration,
                "file_path": track.file_path,
            },
            "audio_url": f"/audio/{filename}",
        })

    async def _send_queue_next(self, track: TrackProfile, transition):
        """Tell the client to queue the next track with transition plan."""
        filename = track.file_path.split("/")[-1]
        await self.ws_mgr.send_to_session(self.session.id, {
            "type": "queue_next",
            "track": {
                "title": track.title,
                "bpm": track.bpm,
                "key": track.key,
                "duration": track.duration,
                "file_path": track.file_path,
            },
            "audio_url": f"/audio/{filename}",
            "transition": transition.to_dict() if transition else None,
        })

    async def _send_chat(self, message: str):
        """Send a chat message from the DJ."""
        await self.ws_mgr.send_to_session(self.session.id, {
            "type": "chat_message",
            "from": "dj",
            "message": message,
        })


# Global registry of active DJ loops
_active_loops: dict[str, DJLoop] = {}


def start_dj_loop(session: Session, store: TrackStore, ws_mgr: ConnectionManager, audio_dir: str = "samples"):
    """Start a DJ loop for a session."""
    if session.id in _active_loops:
        _active_loops[session.id].stop()

    loop = DJLoop(session, store, ws_mgr, audio_dir)
    _active_loops[session.id] = loop
    loop.start()
    return loop


def get_dj_loop(session_id: str) -> DJLoop | None:
    return _active_loops.get(session_id)


def stop_dj_loop(session_id: str):
    loop = _active_loops.pop(session_id, None)
    if loop:
        loop.stop()
