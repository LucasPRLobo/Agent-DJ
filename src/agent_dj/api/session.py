"""Session management for DJ sessions."""

import secrets
import time
from dataclasses import dataclass, field
from enum import Enum

from ..planner.vibe_profile import VibeProfile


class SessionState(str, Enum):
    ONBOARDING = "onboarding"
    PLAYING = "playing"
    PAUSED = "paused"
    ENDED = "ended"


class GuestPermission(str, Enum):
    LISTEN = "listen"
    REQUEST = "request"
    VOTE = "vote"
    COHOST = "cohost"


@dataclass
class Guest:
    token: str
    name: str
    permission: GuestPermission


@dataclass
class SongRequest:
    query: str
    requested_by: str  # guest name or "host"
    votes: int = 0
    timestamp: float = 0.0


@dataclass
class Session:
    id: str
    host_token: str
    state: SessionState = SessionState.ONBOARDING
    created_at: float = 0.0

    # Vibe
    vibe_profile: VibeProfile | None = None

    # Guests
    guests: dict[str, Guest] = field(default_factory=dict)  # token → Guest
    default_guest_permission: GuestPermission = GuestPermission.REQUEST

    # Playback state
    current_track_index: int = -1
    queue: list[str] = field(default_factory=list)  # file paths of upcoming tracks

    # Requests
    song_requests: list[SongRequest] = field(default_factory=list)

    # Chat history
    chat_history: list[dict] = field(default_factory=list)


class SessionManager:
    """Manages all active DJ sessions."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> Session:
        """Create a new DJ session."""
        session_id = secrets.token_urlsafe(8)
        host_token = secrets.token_urlsafe(16)

        session = Session(
            id=session_id,
            host_token=host_token,
            created_at=time.time(),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_session_by_host_token(self, token: str) -> Session | None:
        for session in self._sessions.values():
            if session.host_token == token:
                return session
        return None

    def add_guest(self, session_id: str, name: str, permission: GuestPermission | None = None) -> Guest | None:
        session = self.get_session(session_id)
        if not session:
            return None

        token = secrets.token_urlsafe(12)
        perm = permission or session.default_guest_permission
        guest = Guest(token=token, name=name, permission=perm)
        session.guests[token] = guest
        return guest

    def authenticate(self, session_id: str, token: str) -> tuple[str, str] | None:
        """Authenticate a token. Returns (role, name) or None."""
        session = self.get_session(session_id)
        if not session:
            return None

        if token == session.host_token:
            return ("host", "Host")

        guest = session.guests.get(token)
        if guest:
            return ("guest", guest.name)

        return None

    def get_guest_permission(self, session_id: str, token: str) -> GuestPermission | None:
        session = self.get_session(session_id)
        if not session:
            return None

        if token == session.host_token:
            return GuestPermission.COHOST

        guest = session.guests.get(token)
        return guest.permission if guest else None

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[Session]:
        return list(self._sessions.values())
