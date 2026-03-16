"""WebSocket handler for real-time session communication."""

import json
from dataclasses import dataclass, field

from fastapi import WebSocket


@dataclass
class ConnectionInfo:
    websocket: WebSocket
    session_id: str
    token: str
    role: str  # "host" or "guest"
    name: str


class ConnectionManager:
    """Manages WebSocket connections for all sessions."""

    def __init__(self):
        # session_id → list of connections
        self._connections: dict[str, list[ConnectionInfo]] = {}

    async def connect(self, info: ConnectionInfo) -> None:
        await info.websocket.accept()
        if info.session_id not in self._connections:
            self._connections[info.session_id] = []
        self._connections[info.session_id].append(info)

    def disconnect(self, info: ConnectionInfo) -> None:
        conns = self._connections.get(info.session_id, [])
        self._connections[info.session_id] = [c for c in conns if c.websocket != info.websocket]

    async def send_to_session(self, session_id: str, message: dict) -> None:
        """Send a message to all connections in a session."""
        conns = self._connections.get(session_id, [])
        data = json.dumps(message)
        for conn in conns:
            try:
                await conn.websocket.send_text(data)
            except Exception:
                pass

    async def send_to_host(self, session_id: str, message: dict) -> None:
        """Send a message only to the host."""
        conns = self._connections.get(session_id, [])
        data = json.dumps(message)
        for conn in conns:
            if conn.role == "host":
                try:
                    await conn.websocket.send_text(data)
                except Exception:
                    pass

    async def send_to_connection(self, websocket: WebSocket, message: dict) -> None:
        """Send a message to a specific connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            pass

    def get_session_connections(self, session_id: str) -> list[ConnectionInfo]:
        return self._connections.get(session_id, [])

    def get_guest_count(self, session_id: str) -> int:
        conns = self._connections.get(session_id, [])
        return sum(1 for c in conns if c.role == "guest")
