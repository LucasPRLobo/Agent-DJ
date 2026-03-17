"""FastAPI application — main entry point for the Agent DJ backend."""

import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load .env file

import qrcode
from fastapi import FastAPI, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..analyzer.audio import analyze_track
from ..analyzer.track_store import TrackStore
from ..chat.command_parser import CommandParser
from ..chat.vibe_parser import VibeParser
from ..planner.set_planner import SelectionStrategy, plan_next_tracks, plan_set
from .session import GuestPermission, Session, SessionManager, SessionState, SongRequest
from .websocket import ConnectionInfo, ConnectionManager

# ── App setup ──

app = FastAPI(title="Agent DJ", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
session_mgr = SessionManager()
ws_mgr = ConnectionManager()
track_store = TrackStore(os.environ.get("AGENT_DJ_DB", "tracks.db"))

# Audio files directory
AUDIO_DIR = Path(os.environ.get("AGENT_DJ_AUDIO_DIR", "samples"))

# LLM parsers (initialized lazily)
_vibe_parser: VibeParser | None = None
_command_parser: CommandParser | None = None


def get_vibe_parser() -> VibeParser:
    global _vibe_parser
    if _vibe_parser is None:
        _vibe_parser = VibeParser()
    return _vibe_parser


def get_command_parser() -> CommandParser:
    global _command_parser
    if _command_parser is None:
        _command_parser = CommandParser()
    return _command_parser


# ── Pydantic models ──

class CreateSessionResponse(BaseModel):
    session_id: str
    host_token: str


class JoinRequest(BaseModel):
    name: str


class JoinResponse(BaseModel):
    guest_token: str
    permission: str


class ChatMessage(BaseModel):
    message: str


class SessionInfo(BaseModel):
    session_id: str
    state: str
    guest_count: int
    current_track: str | None
    queue: list[str]


# ── REST endpoints ──

@app.post("/session", response_model=CreateSessionResponse)
async def create_session():
    """Create a new DJ session."""
    session = session_mgr.create_session()
    return CreateSessionResponse(session_id=session.id, host_token=session.host_token)


@app.get("/session/{session_id}")
async def get_session(session_id: str, token: str = Query(...)):
    """Get session state."""
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    auth = session_mgr.authenticate(session_id, token)
    if not auth:
        raise HTTPException(403, "Invalid token")

    current_track = None
    if 0 <= session.current_track_index < len(session.queue):
        current_track = session.queue[session.current_track_index]

    return SessionInfo(
        session_id=session.id,
        state=session.state.value,
        guest_count=ws_mgr.get_guest_count(session_id),
        current_track=current_track,
        queue=session.queue,
    )


@app.post("/session/{session_id}/join", response_model=JoinResponse)
async def join_session(session_id: str, request: JoinRequest):
    """Guest joins a session."""
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    guest = session_mgr.add_guest(session_id, request.name)
    if not guest:
        raise HTTPException(500, "Failed to add guest")

    await ws_mgr.send_to_host(session_id, {
        "type": "guest_joined",
        "name": request.name,
        "guest_count": len(session.guests),
    })

    return JoinResponse(guest_token=guest.token, permission=guest.permission.value)


@app.get("/session/{session_id}/qr")
async def get_qr_code(session_id: str, base_url: str = Query("http://localhost:5173")):
    """Generate QR code for guest invite link."""
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    invite_url = f"{base_url}/guest/{session_id}"

    qr = qrcode.make(invite_url)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve an audio file."""
    file_path = AUDIO_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "Audio file not found")

    return Response(
        content=file_path.read_bytes(),
        media_type="audio/mpeg",
        headers={"Accept-Ranges": "bytes"},
    )


class SearchQuery(BaseModel):
    query: str
    max_results: int = 5


class AddTrackRequest(BaseModel):
    query: str  # song name or YouTube URL


@app.post("/search")
async def search_music(req: SearchQuery):
    """Search for tracks on YouTube."""
    from ..sources.youtube import search_tracks
    results = search_tracks(req.query, req.max_results)
    return [
        {
            "video_id": r.video_id,
            "title": r.title,
            "artist": r.artist,
            "duration": r.duration,
            "url": r.url,
            "thumbnail": r.thumbnail,
        }
        for r in results
    ]


@app.post("/add")
async def add_track(req: AddTrackRequest):
    """Download a track from YouTube, analyze it, and add to library."""
    from ..sources.youtube import download_track

    dl = download_track(req.query, str(AUDIO_DIR))
    if not dl:
        raise HTTPException(500, "Failed to download track")

    try:
        song_title = f"{dl.title} - {dl.artist}" if dl.artist != "Unknown" else dl.title
        profile = analyze_track(dl.file_path, title=song_title)
        track_store.save(profile)
        return {
            "status": "ok",
            "title": profile.title,
            "artist": dl.artist,
            "bpm": profile.bpm,
            "key": profile.key,
            "duration": profile.duration,
            "file_path": profile.file_path,
        }
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/library")
async def get_library():
    """Get all analyzed tracks."""
    tracks = track_store.get_all()
    return [
        {
            "file_path": t.file_path,
            "title": t.title,
            "bpm": t.bpm,
            "key": t.key,
            "duration": t.duration,
            "genres": t.classification.genres[:3] if t.classification else [],
        }
        for t in tracks
    ]


@app.post("/session/{session_id}/chat")
async def chat(session_id: str, msg: ChatMessage, token: str = Query(...)):
    """Send a chat message to the DJ."""
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    auth = session_mgr.authenticate(session_id, token)
    if not auth:
        raise HTTPException(403, "Invalid token")

    role, name = auth

    # During onboarding, use vibe parser (or fallback if no API key)
    if session.state == SessionState.ONBOARDING:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

        if use_llm:
            try:
                parser = get_vibe_parser()

                # Initialize onboarding if first message
                if not session.chat_history:
                    onboarding, opening = parser.start_session()
                    session.chat_history = onboarding.messages
                    await ws_mgr.send_to_session(session_id, {
                        "type": "chat_message",
                        "from": "dj",
                        "message": opening,
                    })

                from ..chat.vibe_parser import OnboardingSession
                onboarding = OnboardingSession(messages=list(session.chat_history))

                dj_response = parser.send_message(onboarding, msg.message)
                session.chat_history = onboarding.messages

                await ws_mgr.send_to_session(session_id, {
                    "type": "chat_message",
                    "from": name,
                    "message": msg.message,
                })

                if onboarding.is_complete and onboarding.vibe_profile:
                    # Show a friendly message instead of raw JSON
                    dj_response = "Got it! Setting up your set now — let's go!"
                    session.vibe_profile = onboarding.vibe_profile
                    session.state = SessionState.PLAYING

                    tracks = track_store.get_all()
                    set_plan = plan_set(tracks, session.vibe_profile, SelectionStrategy.GREEDY_SCORING)
                    session.queue = [t.file_path for t in set_plan.tracks]

                    await ws_mgr.send_to_session(session_id, {
                        "type": "chat_message",
                        "from": "dj",
                        "message": dj_response,
                    })
                    await ws_mgr.send_to_session(session_id, {
                        "type": "session_started",
                        "vibe": session.vibe_profile.to_dict(),
                        "set_plan": set_plan.to_dict(),
                    })

                    return {"response": dj_response, "state": session.state.value}

                # Still onboarding — send the DJ's follow-up question
                await ws_mgr.send_to_session(session_id, {
                    "type": "chat_message",
                    "from": "dj",
                    "message": dj_response,
                })

                return {"response": dj_response, "state": session.state.value}

            except Exception as e:
                # LLM failed — fall through to preset mode
                print(f"LLM error, falling back to preset mode: {e}")
                use_llm = False

        # Fallback: no LLM — use preset based on user message keywords
        if not use_llm:
            from ..planner.vibe_profile import chill_jazz_party, house_party, dinner_ambient

            user_msg = msg.message.lower()
            if any(w in user_msg for w in ["dance", "house", "party", "energy", "electronic"]):
                vibe = house_party()
                vibe_name = "House Party"
            elif any(w in user_msg for w in ["dinner", "calm", "quiet", "background", "ambient"]):
                vibe = dinner_ambient()
                vibe_name = "Dinner Ambient"
            else:
                vibe = chill_jazz_party()
                vibe_name = "Chill Jazz"

            session.vibe_profile = vibe
            session.state = SessionState.PLAYING

            dj_response = f"Got it! Setting up a {vibe_name} vibe for you. Let's go!"

            await ws_mgr.send_to_session(session_id, {
                "type": "chat_message", "from": name, "message": msg.message,
            })
            await ws_mgr.send_to_session(session_id, {
                "type": "chat_message", "from": "dj", "message": dj_response,
            })

            tracks = track_store.get_all()
            set_plan = plan_set(tracks, session.vibe_profile, SelectionStrategy.GREEDY_SCORING)
            session.queue = [t.file_path for t in set_plan.tracks]

            await ws_mgr.send_to_session(session_id, {
                "type": "session_started",
                "vibe": session.vibe_profile.to_dict(),
                "set_plan": set_plan.to_dict(),
            })

            return {"response": dj_response, "state": session.state.value}

    # During playback, handle commands (simple keyword matching if no LLM)
    command_type = "chat"
    command_params: dict = {}
    dj_resp = ""

    user_lower = msg.message.lower()
    if any(w in user_lower for w in ["skip", "next"]):
        command_type = "skip"
    elif any(w in user_lower for w in ["play ", "request "]):
        command_type = "request"
        command_params = {"query": msg.message}
    elif any(w in user_lower for w in ["more energy", "louder", "pump", "harder"]):
        command_type = "adjust_energy"
        command_params = {"direction": "up"}
    elif any(w in user_lower for w in ["chill", "calm", "quiet", "slow"]):
        command_type = "adjust_energy"
        command_params = {"direction": "down"}
    elif any(w in user_lower for w in ["wind down", "end", "finish", "stop"]):
        command_type = "wind_down"

    # Try LLM command parser if available
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            cmd_parser = get_command_parser()
            command = cmd_parser.parse(msg.message)
            command_type = command.type
            command_params = command.parameters
            dj_resp = command.dj_response
        except Exception:
            pass  # fall through to keyword-based parsing above

    await ws_mgr.send_to_session(session_id, {
        "type": "chat_message",
        "from": name,
        "message": msg.message,
    })

    # Handle commands
    if command_type == "skip":
        await ws_mgr.send_to_session(session_id, {"type": "skip"})
    elif command_type == "request":
        perm = session_mgr.get_guest_permission(session_id, token)
        if perm in (GuestPermission.REQUEST, GuestPermission.VOTE, GuestPermission.COHOST):
            req = SongRequest(
                query=command_params.get("query", msg.message),
                requested_by=name,
            )
            session.song_requests.append(req)
            await ws_mgr.send_to_session(session_id, {
                "type": "song_request",
                "query": req.query,
                "from": name,
            })
    elif command_type in ("adjust_energy", "adjust_genre", "adjust_tempo", "wind_down"):
        await ws_mgr.send_to_session(session_id, {
            "type": "dj_command",
            "command": command_type,
            "parameters": command_params,
        })

    if dj_resp:
        await ws_mgr.send_to_session(session_id, {
            "type": "chat_message",
            "from": "dj",
            "message": dj_resp,
        })

    return {"response": dj_resp or "Got it!", "command": command_type}


# ── WebSocket ──

@app.websocket("/session/{session_id}/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str, token: str = Query(...)):
    """WebSocket connection for real-time session updates."""
    session = session_mgr.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    auth = session_mgr.authenticate(session_id, token)
    if not auth:
        await websocket.close(code=4003, reason="Invalid token")
        return

    role, name = auth
    conn_info = ConnectionInfo(
        websocket=websocket,
        session_id=session_id,
        token=token,
        role=role,
        name=name,
    )

    await ws_mgr.connect(conn_info)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type")

            if msg_type == "playback_state":
                # Client reports current playback position
                await ws_mgr.send_to_host(session_id, {
                    "type": "playback_state",
                    "position": message.get("position", 0),
                    "deck": message.get("deck", "A"),
                })

            elif msg_type == "chat":
                # Chat message via WebSocket
                chat_msg = message.get("message", "")
                if chat_msg:
                    # Process through the chat endpoint logic
                    await ws_mgr.send_to_session(session_id, {
                        "type": "chat_message",
                        "from": name,
                        "message": chat_msg,
                    })

            elif msg_type == "vote":
                # Vote on a song request
                perm = session_mgr.get_guest_permission(session_id, token)
                if perm in (GuestPermission.VOTE, GuestPermission.COHOST):
                    request_idx = message.get("request_index", -1)
                    if 0 <= request_idx < len(session.song_requests):
                        session.song_requests[request_idx].votes += 1
                        await ws_mgr.send_to_session(session_id, {
                            "type": "vote_update",
                            "request_index": request_idx,
                            "votes": session.song_requests[request_idx].votes,
                        })

    except WebSocketDisconnect:
        ws_mgr.disconnect(conn_info)
    except Exception:
        ws_mgr.disconnect(conn_info)


# ── Server entry point ──

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the Agent DJ backend server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
