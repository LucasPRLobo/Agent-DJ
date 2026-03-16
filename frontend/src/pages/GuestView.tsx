import { useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { NowPlaying } from "../components/NowPlaying";
import { Queue } from "../components/Queue";
import { Requests } from "../components/Requests";
import { useWebSocket, type WSMessage } from "../hooks/useWebSocket";

const API_BASE = `http://${window.location.hostname}:8000`;

interface TrackInfo {
  title: string;
  bpm: number;
  key: string;
  duration: number;
}

export function GuestView() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [guestToken, setGuestToken] = useState<string | null>(null);
  const [guestName, setGuestName] = useState("");
  const [joining, setJoining] = useState(false);
  const [permission, setPermission] = useState("listen");

  const [currentTrack, setCurrentTrack] = useState<TrackInfo | null>(null);
  const [queue, setQueue] = useState<TrackInfo[]>([]);
  const [currentIndex] = useState(0);
  const [position] = useState(0);
  const [requests, setRequests] = useState<{ query: string; from: string; votes: number }[]>([]);
  const [requestInput, setRequestInput] = useState("");

  const handleWSMessage = useCallback((msg: WSMessage) => {
    switch (msg.type) {
      case "session_started": {
        const plan = msg.set_plan as { tracks: TrackInfo[] };
        if (plan?.tracks) {
          setQueue(plan.tracks);
          if (plan.tracks.length > 0) setCurrentTrack(plan.tracks[0]);
        }
        break;
      }
      case "queue_update": {
        const tracks = msg.tracks as TrackInfo[];
        if (tracks) setQueue(tracks);
        break;
      }
      case "song_request":
        setRequests((prev) => [
          ...prev,
          { query: msg.query as string, from: msg.from as string, votes: 0 },
        ]);
        break;
      case "vote_update":
        setRequests((prev) => {
          const next = [...prev];
          const idx = msg.request_index as number;
          if (next[idx]) next[idx].votes = msg.votes as number;
          return next;
        });
        break;
    }
  }, []);

  const { connected, send } = useWebSocket({
    sessionId: sessionId ?? "",
    token: guestToken ?? "",
    onMessage: handleWSMessage,
  });

  const handleJoin = async () => {
    if (!sessionId || !guestName.trim()) return;
    setJoining(true);
    try {
      const res = await fetch(`${API_BASE}/session/${sessionId}/join`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: guestName.trim() }),
      });
      const data = await res.json();
      setGuestToken(data.guest_token);
      setPermission(data.permission);
    } catch {
      alert("Failed to join session");
    } finally {
      setJoining(false);
    }
  };

  const handleRequest = () => {
    if (!requestInput.trim()) return;
    send({ type: "chat", message: `Play ${requestInput.trim()}` });
    setRequestInput("");
  };

  const handleVote = (index: number) => {
    send({ type: "vote", request_index: index });
  };

  const handleVibeChange = (direction: "up" | "down") => {
    send({
      type: "chat",
      message: direction === "up" ? "More energy!" : "Chill out a bit",
    });
  };

  // --- Join screen ---
  if (!guestToken) {
    return (
      <div className="guest-join">
        <h1>Agent DJ</h1>
        <p>Join the party</p>
        <input
          type="text"
          value={guestName}
          onChange={(e) => setGuestName(e.target.value)}
          placeholder="Your name"
          maxLength={20}
        />
        <button onClick={handleJoin} disabled={joining || !guestName.trim()}>
          {joining ? "Joining..." : "Join"}
        </button>
      </div>
    );
  }

  const canRequest = permission !== "listen";
  const canVote = permission === "vote" || permission === "cohost";

  return (
    <div className="guest-view">
      <header className="guest-header">
        <h1>Agent DJ</h1>
        <span className={`status-dot ${connected ? "connected" : ""}`} />
      </header>

      <NowPlaying
        title={currentTrack?.title ?? "Waiting for DJ..."}
        bpm={currentTrack?.bpm ?? 0}
        musicalKey={currentTrack?.key ?? ""}
        position={position}
        duration={currentTrack?.duration ?? 0}
        isPlaying={!!currentTrack}
      />

      {canRequest && (
        <div className="guest-controls">
          <div className="request-form">
            <input
              type="text"
              value={requestInput}
              onChange={(e) => setRequestInput(e.target.value)}
              placeholder="Request a song..."
            />
            <button onClick={handleRequest} disabled={!requestInput.trim()}>
              Request
            </button>
          </div>
          <div className="vibe-buttons">
            <button onClick={() => handleVibeChange("up")} className="vibe-up">
              More Energy
            </button>
            <button onClick={() => handleVibeChange("down")} className="vibe-down">
              Chill Out
            </button>
          </div>
        </div>
      )}

      <Queue tracks={queue} currentIndex={currentIndex} />
      <Requests requests={requests} onVote={canVote ? handleVote : undefined} canVote={canVote} />
    </div>
  );
}
