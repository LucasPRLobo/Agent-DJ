import { useState, useCallback } from "react";
import { Chat } from "../components/Chat";
import { NowPlaying } from "../components/NowPlaying";
import { Queue } from "../components/Queue";
import { EnergyArc } from "../components/EnergyArc";
import { Requests } from "../components/Requests";
import { useSession } from "../hooks/useSession";
import { useWebSocket, type WSMessage } from "../hooks/useWebSocket";

interface ChatMessage {
  from: string;
  message: string;
}

interface TrackInfo {
  title: string;
  bpm: number;
  key: string;
  duration: number;
  file_path: string;
}

export function HostView() {
  const { session, loading, createSession, sendChat, getQrUrl } = useSession();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentTrack, setCurrentTrack] = useState<TrackInfo | null>(null);
  const [queue, setQueue] = useState<TrackInfo[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [energyArc, setEnergyArc] = useState<number[]>([]);
  const [position, setPosition] = useState(0);
  const [requests, setRequests] = useState<{ query: string; from: string; votes: number }[]>([]);
  const [sessionState, setSessionState] = useState<string>("idle");

  const handleWSMessage = useCallback((msg: WSMessage) => {
    switch (msg.type) {
      case "chat_message":
        setMessages((prev) => [
          ...prev,
          { from: msg.from as string, message: msg.message as string },
        ]);
        break;
      case "session_started": {
        setSessionState("playing");
        const plan = msg.set_plan as { tracks: TrackInfo[] };
        const vibe = msg.vibe as { energy_arc: number[] };
        if (plan?.tracks) {
          setQueue(plan.tracks);
          if (plan.tracks.length > 0) {
            setCurrentTrack(plan.tracks[0]);
            setCurrentIndex(0);
          }
        }
        if (vibe?.energy_arc) setEnergyArc(vibe.energy_arc);
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
      case "playback_state":
        setPosition(msg.position as number);
        break;
    }
  }, []);

  const { connected } = useWebSocket({
    sessionId: session?.sessionId ?? "",
    token: session?.hostToken ?? "",
    onMessage: handleWSMessage,
  });

  const handleSendChat = useCallback(
    async (message: string) => {
      // Optimistically add the user message
      setMessages((prev) => [...prev, { from: "You", message }]);
      await sendChat(message);
    },
    [sendChat]
  );

  // --- Not started yet ---
  if (!session) {
    return (
      <div className="host-landing">
        <h1>Agent DJ</h1>
        <p>AI-powered live DJ for your party</p>
        <button onClick={createSession} disabled={loading} className="start-btn">
          {loading ? "Starting..." : "Start a Session"}
        </button>
      </div>
    );
  }

  const totalDuration = currentTrack?.duration ?? 0;
  const setPosition2 = queue.length > 0 ? currentIndex / queue.length : 0;

  return (
    <div className="host-view">
      <header className="host-header">
        <h1>Agent DJ</h1>
        <div className="header-status">
          <span className={`status-dot ${connected ? "connected" : ""}`} />
          {sessionState === "playing" ? "Live" : "Setting up"}
        </div>
        {session && (
          <img src={getQrUrl()} alt="QR Code" className="qr-code" width={100} height={100} />
        )}
      </header>

      <div className="host-layout">
        <div className="host-left">
          <Chat messages={messages} onSend={handleSendChat} disabled={!connected} />
        </div>
        <div className="host-right">
          <NowPlaying
            title={currentTrack?.title ?? ""}
            bpm={currentTrack?.bpm ?? 0}
            musicalKey={currentTrack?.key ?? ""}
            position={position}
            duration={totalDuration}
            isPlaying={sessionState === "playing"}
          />
          <Queue tracks={queue} currentIndex={currentIndex} />
          <EnergyArc arc={energyArc} currentPosition={setPosition2} />
          <Requests requests={requests} />
        </div>
      </div>
    </div>
  );
}
