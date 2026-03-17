import { useState, useCallback, useRef, useEffect } from "react";
import { Chat } from "../components/Chat";
import { NowPlaying } from "../components/NowPlaying";
import { Queue } from "../components/Queue";
import { EnergyArc } from "../components/EnergyArc";
import { Requests } from "../components/Requests";
import { useSession } from "../hooks/useSession";
import { useWebSocket, type WSMessage } from "../hooks/useWebSocket";
import { AudioEngine } from "../audio/AudioEngine";

const API_BASE = `http://${window.location.hostname}:8000`;

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

interface TransitionInfo {
  type: string;
  duration_seconds: number;
  mix_out_time: number;
  mix_in_time: number;
  target_bpm: number;
}

interface SetPlan {
  tracks: TrackInfo[];
  transitions: TransitionInfo[];
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

  // Audio engine
  const engineRef = useRef<AudioEngine | null>(null);
  const planRef = useRef<SetPlan | null>(null);
  const trackIndexRef = useRef(0);
  const positionTimerRef = useRef<number | null>(null);

  // Clean up audio on unmount
  useEffect(() => {
    return () => {
      engineRef.current?.destroy();
      if (positionTimerRef.current) clearInterval(positionTimerRef.current);
    };
  }, []);

  const getAudioUrl = (filePath: string) => {
    // Extract filename from path like "samples/K0HSD_i2DvA.wav"
    const filename = filePath.split("/").pop() || filePath;
    return `${API_BASE}/audio/${filename}`;
  };

  const startPlayback = useCallback(async (plan: SetPlan) => {
    if (plan.tracks.length === 0) return;

    // Initialize audio engine
    const engine = new AudioEngine();
    await engine.resume();
    engineRef.current = engine;
    planRef.current = plan;
    trackIndexRef.current = 0;

    // Load and play first track
    const firstTrack = plan.tracks[0];
    const url = getAudioUrl(firstTrack.file_path);

    try {
      await engine.loadTrack("A", url);
      engine.setGain("A", 1);
      engine.play("A", 0);
      setCurrentTrack(firstTrack);
      setCurrentIndex(0);

      // Pre-load next track
      if (plan.tracks.length > 1) {
        const nextUrl = getAudioUrl(plan.tracks[1].file_path);
        engine.loadTrack("B", nextUrl);
      }

      // Start position tracking
      positionTimerRef.current = window.setInterval(() => {
        if (!engineRef.current) return;
        const idx = trackIndexRef.current;
        const activeDeck = idx % 2 === 0 ? "A" : "B";
        const pos = engineRef.current.getPosition(activeDeck as "A" | "B");
        setPosition(pos);

        // Check if we should transition
        const plan = planRef.current;
        if (plan && idx < plan.transitions.length) {
          const transition = plan.transitions[idx];
          if (pos >= transition.mix_out_time) {
            doTransition(idx);
          }
        }
      }, 200);
    } catch (err) {
      console.error("Failed to start playback:", err);
    }
  }, []);

  const doTransition = useCallback(async (fromIndex: number) => {
    const engine = engineRef.current;
    const plan = planRef.current;
    if (!engine || !plan) return;

    const nextIndex = fromIndex + 1;
    if (nextIndex >= plan.tracks.length) return;

    // Prevent re-triggering
    if (trackIndexRef.current !== fromIndex) return;
    trackIndexRef.current = nextIndex;

    const fromDeck = fromIndex % 2 === 0 ? "A" : "B";
    const toDeck = fromIndex % 2 === 0 ? "B" : "A";
    const transition = plan.transitions[fromIndex];
    const nextTrack = plan.tracks[nextIndex];

    // Ensure next track is loaded
    const nextUrl = getAudioUrl(nextTrack.file_path);
    try {
      await engine.loadTrack(toDeck as "A" | "B", nextUrl);
    } catch {
      // Already loaded
    }

    // Calculate playback rate for BPM matching
    const rate = transition.target_bpm / (nextTrack.bpm || 120);

    // Execute crossfade transition
    const duration = transition.duration_seconds || 8;
    engine.setGain(toDeck as "A" | "B", 0);
    engine.play(toDeck as "A" | "B", transition.mix_in_time || 0, rate);
    engine.rampGain(toDeck as "A" | "B", 1, duration);
    engine.rampGain(fromDeck as "A" | "B", 0, duration);

    // Update UI
    setCurrentTrack(nextTrack);
    setCurrentIndex(nextIndex);
    setPosition(0);

    // Stop old deck after transition
    setTimeout(() => {
      engine.stopDeck(fromDeck as "A" | "B");
    }, duration * 1000);

    // Pre-load the track after next
    if (nextIndex + 1 < plan.tracks.length) {
      const preloadDeck = fromDeck;
      const preloadUrl = getAudioUrl(plan.tracks[nextIndex + 1].file_path);
      engine.loadTrack(preloadDeck as "A" | "B", preloadUrl);
    }
  }, []);

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
        const plan = msg.set_plan as SetPlan;
        const vibe = msg.vibe as { energy_arc: number[] };
        if (plan?.tracks) {
          setQueue(plan.tracks);
          startPlayback(plan);
        }
        if (vibe?.energy_arc) setEnergyArc(vibe.energy_arc);
        break;
      }
      case "skip": {
        const idx = trackIndexRef.current;
        doTransition(idx);
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
  }, [startPlayback, doTransition]);

  const { connected } = useWebSocket({
    sessionId: session?.sessionId ?? "",
    token: session?.hostToken ?? "",
    onMessage: handleWSMessage,
  });

  const handleSendChat = useCallback(
    async (message: string) => {
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
  const arcPosition = queue.length > 0 ? currentIndex / queue.length : 0;

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
          <EnergyArc arc={energyArc} currentPosition={arcPosition} />
          <Requests requests={requests} />
        </div>
      </div>
    </div>
  );
}
