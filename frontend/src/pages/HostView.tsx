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
  const trackIndexRef = useRef(0);
  const positionTimerRef = useRef<number | null>(null);
  const sendRef = useRef<(msg: { type: string; [key: string]: unknown }) => void>(() => {});

  // Clean up audio on unmount
  useEffect(() => {
    return () => {
      engineRef.current?.destroy();
      if (positionTimerRef.current) clearInterval(positionTimerRef.current);
    };
  }, []);

  const doRollingTransition = useCallback((fromIndex: number, nextTrack: TrackInfo, _audioUrl: string, transition: TransitionInfo) => {
    const engine = engineRef.current;
    if (!engine || trackIndexRef.current !== fromIndex) return;

    const nextIndex = fromIndex + 1;
    trackIndexRef.current = nextIndex;

    const fromDeck = fromIndex % 2 === 0 ? "A" : "B";
    const toDeck = fromIndex % 2 === 0 ? "B" : "A";

    const rate = transition.target_bpm / (nextTrack.bpm || 120);
    const duration = transition.duration_seconds || 8;

    engine.setGain(toDeck as "A" | "B", 0);
    engine.play(toDeck as "A" | "B", transition.mix_in_time || 0, rate);
    engine.rampGain(toDeck as "A" | "B", 1, duration);
    engine.rampGain(fromDeck as "A" | "B", 0, duration);

    setCurrentTrack(nextTrack);
    setCurrentIndex(nextIndex);
    setPosition(0);

    setTimeout(() => {
      engine.stopDeck(fromDeck as "A" | "B");
      // Tell backend the transition is done
      sendRef.current({ type: "transition_complete" });
    }, duration * 1000);
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
        const vibe = msg.vibe as { energy_arc: number[] };
        if (vibe?.energy_arc) setEnergyArc(vibe.energy_arc);
        // Don't start playback here — wait for play_track from DJ loop
        break;
      }

      case "play_track": {
        // DJ loop tells us to play the first track
        const track = msg.track as TrackInfo;
        const audioUrl = `${API_BASE}${msg.audio_url as string}`;
        setCurrentTrack(track);
        setCurrentIndex(0);
        setQueue([track]);
        setPosition(0);

        // Initialize audio engine and play
        (async () => {
          const engine = new AudioEngine();
          await engine.resume();
          engineRef.current = engine;
          trackIndexRef.current = 0;

          await engine.loadTrack("A", audioUrl);
          engine.setGain("A", 1);
          engine.play("A", 0);

          // Start position tracking
          positionTimerRef.current = window.setInterval(() => {
            if (!engineRef.current) return;
            const deck = trackIndexRef.current % 2 === 0 ? "A" : "B";
            setPosition(engineRef.current.getPosition(deck as "A" | "B"));
          }, 200);
        })();
        break;
      }

      case "queue_next": {
        // DJ loop queued the next track
        const track = msg.track as TrackInfo;
        const audioUrl = `${API_BASE}${msg.audio_url as string}`;
        const transition = msg.transition as TransitionInfo | null;

        setQueue(prev => [...prev, track]);

        // Pre-load into standby deck
        const engine = engineRef.current;
        if (!engine) break;

        const currentIdx = trackIndexRef.current;
        const standbyDeck = currentIdx % 2 === 0 ? "B" : "A";

        (async () => {
          await engine.loadTrack(standbyDeck as "A" | "B", audioUrl);

          // If we have transition info, set up auto-transition
          if (transition) {
            const checkInterval = setInterval(() => {
              if (!engineRef.current) { clearInterval(checkInterval); return; }
              if (trackIndexRef.current !== currentIdx) { clearInterval(checkInterval); return; }

              const activeDeck = currentIdx % 2 === 0 ? "A" : "B";
              const pos = engineRef.current.getPosition(activeDeck as "A" | "B");

              if (pos >= transition.mix_out_time) {
                clearInterval(checkInterval);
                doRollingTransition(currentIdx, track, audioUrl, transition);
              }
            }, 200);
          }
        })();
        break;
      }

      case "skip": {
        // TODO: Force immediate transition via rolling planner
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
  }, [doRollingTransition]);

  const { connected, send } = useWebSocket({
    sessionId: session?.sessionId ?? "",
    token: session?.hostToken ?? "",
    onMessage: handleWSMessage,
  });

  // Keep sendRef in sync
  useEffect(() => { sendRef.current = send; }, [send]);

  const handleSendChat = useCallback(
    async (message: string) => {
      // Don't add optimistically — WebSocket broadcast will add it
      await sendChat(message);
    },
    [sendChat]
  );

  const handlePause = useCallback(() => {
    engineRef.current?.pause();
    setSessionState("paused");
  }, []);

  const handleResume = useCallback(() => {
    engineRef.current?.unpause();
    setSessionState("playing");
  }, []);

  const handleSkip = useCallback(() => {
    // Tell backend to skip — DJ loop will send the next track
    sendRef.current({ type: "chat", message: "skip" });
  }, []);

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
            onPause={handlePause}
            onResume={handleResume}
            onSkip={handleSkip}
          />
          <Queue tracks={queue} currentIndex={currentIndex} />
          <EnergyArc arc={energyArc} currentPosition={arcPosition} />
          <Requests requests={requests} />
        </div>
      </div>
    </div>
  );
}
