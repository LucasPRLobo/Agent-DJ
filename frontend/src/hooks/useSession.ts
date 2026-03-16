import { useState, useCallback } from "react";

const API_BASE = `http://${window.location.hostname}:8000`;

interface SessionData {
  sessionId: string;
  hostToken: string;
}

export function useSession() {
  const [session, setSession] = useState<SessionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const createSession = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/session`, { method: "POST" });
      const data = await res.json();
      const sess = { sessionId: data.session_id, hostToken: data.host_token };
      setSession(sess);
      return sess;
    } catch (e) {
      setError("Failed to create session");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const sendChat = useCallback(async (message: string) => {
    if (!session) return null;
    try {
      const res = await fetch(
        `${API_BASE}/session/${session.sessionId}/chat?token=${session.hostToken}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message }),
        }
      );
      return await res.json();
    } catch {
      return null;
    }
  }, [session]);

  const getQrUrl = useCallback(() => {
    if (!session) return "";
    const frontendBase = window.location.origin;
    return `${API_BASE}/session/${session.sessionId}/qr?base_url=${encodeURIComponent(frontendBase)}`;
  }, [session]);

  return { session, loading, error, createSession, sendChat, getQrUrl };
}
