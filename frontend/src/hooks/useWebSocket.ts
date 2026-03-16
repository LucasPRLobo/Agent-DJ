import { useCallback, useEffect, useRef, useState } from "react";

export interface WSMessage {
  type: string;
  [key: string]: unknown;
}

interface UseWebSocketOptions {
  sessionId: string;
  token: string;
  onMessage?: (msg: WSMessage) => void;
}

export function useWebSocket({ sessionId, token, onMessage }: UseWebSocketOptions) {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number | null>(null);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname;
    const port = 8000;
    const url = `${protocol}//${host}:${port}/session/${sessionId}/ws?token=${token}`;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WSMessage;
        onMessage?.(msg);
      } catch {
        console.warn("Invalid WS message:", event.data);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Reconnect after 2 seconds
      reconnectTimer.current = window.setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();
  }, [sessionId, token, onMessage]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((msg: WSMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { connected, send };
}
