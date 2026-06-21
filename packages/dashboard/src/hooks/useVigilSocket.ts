import { useEffect, useRef, useState } from "react";
import type { ConnState, ServerMessage, Step, Thresholds } from "../types";

const DEFAULT_THRESHOLDS: Thresholds = {
  theta_sim: 0.85,
  theta_ent: 0.3,
  window: 5,
  trip_streak: 3,
};

function wsUrl(): string {
  const env = (import.meta as { env?: Record<string, string> }).env?.VITE_VIGIL_WS;
  if (env) return env;
  // Default to the proxy on :8765 (the dashboard usually runs on :5173 in dev).
  const host = window.location.hostname || "localhost";
  return `ws://${host}:8765/ws`;
}

export interface SocketState {
  conn: ConnState;
  steps: Step[];
  priceTable: Record<string, [number, number]>;
  thresholds: Thresholds;
}

const MAX_STEPS = 2000;

export function useVigilSocket(): SocketState {
  const [conn, setConn] = useState<ConnState>("connecting");
  const [steps, setSteps] = useState<Step[]>([]);
  const [priceTable, setPriceTable] = useState<Record<string, [number, number]>>({});
  const [thresholds, setThresholds] = useState<Thresholds>(DEFAULT_THRESHOLDS);
  const retry = useRef<number>(0);
  // Dedupe across reconnects: the server replays its snapshot on every (re)connect, so a step
  // already seen must not be appended again (which would double session/aggregate cost).
  const seen = useRef<Set<string>>(new Set());

  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let timer: ReturnType<typeof setTimeout>;

    const connect = () => {
      if (closed) return;
      setConn("connecting");
      ws = new WebSocket(wsUrl());

      ws.onopen = () => {
        if (closed) return;
        retry.current = 0;
        setConn("open");
      };

      ws.onmessage = (ev) => {
        if (closed) return;
        let msg: ServerMessage;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.type === "hello") {
          setPriceTable(msg.price_table);
          if (msg.thresholds) setThresholds(msg.thresholds);
        } else if (msg.type === "step") {
          const key = `${msg.step.session_id}:${msg.step.step_index}`;
          if (seen.current.has(key)) return;
          seen.current.add(key);
          setSteps((prev) => {
            const next = [...prev, msg.step];
            return next.length > MAX_STEPS ? next.slice(next.length - MAX_STEPS) : next;
          });
        }
      };

      ws.onclose = () => {
        setConn("closed");
        if (closed) return;
        // Exponential-ish backoff, capped.
        const delay = Math.min(1000 * 2 ** retry.current, 8000);
        retry.current += 1;
        timer = setTimeout(connect, delay);
      };

      ws.onerror = () => ws?.close();
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, []);

  return { conn, steps, priceTable, thresholds };
}
