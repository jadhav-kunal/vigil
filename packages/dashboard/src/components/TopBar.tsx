import type { ConnState } from "../types";
import { usd } from "../lib/format";

const CONN_LABEL: Record<ConnState, string> = {
  connecting: "connecting",
  open: "live",
  closed: "offline",
};
const CONN_COLOR: Record<ConnState, string> = {
  connecting: "var(--warn)",
  open: "var(--ok)",
  closed: "var(--danger)",
};

export function TopBar({ conn, totalCost }: { conn: ConnState; totalCost: number }) {
  return (
    <header
      className="flex items-center justify-between px-6 h-14 shrink-0"
      style={{ borderBottom: "1px solid var(--border)" }}
    >
      <div className="flex items-center gap-3">
        <span className="mono text-[15px] tracking-tight" style={{ letterSpacing: "-0.01em" }}>
          vigil
        </span>
        <span
          className="mono text-[10px] uppercase px-1.5 py-0.5 rounded"
          style={{
            letterSpacing: "0.08em",
            color: "var(--text-faint)",
            border: "1px solid var(--border)",
          }}
        >
          instrument panel
        </span>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex flex-col items-end">
          <span
            className="mono text-[10px] uppercase"
            style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
          >
            session spend
          </span>
          <span className="mono text-[22px] leading-none" style={{ color: "var(--text)" }}>
            {usd(totalCost)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={"inline-block w-2 h-2 rounded-full " + (conn === "open" ? "pulse" : "")}
            style={{ background: CONN_COLOR[conn] }}
          />
          <span
            className="mono text-[11px] uppercase"
            style={{ letterSpacing: "0.06em", color: "var(--text-dim)" }}
          >
            {CONN_LABEL[conn]}
          </span>
        </div>
      </div>
    </header>
  );
}
