import type { BreakerInfo, ConnState } from "../types";
import { usd } from "../lib/format";
import { BreakerBadge } from "./BreakerBadge";

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

function Tab({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="mono text-[10px] uppercase px-2 py-0.5 rounded"
      style={{
        letterSpacing: "0.08em",
        color: active ? "var(--text)" : "var(--text-faint)",
        border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
      }}
    >
      {label}
    </button>
  );
}

export function TopBar({
  conn,
  totalCost,
  breaker,
  view,
  onView,
}: {
  conn: ConnState;
  totalCost: number;
  breaker: BreakerInfo | null;
  view: "live" | "docs";
  onView: (v: "live" | "docs") => void;
}) {
  // When the breaker halts a session, freeze the meter green and surface what was capped.
  const halted = breaker?.state === "OPEN";
  const meterColor = halted ? "var(--ok)" : "var(--text)";
  return (
    <header
      className="flex items-center justify-between px-6 h-14 shrink-0"
      style={{ borderBottom: "1px solid var(--border)" }}
    >
      <div className="flex items-center gap-3">
        <span className="mono text-[15px] tracking-tight" style={{ letterSpacing: "-0.01em" }}>
          vigil
        </span>
        <Tab label="live" active={view === "live"} onClick={() => onView("live")} />
        <Tab label="docs" active={view === "docs"} onClick={() => onView("docs")} />
        {breaker && view === "live" && <BreakerBadge state={breaker.state} />}
      </div>

      <div className="flex items-center gap-6">
        {halted && breaker && breaker.savedEstimate > 0 && (
          <div className="flex flex-col items-end">
            <span
              className="mono text-[10px] uppercase"
              style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
            >
              loop cost capped
            </span>
            <span className="mono text-[16px] leading-none" style={{ color: "var(--ok)" }}>
              ~{usd(breaker.savedEstimate)}
            </span>
          </div>
        )}
        <div className="flex flex-col items-end">
          <span
            className="mono text-[10px] uppercase"
            style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
          >
            {halted ? "spend (frozen)" : "session spend"}
          </span>
          <span className="mono text-[22px] leading-none" style={{ color: meterColor }}>
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
