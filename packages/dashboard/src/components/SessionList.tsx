import type { SessionAgg } from "../types";
import { shortTime, usd } from "../lib/format";

export function SessionList({
  sessions,
  selected,
  onSelect,
}: {
  sessions: SessionAgg[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <aside
      className="w-72 shrink-0 overflow-y-auto"
      style={{ borderRight: "1px solid var(--border)" }}
    >
      <div
        className="mono text-[10px] uppercase px-5 py-3 sticky top-0"
        style={{
          letterSpacing: "0.08em",
          color: "var(--text-faint)",
          background: "var(--bg)",
          borderBottom: "1px solid var(--border)",
        }}
      >
        sessions
      </div>
      {sessions.length === 0 && (
        <div className="px-5 py-6 text-[13px]" style={{ color: "var(--text-faint)" }}>
          Waiting for traffic. Point a client at{" "}
          <span className="mono">localhost:8765/v1</span>.
        </div>
      )}
      {sessions.map((s) => {
        const active = s.id === selected;
        return (
          <button
            key={s.id}
            onClick={() => onSelect(s.id)}
            className="w-full text-left px-5 py-3 transition-colors"
            style={{
              background: active ? "var(--surface-2)" : "transparent",
              borderLeft: active ? "2px solid var(--accent)" : "2px solid transparent",
              borderBottom: "1px solid var(--border)",
            }}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="mono text-[12px] truncate" style={{ color: "var(--text)" }}>
                {s.id}
              </span>
              <span className="mono text-[11px]" style={{ color: "var(--text-dim)" }}>
                {usd(s.cost)}
              </span>
            </div>
            <div className="flex items-center justify-between mt-1">
              <span className="mono text-[11px]" style={{ color: "var(--text-faint)" }}>
                {s.steps.length} step{s.steps.length === 1 ? "" : "s"}
              </span>
              <span className="mono text-[11px]" style={{ color: "var(--text-faint)" }}>
                {shortTime(s.lastTs)}
              </span>
            </div>
          </button>
        );
      })}
    </aside>
  );
}
