import { compactNum, usd } from "../lib/format";

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="panel px-5 py-4 flex flex-col gap-1">
      <span
        className="mono text-[10px] uppercase"
        style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
      >
        {label}
      </span>
      <span className="mono text-[20px] leading-none" style={{ color: "var(--text)" }}>
        {value}
      </span>
      {sub && (
        <span className="mono text-[11px]" style={{ color: "var(--text-dim)" }}>
          {sub}
        </span>
      )}
    </div>
  );
}

export function AggregateStrip({
  sessions,
  steps,
  tokens,
  cost,
  saved,
  before,
}: {
  sessions: number;
  steps: number;
  tokens: number;
  cost: number;
  saved: number;
  before: number;
}) {
  const pct = before > 0 ? Math.round((saved / before) * 100) : 0;
  return (
    <div className="grid grid-cols-5 gap-3 px-6 py-4">
      <Stat label="sessions" value={String(sessions)} />
      <Stat label="steps observed" value={String(steps)} />
      <Stat label="tokens in" value={compactNum(tokens)} />
      <Stat
        label="context saved"
        value={compactNum(saved)}
        sub={saved > 0 ? `${pct}% of input` : "no redundancy"}
      />
      <Stat label="cost" value={usd(cost)} />
    </div>
  );
}
