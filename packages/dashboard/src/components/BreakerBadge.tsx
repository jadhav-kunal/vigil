import type { BreakerState } from "../types";

const STYLE: Record<BreakerState, { label: string; color: string; bg: string; pulse: boolean }> = {
  CLOSED: { label: "closed", color: "var(--ok)", bg: "rgba(62,207,142,0.10)", pulse: false },
  HALF_OPEN: { label: "mitigating", color: "var(--warn)", bg: "rgba(229,181,103,0.12)", pulse: true },
  OPEN: { label: "halted", color: "var(--danger)", bg: "rgba(229,72,77,0.14)", pulse: true },
};

export function BreakerBadge({ state }: { state: BreakerState }) {
  const s = STYLE[state];
  return (
    <span
      className="mono text-[10px] uppercase px-2 py-0.5 rounded inline-flex items-center gap-1.5"
      style={{ letterSpacing: "0.06em", color: s.color, background: s.bg }}
    >
      <span
        className={"inline-block w-1.5 h-1.5 rounded-full " + (s.pulse ? "pulse" : "")}
        style={{ background: s.color }}
      />
      breaker · {s.label}
    </span>
  );
}
