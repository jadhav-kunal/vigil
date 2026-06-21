import type { Step, Thresholds } from "../types";

// Hand-rolled SVG trajectory chart (no chart library). Two lines over the step sequence:
// final score (similarity after the state penalty) and tool-diversity entropy, with dashed
// threshold lines and a red band on each breaching step.
const W = 100;
const H = 40;
const PAD_Y = 3;

export function TrajectoryChart({ steps, thresholds }: { steps: Step[]; thresholds: Thresholds }) {
  const finals = steps.map((s) => s.final_score ?? 0);
  const entropies = steps.map((s) => s.tool_entropy ?? 0);
  const yMax = Math.max(1.0, thresholds.theta_sim, ...entropies, ...finals) * 1.08;

  const x = (i: number) => (steps.length <= 1 ? 0 : (i / (steps.length - 1)) * W);
  const y = (v: number) => H - PAD_Y - (v / yMax) * (H - 2 * PAD_Y);

  const line = (vals: number[]) =>
    vals.length > 1 ? vals.map((v, i) => `${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(" ") : "";

  const tripped = steps.some((s) => s.watchdog_tripped);

  return (
    <div className="panel px-5 py-4">
      <div className="flex items-center justify-between mb-2">
        <span
          className="mono text-[10px] uppercase"
          style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
        >
          trajectory · similarity vs tool diversity
        </span>
        <div className="flex items-center gap-4">
          <Legend color="var(--accent)" label="final score" />
          <Legend color="#4bb1c9" label="tool entropy" />
          {tripped && (
            <span
              className="mono text-[10px] uppercase px-1.5 py-0.5 rounded pulse"
              style={{
                letterSpacing: "0.06em",
                color: "var(--danger)",
                background: "rgba(229,72,77,0.12)",
              }}
            >
              loop detected
            </span>
          )}
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="w-full h-28">
        {/* breach bands */}
        {steps.map((s, i) =>
          s.watchdog_breach ? (
            <rect
              key={i}
              x={Math.max(0, x(i) - 0.6)}
              y={0}
              width={1.2}
              height={H}
              fill="var(--danger)"
              opacity={0.14}
            />
          ) : null,
        )}

        {/* threshold lines (dashed) */}
        <ThresholdLine yv={y(thresholds.theta_sim)} color="var(--accent)" />
        <ThresholdLine yv={y(thresholds.theta_ent)} color="#4bb1c9" />

        {/* metric lines */}
        {line(finals) && (
          <polyline
            points={line(finals)}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={1.3}
            vectorEffect="non-scaling-stroke"
          />
        )}
        {line(entropies) && (
          <polyline
            points={line(entropies)}
            fill="none"
            stroke="#4bb1c9"
            strokeWidth={1.3}
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>

      <div className="flex justify-between mt-1">
        <span className="mono text-[10px]" style={{ color: "var(--text-faint)" }}>
          θsim {thresholds.theta_sim} · θent {thresholds.theta_ent} · W{thresholds.window} · K
          {thresholds.trip_streak}
        </span>
        <span className="mono text-[10px]" style={{ color: "var(--text-faint)" }}>
          {steps.length} steps
        </span>
      </div>
    </div>
  );
}

function ThresholdLine({ yv, color }: { yv: number; color: string }) {
  return (
    <line
      x1={0}
      x2={W}
      y1={yv}
      y2={yv}
      stroke={color}
      strokeWidth={0.6}
      strokeDasharray="2 2"
      opacity={0.5}
      vectorEffect="non-scaling-stroke"
    />
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block w-2.5 h-[2px]" style={{ background: color }} />
      <span className="mono text-[10px]" style={{ color: "var(--text-dim)" }}>
        {label}
      </span>
    </span>
  );
}
