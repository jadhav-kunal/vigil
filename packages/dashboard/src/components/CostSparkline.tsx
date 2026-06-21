import type { Step } from "../types";
import { usd } from "../lib/format";

// Hand-rolled SVG: cumulative spend across the selected step sequence. No chart library.
export function CostSparkline({ steps }: { steps: Step[] }) {
  const w = 100;
  const h = 36;
  let cum = 0;
  const cumulative = steps.map((s) => (cum += s.cost_usd || 0));
  const max = Math.max(cumulative[cumulative.length - 1] ?? 0, 1e-9);

  const points =
    cumulative.length > 1
      ? cumulative
          .map((c, i) => {
            const x = (i / (cumulative.length - 1)) * w;
            const y = h - (c / max) * (h - 4) - 2;
            return `${x.toFixed(2)},${y.toFixed(2)}`;
          })
          .join(" ")
      : "";

  return (
    <div className="panel px-5 py-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span
          className="mono text-[10px] uppercase"
          style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
        >
          cumulative spend
        </span>
        <span className="mono text-[12px]" style={{ color: "var(--text-dim)" }}>
          {usd(cum)}
        </span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="w-full h-10">
        {points && (
          <>
            <polyline
              points={points}
              fill="none"
              stroke="var(--accent)"
              strokeWidth={1.2}
              vectorEffect="non-scaling-stroke"
            />
            <polyline
              points={`0,${h} ${points} ${w},${h}`}
              fill="var(--accent)"
              opacity={0.08}
              stroke="none"
            />
          </>
        )}
      </svg>
    </div>
  );
}
