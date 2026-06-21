import { useEffect, useMemo, useState } from "react";
import { useVigilSocket } from "./hooks/useVigilSocket";
import type { SessionAgg, Step } from "./types";
import { TopBar } from "./components/TopBar";
import { AggregateStrip } from "./components/AggregateStrip";
import { CostSparkline } from "./components/CostSparkline";
import { TrajectoryChart } from "./components/TrajectoryChart";
import { SessionList } from "./components/SessionList";
import { StepLog } from "./components/StepLog";
import { StepDetail } from "./components/StepDetail";

function aggregate(steps: Step[]): SessionAgg[] {
  const map = new Map<string, SessionAgg>();
  for (const s of steps) {
    let agg = map.get(s.session_id);
    if (!agg) {
      agg = {
        id: s.session_id,
        steps: [],
        cost: 0,
        tokensBefore: 0,
        tokensAfter: 0,
        lastTs: s.timestamp,
        models: new Set(),
      };
      map.set(s.session_id, agg);
    }
    agg.steps.push(s);
    agg.cost += s.cost_usd || 0;
    agg.tokensBefore += s.tokens_before_compression ?? 0;
    agg.tokensAfter += s.tokens_after_compression ?? 0;
    agg.models.add(s.model_used);
    if (s.timestamp > agg.lastTs) agg.lastTs = s.timestamp;
  }
  for (const agg of map.values()) agg.steps.sort((a, b) => a.step_index - b.step_index);
  return [...map.values()].sort((a, b) => (a.lastTs < b.lastTs ? 1 : -1));
}

export default function App() {
  const { conn, steps, thresholds } = useVigilSocket();
  const sessions = useMemo(() => aggregate(steps), [steps]);

  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  // Auto-select the most recently active session once traffic appears.
  useEffect(() => {
    if (selectedSession === null && sessions.length > 0) {
      setSelectedSession(sessions[0].id);
    }
  }, [sessions, selectedSession]);

  const current = sessions.find((s) => s.id === selectedSession) ?? null;
  const currentSteps = current?.steps ?? [];
  // selectedStep is a stable step_index value (unique per session), not an array position, so
  // it stays correct as the steps array grows, reorders, or is windowed.
  const detailStep =
    selectedStep !== null ? (currentSteps.find((s) => s.step_index === selectedStep) ?? null) : null;

  const totals = useMemo(() => {
    let cost = 0;
    let tokens = 0;
    for (const s of steps) {
      cost += s.cost_usd || 0;
      tokens += s.prompt_tokens ?? 0;
    }
    return { cost, tokens };
  }, [steps]);

  return (
    <div className="h-full flex flex-col" style={{ background: "var(--bg)" }}>
      <TopBar conn={conn} totalCost={current?.cost ?? 0} />

      <div className="grid grid-cols-4 gap-3 px-6 pt-4">
        <div className="col-span-3">
          <AggregateStrip
            sessions={sessions.length}
            steps={steps.length}
            tokens={totals.tokens}
            cost={totals.cost}
          />
        </div>
        <div className="pt-4 pr-0">
          <CostSparkline steps={currentSteps} />
        </div>
      </div>

      <div className="px-6 pb-3">
        <TrajectoryChart steps={currentSteps} thresholds={thresholds} />
      </div>

      <div className="flex flex-1 min-h-0 mt-1" style={{ borderTop: "1px solid var(--border)" }}>
        <SessionList
          sessions={sessions}
          selected={selectedSession}
          onSelect={(id) => {
            setSelectedSession(id);
            setSelectedStep(null);
          }}
        />
        <StepLog
          steps={currentSteps}
          selectedStepIndex={selectedStep}
          onSelect={(stepIndex) => setSelectedStep(stepIndex)}
        />
        <StepDetail step={detailStep} onClose={() => setSelectedStep(null)} />
      </div>
    </div>
  );
}
