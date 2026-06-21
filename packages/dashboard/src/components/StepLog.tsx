import type { Step } from "../types";
import { modelChip, shortModel } from "../lib/models";
import { compactNum, shortTime, truncate, usd } from "../lib/format";

function ModelChip({ model }: { model: string }) {
  const c = modelChip(model);
  return (
    <span
      className="mono text-[10px] px-1.5 py-0.5 rounded inline-flex items-center gap-1"
      style={{ background: c.bg, color: c.fg }}
    >
      <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: c.dot }} />
      {shortModel(model)}
    </span>
  );
}

function StepRow({
  step,
  active,
  onClick,
}: {
  step: Step;
  active: boolean;
  onClick: () => void;
}) {
  const tokens = (step.prompt_tokens ?? 0) + (step.completion_tokens ?? 0);
  return (
    <button
      onClick={onClick}
      className="w-full text-left px-5 py-3 fade-in transition-colors divide-row"
      style={{ background: active ? "var(--surface-2)" : "transparent" }}
    >
      <div className="flex items-center gap-3">
        <span
          className="mono text-[11px] w-7 shrink-0 text-right"
          style={{ color: "var(--text-faint)" }}
        >
          {step.step_index}
        </span>
        <ModelChip model={step.model_used} />
        {step.tool_name && (
          <span className="mono text-[11px]" style={{ color: "var(--text-dim)" }}>
            {step.tool_name}
            {step.caused_state_mutation && (
              <span style={{ color: "var(--warn)" }}> ·write</span>
            )}
          </span>
        )}
        <span className="flex-1" />
        <span className="mono text-[11px]" style={{ color: "var(--text-faint)" }}>
          {compactNum(tokens)} tok
        </span>
        <span className="mono text-[11px] w-16 text-right" style={{ color: "var(--text-dim)" }}>
          {usd(step.cost_usd)}
        </span>
        <span className="mono text-[10px] w-16 text-right" style={{ color: "var(--text-faint)" }}>
          {shortTime(step.timestamp)}
        </span>
      </div>
      {(step.assistant_text || step.tool_name) && (
        <div className="mt-1.5 pl-10 text-[12px]" style={{ color: "var(--text-dim)" }}>
          {truncate(step.assistant_text || JSON.stringify(step.tool_args ?? {}))}
        </div>
      )}
    </button>
  );
}

export function StepLog({
  steps,
  selectedStepIndex,
  onSelect,
}: {
  steps: Step[];
  selectedStepIndex: number | null;
  onSelect: (stepIndex: number) => void;
}) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div
        className="mono text-[10px] uppercase px-5 py-3 sticky top-0 flex justify-between"
        style={{
          letterSpacing: "0.08em",
          color: "var(--text-faint)",
          background: "var(--bg)",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <span>trajectory</span>
        <span>{steps.length} steps</span>
      </div>
      {steps.length === 0 ? (
        <div className="px-5 py-8 text-[13px]" style={{ color: "var(--text-faint)" }}>
          No steps yet for this session.
        </div>
      ) : (
        steps.map((s) => (
          <StepRow
            key={`${s.session_id}-${s.step_index}`}
            step={s}
            active={s.step_index === selectedStepIndex}
            onClick={() => onSelect(s.step_index)}
          />
        ))
      )}
    </div>
  );
}
