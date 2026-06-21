import type { Step } from "../types";
import { modelChip, shortModel } from "../lib/models";
import { shortTime, usd } from "../lib/format";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span
        className="mono text-[10px] uppercase"
        style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
      >
        {label}
      </span>
      <span className="mono text-[12px]" style={{ color: "var(--text)" }}>
        {children}
      </span>
    </div>
  );
}

export function StepDetail({ step, onClose }: { step: Step | null; onClose: () => void }) {
  if (!step) {
    return (
      <aside
        className="w-80 shrink-0 p-5 overflow-y-auto"
        style={{ borderLeft: "1px solid var(--border)" }}
      >
        <div className="text-[13px]" style={{ color: "var(--text-faint)" }}>
          Select a step to inspect its request, model, tokens, and cost.
        </div>
      </aside>
    );
  }
  const c = modelChip(step.model_used);
  const rerouted = step.model_requested !== step.model_used;
  return (
    <aside
      className="w-80 shrink-0 p-5 overflow-y-auto fade-in flex flex-col gap-4"
      style={{ borderLeft: "1px solid var(--border)" }}
    >
      <div className="flex items-center justify-between">
        <span className="mono text-[13px]" style={{ color: "var(--text)" }}>
          step {step.step_index}
        </span>
        <button
          onClick={onClose}
          className="mono text-[11px] px-2 py-0.5 rounded"
          style={{ color: "var(--text-dim)", border: "1px solid var(--border)" }}
        >
          close
        </button>
      </div>

      <Field label="model used">
        <span style={{ color: c.fg }}>{shortModel(step.model_used)}</span>
        {rerouted && (
          <span style={{ color: "var(--text-faint)" }}>
            {" "}
            (requested {shortModel(step.model_requested)})
          </span>
        )}
      </Field>

      <div className="grid grid-cols-2 gap-4">
        <Field label="prompt tok">{step.prompt_tokens ?? "—"}</Field>
        <Field label="completion tok">{step.completion_tokens ?? "—"}</Field>
        <Field label="cost">{usd(step.cost_usd)}</Field>
        <Field label="time">{shortTime(step.timestamp)}</Field>
      </div>

      {step.tool_name && (
        <Field label="tool call">
          {step.tool_name}
          {step.caused_state_mutation && (
            <span style={{ color: "var(--warn)" }}> · state-mutating</span>
          )}
        </Field>
      )}

      {step.tool_args && (
        <div className="flex flex-col gap-1">
          <span
            className="mono text-[10px] uppercase"
            style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
          >
            tool args
          </span>
          <pre
            className="mono text-[11px] p-3 rounded overflow-x-auto"
            style={{ background: "var(--surface-2)", color: "var(--text-dim)" }}
          >
            {JSON.stringify(step.tool_args, null, 2)}
          </pre>
        </div>
      )}

      {step.assistant_text && (
        <div className="flex flex-col gap-1">
          <span
            className="mono text-[10px] uppercase"
            style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
          >
            assistant text
          </span>
          <div
            className="text-[12px] p-3 rounded"
            style={{ background: "var(--surface-2)", color: "var(--text-dim)", lineHeight: 1.6 }}
          >
            {step.assistant_text}
          </div>
        </div>
      )}
    </aside>
  );
}
