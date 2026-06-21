import { useState } from "react";

// The canonical AI coding-agent integration prompt — keep in sync with cli/vigil.js and the README.
const AGENT_PROMPT = `Integrate Vigil into this codebase. Vigil is a transparent LLM proxy that
watches an agent's trajectory and halts runaway loops before they burn budget. It speaks the
OpenAI and Anthropic APIs and forwards to the real provider, so the ONLY change needed is the
client's base URL — do not change prompts, models, keys, or any other logic.

Steps:
1. Find every place an OpenAI or Anthropic client is constructed (e.g. OpenAI(), AsyncOpenAI(),
   Anthropic(), LangChain/LlamaIndex model configs, or a raw base_url/baseURL setting).
2. Point base_url at the Vigil proxy, read from an env var so it is easy to toggle:
     OpenAI:    base_url = os.environ.get("VIGIL_BASE_URL", "http://localhost:8765/v1")
     Anthropic: base_url = os.environ.get("VIGIL_BASE_URL", "http://localhost:8765")
3. Leave the API key exactly as-is — Vigil passes it straight through and never stores it.
4. Recommended: set a per-run session id header so Vigil groups one agent run into one trajectory:
     default_headers={"x-vigil-session-id": "<a stable id for this run>"}
5. Verify: curl http://localhost:8765/health returns {"status":"ok"}; after a request,
   curl http://localhost:8765/metrics/session/<id> shows the captured steps.

Make the minimal edit, keep it behind the VIGIL_BASE_URL env var, and do not alter behavior.`;

function Copy({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(text).then(
          () => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1400);
          },
          () => {}
        );
      }}
      className="mono text-[10px] uppercase px-2 py-0.5 rounded"
      style={{
        letterSpacing: "0.06em",
        color: copied ? "var(--ok)" : "var(--text-dim)",
        border: "1px solid var(--border)",
      }}
    >
      {copied ? "copied" : "copy"}
    </button>
  );
}

function Block({ title, code }: { title?: string; code: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      {title && (
        <div className="flex items-center justify-between">
          <span
            className="mono text-[10px] uppercase"
            style={{ letterSpacing: "0.08em", color: "var(--text-faint)" }}
          >
            {title}
          </span>
          <Copy text={code} />
        </div>
      )}
      <pre
        className="mono text-[12px] p-3 rounded overflow-x-auto"
        style={{ background: "var(--surface-2)", color: "var(--text-dim)", lineHeight: 1.6 }}
      >
        {code}
      </pre>
    </div>
  );
}

function Section({ n, title, children }: { n: string; title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="mono text-[13px]" style={{ color: "var(--text)" }}>
        <span style={{ color: "var(--accent)" }}>{n}</span> {title}
      </h2>
      {children}
    </section>
  );
}

export function DocsPanel() {
  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-8 py-8 flex flex-col gap-9">
        <div className="flex flex-col gap-2">
          <h1 className="mono text-[18px]" style={{ color: "var(--text)" }}>
            Integrate Vigil
          </h1>
          <p className="text-[13px]" style={{ color: "var(--text-dim)", lineHeight: 1.6 }}>
            Vigil is a drop-in proxy. Change one line — your client's base URL — and keep your API
            key. Vigil forwards to your real provider, watches every step, and halts runaway loops.
          </p>
        </div>

        <Section n="1." title="One-line change">
          <Block
            title="python · openai"
            code={'from openai import OpenAI\n\nclient = OpenAI(\n    base_url="http://localhost:8765/v1",  # the only change\n    api_key="sk-...",                      # passed straight through, never stored\n)'}
          />
          <Block
            title="python · anthropic"
            code={'from anthropic import Anthropic\n\nclient = Anthropic(base_url="http://localhost:8765")'}
          />
          <Block
            title="node · openai"
            code={"import OpenAI from 'openai';\n\nconst client = new OpenAI({\n  baseURL: 'http://localhost:8765/v1',\n  apiKey: process.env.OPENAI_API_KEY,\n});"}
          />
        </Section>

        <Section n="2." title="CLI — npx">
          <p className="text-[12px]" style={{ color: "var(--text-dim)" }}>
            From a clone, use <span className="mono">node cli/vigil.js &lt;cmd&gt;</span>; once
            published, <span className="mono">npx vigil &lt;cmd&gt;</span>.
          </p>
          <Block
            code={"npx vigil init      # print the one-line base_url integration\nnpx vigil prompt    # copy-paste prompt for an AI coding agent\nnpx vigil demo      # scripted loop -> watch the breaker trip + freeze cost"}
          />
        </Section>

        <Section n="3." title="Give this to your AI coding agent">
          <p className="text-[12px]" style={{ color: "var(--text-dim)" }}>
            Paste into Claude Code, Cursor, or any coding agent to wire Vigil into an existing app.
          </p>
          <Block title="integration prompt" code={AGENT_PROMPT} />
        </Section>

        <Section n="4." title="Verify & inspect">
          <Block
            code={"curl http://localhost:8765/health                 # {\"status\":\"ok\"}\ncurl http://localhost:8765/metrics/aggregate      # cross-session totals (counts only)\ncurl http://localhost:8765/metrics/session/<id>   # one trajectory's steps + savings"}
          />
        </Section>
      </div>
    </div>
  );
}
