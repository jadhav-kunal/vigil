# Vigil

**A transparent LLM proxy that catches your agent looping — and stops it — while it runs.**

AI agents loop and burn money. Every shipping tool either tells you *after* the fact
(Langfuse, Arize, LangSmith) or counts tokens but not *meaning* (budget caps a reworded loop
sails straight past). Vigil sits in the empty quadrant: **in-flight + semantic**. It watches
the trajectory as it happens, detects when the agent is going in circles, and intervenes —
downgrading the model, stripping write tools, or halting outright — before the bill runs away.

## 30-second quickstart

Change **one line** — the `base_url` of your existing OpenAI/Anthropic client. Nothing else.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8765/v1",   # <- the only change
    api_key="sk-...",                       # your real provider key, passed straight through
)
```

Start the proxy:

```bash
uv sync
uv run uvicorn vigil_proxy.app:app --host 0.0.0.0 --port 8765
# health check
curl localhost:8765/health      # -> {"status":"ok"}
```

Your agent behaves identically — except Vigil now watches every step, and when the trajectory
degenerates it acts.

## Run it locally — step by step

**Prerequisites:** Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) for the proxy; Node 18+
for the dashboard (optional). No Redis, no cloud account, no extra keys.

### 1. Install the proxy

```bash
cd vigil
uv sync                 # creates .venv and installs the proxy + deps
cp .env.example .env     # optional — the defaults already target OpenAI
```

### 2. Start the proxy

```bash
uv run uvicorn vigil_proxy.app:app --host 0.0.0.0 --port 8765
curl localhost:8765/health        # -> {"status":"ok"}
```

> First start downloads the embedding model `all-MiniLM-L6-v2` (~90 MB, one time). To skip it
> for a quick spin (uses a deterministic hashing embedder instead), prefix the command with
> `VIGIL_EMBED_HASHING=true`.

By default the proxy forwards to OpenAI (`OPENAI_BASE_URL=https://api.openai.com/v1`). Point it
anywhere OpenAI-compatible by editing `.env`.

### 3. Send traffic through it

Either change your client's `base_url` (the quickstart above), or test directly with curl:

```bash
curl localhost:8765/v1/chat/completions \
  -H "authorization: Bearer $OPENAI_API_KEY" \
  -H "x-vigil-session-id: demo" \
  -H "content-type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'
```

The optional `x-vigil-session-id` header groups requests into one trajectory (any string; one is
auto-generated if omitted). Your provider key is passed straight through in `Authorization` and
is **never stored**.

### 4. Watch it live — the dashboard (optional)

```bash
cd packages/dashboard
npm install
npm run dev                       # http://localhost:5173
```

It connects to the proxy's WebSocket at `ws://localhost:8765/ws` and streams every step — cost
meter, similarity/entropy charts, breaker state, per-step model and token counts. (If the proxy
isn't on `localhost`, set `VITE_VIGIL_WS=ws://host:8765/ws` before `npm run dev`.)

### 5. Inspect and act on a session

```bash
curl localhost:8765/metrics/session/demo              # steps, tokens, tokens_saved, cost
curl localhost:8765/sessions/demo/breaker             # breaker state + post-mortem
curl -X POST localhost:8765/sessions/demo/replay      # cached-trace replay (zero upstream calls)
curl -X POST localhost:8765/sessions/demo/fork \
  -H "content-type: application/json" -H "authorization: Bearer $OPENAI_API_KEY" \
  -d '{"step_index":0,"model":"gpt-4o-mini"}'          # counterfactual model fork (one call)
curl -X POST localhost:8765/sessions/demo/override    # reset a tripped breaker to CLOSED
```

### 6. See a loop get caught (no API key needed)

The deterministic benchmark drives scripted looping / healthy / normal trajectories through the
**real** watchdog, breaker, compressor and governor — proving the breaker trips on loops, stays
quiet on healthy work, and quantifying the savings:

```bash
uv pip install -e ".[eval]"       # matplotlib + scipy (plots + stats)
uv run python -m eval.benchmark --seeds 20
ls eval/out/                       # savings_table.md, ablation.md, net_savings.md, *.png, ...
```

### 7. Run the tests

```bash
uv run pytest -q                   # full suite (proxy + eval)
# full quality gate, as run before every commit:
uv run ruff check packages/proxy eval && uv run black --check packages/proxy eval \
  && uv run mypy && uv run pytest -q
```

## Endpoints

| Method & path | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible proxy (streaming + non-streaming) |
| `POST /v1/messages` | Anthropic-compatible proxy |
| `GET /health` | Liveness check |
| `GET /metrics/session/{id}` | Per-session steps, tokens, compression savings, cost |
| `GET /metrics/aggregate` | Cross-session totals (counts only — never prompt content) |
| `GET /sessions/{id}/breaker` | Breaker state, trip step, post-mortem |
| `POST /sessions/{id}/override` | Reset a tripped breaker to CLOSED |
| `POST /sessions/{id}/replay` | Cached-trace replay — rebuilds the trajectory, zero upstream calls |
| `POST /sessions/{id}/fork` | Re-run one step with a swapped model; diffs reasoning vs tool output |
| `WS /ws` | Live step/cost/breaker stream for the dashboard |

## Configuration

Everything is configured through environment variables (see `.env.example` for the full,
commented list). The defaults boot a fully working local proxy. The notable toggles:

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` | OpenAI / Anthropic | Upstream the proxy forwards to (any OpenAI-compatible host) |
| `VIGIL_ALLOW_UPSTREAM_HEADER` | `false` | Allow per-request routing via the `x-vigil-upstream` header (constrain with `VIGIL_UPSTREAM_ALLOWLIST`) |
| `VIGIL_COMPRESS_ENABLED` | `true` | Layer-1 loop-aware context compression (free, structural) |
| `VIGIL_GOVERNOR_ENABLED` | `false` | Per-step model routing to the cheapest adequate model |
| `VIGIL_FORENSICS_ENABLED` | `true` | Cache exchanges for replay/fork |
| `REDIS_LANGCACHE_*` | unset | Semantic cache (M4): serve repeats from cache, skip upstream |
| `VIGIL_EMBED_HASHING` | `false` | Use the offline hashing embedder (skip the ML model download) |
| `VIGIL_WINDOW` / `VIGIL_TRIP_STREAK` / `VIGIL_THETA_SIM` / `VIGIL_THETA_ENT` | `5 / 3 / 0.85 / 0.30` | Watchdog detection thresholds |
| `VIGIL_JUDGE_*` | unset | Optional LLM goal-judge (degrades to cosine+entropy if absent) |

**Where the provider URL comes from:** Vigil forwards to the upstream set by `OPENAI_BASE_URL` /
`ANTHROPIC_BASE_URL` (the request only carries the *key*, in `Authorization`, passed through and
never stored). Point those at any OpenAI-compatible host (Azure, OpenRouter, vLLM, …). For
**per-request** routing, enable `VIGIL_ALLOW_UPSTREAM_HEADER=true` and send a header:

```bash
curl http://localhost:8765/v1/chat/completions \
  -H "authorization: Bearer $KEY" \
  -H "x-vigil-upstream: https://openrouter.ai/api/v1" \
  -H "content-type: application/json" \
  -d '{"model":"...","messages":[...]}'
```

Constrain it with `VIGIL_UPSTREAM_ALLOWLIST` (comma-separated URL prefixes) — an unconstrained
upstream is an SSRF risk. The `x-vigil-*` control headers are stripped before forwarding.

## CLI

A dependency-free Node CLI for setup and a self-contained demo. From a clone use
`node cli/vigil.js <cmd>`; once published, `npx vigil <cmd>`.

```bash
npx vigil init      # print the one-line base_url integration (OpenAI/Anthropic, Python/Node)
npx vigil prompt    # print a copy-paste prompt for an AI coding agent (below)
npx vigil demo      # scripted runaway loop -> watch the breaker trip and freeze the cost meter
```

`vigil demo` starts a looping mock upstream, drives a running proxy with one session, and shows
the breaker move `CLOSED → HALF_OPEN → OPEN` (by ~step 7) with the projected cost it capped — no
API key required. The same setup and prompt are also on the dashboard's **docs** tab.

## Integrate with an AI coding agent

Paste this into Claude Code, Cursor, or any coding agent to wire Vigil into an existing app:

```text
Integrate Vigil into this codebase. Vigil is a transparent LLM proxy that watches an agent's
trajectory and halts runaway loops before they burn budget. It speaks the OpenAI and Anthropic
APIs and forwards to the real provider, so the ONLY change needed is the client's base URL — do
not change prompts, models, keys, or any other logic.

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

Make the minimal edit, keep it behind the VIGIL_BASE_URL env var, and do not alter behavior.
```

## What it does (plainly)

- **Watches** every request/response as a step in a trajectory.
- **Detects** semantic loops and meltdowns in flight (embedding similarity + tool-diversity
  entropy + a state-mutation penalty + a periodic goal-judge), not just token counts.
- **Intervenes** via a circuit breaker: downgrade the model, go read-only, or stop and return
  a post-mortem — capping a runaway loop's cost.
- **Routes** each step to the cheapest model that fits the work (effort governor).
- **Compresses** redundant loop context losslessly before it hits the upstream.
- **Records** everything for deterministic, side-effect-free **cached-trace replay** and
  counterfactual model forks.

## Honest scope

- Vigil is the **reliability** layer (*is the agent making progress?*), **not** the
  **quality** layer (*is the answer correct?*). Correctness is left to eval tools.
- Replay is **cached-trace replay**, not environment replication — we re-feed cached tool
  outputs and never re-touch your production environment.
- Pure polling/batch loops are *correct* behavior that *looks* stuck. Vigil ships explicit
  mitigations (state-mutation penalty, goal-judge, a `polling` session mode, manual override)
  rather than pretending false positives are impossible.
- The free core (loop breaker, effort governor, structural dedup) is net-positive
  unconditionally. The **paid layers** (ML compression, semantic cache) are
  **regime-dependent** — they net positive only with large inputs / expensive models / high
  cache-hit rates. The evaluation harness reports the break-even per workload; we never claim
  a flat saving.

## Benchmark — proving the savings

The savings claims are backed by a deterministic, offline benchmark (no network, no real LLM —
a scripted mock upstream replays labeled trajectories, so the whole thing is reproducible from a
seed). Run it with the commands in [step 6](#6-see-a-loop-get-caught-no-api-key-needed) above.

It runs an **ablation ladder** (C0 control → each mechanism alone → full Vigil) over three
datasets — looping sessions (must trip), healthy-repetitive sessions (must **not** trip), and
normal tasks (verifiable success) — on a paired design, and writes six artifacts to `eval/out/`:
a savings table with bootstrap 95% CIs, a per-mechanism ablation, a **net** accounting (savings
minus Vigil's own overhead) with break-even regimes, a detection report (confusion matrix split
into *cosine/entropy math only* vs *math + goal-judge*, so the math isn't credited with what the
judge adds), `results.json`, and PNG plots. The headline carries a paired Wilcoxon test, and
compression ratios are reported **per dataset** (looping context compresses far more than normal
context — conflating them is the dishonest move we avoid).

## Local vs. full mode

The same binary runs both. With nothing but a SQLite file and one provider key it boots and
demos fully. Redis, Sentry, Phoenix/Arize tracing, ML compression, semantic cache, and the
goal-judge LLM are all env-gated — set a key to light one up, leave it unset and that
capability is silently skipped.

## Status

Active development. Working today: the pass-through proxy + step capture, the live dashboard, the
semantic watchdog, the circuit breaker, loop-aware compression, the effort governor, the
deterministic evaluation harness, and cached-trace replay + fork. Next up: env-gated sponsor
integrations, aggregate metrics + a one-command CLI demo, and an optional Redis backend.
