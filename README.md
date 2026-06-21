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

## Local vs. full mode

The same binary runs both. With nothing but a SQLite file and one provider key it boots and
demos fully. Redis, Sentry, Phoenix/Arize tracing, ML compression, semantic cache, and the
goal-judge LLM are all env-gated — set a key to light one up, leave it unset and that
capability is silently skipped.

## Status

Early development. See the build slices in the project plan.
