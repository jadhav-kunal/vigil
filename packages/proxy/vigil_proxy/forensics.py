"""Forensic replay & fork (spec 4.7).

Every successful exchange is cached content-addressed by a hash of the salient request (model +
messages + tools). Two read-only forensic operations then run off that cache:

  - REPLAY: reconstruct a recorded session entirely from the cache — zero new API calls, zero
    side effects ("cached-trace replay"). Deterministic, so it yields a stable trace hash.
  - FORK at step N: re-issue step N's request with a different model while holding the upstream
    context (and thus the cached tool outputs that produced it) constant. The diff isolates a
    model-reasoning change from tool-output variance, because the tool outputs did not move.

The cache stores request BODIES only — never the caller's provider key (keys ride in headers,
which Vigil never persists). Fork is the one operation that may call upstream (once per forked
step); it requires the caller to supply their own Authorization, exactly like a normal request.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from .normalize import build_step, normalize_anthropic_request, normalize_openai_request
from .pricing import PriceTable, estimate_cost
from .store import Store

# An async function that takes a request body and returns the upstream response JSON.
GenerateFn = Callable[[dict], Awaitable[dict]]


def canonical_request(body: dict) -> str:
    """A stable, key-free serialization of the parts of a request that determine the response."""
    salient = {
        "model": body.get("model"),
        "messages": body.get("messages"),
        "tools": body.get("tools"),
    }
    return json.dumps(salient, sort_keys=True, separators=(",", ":"), default=str)


def request_hash(body: dict) -> str:
    return hashlib.sha256(canonical_request(body).encode()).hexdigest()


@dataclass
class ReplayStep:
    step_index: int
    model: str
    tool_name: str | None
    assistant_text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float


@dataclass
class ReplayResult:
    session_id: str
    steps: list[ReplayStep]
    trace_hash: str
    upstream_calls: int

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "steps": [asdict(s) for s in self.steps],
            "trace_hash": self.trace_hash,
            "upstream_calls": self.upstream_calls,
        }


@dataclass
class ForkResult:
    session_id: str
    step_index: int
    original_model: str
    forked_model: str
    original: dict
    forked: dict
    same_tool: bool
    same_args: bool
    divergence: str

    def as_dict(self) -> dict:
        return asdict(self)


class Forensics:
    def __init__(self, store: Store, price_table: PriceTable) -> None:
        self._store = store
        self._prices = price_table

    async def record(
        self, session_id: str, step_index: int, request: dict, response: dict, model: str | None
    ) -> None:
        await self._store.cache_exchange(
            session_id, step_index, request_hash(request), request, response, model
        )

    async def replay(self, session_id: str) -> ReplayResult | None:
        exchanges = await self._store.get_exchanges(session_id)
        if not exchanges:
            return None
        steps: list[ReplayStep] = []
        for ex in exchanges:
            # Serve from the content-addressed cache; fall back to the row's own response only if
            # the hash index somehow missed (it never should) — still zero upstream calls.
            cached = await self._store.get_cached_response(ex["request_hash"]) or ex["response"]
            steps.append(self._summarize(ex["step_index"], ex["request"], cached, ex["model"]))
        trace_hash = hashlib.sha256(
            json.dumps([asdict(s) for s in steps], sort_keys=True).encode()
        ).hexdigest()
        return ReplayResult(session_id, steps, trace_hash, upstream_calls=0)

    async def fork(
        self, session_id: str, step_index: int, new_model: str, generate: GenerateFn
    ) -> ForkResult | None:
        exchanges = await self._store.get_exchanges(session_id)
        target = next((e for e in exchanges if e["step_index"] == step_index), None)
        if target is None:
            return None

        original_model = target["model"] or str(target["request"].get("model", ""))
        forked_request = {**target["request"], "model": new_model}
        forked_response = await generate(forked_request)  # the one upstream call a fork may make

        orig = self._action(target["request"], target["response"])
        fork = self._action(forked_request, forked_response)
        same_tool = orig["tool_name"] == fork["tool_name"]
        same_args = orig["tool_args"] == fork["tool_args"]
        if same_tool and same_args:
            divergence = (
                "Same action under identical cached context — model reasoning is stable across "
                "the swap; any earlier failure is not isolated to this step's reasoning."
            )
        else:
            divergence = (
                "Model reasoning diverged: the tool outputs were held constant (cached), so this "
                "difference is attributable to the model, not to tool-output variance."
            )
        return ForkResult(
            session_id,
            step_index,
            original_model,
            new_model,
            orig,
            fork,
            same_tool,
            same_args,
            divergence,
        )

    # ----------------------------------------------------------------- internals

    def _summarize(
        self, step_index: int, request: dict, response: dict, model: str | None
    ) -> ReplayStep:
        step = self._build(request, response, model)
        return ReplayStep(
            step_index=step_index,
            model=step.model_used,
            tool_name=step.tool_name,
            assistant_text=step.assistant_text,
            prompt_tokens=step.prompt_tokens,
            completion_tokens=step.completion_tokens,
            cost_usd=estimate_cost(
                step.model_used, step.prompt_tokens, step.completion_tokens, self._prices
            ),
        )

    def _action(self, request: dict, response: dict) -> dict:
        step = self._build(request, response, request.get("model"))
        return {
            "tool_name": step.tool_name,
            "tool_args": step.tool_args,
            "assistant_text": step.assistant_text,
        }

    def _build(self, request: dict, response: dict, model: str | None):
        # Anthropic responses carry a top-level "content" list; OpenAI carry "choices".
        if "choices" in response:
            req = normalize_openai_request(request)
        elif "content" in response:
            req = normalize_anthropic_request(request)
        else:
            req = normalize_openai_request(request)
        return build_step(
            req=req,
            response=response,
            session_id="replay",
            step_index=0,
            model_used=model or req.model,
        )
