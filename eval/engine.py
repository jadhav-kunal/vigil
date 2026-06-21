"""The benchmark engine: run one scripted scenario under one condition, in-process, no network.

It reuses Vigil's real pure cores so the numbers reflect the shipped code:
  - compressor.compress_messages           (M1)
  - governor.Governor / classify           (M3)
  - watchdog math + HashingEmbedder         (M2 detection)
  - breaker.transition (the real FSM)       (M2 intervention)
A deterministic semantic cache (M4) and a goal-judge oracle (keyed to scripted progress) stand in
for their networked counterparts; both are called out in the report's threats-to-validity.

Cost emerges honestly from the mechanisms, never scripted: prompt tokens are measured on the
ACTUAL forwarded (possibly compressed) context, the model price comes from whatever model routing
chose, and a halted loop simply executes fewer steps. Vigil's own overhead (goal-judge calls,
cache lookups) is accounted separately so the report can show a NET figure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from vigil_proxy.breaker import CLOSED, OPEN, BreakerSnapshot, is_mitigating, is_open, transition
from vigil_proxy.compressor import compress_messages
from vigil_proxy.embedder import HashingEmbedder
from vigil_proxy.governor import Governor
from vigil_proxy.normalize import estimate_messages_tokens
from vigil_proxy.pricing import PriceTable, price_for
from vigil_proxy.watchdog import (
    final_score,
    is_breach,
    mean_similarity,
    shannon_entropy,
    state_penalty,
)

from .conditions import Condition
from .datasets import Scenario

# Vigil overhead unit costs (USD). The goal-judge is the env-gated LLM variant; the heuristic
# governor is free. The cache lookup (embed + ANN) is paid on every request, hit or miss.
JUDGE_CALL_USD = 0.00012
CACHE_LOOKUP_USD = 0.00002
CACHE_SIM_THRESHOLD = 0.92

BASE_MODEL = "gpt-4o"
DOWNGRADE_MODEL = "gpt-4o-mini"


@dataclass
class RunRecord:
    scenario_id: str
    dataset: str
    archetype: str
    condition: str
    is_loop: bool
    seed: int
    steps_executed: int
    budget_steps: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_before_compression: int = 0
    tokens_after_compression: int = 0
    llm_cost_usd: float = 0.0
    overhead_usd: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    opened: bool = False
    opened_by: str | None = None  # "math" | "judge"
    trip_step: int | None = None
    inevitable_step: int | None = None
    task_success: bool | None = None
    transitions: list[str] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return round(self.llm_cost_usd + self.overhead_usd, 8)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def compression_ratio(self) -> float | None:
        b = self.tokens_before_compression
        return (self.tokens_after_compression / b) if b else None


class _Detector:
    """A synchronous mirror of analyzer.Analyzer over the real watchdog math (deterministic
    HashingEmbedder), so the engine stays sync and fast while using the shipped formulas."""

    def __init__(self, window: int, trip_streak: int, theta_sim: float, theta_ent: float) -> None:
        self._embedder = HashingEmbedder()
        self._window = window
        self._k = trip_streak
        self._theta_sim = theta_sim
        self._theta_ent = theta_ent
        self._embeds: list[np.ndarray] = []
        self._tools: list[str | None] = []
        self._streak = 0

    def observe(self, text: str, tool: str | None, mutated: bool) -> tuple[bool, bool]:
        """Feed one step; return (breach, tripped)."""
        emb = self._embedder.encode(text)
        self._embeds.append(emb)
        self._tools.append(tool)
        win_emb = self._embeds[-self._window :]
        win_tools = self._tools[-self._window :]
        prior = win_emb[:-1]
        sc = mean_similarity(emb, prior)
        h = shannon_entropy(win_tools)
        s_final = final_score(sc, state_penalty(mutated))
        breach = is_breach(s_final, h, self._theta_sim, self._theta_ent)
        self._streak = self._streak + 1 if breach else 0
        return breach, self._streak >= self._k


class _SemanticCache:
    def __init__(self) -> None:
        self._embedder = HashingEmbedder()
        self._keys: list[np.ndarray] = []

    def lookup_or_store(self, query: str) -> bool:
        """Return True on a hit (a semantically-similar query was seen before)."""
        emb = self._embedder.encode(query)
        for k in self._keys:
            denom = float(np.linalg.norm(emb) * np.linalg.norm(k))
            if denom and float(np.dot(emb, k)) / denom >= CACHE_SIM_THRESHOLD:
                return True
        self._keys.append(emb)
        return False


def run_scenario(
    scenario: Scenario,
    condition: Condition,
    *,
    seed: int,
    price_table: PriceTable,
    window: int = 5,
    trip_streak: int = 3,
    theta_sim: float = 0.85,
    theta_ent: float = 0.30,
    recovery_steps: int = 3,
    judge_cadence: int = 5,
    judge_low: float = 0.4,
) -> RunRecord:
    rec = RunRecord(
        scenario.scenario_id,
        scenario.dataset,
        scenario.archetype,
        condition.id,
        scenario.is_loop,
        seed,
        0,
        scenario.budget_steps,
        inevitable_step=scenario.inevitable_step,
    )
    governor = Governor({"openai": _gov_map()}, price_table) if condition.governor else None
    detector = _Detector(window, trip_streak, theta_sim, theta_ent) if condition.breaker else None
    cache = _SemanticCache() if condition.cache else None
    snap = BreakerSnapshot(session_id=scenario.scenario_id)
    consecutive_low = 0

    history: list[dict] = [
        {"role": "system", "content": scenario.system_prompt},
        {"role": "user", "content": scenario.user_goal},
    ]
    last_text = ""

    for i, step in enumerate(scenario.steps):
        # --- breaker gate (intervention takes effect before the request, like the real proxy) ---
        if condition.breaker and is_open(snap.state):
            break  # loop halted: the remaining budget is saved
        model = BASE_MODEL
        if condition.breaker and is_mitigating(snap.state):
            model = DOWNGRADE_MODEL  # HALF_OPEN read-only/downgrade

        # --- request shaping: governor routing then compression (matches _proxy order) ---
        request = {"model": model, "messages": history, "tools": scenario.tools}
        if governor is not None and snap.state == CLOSED:
            decision = governor.decide(scenario.scenario_id, "openai", request)
            model = decision.model

        before = estimate_messages_tokens(history)
        if condition.compressor:
            forwarded, _ = compress_messages(
                history, min_tool_bytes=4000, floor_messages=6, dedup_min_run=3
            )
        else:
            forwarded = history
        after = estimate_messages_tokens(forwarded)
        rec.tokens_before_compression += before
        rec.tokens_after_compression += after

        # --- semantic cache (M4): a hit skips the LLM entirely ---
        cache_query = step.tool_result or step.assistant_text or last_text
        hit = False
        if cache is not None:
            rec.overhead_usd += CACHE_LOOKUP_USD
            hit = cache.lookup_or_store(cache_query)
            if hit:
                rec.cache_hits += 1
            else:
                rec.cache_misses += 1

        in_rate, out_rate = price_for(model, price_table)
        prompt_tokens = after
        completion_tokens = 0 if hit else step.completion_tokens
        rec.prompt_tokens += 0 if hit else prompt_tokens
        rec.completion_tokens += completion_tokens
        if not hit:
            rec.llm_cost_usd += (prompt_tokens / 1000.0) * in_rate
        rec.llm_cost_usd += (completion_tokens / 1000.0) * out_rate

        # --- grow the agent's memory with the scripted response (full, uncompressed, real shape
        # so the compressor sees genuine assistant-tool_call + tool cycles to collapse) ---
        if step.tool_name:
            history = [
                *history,
                {
                    "role": "assistant",
                    "content": step.assistant_text or None,
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {
                                "name": step.tool_name,
                                "arguments": json.dumps(step.tool_args or {}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": f"c{i}", "content": step.tool_result},
            ]
        else:
            history = [*history, {"role": "assistant", "content": step.assistant_text}]
        last_text = step.assistant_text
        rec.steps_executed = i + 1

        # --- analysis path: watchdog + goal-judge -> breaker FSM ---
        if condition.breaker and detector is not None:
            breach, tripped = detector.observe(
                step.assistant_text, step.tool_name, step.caused_state_mutation
            )
            force_open = False
            if (i + 1) % judge_cadence == 0:
                rec.overhead_usd += JUDGE_CALL_USD
                score = 1.0 if step.progressed else 0.1
                consecutive_low = consecutive_low + 1 if score < judge_low else 0
                force_open = consecutive_low >= 2
            prev_state = snap.state
            snap, label = transition(
                snap,
                step_index=i,
                tripped=tripped,
                breach=breach,
                recovery_steps=recovery_steps,
                force_open=force_open,
            )
            if label:
                rec.transitions.append(f"{i}:{label}")
            if snap.trip_step_index is not None and rec.trip_step is None:
                rec.trip_step = snap.trip_step_index
            if snap.state == OPEN and prev_state != OPEN and not rec.opened:
                rec.opened = True
                rec.opened_by = "judge" if force_open else "math"

    # task success (D3): completed all steps and the final assistant text carries the token.
    if scenario.success_token is not None:
        completed = rec.steps_executed == scenario.budget_steps
        rec.task_success = completed and scenario.success_token in scenario.steps[-1].assistant_text
    return rec


def _gov_map() -> dict[str, str]:
    return {
        "PLANNING": BASE_MODEL,
        "TOOL_USE": BASE_MODEL,
        "EXTRACTION": DOWNGRADE_MODEL,
        "VERIFICATION": DOWNGRADE_MODEL,
    }
