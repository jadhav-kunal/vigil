"""Effort governor (spec 4.6): per-step model routing to cut cost.

Each request is classified into an effort tier by a zero-latency heuristic, then routed to the
cheapest model adequate for that tier by rewriting the request `model` before forwarding. The
classifier is pure and deterministic; an optional LLM classifier (behind the judge env vars) is
a documented seam, intentionally NOT wired on the hot path so routing adds no latency.

Tiers (spec 4.6):
  PLANNING      -> frontier   (open-ended reasoning / first turn / explicit planning intent)
  TOOL_USE      -> mid        (a tool-driven step; the default for ambiguous follow-ups)
  EXTRACTION    -> cheap      (pull/parse/summarize from provided content)
  VERIFICATION  -> cheap, but escalate to frontier when verification repeats (it isn't resolving)

Two safety rules make routing strictly opt-in and strictly cost-reducing:
  - governor is gated by VIGIL_GOVERNOR_ENABLED (default off) and skipped while the breaker is
    mitigating (don't fight the breaker's downgrade);
  - the no-upgrade guard never routes to a model more expensive than the one requested, so the
    governor can only ever lower cost, never raise it (and never silently upgrades a cheap model).

The CLASSIFICATION-ORDERING INVARIANT (CLAUDE.md / spec): EXTRACTION is tested BEFORE VERIFICATION,
so a step that reads as both ("extract the totals and verify them") routes as EXTRACTION.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .pricing import PriceTable, price_for
from .settings import Settings

PLANNING = "PLANNING"
TOOL_USE = "TOOL_USE"
EXTRACTION = "EXTRACTION"
VERIFICATION = "VERIFICATION"

# Ordered keyword cues for the heuristic classifier. Order of the *checks* matters (see invariant);
# order within a set does not.
_PLANNING_KW = (
    "plan",
    "design",
    "architect",
    "strategy",
    "approach",
    "figure out",
    "decide how",
    "how should",
    "break down",
    "outline",
)
_EXTRACTION_KW = (
    "extract",
    "parse",
    "summarize",
    "summarise",
    "list the",
    "pull out",
    "retrieve",
    "what is the",
    "what are the",
    "find the",
)
_VERIFICATION_KW = (
    "verify",
    "validate",
    "confirm",
    "double-check",
    "double check",
    "check that",
    "check whether",
    "is this correct",
    "make sure",
    "ensure",
)

# provider -> tier -> model id. Override wholesale via VIGIL_GOVERNOR_MODEL_MAP (JSON).
DEFAULT_GOVERNOR_MODELS: dict[str, dict[str, str]] = {
    "openai": {
        PLANNING: "gpt-4o",
        TOOL_USE: "gpt-4o",
        EXTRACTION: "gpt-4o-mini",
        VERIFICATION: "gpt-4o-mini",
    },
    "anthropic": {
        PLANNING: "claude-3-5-sonnet-latest",
        TOOL_USE: "claude-3-5-sonnet-latest",
        EXTRACTION: "claude-3-5-haiku-latest",
        VERIFICATION: "claude-3-5-haiku-latest",
    },
}


@dataclass
class GovernorDecision:
    effort_class: str
    model: str  # the model to actually forward (== requested when not routed)
    routed: bool  # the model was rewritten
    escalated: bool  # a repeated verification was bumped to the frontier tier
    reason: str


# --------------------------------------------------------------------------- pure classifier


def classify(messages: list[dict], tools: object, *, is_first_turn: bool) -> str:
    """Heuristic effort classification from the request alone (no response, zero latency)."""
    focus = _focus_text(messages).lower()

    if _matches(focus, _PLANNING_KW):
        return PLANNING
    # EXTRACTION is checked BEFORE VERIFICATION (classification-ordering invariant).
    if _matches(focus, _EXTRACTION_KW):
        return EXTRACTION
    if _matches(focus, _VERIFICATION_KW):
        return VERIFICATION
    if isinstance(tools, list) and tools:
        return TOOL_USE
    if is_first_turn:
        return PLANNING
    # Ambiguous follow-up: default to the mid tier so the no-route/strong model is used (we only
    # route to a cheaper model on a positive cheap-step signal).
    return TOOL_USE


def is_first_turn(messages: list[dict]) -> bool:
    return not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages)


def _matches(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


def _focus_text(messages: list[dict]) -> str:
    """The most recent instruction/observation the upcoming step responds to."""
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        text = _content_text(m.get("content"))
        if text:
            return text
    return ""


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(p for p in parts if p)
    return ""


# --------------------------------------------------------------------------- stateful router


class Governor:
    """Classifies and routes per request, remembering the last tier per session for escalation."""

    def __init__(self, models: dict[str, dict[str, str]], price_table: PriceTable) -> None:
        self._models = models
        self._prices = price_table
        self._last_class: dict[str, str] = {}

    def decide(self, session_id: str, provider: str, parsed: dict) -> GovernorDecision:
        messages = parsed.get("messages")
        messages = messages if isinstance(messages, list) else []
        requested = str(parsed.get("model", ""))

        effort = classify(messages, parsed.get("tools"), is_first_turn=is_first_turn(messages))
        last = self._last_class.get(session_id)
        # A verification that immediately follows another verification isn't resolving -> escalate.
        escalated = effort == VERIFICATION and last == VERIFICATION
        self._last_class[session_id] = effort

        tier = PLANNING if escalated else effort
        target = self._models.get(provider, {}).get(tier, requested)

        # No-upgrade guard: only ever route to a strictly cheaper model.
        if target and target != requested and not self._is_cheaper(target, requested):
            target = requested

        routed = bool(target) and target != requested
        reason = f"{effort}{'+escalate' if escalated else ''}->{tier}"
        return GovernorDecision(effort, target or requested, routed, escalated, reason)

    def reset(self, session_id: str) -> None:
        self._last_class.pop(session_id, None)

    def _is_cheaper(self, candidate: str, baseline: str) -> bool:
        ci, co = price_for(candidate, self._prices)
        bi, bo = price_for(baseline, self._prices)
        return (ci + co) < (bi + bo)


def load_governor_models(settings: Settings) -> dict[str, dict[str, str]]:
    """Defaults, optionally replaced per provider by the VIGIL_GOVERNOR_MODEL_MAP JSON override."""
    models = {p: dict(tiers) for p, tiers in DEFAULT_GOVERNOR_MODELS.items()}
    raw = settings.governor_model_map
    if raw:
        try:
            override = json.loads(raw)
            for provider, tiers in override.items():
                models.setdefault(provider, {}).update({str(k): str(v) for k, v in tiers.items()})
        except (ValueError, AttributeError):
            pass
    return models


def make_governor(settings: Settings, price_table: PriceTable) -> Governor:
    return Governor(load_governor_models(settings), price_table)
