"""Internal data model. Both OpenAI- and Anthropic-format traffic normalizes into `Step`."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class Step(BaseModel):
    """One request/response turn in an agent trajectory.

    The unit the watchdog, governor, compressor, and forensics all operate on.
    """

    session_id: str
    step_index: int
    model_requested: str
    model_used: str
    tool_name: str | None = None
    tool_args: dict | None = None
    assistant_text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_before_compression: int | None = None
    tokens_after_compression: int | None = None
    timestamp: str = Field(default_factory=_utcnow_iso)
    caused_state_mutation: bool = False
    # Watchdog metrics (spec 4.2), filled by the analyzer on the analysis path.
    sim_score: float | None = None
    tool_entropy: float | None = None
    state_penalty: float | None = None
    final_score: float | None = None
    watchdog_breach: bool = False
    watchdog_streak: int = 0
    watchdog_tripped: bool = False
    # Populated by later slices; carried here so the schema is stable.
    breaker_override: bool = False
    breaker_state: str | None = None

    # The text used by the watchdog to embed this step (spec 4.2):
    #   tool_name + " " + json(tool_args) + " " + assistant_text
    def embedding_text(self) -> str:
        args = json.dumps(self.tool_args, separators=(",", ":")) if self.tool_args else ""
        return f"{self.tool_name or ''} {args} {self.assistant_text}".strip()


class NormalizedRequest(BaseModel):
    """A provider request flattened to the fields Vigil reasons about."""

    provider: str  # "openai" | "anthropic"
    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    stream: bool = False
    # The full original body, forwarded upstream (possibly after model/context rewrite).
    raw: dict
