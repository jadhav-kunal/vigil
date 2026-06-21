"""Normalize OpenAI and Anthropic request/response shapes into Vigil's internal model.

Pure functions, no I/O — unit-testable. Token estimation falls back to a char heuristic when
the provider omits usage (e.g. streaming without `stream_options.include_usage`).
"""

from __future__ import annotations

import json
from typing import Any

from .models import NormalizedRequest, Step
from .state_mutation import caused_state_mutation


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for when provider usage is absent."""
    return max(1, len(text) // 4) if text else 0


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate the token footprint of the whole message array as sent on the wire.

    Unlike `estimate_prompt_tokens` (text only, used to approximate provider prompt usage), this
    counts the full serialized payload — tool_calls and tool results included — so it reflects the
    real context size Layer 1 compression shrinks. This is the before/after savings measurement.
    """
    return estimate_tokens(json.dumps(messages, separators=(",", ":"), default=str))


def _message_text(content: Any) -> str:
    """Flatten a message `content` (str, or a list of OpenAI/Anthropic content blocks)."""
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


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Estimate input tokens from the request messages when the provider omits usage."""
    text = " ".join(_message_text(m.get("content")) for m in messages if isinstance(m, dict))
    return estimate_tokens(text)


# --------------------------------------------------------------------------- requests


def _normalize_request(body: dict, provider: str) -> NormalizedRequest:
    return NormalizedRequest(
        provider=provider,
        model=str(body.get("model", "")),
        messages=list(body.get("messages", []) or []),
        tools=body.get("tools"),
        stream=bool(body.get("stream", False)),
        raw=body,
    )


def normalize_openai_request(body: dict) -> NormalizedRequest:
    return _normalize_request(body, "openai")


def normalize_anthropic_request(body: dict) -> NormalizedRequest:
    return _normalize_request(body, "anthropic")


# --------------------------------------------------------------------------- responses


def _openai_extract(resp: dict) -> tuple[str, str | None, dict | None, int | None, int | None]:
    """-> (assistant_text, tool_name, tool_args, prompt_tokens, completion_tokens)."""
    text = ""
    tool_name: str | None = None
    tool_args: dict | None = None

    choices = resp.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            fn = (tool_calls[0] or {}).get("function") or {}
            tool_name = fn.get("name")
            raw_args = fn.get("arguments")
            tool_args = _maybe_json(raw_args)
        elif msg.get("function_call"):  # legacy
            fc = msg["function_call"]
            tool_name = fc.get("name")
            tool_args = _maybe_json(fc.get("arguments"))

    usage = resp.get("usage") or {}
    return (
        text or "",
        tool_name,
        tool_args,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )


def _anthropic_extract(resp: dict) -> tuple[str, str | None, dict | None, int | None, int | None]:
    text_parts: list[str] = []
    tool_name: str | None = None
    tool_args: dict | None = None

    for block in resp.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use" and tool_name is None:
            tool_name = block.get("name")
            inp = block.get("input")
            tool_args = inp if isinstance(inp, dict) else None

    usage = resp.get("usage") or {}
    return (
        " ".join(p for p in text_parts if p).strip(),
        tool_name,
        tool_args,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
    )


def _maybe_json(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return None


def build_step(
    *,
    req: NormalizedRequest,
    response: dict,
    session_id: str,
    step_index: int,
    model_used: str | None = None,
    state_mutation_override: bool | None = None,
    tokens_before_compression: int | None = None,
    tokens_after_compression: int | None = None,
) -> Step:
    """Assemble a `Step` from a normalized request and the upstream JSON response.

    ``tokens_before/after_compression`` are the same-estimator measurements of the message array
    before and after Layer 1 compression (slice 5); when absent (compression disabled), both
    default to the prompt size so the columns are never NULL and savings reads as zero.
    """
    if req.provider == "anthropic":
        text, tool_name, tool_args, ptok, ctok = _anthropic_extract(response)
    else:
        text, tool_name, tool_args, ptok, ctok = _openai_extract(response)

    if ctok is None and text:
        ctok = estimate_tokens(text)
    # Streaming responses (and some providers) omit usage; estimate the prompt side from the
    # request so token accounting / cost / compression metrics are never silently NULL.
    if ptok is None:
        ptok = estimate_prompt_tokens(req.messages)

    mutated = caused_state_mutation(tool_name, metadata_override=state_mutation_override)
    before = tokens_before_compression if tokens_before_compression is not None else ptok
    after = tokens_after_compression if tokens_after_compression is not None else ptok

    return Step(
        session_id=session_id,
        step_index=step_index,
        model_requested=req.model,
        model_used=model_used or req.model,
        tool_name=tool_name,
        tool_args=tool_args,
        assistant_text=text,
        prompt_tokens=ptok,
        completion_tokens=ctok,
        tokens_before_compression=before,
        tokens_after_compression=after,
        caused_state_mutation=mutated,
    )
