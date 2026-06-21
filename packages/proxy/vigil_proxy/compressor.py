"""Layer 1 context compression (spec 4.5): structural, loop-aware, lossless-to-outcome.

Two transforms applied to the messages sent UPSTREAM (never to what the agent sees):
  1. Collapse runs of repeated, identical tool-call cycles into a single marker (OpenAI shape).
  2. Truncate oversized tool outputs to head+tail with an elision marker (both shapes).

Pure functions, no I/O — unit-testable. Hard safety guards:
  - never touch the last message (the current turn) or system messages;
  - refuse to compress a short conversation (below a message floor);
  - copy-on-write — the caller's original message dicts are never mutated, so the
    pre-compression token estimate stays honest.

Outcome-preservation is proven empirically in the eval harness (slice 7), not asserted here.
Layer 2 (Token Company, env-gated) lives in slice 9; Layer 1 is always-on and free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

_MARKER = "[vigil-compressed]"


@dataclass
class CompressionStats:
    """What Layer 1 did to one request — the savings proof is measured at the wire, not here."""

    collapsed_runs: int = 0
    dropped_messages: int = 0
    truncated_outputs: int = 0

    @property
    def changed(self) -> bool:
        return self.dropped_messages > 0 or self.truncated_outputs > 0


def compress_messages(
    messages: list[dict],
    *,
    min_tool_bytes: int,
    floor_messages: int,
    dedup_min_run: int,
) -> tuple[list[dict], CompressionStats]:
    """Return (possibly-compressed messages, stats).

    The last message and all system messages are always preserved verbatim. When nothing
    changes, the input list is returned unchanged (identity), so callers can cheaply skip work.
    """
    stats = CompressionStats()
    if not isinstance(messages, list) or len(messages) <= floor_messages:
        return messages, stats

    last = messages[-1]
    region = messages[:-1]
    # 1. collapse repeated identical tool-call cycles (drops whole cycles -> stays request-valid)
    region = _collapse_cycles(region, dedup_min_run, stats)
    # 2. truncate oversized tool outputs in whatever remains (never the protected last message)
    region = [_truncate_message(m, min_tool_bytes, stats) for m in region]

    if not stats.changed:
        return messages, stats
    return [*region, last], stats


# --------------------------------------------------------------------------- collapse


def _collapse_cycles(messages: list[dict], min_run: int, stats: CompressionStats) -> list[dict]:
    """Collapse maximal runs of identical assistant-tool_call + tool-result cycles (OpenAI).

    A run is collapsed only when it repeats at least ``min_run`` times; the first cycle is kept
    intact and the rest are replaced by a single assistant text marker. Whole cycles are removed
    together (the assistant tool_call and the tool messages answering it) so the upstream request
    never ends up with an orphaned tool_call or tool result.
    """
    out: list[dict] = []
    i = 0
    n = len(messages)
    while i < n:
        block = _cycle_at(messages, i)
        if block is None:
            out.append(messages[i])
            i += 1
            continue
        end, sig, names = block
        run: list[tuple[int, int]] = [(i, end)]
        k = end
        while k < n:
            nxt = _cycle_at(messages, k)
            if nxt is None or nxt[1] != sig:
                break
            run.append((k, nxt[0]))
            k = nxt[0]

        if len(run) >= min_run:
            first_start, first_end = run[0]
            out.extend(messages[first_start:first_end])  # keep the first cycle verbatim
            dropped_cycles = len(run) - 1
            stats.collapsed_runs += 1
            stats.dropped_messages += sum(e - s for s, e in run[1:])
            label = ", ".join(dict.fromkeys(names)) or "tool"
            out.append(
                {
                    "role": "assistant",
                    "content": (
                        f"{_MARKER} omitted {dropped_cycles} repeated `{label}` "
                        "call(s) that returned identical results"
                    ),
                }
            )
            i = k
        else:
            out.extend(messages[i:end])
            i = end
    return out


def _cycle_at(messages: list[dict], i: int) -> tuple[int, tuple, tuple[str, ...]] | None:
    """If messages[i] starts an assistant-tool_call cycle, return (end_exclusive, sig, names)."""
    msg = messages[i]
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return None
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    j = i + 1
    results: list[str] = []
    while j < len(messages):
        m = messages[j]
        if not isinstance(m, dict) or m.get("role") != "tool":
            break
        results.append(_norm(m.get("content")))
        j += 1
    if not results:  # an unanswered tool_call (e.g. the pending final turn) is never collapsed
        return None

    names = tuple(_tool_call_names(tool_calls))
    sig = (names, tuple(results), _norm(msg.get("content")))
    return j, sig, names


def _tool_call_names(tool_calls: list) -> list[str]:
    names: list[str] = []
    for tc in tool_calls:
        name = None
        if isinstance(tc, dict):
            fn = tc.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
        names.append(str(name) if name else "tool")
    return names


def _norm(content: object) -> str:
    if isinstance(content, str):
        return " ".join(content.split())
    if content is None:
        return ""
    return json.dumps(content, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- truncate


def _truncate_message(msg: dict, min_bytes: int, stats: CompressionStats) -> dict:
    """Truncate oversized tool-output text, copy-on-write. Non-tool messages pass through."""
    if not isinstance(msg, dict):
        return msg
    content = msg.get("content")

    if msg.get("role") == "tool" and isinstance(content, str):
        new, changed = _truncate_text(content, min_bytes)
        if changed:
            stats.truncated_outputs += 1
            return {**msg, "content": new}
        return msg

    # Anthropic: tool results are tool_result blocks inside a user-message content list.
    if isinstance(content, list):
        new_blocks: list = []
        block_changed = False
        for block in content:
            updated = _truncate_tool_result_block(block, min_bytes, stats)
            block_changed = block_changed or (updated is not block)
            new_blocks.append(updated)
        if block_changed:
            return {**msg, "content": new_blocks}
    return msg


def _truncate_tool_result_block(block: object, min_bytes: int, stats: CompressionStats) -> object:
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return block
    bc = block.get("content")
    if isinstance(bc, str):
        new, changed = _truncate_text(bc, min_bytes)
        if changed:
            stats.truncated_outputs += 1
            return {**block, "content": new}
        return block
    if isinstance(bc, list):
        new_inner: list = []
        inner_changed = False
        for tb in bc:
            if isinstance(tb, dict) and isinstance(tb.get("text"), str):
                new, changed = _truncate_text(tb["text"], min_bytes)
                if changed:
                    stats.truncated_outputs += 1
                    new_inner.append({**tb, "text": new})
                    inner_changed = True
                    continue
            new_inner.append(tb)
        if inner_changed:
            return {**block, "content": new_inner}
    return block


def _truncate_text(text: str, min_bytes: int) -> tuple[str, bool]:
    raw = text.encode("utf-8")
    if len(raw) <= min_bytes:
        return text, False
    half = max(1, min_bytes // 2)
    head = raw[:half].decode("utf-8", "ignore")
    tail = raw[-half:].decode("utf-8", "ignore")
    elided = len(raw) - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
    return f"{head}\n{_MARKER} truncated {elided} bytes\n{tail}", True
