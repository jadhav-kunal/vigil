"""Reconstruct a full completion from streamed SSE deltas, for analysis only.

The client receives every byte unmodified and without delay (Invariant I1). We *tee* the
stream: a lightweight accumulator parses `data:` events as they fly past and, once the stream
ends, produces a response dict shaped like the non-streaming JSON so `normalize.build_step`
can treat both paths identically.
"""

from __future__ import annotations

import json
from typing import Any


class _SSEDecoder:
    """Incremental SSE parser. Feed raw bytes; yields decoded `data:` JSON objects."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: bytes) -> list[dict]:
        events: list[dict] = []
        self._buf += chunk.decode("utf-8", errors="replace")
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
        return events


class OpenAIStreamAccumulator:
    def __init__(self) -> None:
        self._dec = _SSEDecoder()
        self._content: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._usage: dict | None = None

    def feed(self, chunk: bytes) -> None:
        for event in self._dec.feed(chunk):
            if event.get("usage"):
                self._usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    self._content.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = self._tool_calls.setdefault(idx, {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

    def to_response(self) -> dict:
        message: dict[str, Any] = {"content": "".join(self._content)}
        if self._tool_calls:
            message["tool_calls"] = [
                {"function": {"name": v["name"], "arguments": v["arguments"]}}
                for _, v in sorted(self._tool_calls.items())
            ]
        resp: dict[str, Any] = {"choices": [{"message": message}]}
        if self._usage:
            resp["usage"] = self._usage
        return resp


class AnthropicStreamAccumulator:
    def __init__(self) -> None:
        self._dec = _SSEDecoder()
        self._text: list[str] = []
        self._tool_name: str | None = None
        self._tool_json: list[str] = []
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None

    def feed(self, chunk: bytes) -> None:
        for event in self._dec.feed(chunk):
            etype = event.get("type")
            if etype == "message_start":
                usage = (event.get("message") or {}).get("usage") or {}
                self._input_tokens = usage.get("input_tokens")
            elif etype == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    self._tool_name = block.get("name")
            elif etype == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    self._text.append(delta.get("text", ""))
                elif delta.get("type") == "input_json_delta":
                    self._tool_json.append(delta.get("partial_json", ""))
            elif etype == "message_delta":
                usage = event.get("usage") or {}
                if usage.get("output_tokens") is not None:
                    self._output_tokens = usage["output_tokens"]

    def to_response(self) -> dict:
        content: list[dict[str, Any]] = []
        text = "".join(self._text)
        if text:
            content.append({"type": "text", "text": text})
        if self._tool_name:
            try:
                parsed = json.loads("".join(self._tool_json)) if self._tool_json else {}
            except json.JSONDecodeError:
                parsed = {}
            content.append({"type": "tool_use", "name": self._tool_name, "input": parsed})
        return {
            "content": content,
            "usage": {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
            },
        }
