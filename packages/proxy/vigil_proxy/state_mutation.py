"""Infer whether a tool call mutated external state.

Used for the watchdog's state-mutation penalty (spec 4.2) and for the breaker's read-only
mode. The verb match is on word boundaries (tool names split on `_`/`-`/camelCase), so a
read-only name like `find` is never matched by the substring `in` inside it.
"""

from __future__ import annotations

import re

# Verbs whose presence as a whole token implies a write/mutation.
MUTATING_VERBS: frozenset[str] = frozenset(
    {
        "write",
        "create",
        "update",
        "delete",
        "insert",
        "remove",
        "drop",
        "set",
        "put",
        "post",
        "patch",
        "add",
        "edit",
        "modify",
        "append",
        "send",
        "commit",
        "push",
        "deploy",
        "execute",
        "run",
        "upload",
        "rename",
        "move",
        "destroy",
        "purge",
        "truncate",
        "save",
        "publish",
    }
)


def _tokenize(name: str) -> list[str]:
    """Split a tool name into lowercase word tokens (handles snake, kebab, camelCase)."""
    if not name:
        return []
    # camelCase / PascalCase -> space-separated
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    parts = re.split(r"[^A-Za-z0-9]+", spaced)
    return [p.lower() for p in parts if p]


def caused_state_mutation(
    tool_name: str | None,
    *,
    metadata_override: bool | None = None,
) -> bool:
    """Return True if the tool call likely mutated state.

    Caller-supplied `metadata_override` (e.g. a `x-vigil-state-mutation` hint) wins when given.
    Read-only steps (no tool call) get False.
    """
    if metadata_override is not None:
        return metadata_override
    if not tool_name:
        return False
    tokens = set(_tokenize(tool_name))
    return bool(tokens & MUTATING_VERBS)
