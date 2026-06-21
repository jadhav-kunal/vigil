"""The ablation ladder (eval design §5). Each condition toggles exactly the mechanisms it
isolates; the task stream and seeds are identical across conditions (a paired design), so the
delta between adjacent rows attributes a saving to one mechanism."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    id: str
    label: str
    breaker: bool  # M2 — loop interruption
    governor: bool  # M3 — per-step routing
    compressor: bool  # M1 — context compression
    cache: bool  # M4 — semantic cache


CONDITIONS: list[Condition] = [
    Condition("C0", "Control", False, False, False, False),
    Condition("C1", "Breaker", True, False, False, False),
    Condition("C2", "Governor", False, True, False, False),
    Condition("C3", "Compressor", False, False, True, False),
    Condition("C4", "Cache", False, False, False, True),
    Condition("C5", "Full", True, True, True, True),
]

BY_ID = {c.id: c for c in CONDITIONS}
