"""Circuit breaker finite state machine (spec 4.3) — pure transition logic, unit-testable.

    CLOSED ──trip──▶ HALF_OPEN ──recovery fails (R breaching steps)──▶ OPEN
       ▲                  │
       └─ recovery ok ────┘

CLOSED -> HALF_OPEN when the watchdog trips (K consecutive breaches). HALF_OPEN grants R
recovery steps with mitigations applied (model downgrade + write-tool strip); a non-breaching
step means the agent recovered (-> CLOSED), while R consecutive breaching steps means it did
not (-> OPEN). OPEN is terminal until a manual override. The goal-judge can force OPEN directly.

The stateful glue (persistence, judge calls, side-effects, broadcasts) lives in
breaker_manager.py; this module is just the math.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

CLOSED = "CLOSED"
HALF_OPEN = "HALF_OPEN"
OPEN = "OPEN"


@dataclass
class BreakerSnapshot:
    session_id: str
    state: str = CLOSED
    recovery_remaining: int = 0
    trip_step_index: int | None = None
    consecutive_low_judge: int = 0
    saved_estimate_usd: float = 0.0
    post_mortem: dict | None = field(default=None)


def transition(
    snap: BreakerSnapshot,
    *,
    step_index: int,
    tripped: bool,
    breach: bool,
    recovery_steps: int,
    force_open: bool = False,
) -> tuple[BreakerSnapshot, str | None]:
    """Compute the next snapshot and the transition label (None if no state change).

    `tripped` (watchdog: K consecutive breaches) drives CLOSED -> HALF_OPEN. `breach` (this step
    is still looping) drives the HALF_OPEN recovery countdown. `force_open` (two low goal-judge
    scores) escalates straight to OPEN from any non-OPEN state.
    """
    if snap.state == OPEN:
        return snap, None

    if force_open:
        return replace(snap, state=OPEN, recovery_remaining=0), f"{snap.state}->OPEN(judge)"

    if snap.state == CLOSED:
        if tripped:
            return (
                replace(
                    snap,
                    state=HALF_OPEN,
                    recovery_remaining=recovery_steps,
                    trip_step_index=step_index,
                ),
                "CLOSED->HALF_OPEN",
            )
        return snap, None

    # HALF_OPEN
    if not breach:
        return replace(snap, state=CLOSED, recovery_remaining=0), "HALF_OPEN->CLOSED"
    remaining = snap.recovery_remaining - 1
    if remaining <= 0:
        return replace(snap, state=OPEN, recovery_remaining=0), "HALF_OPEN->OPEN"
    return replace(snap, recovery_remaining=remaining), None


def is_mitigating(state: str) -> bool:
    """HALF_OPEN forwards requests but with mitigations applied."""
    return state == HALF_OPEN


def is_open(state: str) -> bool:
    return state == OPEN
