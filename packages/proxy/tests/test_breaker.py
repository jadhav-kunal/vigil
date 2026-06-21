"""Circuit breaker FSM (spec 4.3) — pure transitions."""

from vigil_proxy.breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    BreakerSnapshot,
    transition,
)


def snap(**kw):
    return BreakerSnapshot(session_id="s", **kw)


def test_closed_stays_closed_without_trip():
    s, label = transition(snap(), step_index=2, tripped=False, breach=False, recovery_steps=3)
    assert s.state == CLOSED and label is None


def test_trip_opens_half_open_with_recovery_budget():
    s, label = transition(snap(), step_index=3, tripped=True, breach=True, recovery_steps=3)
    assert s.state == HALF_OPEN
    assert s.recovery_remaining == 3
    assert s.trip_step_index == 3
    assert label == "CLOSED->HALF_OPEN"


def test_half_open_recovers_on_non_breach():
    s = snap(state=HALF_OPEN, recovery_remaining=3)
    s2, label = transition(s, step_index=5, tripped=False, breach=False, recovery_steps=3)
    assert s2.state == CLOSED and label == "HALF_OPEN->CLOSED"


def test_half_open_opens_after_r_breaching_steps():
    s = snap(state=HALF_OPEN, recovery_remaining=3, trip_step_index=3)
    labels = []
    for i in range(4, 8):
        s, label = transition(s, step_index=i, tripped=True, breach=True, recovery_steps=3)
        labels.append(label)
    # 3 breaching recovery steps exhaust the budget -> OPEN on the third.
    assert s.state == OPEN
    assert "HALF_OPEN->OPEN" in labels
    assert labels.count("HALF_OPEN->OPEN") == 1  # transitions exactly once


def test_open_is_terminal():
    s = snap(state=OPEN)
    s2, label = transition(s, step_index=9, tripped=True, breach=True, recovery_steps=3)
    assert s2.state == OPEN and label is None


def test_force_open_from_judge():
    s, label = transition(
        snap(), step_index=5, tripped=False, breach=False, recovery_steps=3, force_open=True
    )
    assert s.state == OPEN and label == "CLOSED->OPEN(judge)"


def test_trips_by_step_seven_in_sabotage_shape():
    # A sustained loop: trip at step 3 (K=3), then 3 breaching recovery steps -> OPEN by step 6.
    s = snap()
    opened_at = None
    for i in range(8):
        tripped = i >= 3  # watchdog level after K
        s, label = transition(s, step_index=i, tripped=tripped, breach=tripped, recovery_steps=3)
        if s.state == OPEN and opened_at is None:
            opened_at = i
            break
    assert opened_at is not None and opened_at <= 7
