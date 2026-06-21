"""BreakerManager: FSM integration over a real SQLite store, savings estimate, override, judge."""

import pytest

from vigil_proxy.analyzer import WatchdogResult
from vigil_proxy.breaker import CLOSED, HALF_OPEN, OPEN
from vigil_proxy.breaker_manager import BreakerManager
from vigil_proxy.hub import Broadcaster
from vigil_proxy.models import Step
from vigil_proxy.pricing import DEFAULT_PRICE_TABLE
from vigil_proxy.settings import Settings
from vigil_proxy.store import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "b.db"))
    await s.init()
    yield s
    await s.close()


def manager(store, **kw):
    settings = Settings()
    for k, v in kw.pop("settings", {}).items():
        setattr(settings, k, v)
    return BreakerManager(
        store=store,
        broadcaster=Broadcaster(),
        price_table=DEFAULT_PRICE_TABLE,
        settings=settings,
        judge=kw.get("judge"),
        analyzer=kw.get("analyzer"),
    )


def loop_result(breach: bool, tripped: bool):
    return WatchdogResult(
        sim_score=1.0,
        tool_entropy=0.0,
        state_penalty=0.0,
        final_score=1.0,
        breach=breach,
        streak=99 if tripped else 0,
        tripped=tripped,
    )


async def _append(store, i):
    s = Step(
        session_id="loop",
        step_index=0,
        model_requested="gpt-4o",
        model_used="gpt-4o",
        tool_name="check_status",
        assistant_text="still pending",
        prompt_tokens=40,
        completion_tokens=10,
    )
    s.step_index = await store.append_step(s)
    return s


async def test_sabotage_opens_by_step_seven(store):
    m = manager(store)
    states = []
    for i in range(8):
        step = await _append(store, i)
        tripped = i >= 3  # watchdog level after K=3
        snap, _ = await m.record(step, loop_result(breach=tripped, tripped=tripped))
        states.append(snap.state)
        if snap.state == OPEN:
            break
    assert OPEN in states
    open_at = states.index(OPEN)
    assert open_at <= 7
    assert states[:3] == [CLOSED, CLOSED, CLOSED]
    assert HALF_OPEN in states  # passed through HALF_OPEN first


async def test_open_builds_post_mortem_and_savings(store):
    m = manager(store)
    snap = None
    for i in range(8):
        step = await _append(store, i)
        tripped = i >= 3
        snap, _ = await m.record(step, loop_result(breach=tripped, tripped=tripped))
        if snap.state == OPEN:
            break
    assert snap is not None and snap.state == OPEN
    assert snap.post_mortem is not None
    assert snap.post_mortem["tripped_at_step"] == 3
    assert len(snap.post_mortem["metric_series"]) >= 6
    assert snap.saved_estimate_usd > 0  # projected loop cost capped


async def test_open_state_survives_restart_via_hydration(store):
    # Drive one manager to OPEN, then simulate a restart with a fresh manager + same store.
    m1 = manager(store)
    for i in range(8):
        step = await _append(store, i)
        tripped = i >= 3
        snap, _ = await m1.record(step, loop_result(breach=tripped, tripped=tripped))
        if snap.state == OPEN:
            break
    m2 = manager(store)  # fresh in-memory cache, same persisted store
    assert m2.get("loop").state == CLOSED  # not yet hydrated
    await m2.ensure_loaded("loop")
    assert m2.get("loop").state == OPEN  # rehydrated -> still blocks


async def test_override_resets_to_closed_and_flags_next_step(store):
    m = manager(store)
    for i in range(8):
        step = await _append(store, i)
        tripped = i >= 3
        snap, _ = await m.record(step, loop_result(breach=tripped, tripped=tripped))
        if snap.state == OPEN:
            break
    fresh = await m.override("loop")
    assert fresh.state == CLOSED
    # The next captured step is flagged as override.
    nxt = await _append(store, 99)
    await m.record(nxt, loop_result(breach=False, tripped=False))
    assert nxt.breaker_override is True


async def test_half_open_recovery_returns_to_closed(store):
    m = manager(store)
    snap = None
    # Trip, then a non-breaching step should recover.
    for i, (br, tr) in enumerate(
        [(False, False), (False, False), (False, False), (True, True), (False, False)]
    ):
        step = await _append(store, i)
        snap, _ = await m.record(step, loop_result(breach=br, tripped=tr))
    assert snap is not None and snap.state == CLOSED  # recovered out of HALF_OPEN


class _FakeJudge:
    def __init__(self, score):
        self._score = score

    async def score(self, goal, recent):
        return self._score


async def test_goal_judge_two_low_scores_force_open(store):
    m = manager(store, judge=_FakeJudge(0.1), settings={"goal_judge_cadence": 1})
    snap = None
    for i in range(4):
        step = await _append(store, i)
        # No watchdog trip at all; only the judge drives the escalation.
        snap, _ = await m.record(step, loop_result(breach=False, tripped=False))
        if snap.state == OPEN:
            break
    assert snap is not None and snap.state == OPEN  # two consecutive low scores -> OPEN
