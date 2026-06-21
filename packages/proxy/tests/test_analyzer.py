"""Analyzer: stateful watchdog over a deterministic hashing embedder (no model, no network)."""

import pytest

from vigil_proxy.analyzer import Analyzer
from vigil_proxy.embedder import HashingEmbedder
from vigil_proxy.models import Step


def make_analyzer(**kw):
    defaults = dict(window=5, trip_streak=3, theta_sim=0.85, theta_ent=0.30)
    defaults.update(kw)
    return Analyzer(HashingEmbedder(), **defaults)


def step(session, text, tool=None, mutation=False):
    return Step(
        session_id=session,
        step_index=0,
        model_requested="m",
        model_used="m",
        tool_name=tool,
        assistant_text=text,
        caused_state_mutation=mutation,
    )


async def test_tight_loop_trips_after_k_breaches():
    a = make_analyzer()
    results = []
    for _ in range(4):
        results.append(
            await a.analyze(step("loop", "checking status: still pending", tool="check_status"))
        )
    # First step: no prior, sim 0, no breach. Then identical repeats -> sim 1.0, entropy 0.
    assert results[0].breach is False
    assert results[1].breach and results[1].streak == 1
    assert results[2].streak == 2 and results[2].tripped is False
    assert results[3].streak == 3 and results[3].tripped is True  # K=3 consecutive breaches


async def test_healthy_varied_work_does_not_trip():
    a = make_analyzer()
    # A paginator: distinct content and distinct tools each step -> low similarity, high entropy.
    tools = ["read_page", "parse_table", "store_row", "next_page", "read_page"]
    tripped = False
    for i, tool in enumerate(tools):
        r = await a.analyze(
            step("healthy", f"processing record number {i} with unique data {i}", tool=tool)
        )
        tripped = tripped or r.tripped
    assert tripped is False


async def test_state_mutating_loop_is_not_flagged():
    a = make_analyzer()
    # Identical write calls: high similarity, zero entropy, but the state penalty saves it.
    tripped = False
    for _ in range(5):
        r = await a.analyze(
            step("writer", "writing row to database", tool="db_insert", mutation=True)
        )
        tripped = tripped or r.tripped
        assert r.state_penalty == 0.30
    assert tripped is False


async def test_streak_resets_on_non_breach():
    a = make_analyzer(trip_streak=3)
    await a.analyze(step("s", "same repeated text", tool="t"))
    r1 = await a.analyze(step("s", "same repeated text", tool="t"))
    assert r1.streak == 1
    # A divergent step breaks the streak.
    r2 = await a.analyze(step("s", "completely different unrelated content here", tool="other"))
    assert r2.streak == 0


async def test_reset_clears_session_state():
    a = make_analyzer()
    for _ in range(3):
        await a.analyze(step("x", "loop text", tool="t"))
    a.reset("x")
    r = await a.analyze(step("x", "loop text", tool="t"))
    # After reset there is no prior step, so similarity is 0 and no breach.
    assert r.sim_score == 0.0 and r.breach is False


@pytest.mark.parametrize("mode", ["independent"])
async def test_sessions_are_independent(mode):
    a = make_analyzer()
    await a.analyze(step("A", "loopy", tool="t"))
    r = await a.analyze(step("B", "fresh session first step", tool="t"))
    assert r.sim_score == 0.0  # B's first step never compares against A


async def test_concurrent_same_session_is_serialized():
    import asyncio

    a = make_analyzer(trip_streak=3)
    # Fire 6 identical-loop analyses concurrently. The per-session lock must serialize them so
    # the streak increments cleanly and the trip still fires, despite the embed await.
    results = await asyncio.gather(
        *[a.analyze(step("conc", "identical looping content", tool="t")) for _ in range(6)]
    )
    streaks = sorted(r.streak for r in results)
    assert streaks == [0, 1, 2, 3, 4, 5]  # no lost or duplicated increments
    assert any(r.tripped for r in results)
