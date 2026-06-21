"""SQLite store roundtrip, step indexing, and idempotent migration."""

import pytest

from vigil_proxy.models import Step
from vigil_proxy.store import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "t.db"))
    await s.init()
    yield s
    await s.close()


async def test_roundtrip(store):
    step = Step(
        session_id="s1",
        step_index=0,
        model_requested="gpt-4o",
        model_used="gpt-4o-mini",
        tool_name="write_file",
        tool_args={"path": "x"},
        assistant_text="done",
        prompt_tokens=10,
        completion_tokens=3,
        tokens_before_compression=10,
        tokens_after_compression=10,
        caused_state_mutation=True,
    )
    rowid = await store.add_step(step)
    assert rowid > 0
    got = await store.get_steps("s1")
    assert len(got) == 1
    assert got[0].tool_name == "write_file"
    assert got[0].tool_args == {"path": "x"}
    assert got[0].caused_state_mutation is True
    assert got[0].model_used == "gpt-4o-mini"


async def test_step_index_monotonic_per_session(store):
    assert await store.next_step_index("s1") == 0
    await store.add_step(Step(session_id="s1", step_index=0, model_requested="m", model_used="m"))
    assert await store.next_step_index("s1") == 1
    await store.add_step(Step(session_id="s1", step_index=1, model_requested="m", model_used="m"))
    assert await store.next_step_index("s1") == 2
    # Separate session has its own counter.
    assert await store.next_step_index("s2") == 0


async def test_append_step_assigns_monotonic_index(store):
    idxs = []
    for _ in range(3):
        i = await store.append_step(
            Step(session_id="s1", step_index=0, model_requested="m", model_used="m")
        )
        idxs.append(i)
    assert idxs == [0, 1, 2]
    # Independent session starts its own sequence.
    assert (
        await store.append_step(
            Step(session_id="s2", step_index=0, model_requested="m", model_used="m")
        )
        == 0
    )


async def test_append_step_is_race_safe_under_gather(store):
    import asyncio

    async def one():
        return await store.append_step(
            Step(session_id="race", step_index=0, model_requested="m", model_used="m")
        )

    results = await asyncio.gather(*[one() for _ in range(10)])
    # The UNIQUE index + atomic assignment guarantee 10 distinct indices, no collision.
    assert sorted(results) == list(range(10))
    steps = await store.get_steps("race")
    assert [s.step_index for s in steps] == list(range(10))


async def test_list_sessions(store):
    await store.add_step(Step(session_id="a", step_index=0, model_requested="m", model_used="m"))
    await store.add_step(Step(session_id="b", step_index=0, model_requested="m", model_used="m"))
    assert await store.list_sessions() == ["a", "b"]


async def test_migration_is_idempotent(tmp_path):
    path = str(tmp_path / "mig.db")
    s1 = SQLiteStore(path)
    await s1.init()
    await s1.add_step(Step(session_id="s", step_index=0, model_requested="m", model_used="m"))
    await s1.close()
    # Re-init against an existing db must be a no-op, not an error.
    s2 = SQLiteStore(path)
    await s2.init()
    await s2.init()  # twice, deliberately
    assert len(await s2.get_steps("s")) == 1
    await s2.close()
