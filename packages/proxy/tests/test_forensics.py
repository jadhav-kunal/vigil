"""Forensics (spec 4.7): the content-addressed exchange cache, deterministic cached-trace replay
(zero upstream calls), and fork-at-step-N model swap. The load-bearing guarantees are that replay
never calls upstream and that the cache stores no provider key."""

import pytest

from vigil_proxy.forensics import Forensics, canonical_request, request_hash
from vigil_proxy.pricing import DEFAULT_PRICE_TABLE
from vigil_proxy.store import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "f.db"))
    await s.init()
    yield s
    await s.close()


def _req(model="gpt-4o", text="check the status"):
    return {"model": model, "messages": [{"role": "user", "content": text}]}


def _resp(text="all green", tool=None):
    msg = {"role": "assistant", "content": text}
    if tool:
        msg["tool_calls"] = [
            {"id": "c", "type": "function", "function": {"name": tool, "arguments": "{}"}}
        ]
        msg["content"] = None
    return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}


def test_request_hash_is_stable_and_ignores_key_ordering():
    a = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "tools": None}
    b = {"messages": [{"role": "user", "content": "hi"}], "tools": None, "model": "gpt-4o"}
    assert request_hash(a) == request_hash(b)
    assert "content" in canonical_request(a)


async def test_cache_exchange_is_idempotent(store):
    await store.cache_exchange("s", 0, request_hash(_req()), _req(), _resp(), "gpt-4o")
    await store.cache_exchange(
        "s", 0, request_hash(_req()), _req(), _resp(), "gpt-4o"
    )  # re-capture
    exchanges = await store.get_exchanges("s")
    assert len(exchanges) == 1  # (session, step) is unique — no duplicate row


async def test_cache_stores_no_provider_key(store):
    f = Forensics(store, DEFAULT_PRICE_TABLE)
    await f.record("s", 0, _req(), _resp(), "gpt-4o")
    exchanges = await store.get_exchanges("s")
    blob = str(exchanges[0])
    assert "sk-" not in blob and "authorization" not in blob.lower()  # body only, never the key


async def test_replay_reconstructs_trajectory_without_upstream(store):
    f = Forensics(store, DEFAULT_PRICE_TABLE)
    await f.record("s", 0, _req(text="step one"), _resp("did one", tool="lookup"), "gpt-4o")
    await f.record("s", 1, _req(text="step two"), _resp("did two"), "gpt-4o-mini")

    result = await f.replay("s")
    assert result is not None
    assert result.upstream_calls == 0  # cached-trace replay never calls upstream
    assert [s.step_index for s in result.steps] == [0, 1]
    assert result.steps[0].tool_name == "lookup"
    assert result.steps[0].model == "gpt-4o"
    assert result.steps[1].assistant_text == "did two"
    assert all(s.cost_usd >= 0 for s in result.steps)


async def test_replay_is_deterministic(store):
    f = Forensics(store, DEFAULT_PRICE_TABLE)
    await f.record("s", 0, _req(), _resp(), "gpt-4o")
    a = await f.replay("s")
    b = await f.replay("s")
    assert a is not None and b is not None
    assert a.trace_hash == b.trace_hash  # stable hash over identical cached content


async def test_replay_missing_session_returns_none(store):
    f = Forensics(store, DEFAULT_PRICE_TABLE)
    assert await f.replay("nope") is None


async def test_fork_swaps_model_and_flags_divergence(store):
    f = Forensics(store, DEFAULT_PRICE_TABLE)
    await f.record("s", 0, _req(text="fix it"), _resp("retrying", tool="retry"), "gpt-4o")

    calls = {"n": 0}

    async def fake_generate(forked_request):
        calls["n"] += 1
        assert forked_request["model"] == "gpt-4o-mini"  # the swapped model was forwarded
        return _resp("found the bug", tool="read_logs")  # a different action

    result = await f.fork("s", 0, "gpt-4o-mini", fake_generate)
    assert result is not None
    assert calls["n"] == 1  # a fork makes exactly one upstream call
    assert result.original_model == "gpt-4o" and result.forked_model == "gpt-4o-mini"
    assert result.original["tool_name"] == "retry"
    assert result.forked["tool_name"] == "read_logs"
    assert result.same_tool is False
    assert "attributable to the model" in result.divergence


async def test_fork_same_action_reports_stability(store):
    f = Forensics(store, DEFAULT_PRICE_TABLE)
    await f.record("s", 0, _req(), _resp("ok", tool="lookup"), "gpt-4o")

    async def fake_generate(forked_request):
        return _resp("ok", tool="lookup")  # identical action under identical context

    result = await f.fork("s", 0, "gpt-4o-mini", fake_generate)
    assert result is not None
    assert result.same_tool and result.same_args
    assert "stable" in result.divergence


async def test_fork_missing_step_returns_none(store):
    f = Forensics(store, DEFAULT_PRICE_TABLE)
    await f.record("s", 0, _req(), _resp(), "gpt-4o")

    async def fake_generate(_):
        raise AssertionError("must not call upstream when the step is absent")

    assert await f.fork("s", 5, "gpt-4o-mini", fake_generate) is None
