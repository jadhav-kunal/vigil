"""Proxy app: health, unary passthrough (unmodified), and background step capture."""

import json
import time

import httpx
from fastapi.testclient import TestClient

from vigil_proxy.analyzer import Analyzer
from vigil_proxy.app import CaptureCtx, _capture, app
from vigil_proxy.breaker_manager import BreakerManager
from vigil_proxy.embedder import HashingEmbedder
from vigil_proxy.hub import Broadcaster
from vigil_proxy.normalize import normalize_openai_request
from vigil_proxy.pricing import DEFAULT_PRICE_TABLE
from vigil_proxy.settings import Settings
from vigil_proxy.store import SQLiteStore


def _wait_metrics(client: TestClient, session_id: str, *, tries: int = 200) -> dict:
    """Poll the metrics endpoint until the background capture task (Invariant I1: analysis runs
    off the response path) has persisted at least one step. Each GET pumps the event loop and the
    sleep yields wall-time to the embedding thread, so the wait is robust under full-suite load."""
    m = {"steps": 0}
    for _ in range(tries):
        m = client.get(f"/metrics/session/{session_id}").json()
        if m.get("steps", 0) >= 1:
            return m
        time.sleep(0.01)
    return m


def _test_analyzer():
    return Analyzer(HashingEmbedder(), window=5, trip_streak=3, theta_sim=0.85, theta_ent=0.30)


def _test_breaker(store, broadcaster, analyzer):
    return BreakerManager(
        store=store,
        broadcaster=broadcaster,
        price_table=DEFAULT_PRICE_TABLE,
        settings=Settings(),
        judge=None,
        analyzer=analyzer,
    )


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_unary_passthrough_is_unmodified(monkeypatch):
    upstream_body = {
        "id": "chatcmpl-1",
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer sk-test"  # key passed through
        return httpx.Response(200, json=upstream_body)

    with TestClient(app) as client:
        # Swap the upstream client for a mocked transport after lifespan startup.
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        r = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk-test", "x-vigil-session-id": "s1"},
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert r.json() == upstream_body  # body returned byte-for-byte equal


async def test_capture_persists_and_broadcasts_step(tmp_path):
    store = SQLiteStore(str(tmp_path / "c.db"))
    await store.init()
    sent: list[dict] = []

    class FakeBroadcaster(Broadcaster):
        async def broadcast(self, message):
            sent.append(message)

    analyzer = _test_analyzer()
    broadcaster = FakeBroadcaster()
    ctx = CaptureCtx(
        store=store,
        broadcaster=broadcaster,
        price_table=DEFAULT_PRICE_TABLE,
        analyzer=analyzer,
        breaker=_test_breaker(store, broadcaster, analyzer),
    )
    req = normalize_openai_request({"model": "gpt-4o", "messages": []})
    resp = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    await _capture(ctx, req, resp, "sess-x", None)
    steps = await store.get_steps("sess-x")
    assert len(steps) == 1
    assert steps[0].assistant_text == "ok"
    assert steps[0].step_index == 0
    # The step was broadcast with a computed cost and watchdog metrics.
    assert len(sent) == 1 and sent[0]["type"] == "step"
    assert "cost_usd" in sent[0]["step"]
    assert "sim_score" in sent[0]["step"] and "final_score" in sent[0]["step"]
    await store.close()


def test_session_metrics_endpoint():
    body = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with TestClient(app) as client:
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer k", "x-vigil-session-id": "m1"},
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "q"}]},
        )
        m = _wait_metrics(client, "m1")
        assert m["session_id"] == "m1"
        assert m["steps"] >= 1
        assert m["cost_usd"] > 0


def test_compression_collapses_loop_on_the_wire_and_records_savings():
    """Slice 5: a looping conversation is compressed BEFORE forwarding upstream, the agent still
    gets the upstream response verbatim, and the step records tokens_before > tokens_after."""
    forwarded: dict = {}
    upstream = {
        "choices": [{"message": {"content": "done"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 2},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded["body"] = json.loads(request.content)
        return httpx.Response(200, json=upstream)

    msgs = [
        {"role": "system", "content": "You are a release agent."},
        {"role": "user", "content": "Ship it."},
    ]
    for _ in range(6):
        msgs += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "check_status", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c", "content": "still pending"},
        ]
    msgs.append({"role": "user", "content": "Any update?"})

    with TestClient(app) as client:
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        r = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer k", "x-vigil-session-id": "loop1"},
            json={"model": "gpt-4o", "messages": msgs},
        )
        assert (
            r.json()["choices"][0]["message"]["content"] == "done"
        )  # agent sees upstream verbatim
        fwd = forwarded["body"]["messages"]
        assert len(fwd) < len(msgs)  # collapsed cycles were dropped before forwarding
        markers = [m for m in fwd if str(m.get("content", "")).startswith("[vigil-compressed]")]
        assert len(markers) == 1
        # every remaining tool message still follows an assistant tool_call (request stays valid)
        for i, m in enumerate(fwd):
            if m.get("role") == "tool":
                assert fwd[i - 1].get("role") == "assistant" and fwd[i - 1].get("tool_calls")

        m = _wait_metrics(client, "loop1")
        assert m["tokens_before_compression"] > m["tokens_after_compression"]
        assert m["tokens_saved"] > 0


def test_governor_routes_extraction_to_cheaper_model_on_the_wire():
    """Slice 6: with the governor enabled, an EXTRACTION step has its model rewritten to a cheaper
    one BEFORE forwarding, and the step records the routed model_used."""
    forwarded: dict = {}
    upstream = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded["body"] = json.loads(request.content)
        return httpx.Response(200, json=upstream)

    with TestClient(app) as client:
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.state.settings.governor_enabled = True
        try:
            r = client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer k", "x-vigil-session-id": "gov1"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "extract the totals from the report"}],
                },
            )
            assert r.json() == upstream  # agent still gets the upstream response verbatim
            assert forwarded["body"]["model"] == "gpt-4o-mini"  # routed down before forwarding
            m = _wait_metrics(client, "gov1")
            assert m["models_used"] == ["gpt-4o-mini"]  # model_used reflects the routed model
        finally:
            app.state.settings.governor_enabled = False  # don't leak the toggle to other tests


def test_layer2_compression_runs_on_the_request_path_when_enabled():
    """Slice 9: a configured Layer-2 compressor (Token Company) runs after Layer 1 on the request
    path and shrinks the forwarded body. Here a fake L2 stands in for the paid API."""
    forwarded: dict = {}
    upstream = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5}}

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded["body"] = json.loads(request.content)
        return httpx.Response(200, json=upstream)

    class FakeL2:
        async def compress(self, messages):
            return messages[:-1], True  # drop one message to prove L2 ran

    with TestClient(app) as client:
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.state.l2 = FakeL2()
        try:
            msgs = [{"role": "user", "content": f"m{i}"} for i in range(3)]
            client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer k", "x-vigil-session-id": "l2"},
                json={"model": "gpt-4o", "messages": msgs},
            )
            assert len(forwarded["body"]["messages"]) == 2  # L2 dropped one before forwarding
        finally:
            app.state.l2 = None  # don't leak into other tests


def test_replay_endpoint_reconstructs_session_with_zero_upstream_calls():
    """Slice 8: after a live request is captured, /replay rebuilds the trajectory purely from the
    forensic cache — and crucially never touches upstream (the transport raises if called)."""
    upstream = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 5},
    }
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=upstream)

    with TestClient(app) as client:
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer k", "x-vigil-session-id": "rep1"},
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "q"}]},
        )
        _wait_metrics(client, "rep1")  # let the background capture cache the exchange
        calls_before_replay = calls["n"]

        # Poll replay until the exchange is recorded (recorded just after the step is persisted).
        for _ in range(200):
            r = client.post("/sessions/rep1/replay")
            if r.status_code == 200 and r.json()["steps"]:
                break
            time.sleep(0.01)
        body = r.json()
        assert r.status_code == 200
        assert body["upstream_calls"] == 0
        assert calls["n"] == calls_before_replay  # replay made NO new upstream calls
        assert len(body["steps"]) == 1
        assert body["steps"][0]["assistant_text"] == "answer"
        assert isinstance(body["trace_hash"], str) and len(body["trace_hash"]) == 64


def test_metrics_aggregate_counts_only_no_content():
    """Slice 10: /metrics/aggregate reports cross-session COUNTS and never echoes prompt/tool
    content (the privacy invariant)."""
    body = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with TestClient(app) as client:
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer k", "x-vigil-session-id": "agg1"},
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "secret prompt"}]},
        )
        _wait_metrics(client, "agg1")
        agg = client.get("/metrics/aggregate").json()
        assert agg["sessions"] >= 1 and agg["steps"] >= 1
        assert agg["cost_usd"] > 0
        assert "gpt-4o" in agg["models_used"]
        # The privacy invariant: no message text anywhere in the aggregate payload.
        assert "secret prompt" not in json.dumps(agg)
        assert "answer" not in json.dumps(agg)


def test_replay_unknown_session_is_404():
    with TestClient(app) as client:
        r = client.post("/sessions/does-not-exist/replay")
        assert r.status_code == 404


def test_override_endpoint_resets_breaker():
    with TestClient(app) as client:
        r = client.post("/sessions/any-session/override")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "CLOSED" and body["override"] is True
        # Status endpoint reflects the reset.
        s = client.get("/sessions/any-session/breaker").json()
        assert s["state"] == "CLOSED"


def test_ws_hello_and_live_step():
    body = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with TestClient(app) as client:
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello" and "price_table" in hello
            snap = ws.receive_json()
            assert snap["type"] == "snapshot"
            client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer k", "x-vigil-session-id": "w1"},
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "q"}]},
            )
            # Drain any snapshot replays until the live w1 step arrives.
            for _ in range(50):
                evt = ws.receive_json()
                if evt.get("type") == "step" and evt["step"]["session_id"] == "w1":
                    break
            else:
                raise AssertionError("did not receive live w1 step over the websocket")
