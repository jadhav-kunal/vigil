"""Proxy app: health, unary passthrough (unmodified), and background step capture."""

import httpx
from fastapi.testclient import TestClient

from vigil_proxy.analyzer import Analyzer
from vigil_proxy.app import CaptureCtx, _capture, app
from vigil_proxy.embedder import HashingEmbedder
from vigil_proxy.hub import Broadcaster
from vigil_proxy.normalize import normalize_openai_request
from vigil_proxy.pricing import DEFAULT_PRICE_TABLE
from vigil_proxy.store import SQLiteStore


def _test_analyzer():
    return Analyzer(HashingEmbedder(), window=5, trip_streak=3, theta_sim=0.85, theta_ent=0.30)


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

    ctx = CaptureCtx(
        store=store,
        broadcaster=FakeBroadcaster(),
        price_table=DEFAULT_PRICE_TABLE,
        analyzer=_test_analyzer(),
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
        m = client.get("/metrics/session/m1").json()
        assert m["session_id"] == "m1"
        assert m["steps"] >= 1
        assert m["cost_usd"] > 0


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
