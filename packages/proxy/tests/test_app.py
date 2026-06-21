"""Proxy app: health, unary passthrough (unmodified), and background step capture."""

import httpx
from fastapi.testclient import TestClient

from vigil_proxy.app import _capture, app
from vigil_proxy.normalize import normalize_openai_request
from vigil_proxy.store import SQLiteStore


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


async def test_capture_persists_step(tmp_path):
    store = SQLiteStore(str(tmp_path / "c.db"))
    await store.init()
    req = normalize_openai_request({"model": "gpt-4o", "messages": []})
    resp = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    await _capture(store, req, resp, "sess-x", None)
    steps = await store.get_steps("sess-x")
    assert len(steps) == 1
    assert steps[0].assistant_text == "ok"
    assert steps[0].step_index == 0
    await store.close()
