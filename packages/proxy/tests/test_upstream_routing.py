"""Per-request provider routing (x-vigil-upstream), gated by VIGIL_ALLOW_UPSTREAM_HEADER and an
optional allowlist. Covers the pure allowlist logic and the proxy behavior: the override is
honored only when enabled+allowed, and Vigil's own x-vigil-* headers never leak upstream."""

import httpx
from fastapi.testclient import TestClient

from vigil_proxy.app import _upstream_allowed, app
from vigil_proxy.settings import Settings

UPSTREAM = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3}}


def _post_capture(headers):
    """POST through the proxy with a recording upstream; return the captured httpx.Request."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json=UPSTREAM)

    with TestClient(app) as client:
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer k", **headers},
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    return seen


# --------------------------------------------------------------------------- pure allowlist


def test_allowlist_allows_any_when_unset():
    assert _upstream_allowed("https://api.openai.com/v1", Settings()) is True


def test_allowlist_rejects_non_http_scheme():
    assert _upstream_allowed("ftp://api.openai.com", Settings()) is False


def test_allowlist_enforces_prefixes():
    s = Settings()
    s.upstream_allowlist = "https://api.openai.com, https://openrouter.ai/api"
    assert _upstream_allowed("https://api.openai.com/v1", s) is True
    assert _upstream_allowed("https://openrouter.ai/api/v1", s) is True
    assert _upstream_allowed("https://evil.example.com/v1", s) is False


# --------------------------------------------------------------------------- proxy behavior


def test_override_ignored_when_disabled():
    # Default settings: allow_upstream_header is False -> the header is ignored, default upstream used.
    seen = _post_capture({"x-vigil-upstream": "https://evil.example.com/v1"})
    assert "api.openai.com" in seen["url"]
    assert "evil.example.com" not in seen["url"]


def test_override_honored_when_enabled():
    with TestClient(app):  # ensure lifespan settings exist
        app.state.settings.allow_upstream_header = True
        try:
            seen = _post_capture({"x-vigil-upstream": "https://openrouter.ai/api/v1"})
            assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
        finally:
            app.state.settings.allow_upstream_header = False


def test_override_rejected_by_allowlist_falls_back_to_default():
    with TestClient(app):
        app.state.settings.allow_upstream_header = True
        app.state.settings.upstream_allowlist = "https://api.openai.com"
        try:
            seen = _post_capture({"x-vigil-upstream": "https://evil.example.com/v1"})
            assert "api.openai.com" in seen["url"]  # rejected override -> default
        finally:
            app.state.settings.allow_upstream_header = False
            app.state.settings.upstream_allowlist = None


def test_vigil_control_headers_are_not_forwarded_upstream():
    seen = _post_capture({"x-vigil-session-id": "s1", "x-vigil-upstream": "https://x/v1"})
    fwd = {k.lower() for k in seen["headers"]}
    assert "x-vigil-session-id" not in fwd
    assert "x-vigil-upstream" not in fwd
    assert "authorization" in fwd  # the provider key IS forwarded
