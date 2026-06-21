"""Sponsor adapters (spec 4.9): each is env-gated, a silent no-op without its key, and
individually testable. None of the heavy SDKs are installed here, so the guarded-import paths
(Sentry, tracing) are exercised for real and must degrade to None rather than crash."""

import json

import httpx
import pytest

from vigil_proxy.integrations.compression_l2 import TokenCompressor, make_l2_compressor
from vigil_proxy.integrations.sentry_sink import SentrySink, make_sentry
from vigil_proxy.integrations.tracing import Tracer, make_tracer
from vigil_proxy.judge import AnthropicGoalJudge, GoalJudge, make_judge
from vigil_proxy.models import Step
from vigil_proxy.settings import Settings

# --------------------------------------------------------------------------- L2 compression


def test_l2_disabled_without_key():
    assert make_l2_compressor(Settings()) is None


def test_l2_enabled_with_key():
    s = Settings()
    s.ttc_api_key = "ttc-test"
    assert isinstance(make_l2_compressor(s), TokenCompressor)


async def test_l2_compresses_via_api():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer k"
        body = json.loads(request.content)
        return httpx.Response(200, json={"messages": body["messages"][:1]})  # drop one message

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tc = TokenCompressor(api_key="k", base_url="http://ttc", client=client)
    out, changed = await tc.compress(
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    )
    assert changed and len(out) == 1
    await client.aclose()


async def test_l2_failure_returns_input_unchanged():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    msgs = [{"role": "user", "content": "a"}]
    out, changed = await TokenCompressor(
        api_key="k", base_url="http://ttc", client=client
    ).compress(msgs)
    assert (
        out is msgs and changed is False
    )  # best-effort: a flaky paid layer never breaks the proxy
    await client.aclose()


# --------------------------------------------------------------------------- Sentry


def _sentry_sdk_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentry_sdk") is not None


def test_sentry_disabled_without_dsn():
    assert make_sentry(Settings()) is None


@pytest.mark.skipif(_sentry_sdk_installed(), reason="exercises the SDK-absent path only")
def test_sentry_dsn_set_but_sdk_missing_degrades_to_none():
    s = Settings()
    s.sentry_dsn = "https://example@sentry.io/123"
    assert make_sentry(s) is None  # sentry_sdk not installed -> graceful no-op, not a crash


@pytest.mark.skipif(not _sentry_sdk_installed(), reason="needs the [sentry] extra installed")
def test_sentry_enabled_when_dsn_set_and_sdk_present(monkeypatch):
    import sentry_sdk

    captured = {}
    # Stub init so the test never spins up a real Sentry client (no global state, no atexit flush).
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: captured.update(kw))
    s = Settings()
    s.sentry_dsn = "https://example@sentry.io/123"
    sink = make_sentry(s)
    assert isinstance(sink, SentrySink)
    assert captured["dsn"] == s.sentry_dsn


def test_sentry_sink_captures_breaker_open():
    class FakeSdk:
        def __init__(self):
            self.context = None
            self.messages = []

        def set_context(self, name, data):
            self.context = (name, data)

        def capture_message(self, msg, level):
            self.messages.append((msg, level))

        def flush(self, timeout=None):
            self.flushed = True

    sdk = FakeSdk()
    SentrySink(sdk).capture_breaker_open("sess-1", {"reason": "loop"})
    assert sdk.context[0] == "vigil_breaker"
    assert sdk.messages and sdk.messages[0][1] == "error" and "sess-1" in sdk.messages[0][0]


# --------------------------------------------------------------------------- tracing


def test_tracing_disabled_when_unconfigured():
    assert make_tracer(Settings()) is None


def test_tracing_configured_but_sdk_missing_degrades_to_none():
    s = Settings()
    s.phoenix_collector_endpoint = "http://localhost:6006"
    assert s.tracing_enabled is True
    assert make_tracer(s) is None  # opentelemetry not installed -> no-op, never breaks local mode


def test_tracer_records_a_span_without_raising():
    class FakeSpan:
        def __init__(self):
            self.attrs = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def set_attribute(self, k, v):
            self.attrs[k] = v

    class FakeOtel:
        def __init__(self):
            self.span = FakeSpan()

        def start_as_current_span(self, name):
            return self.span

    otel = FakeOtel()
    step = Step(
        session_id="s",
        step_index=2,
        model_requested="gpt-4o",
        model_used="gpt-4o-mini",
        tool_name="lookup",
        final_score=0.9,
        breaker_state="HALF_OPEN",
    )
    Tracer(otel).record_step(step)
    assert otel.span.attrs["session.id"] == "s"
    assert otel.span.attrs["llm.model_name"] == "gpt-4o-mini"
    assert otel.span.attrs["vigil.breaker_state"] == "HALF_OPEN"


# --------------------------------------------------------------------------- Anthropic judge


def test_make_judge_selects_anthropic_provider():
    s = Settings()
    s.judge_provider = "anthropic"
    s.judge_base_url = "https://api.anthropic.com"
    s.judge_api_key = "k"
    s.judge_model = "claude-3-5-haiku-latest"
    assert isinstance(make_judge(s), AnthropicGoalJudge)


def test_make_judge_defaults_to_openai():
    s = Settings()
    s.judge_base_url = "https://api.openai.com/v1"
    s.judge_api_key = "k"
    s.judge_model = "gpt-4o-mini"
    assert isinstance(make_judge(s), GoalJudge)


async def test_anthropic_judge_parses_score():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "k"
        assert request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "0.2"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    judge = AnthropicGoalJudge(
        base_url="https://api.anthropic.com",
        api_key="k",
        model="claude-3-5-haiku-latest",
        client=client,
    )
    assert await judge.score("ship the release", []) == pytest.approx(0.2)
    await client.aclose()
