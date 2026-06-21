"""Tracing (spec 4.9): vendor-neutral OpenInference/OTel, exporter chosen by env.

There is ONE instrumentation path (plain OTLP/HTTP). Whether spans go to a local Phoenix
collector (``PHOENIX_COLLECTOR_ENDPOINT``) or to Arize AX (``ARIZE_SPACE_ID`` + ``ARIZE_API_KEY``)
is purely a matter of the exporter endpoint + headers — no vendor SDK is imported, so a wrong
import can never break local Phoenix (the footgun CLAUDE.md explicitly bans). If the OTel SDK
isn't installed, this degrades to a silent no-op (Invariant I2).
"""

from __future__ import annotations

from typing import Any

from ..logging_config import get_logger, log_event
from ..models import Step
from ..settings import Settings

logger = get_logger("tracing")


def _exporter_target(settings: Settings) -> tuple[str, dict[str, str]]:
    """(endpoint, headers) for the configured backend. Phoenix wins if both are set."""
    if settings.phoenix_collector_endpoint:
        headers = {"api_key": settings.phoenix_api_key} if settings.phoenix_api_key else {}
        return settings.phoenix_collector_endpoint.rstrip("/") + "/v1/traces", headers
    return (
        "https://otlp.arize.com/v1/traces",
        {"space_id": settings.arize_space_id or "", "api_key": settings.arize_api_key or ""},
    )


class Tracer:
    def __init__(self, otel_tracer: Any) -> None:
        self._tracer = otel_tracer

    def record_step(self, step: Step) -> None:
        """Emit one span per captured step. Best-effort — tracing never breaks capture."""
        try:
            with self._tracer.start_as_current_span("vigil.step") as span:
                span.set_attribute("session.id", step.session_id)
                span.set_attribute("step.index", step.step_index)
                span.set_attribute("llm.model_name", step.model_used or "")
                if step.tool_name:
                    span.set_attribute("tool.name", step.tool_name)
                if step.final_score is not None:
                    span.set_attribute("vigil.final_score", step.final_score)
                if step.tool_entropy is not None:
                    span.set_attribute("vigil.tool_entropy", step.tool_entropy)
                span.set_attribute("vigil.breaker_state", step.breaker_state or "CLOSED")
                span.set_attribute("vigil.watchdog_tripped", step.watchdog_tripped)
        except Exception as exc:  # never let an exporter hiccup touch the capture path
            log_event(logger, 30, "tracing.error", error=str(exc))


def make_tracer(settings: Settings) -> Tracer | None:
    """None unless a backend is configured AND the OTel SDK is importable."""
    if not settings.tracing_enabled:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log_event(logger, 30, "tracing.sdk_missing", hint="pip install vigil-proxy[tracing]")
        return None

    endpoint, headers = _exporter_target(settings)
    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.tracing_service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
    )
    trace.set_tracer_provider(provider)
    backend = "phoenix" if settings.phoenix_collector_endpoint else "arize"
    log_event(logger, 20, "tracing.enabled", backend=backend, endpoint=endpoint)
    return Tracer(trace.get_tracer("vigil"))
