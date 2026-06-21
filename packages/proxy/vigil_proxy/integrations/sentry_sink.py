"""Sentry agent-failure reporting (spec 4.9), env-gated on ``SENTRY_DSN``.

When the breaker trips to OPEN, Vigil raises a Sentry issue carrying the post-mortem. The
``sentry_sdk`` import is guarded: if the DSN is set but the SDK isn't installed, we log a warning
once and stay a no-op rather than crashing (Invariant I2 / CLAUDE.md: an absent optional SDK
never breaks local mode).
"""

from __future__ import annotations

from typing import Any

from ..logging_config import get_logger, log_event
from ..settings import Settings

logger = get_logger("sentry")


class SentrySink:
    def __init__(self, sdk: Any) -> None:
        self._sdk = sdk

    def capture_breaker_open(self, session_id: str, post_mortem: dict | None) -> None:
        try:
            self._sdk.set_context("vigil_breaker", post_mortem or {"session_id": session_id})
            self._sdk.capture_message(
                f"Vigil breaker OPEN — semantic loop halted (session {session_id})", level="error"
            )
            # Flush so the event is delivered even if the process is short-lived (serverless) or
            # is torn down right after the halt. A breaker OPEN is rare, so the brief wait is fine.
            self._sdk.flush(timeout=3.0)
            log_event(logger, 20, "sentry.captured", session=session_id)
        except Exception as exc:  # best-effort; never break the breaker path
            log_event(logger, 30, "sentry.error", error=str(exc))


def make_sentry(settings: Settings) -> SentrySink | None:
    """None unless SENTRY_DSN is set and sentry_sdk is importable."""
    if not settings.sentry_dsn:
        return None
    try:
        import sentry_sdk
    except ImportError:
        log_event(logger, 30, "sentry.sdk_missing", hint="pip install vigil-proxy[sentry]")
        return None
    sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.0)
    log_event(logger, 20, "sentry.enabled")
    return SentrySink(sentry_sdk)
