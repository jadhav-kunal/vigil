"""Compression Layer 2 — The Token Company (spec 4.5 / 4.9), env-gated on ``TTC_API_KEY``.

Layer 1 (structural dedup) is always-on and free; Layer 2 is an optional paid pass over an HTTP
API that further shrinks the messages before they are forwarded. It runs on the request path
(it must, to shrink what is sent), so it adds one network round-trip — which is why it is opt-in.
Best-effort: any error returns the input unchanged, so a flaky Layer-2 endpoint never breaks the
proxy. The honest break-even (removed tokens × input price vs. the per-token TTC charge) lives in
the eval harness; this module just does the call.
"""

from __future__ import annotations

import httpx

from ..logging_config import get_logger, log_event
from ..settings import Settings

logger = get_logger("compress.l2")

_DEFAULT_BASE_URL = "https://api.tokencompany.ai"


class TokenCompressor:
    def __init__(
        self, *, api_key: str, base_url: str, client: httpx.AsyncClient | None = None
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def compress(self, messages: list[dict]) -> tuple[list[dict], bool]:
        """Return (messages, changed). On any failure, return the input unchanged."""
        if not messages:
            return messages, False
        owns = self._client is None
        client = self._client or httpx.AsyncClient(timeout=20.0)
        try:
            resp = await client.post(
                f"{self._base_url}/v1/compress",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"messages": messages},
            )
            resp.raise_for_status()
            out = resp.json().get("messages")
            if isinstance(out, list) and out:
                return out, out != messages
            return messages, False
        except Exception as exc:  # best-effort; a paid layer must never break the free proxy
            log_event(logger, 30, "compress.l2.error", error=str(exc))
            return messages, False
        finally:
            if owns:
                await client.aclose()


def make_l2_compressor(settings: Settings) -> TokenCompressor | None:
    """None unless TTC_API_KEY is set (then Layer 2 runs after Layer 1)."""
    if not settings.ttc_api_key:
        return None
    base = settings.ttc_base_url or _DEFAULT_BASE_URL
    log_event(logger, 20, "compress.l2.enabled", base_url=base)
    return TokenCompressor(api_key=settings.ttc_api_key, base_url=base)
