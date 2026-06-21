"""Semantic cache — Redis LangCache (M4, spec 4.4 / 4.9), env-gated on ``REDIS_LANGCACHE_*``.

A pre-proxy lookup: before forwarding a (non-streaming) request, search LangCache for a
semantically-similar prior request scoped by hard metadata (the model). On a hit above the
similarity threshold we serve the cached response and skip the upstream call entirely; on a miss
we forward and store the response for next time.

LangCache is a hosted REST service, so this is a thin httpx client — no Redis SDK needed. It is
best-effort: any error is treated as a miss, so the cache can never break the proxy. Honest
economics (the eval reports the break-even): a hit saves the whole LLM call, but EVERY lookup —
hit or miss — costs a round-trip, so it only nets positive above a break-even hit rate. Scoping by
model avoids serving one model's answer for another's request; a wrong semantic hit is the real
correctness risk, which is why the similarity threshold is conservative and configurable.
"""

from __future__ import annotations

import json

import httpx

from ..logging_config import get_logger, log_event
from ..settings import Settings

logger = get_logger("semcache")


class LangCacheClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        cache_id: str,
        min_similarity: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._cache_id = cache_id
        self._min_sim = min_similarity
        self._client = client

    def _url(self, suffix: str) -> str:
        return f"{self._base}/v1/caches/{self._cache_id}/entries{suffix}"

    async def search(self, prompt: str, attributes: dict) -> dict | None:
        """Return the cached response dict on a hit above the threshold, else None (a miss)."""
        owns = self._client is None
        client = self._client or httpx.AsyncClient(timeout=10.0)
        try:
            resp = await client.post(
                self._url("/search"),
                headers={"Authorization": f"Bearer {self._key}"},
                json={"prompt": prompt, "attributes": attributes},
            )
            resp.raise_for_status()
            data = resp.json().get("data") or []
            if not data:
                return None
            top = data[0]
            if float(top.get("similarity", 0.0)) < self._min_sim:
                return None
            raw = top.get("response")
            if isinstance(raw, str):
                return json.loads(raw)
            return raw if isinstance(raw, dict) else None
        except Exception as exc:  # best-effort: a flaky cache is a miss, never an error
            log_event(logger, 30, "semcache.search_error", error=str(exc))
            return None
        finally:
            if owns:
                await client.aclose()

    async def store(self, prompt: str, response: dict, attributes: dict) -> None:
        owns = self._client is None
        client = self._client or httpx.AsyncClient(timeout=10.0)
        try:
            await client.post(
                self._url(""),
                headers={"Authorization": f"Bearer {self._key}"},
                json={"prompt": prompt, "response": json.dumps(response), "attributes": attributes},
            )
        except Exception as exc:  # best-effort; storing is never on the response path
            log_event(logger, 30, "semcache.store_error", error=str(exc))
        finally:
            if owns:
                await client.aclose()


def make_semantic_cache(settings: Settings) -> LangCacheClient | None:
    """None unless all REDIS_LANGCACHE_* vars are set."""
    if not settings.langcache_enabled:
        return None
    assert (
        settings.redis_langcache_url
        and settings.redis_langcache_api_key
        and settings.redis_langcache_cache_id
    )
    log_event(
        logger, 20, "semcache.enabled", min_similarity=settings.redis_langcache_min_similarity
    )
    return LangCacheClient(
        base_url=settings.redis_langcache_url,
        api_key=settings.redis_langcache_api_key,
        cache_id=settings.redis_langcache_cache_id,
        min_similarity=settings.redis_langcache_min_similarity,
    )
