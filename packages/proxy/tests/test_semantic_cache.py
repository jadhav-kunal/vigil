"""Semantic cache (M4, Redis LangCache) — env-gated, best-effort, scoped by model. Tests cover the
hit/miss/threshold logic, the store call, graceful failure, and the factory gating."""

import json

import httpx

from vigil_proxy.integrations.semantic_cache import LangCacheClient, make_semantic_cache
from vigil_proxy.settings import Settings


def _client(handler) -> LangCacheClient:
    return LangCacheClient(
        base_url="http://langcache",
        api_key="k",
        cache_id="c1",
        min_similarity=0.9,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_make_semantic_cache_disabled_without_keys():
    assert make_semantic_cache(Settings()) is None


def test_make_semantic_cache_enabled_with_keys():
    s = Settings()
    s.redis_langcache_url = "http://langcache"
    s.redis_langcache_api_key = "k"
    s.redis_langcache_cache_id = "c1"
    assert isinstance(make_semantic_cache(s), LangCacheClient)


async def test_search_hit_above_threshold():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/entries/search")
        assert json.loads(request.content)["attributes"] == {"model": "gpt-4o"}
        return httpx.Response(
            200,
            json={"data": [{"similarity": 0.97, "response": json.dumps({"choices": ["cached"]})}]},
        )

    cache = _client(handler)
    hit = await cache.search("are we done?", {"model": "gpt-4o"})
    assert hit == {"choices": ["cached"]}
    await cache._client.aclose()


async def test_search_miss_below_threshold():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"similarity": 0.5, "response": "{}"}]})

    cache = _client(handler)
    assert await cache.search("q", {"model": "gpt-4o"}) is None  # 0.5 < 0.9 -> miss
    await cache._client.aclose()


async def test_search_empty_is_miss():
    cache = _client(lambda request: httpx.Response(200, json={"data": []}))
    assert await cache.search("q", {"model": "gpt-4o"}) is None
    await cache._client.aclose()


async def test_search_error_is_miss_not_a_crash():
    cache = _client(lambda request: httpx.Response(500))
    assert await cache.search("q", {"model": "gpt-4o"}) is None  # never breaks the proxy
    await cache._client.aclose()


async def test_store_posts_prompt_and_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx.Response(201, json={"ok": True})

    cache = _client(handler)
    await cache.store("q", {"choices": ["x"]}, {"model": "gpt-4o"})
    assert seen["path"].endswith("/entries")
    assert seen["body"]["prompt"] == "q"
    assert json.loads(seen["body"]["response"]) == {"choices": ["x"]}
    await cache._client.aclose()
