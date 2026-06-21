"""Vigil proxy — FastAPI app.

REQUEST PATH (must stay non-blocking on analysis): intercept -> forward to the real upstream
with the caller's key passed through unchanged -> stream/return the response to the agent
UNMODIFIED. ANALYSIS PATH: reconstruct the Step, persist it, and broadcast it to dashboards in
a background task that never blocks the response (Invariant I1). Later slices hook the
watchdog/breaker/governor into the same two paths.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.websockets import WebSocketDisconnect

from .analyzer import Analyzer, make_analyzer
from .breaker import CLOSED, is_mitigating, is_open
from .breaker_manager import BreakerManager, make_breaker
from .compressor import compress_messages
from .embedder import make_embedder
from .forensics import Forensics
from .governor import Governor, make_governor
from .hub import Broadcaster, step_event
from .integrations.compression_l2 import TokenCompressor, make_l2_compressor
from .integrations.semantic_cache import LangCacheClient, make_semantic_cache
from .integrations.tracing import Tracer, make_tracer
from .judge import make_judge
from .logging_config import get_logger, log_event, set_level
from .normalize import (
    build_step,
    estimate_messages_tokens,
    normalize_anthropic_request,
    normalize_openai_request,
)
from .pricing import PriceTable, estimate_cost, load_price_table
from .settings import Settings, get_settings
from .state_mutation import caused_state_mutation
from .store import Store, make_store
from .streaming import AnthropicStreamAccumulator, OpenAIStreamAccumulator

logger = get_logger("proxy")

# Hop-by-hop / length headers we must not relay verbatim (httpx recomputes them; content has
# already been decoded so a stale content-encoding/length would corrupt the response).
_DROP_REQUEST_HEADERS = {"host", "content-length", "connection", "accept-encoding"}
_DROP_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
}

# Most recent steps replayed to a dashboard when it first connects.
_SNAPSHOT_LIMIT = 200

# Keep strong refs to in-flight background tasks so they are not garbage-collected.
_bg_tasks: set[asyncio.Task] = set()


@dataclass
class CaptureCtx:
    """Everything the analysis path needs, bundled so signatures stay small."""

    store: Store
    broadcaster: Broadcaster
    price_table: PriceTable
    analyzer: Analyzer
    breaker: BreakerManager
    model_used: str | None = None
    # Same-estimator measurement of the message array before/after Layer 1 compression.
    tokens_before: int | None = None
    tokens_after: int | None = None
    forensics: Forensics | None = None
    # The original (pre-compression/route) request body, cached for forensic replay/fork.
    original_request: dict | None = None
    l2: TokenCompressor | None = None
    tracer: Tracer | None = None
    cache: LangCacheClient | None = None
    # The canonical prompt key for the semantic cache, and whether this turn was a cache hit.
    cache_prompt: str | None = None
    served_from_cache: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    set_level(settings.log_level)
    app.state.settings = settings
    app.state.store = await make_store(settings)
    app.state.broadcaster = Broadcaster()
    app.state.price_table = load_price_table(settings)
    app.state.analyzer = make_analyzer(settings, make_embedder(settings))
    app.state.governor = make_governor(settings, app.state.price_table)
    app.state.forensics = Forensics(app.state.store, app.state.price_table)
    app.state.l2 = make_l2_compressor(settings)
    app.state.tracer = make_tracer(settings)
    app.state.cache = make_semantic_cache(settings)
    app.state.breaker = make_breaker(
        settings,
        app.state.store,
        app.state.broadcaster,
        app.state.price_table,
        make_judge(settings),
        app.state.analyzer,
    )
    app.state.http = httpx.AsyncClient(timeout=settings.upstream_timeout_s)
    log_event(logger, 20, "proxy.start", port=settings.port, redis=settings.use_redis)
    try:
        yield
    finally:
        # Let any trailing capture tasks finish before tearing down the store.
        if _bg_tasks:
            await asyncio.gather(*list(_bg_tasks), return_exceptions=True)
        await app.state.http.aclose()
        await app.state.store.close()


app = FastAPI(title="Vigil", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def openai_chat(request: Request) -> Response:
    return await _proxy(request, provider="openai")


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Response:
    return await _proxy(request, provider="anthropic")


@app.get("/metrics/session/{session_id}")
async def session_metrics(session_id: str, request: Request) -> dict:
    store: Store = request.app.state.store
    table: PriceTable = request.app.state.price_table
    steps = await store.get_steps(session_id)
    cost = sum(
        estimate_cost(s.model_used, s.prompt_tokens, s.completion_tokens, table) for s in steps
    )
    before = sum(s.tokens_before_compression or 0 for s in steps)
    after = sum(s.tokens_after_compression or 0 for s in steps)
    return {
        "session_id": session_id,
        "steps": len(steps),
        "models_used": sorted({s.model_used for s in steps if s.model_used}),
        "tokens_before_compression": before,
        "tokens_after_compression": after,
        "tokens_saved": max(0, before - after),
        "completion_tokens": sum(s.completion_tokens or 0 for s in steps),
        "cost_usd": round(cost, 6),
    }


@app.get("/metrics/aggregate")
async def metrics_aggregate(request: Request) -> dict:
    """Cross-session totals (spec 4.8) — COUNTS ONLY, never prompt or tool content."""
    store: Store = request.app.state.store
    table: PriceTable = request.app.state.price_table
    agg = await store.aggregate()
    cost = sum(
        estimate_cost(model, m["prompt_tokens"], m["completion_tokens"], table)
        for model, m in agg["by_model"].items()
    )
    before = agg["tokens_before_compression"]
    after = agg["tokens_after_compression"]
    return {
        "sessions": agg["sessions"],
        "steps": agg["steps"],
        "prompt_tokens": agg["prompt_tokens"],
        "completion_tokens": agg["completion_tokens"],
        "tokens_before_compression": before,
        "tokens_after_compression": after,
        "tokens_saved": max(0, before - after),
        "cache_hits": agg["cache_hits"],
        "breaker_open_sessions": agg["breaker_open_sessions"],
        "models_used": sorted(agg["by_model"].keys()),
        "model_step_counts": {m: v["steps"] for m, v in agg["by_model"].items()},
        "cost_usd": round(cost, 6),
    }


@app.post("/sessions/{session_id}/override")
async def breaker_override(session_id: str, request: Request) -> dict:
    """Manual override (spec 4.3): reset the breaker FSM to CLOSED for this session."""
    breaker: BreakerManager = request.app.state.breaker
    snap = await breaker.override(session_id)
    return {"session_id": session_id, "state": snap.state, "override": True}


@app.get("/sessions/{session_id}/breaker")
async def breaker_status(session_id: str, request: Request) -> dict:
    breaker: BreakerManager = request.app.state.breaker
    snap = breaker.get(session_id)
    return {
        "session_id": session_id,
        "state": snap.state,
        "trip_step_index": snap.trip_step_index,
        "saved_estimate_usd": snap.saved_estimate_usd,
        "post_mortem": snap.post_mortem,
    }


@app.post("/sessions/{session_id}/replay")
async def session_replay(session_id: str, request: Request) -> Response:
    """Cached-trace replay (spec 4.7): reconstruct the session entirely from the forensic cache.
    Zero new API calls, zero side effects — deterministic, so it returns a stable trace hash."""
    forensics: Forensics = request.app.state.forensics
    result = await forensics.replay(session_id)
    if result is None:
        return JSONResponse(
            {"error": "no cached exchanges for this session", "session_id": session_id},
            status_code=404,
        )
    log_event(logger, 20, "forensics.replay", session=session_id, steps=len(result.steps))
    return JSONResponse(result.as_dict())


@app.post("/sessions/{session_id}/fork")
async def session_fork(session_id: str, request: Request) -> Response:
    """Fork at step N with a different model (spec 4.7). Replays the recorded context up to N and
    re-issues that one request with the swapped model; tool outputs are held constant (cached), so
    the diff isolates a model-reasoning change. This is the one forensic op that calls upstream
    (once) — the caller supplies their own Authorization, exactly like a normal request."""
    forensics: Forensics = request.app.state.forensics
    settings: Settings = request.app.state.settings
    http: httpx.AsyncClient = request.app.state.http
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        body = {}
    step_index = body.get("step_index")
    new_model = body.get("model")
    if not isinstance(step_index, int) or not isinstance(new_model, str) or not new_model:
        return JSONResponse(
            {"error": "fork requires integer 'step_index' and string 'model'"}, status_code=400
        )
    headers = _forward_headers(request)

    async def generate(forked_request: dict) -> dict:
        model = str(forked_request.get("model", "")).lower()
        if "claude" in model:
            base = settings.anthropic_base_url.rstrip("/")
            fork_url = f"{base}/v1/messages"
        else:
            base = settings.openai_base_url.rstrip("/")
            fork_url = f"{base}/chat/completions"
        resp = await http.post(fork_url, headers=headers, json=forked_request)
        resp.raise_for_status()
        return resp.json()

    try:
        result = await forensics.fork(session_id, step_index, new_model, generate)
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"fork upstream call failed: {exc}"}, status_code=502)
    if result is None:
        return JSONResponse(
            {
                "error": "no cached exchange at that step",
                "session_id": session_id,
                "step": step_index,
            },
            status_code=404,
        )
    log_event(
        logger,
        20,
        "forensics.fork",
        session=session_id,
        step=step_index,
        to=new_model,
        diverged=not (result.same_tool and result.same_args),
    )
    return JSONResponse(result.as_dict())


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    broadcaster: Broadcaster = websocket.app.state.broadcaster
    store: Store = websocket.app.state.store
    table: PriceTable = websocket.app.state.price_table
    settings: Settings = websocket.app.state.settings
    await broadcaster.accept(websocket)
    try:
        await websocket.send_json(
            {
                "type": "hello",
                "price_table": _table_json(table),
                "thresholds": {
                    "theta_sim": settings.theta_sim,
                    "theta_ent": settings.theta_ent,
                    "window": settings.window,
                    "trip_streak": settings.trip_streak,
                },
            }
        )
        await _send_snapshot(websocket, store, table)
        # Join the fan-out only after the snapshot is fully sent, so a live broadcast can never
        # send on this socket concurrently with the snapshot loop.
        broadcaster.register(websocket)
        # The dashboard is receive-only; keep the socket alive until it disconnects.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
    except Exception:  # any client error -> drop the connection, never crash the server
        broadcaster.disconnect(websocket)


# --------------------------------------------------------------------------- internals


def _table_json(table: PriceTable) -> dict[str, list[float]]:
    return {model: [rates[0], rates[1]] for model, rates in table.items()}


async def _send_snapshot(websocket: WebSocket, store: Store, table: PriceTable) -> None:
    """Replay recent steps so a freshly opened dashboard is not blank.

    Bounded query (recent_steps) so a long history never loads fully into memory. The client
    dedupes by (session_id, step_index), so a step that also arrives live is harmless.
    """
    recent = await store.recent_steps(_SNAPSHOT_LIMIT)
    await websocket.send_json({"type": "snapshot", "count": len(recent)})
    for step in recent:
        await websocket.send_json(step_event(step, table))


def _session_id(request: Request) -> str:
    return request.headers.get("x-vigil-session-id") or f"sess-{uuid.uuid4().hex[:12]}"


def _state_mutation_override(request: Request) -> bool | None:
    raw = request.headers.get("x-vigil-state-mutation")
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes")


def _forward_headers(request: Request) -> dict[str, str]:
    # Drop hop-by-hop/length headers and Vigil's own control headers (x-vigil-*) so they never
    # leak to the provider.
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _DROP_REQUEST_HEADERS and not k.lower().startswith("x-vigil-")
    }


def _upstream_allowed(base: str, settings: Settings) -> bool:
    if not (base.startswith("http://") or base.startswith("https://")):
        return False
    allow = settings.upstream_allowlist
    if not allow:
        return True  # header enabled but no allowlist => any URL (documented SSRF caveat)
    prefixes = [p.strip().rstrip("/") for p in allow.split(",") if p.strip()]
    return any(base.rstrip("/").startswith(p) for p in prefixes)


def _resolve_upstream(request: Request, provider: str, settings: Settings) -> str:
    """The full upstream URL for this request. Honors the per-request `x-vigil-upstream` override
    when enabled and allowed; otherwise uses the configured per-provider base."""
    default_base = (
        settings.anthropic_base_url if provider == "anthropic" else settings.openai_base_url
    )
    base = default_base
    override = request.headers.get("x-vigil-upstream")
    if override and settings.allow_upstream_header:
        if _upstream_allowed(override, settings):
            base = override
            log_event(logger, 20, "upstream.override", provider=provider, base=override)
        else:
            log_event(logger, 30, "upstream.override_rejected", base=override)
    base = base.rstrip("/")
    return f"{base}/v1/messages" if provider == "anthropic" else f"{base}/chat/completions"


def _cache_prompt(messages: list[dict]) -> str:
    """The semantic-cache key: the most recent message text (the query the step answers)."""
    for m in reversed(messages):
        if isinstance(m, dict):
            content = m.get("content")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                text = " ".join(
                    str(b.get("text", "")) for b in content if isinstance(b, dict)
                ).strip()
                if text:
                    return text
    return ""


def _extract_goal(messages: list[dict]) -> str:
    """The original task = the first user message; used by the goal-judge."""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    str(b.get("text", "")) for b in content if isinstance(b, dict)
                ).strip()
    return ""


def _apply_mitigations(parsed: dict, provider: str, settings: Settings) -> tuple[dict, str]:
    """HALF_OPEN side-effects: downgrade the model and strip write/mutating tools (read-only)."""
    downgrade = (
        settings.breaker_downgrade_anthropic
        if provider == "anthropic"
        else settings.breaker_downgrade_openai
    )
    parsed = dict(parsed)
    parsed["model"] = downgrade
    tools = parsed.get("tools")
    if isinstance(tools, list):
        parsed["tools"] = [t for t in tools if not _tool_is_mutating(t)]
    return parsed, downgrade


def _route_request(
    parsed: dict, provider: str, ctx: CaptureCtx, governor: Governor, session_id: str
) -> tuple[dict, bool]:
    """Effort governor (spec 4.6): classify the step and route to a cheaper model when warranted.
    Returns the (possibly model-rewritten) body and whether it was rewritten. Only ever downgrades
    (the governor's no-upgrade guard), so `model_used` here is always <= the requested cost."""
    requested = parsed.get("model")
    decision = governor.decide(session_id, provider, parsed)
    if not decision.routed:
        return parsed, False
    parsed = dict(parsed)
    parsed["model"] = decision.model
    ctx.model_used = decision.model
    log_event(
        logger,
        20,
        "governor.route",
        session=session_id,
        effort=decision.effort_class,
        requested=requested,
        routed_to=decision.model,
        escalated=decision.escalated,
    )
    return parsed, True


async def _compress_request(
    parsed: dict, ctx: CaptureCtx, settings: Settings, session_id: str
) -> bool:
    """Compression on the request path: Layer 1 (free, structural) then optional Layer 2 (Token
    Company, env-gated). Records before/after token estimates (the same-estimator savings proof)
    on ctx and returns True if the body was rewritten."""
    messages = parsed.get("messages")
    if not isinstance(messages, list) or (not settings.compress_enabled and ctx.l2 is None):
        return False
    ctx.tokens_before = estimate_messages_tokens(messages)
    working = messages
    changed = False

    if settings.compress_enabled:
        compressed, stats = compress_messages(
            working,
            min_tool_bytes=settings.compress_min_tool_bytes,
            floor_messages=settings.compress_floor_messages,
            dedup_min_run=settings.compress_dedup_min_run,
        )
        if stats.changed:
            working = compressed
            changed = True
            log_event(
                logger,
                20,
                "compress.layer1",
                session=session_id,
                collapsed_runs=stats.collapsed_runs,
                dropped_messages=stats.dropped_messages,
                truncated_outputs=stats.truncated_outputs,
            )

    if ctx.l2 is not None:
        l2_out, l2_changed = await ctx.l2.compress(working)
        if l2_changed:
            working = l2_out
            changed = True
            log_event(logger, 20, "compress.layer2", session=session_id)

    ctx.tokens_after = estimate_messages_tokens(working)
    if changed:
        parsed["messages"] = working
    return changed


def _tool_is_mutating(tool: dict) -> bool:
    if not isinstance(tool, dict):
        return False
    # OpenAI: {"type":"function","function":{"name":...}}. Anthropic: {"name":...}.
    name = tool.get("name")
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name", name)
    return caused_state_mutation(name if isinstance(name, str) else None)


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in upstream.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}


async def _proxy(request: Request, *, provider: str) -> Response:
    settings: Settings = request.app.state.settings
    http: httpx.AsyncClient = request.app.state.http
    breaker: BreakerManager = request.app.state.breaker
    ctx = CaptureCtx(
        store=request.app.state.store,
        broadcaster=request.app.state.broadcaster,
        price_table=request.app.state.price_table,
        analyzer=request.app.state.analyzer,
        breaker=breaker,
        l2=request.app.state.l2,
        tracer=request.app.state.tracer,
        cache=request.app.state.cache,
    )

    body = await request.body()
    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    if provider == "anthropic":
        req = normalize_anthropic_request(parsed)
    else:
        req = normalize_openai_request(parsed)
    url = _resolve_upstream(request, provider, settings)

    session_id = _session_id(request)
    mutation_override = _state_mutation_override(request)
    headers = _forward_headers(request)
    breaker.remember_goal(session_id, _extract_goal(req.messages))

    if settings.forensics_enabled:
        ctx.forensics = request.app.state.forensics
        # Snapshot the ORIGINAL request (before any mitigation/route/compression rewrite) for
        # faithful replay; the message list is never mutated downstream (compression is copy-on-
        # write), so holding its reference is safe. No headers here => no provider key is cached.
        ctx.original_request = {
            "model": parsed.get("model"),
            "messages": parsed.get("messages"),
            "tools": parsed.get("tools"),
        }

    # Breaker gate (reads in-memory state; intervention took effect on a prior step).
    await breaker.ensure_loaded(session_id)  # survive restarts: rehydrate an OPEN breaker
    snap = breaker.get(session_id)
    if is_open(snap.state):
        log_event(logger, 30, "breaker.blocked", session=session_id)
        return JSONResponse(
            {
                "error": {
                    "type": "vigil_breaker_open",
                    "message": "Vigil halted this session: a semantic loop was detected. "
                    "POST /sessions/{id}/override to resume.",
                    "session_id": session_id,
                },
                "vigil_post_mortem": snap.post_mortem,
            },
            status_code=503,
        )

    # Semantic cache (M4): a hit serves the cached response and skips upstream entirely. Only when
    # CLOSED (don't bypass the breaker's mitigation) and non-streaming. The lookup is paid overhead
    # on every request; the eval reports the break-even hit rate.
    if ctx.cache is not None and not req.stream and snap.state == CLOSED:
        ctx.cache_prompt = _cache_prompt(req.messages)
        cached = await ctx.cache.search(ctx.cache_prompt, {"model": req.model})
        if cached is not None:
            ctx.served_from_cache = True
            _schedule_capture(ctx, req, cached, session_id, mutation_override)
            log_event(logger, 20, "semcache.hit", session=session_id)
            return JSONResponse(cached)

    rewritten = False
    if is_mitigating(snap.state):
        # Breaker mitigation owns the model while HALF_OPEN; the governor stays out of its way.
        parsed, ctx.model_used = _apply_mitigations(parsed, provider, settings)
        rewritten = True
        log_event(logger, 20, "breaker.mitigate", session=session_id, downgraded_to=ctx.model_used)
    elif settings.governor_enabled:
        parsed, routed = _route_request(
            parsed, provider, ctx, request.app.state.governor, session_id
        )
        rewritten = routed or rewritten

    rewritten = await _compress_request(parsed, ctx, settings, session_id) or rewritten
    if rewritten:
        body = json.dumps(parsed).encode()

    if req.stream:
        return await _proxy_streaming(
            http, url, headers, body, req, ctx, session_id, mutation_override
        )
    return await _proxy_unary(http, url, headers, body, req, ctx, session_id, mutation_override)


async def _proxy_unary(
    http, url, headers, body, req, ctx, session_id, mutation_override
) -> Response:
    try:
        upstream = await http.post(url, headers=headers, content=body)
    except httpx.HTTPError as exc:
        log_event(logger, 40, "proxy.upstream_error", url=url, error=str(exc))
        return JSONResponse({"error": f"upstream request failed: {exc}"}, status_code=502)

    # Forward to the agent immediately; analyze in the background (Invariant I1).
    if upstream.status_code == 200:
        try:
            resp_json = upstream.json()
            _schedule_capture(ctx, req, resp_json, session_id, mutation_override)
        except (json.JSONDecodeError, ValueError):
            pass

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
        media_type=upstream.headers.get("content-type"),
    )


async def _proxy_streaming(
    http, url, headers, body, req, ctx, session_id, mutation_override
) -> Response:
    upstream_req = http.build_request("POST", url, headers=headers, content=body)
    try:
        upstream = await http.send(upstream_req, stream=True)
    except httpx.HTTPError as exc:
        log_event(logger, 40, "proxy.upstream_error", url=url, error=str(exc))
        return JSONResponse({"error": f"upstream request failed: {exc}"}, status_code=502)

    accumulator = (
        AnthropicStreamAccumulator() if req.provider == "anthropic" else OpenAIStreamAccumulator()
    )

    async def tee() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                if upstream.status_code == 200:
                    accumulator.feed(chunk)
                yield chunk  # client gets every byte, undelayed
        finally:
            await upstream.aclose()
            if upstream.status_code == 200:
                _schedule_capture(
                    ctx, req, accumulator.to_response(), session_id, mutation_override
                )

    return StreamingResponse(
        tee(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


def _schedule_capture(ctx, req, resp_json, session_id, mutation_override) -> None:
    task = asyncio.create_task(_capture(ctx, req, resp_json, session_id, mutation_override))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _capture(ctx: CaptureCtx, req, resp_json, session_id, mutation_override) -> None:
    try:
        step = build_step(
            req=req,
            response=resp_json,
            session_id=session_id,
            step_index=0,  # real index assigned atomically by append_step
            model_used=ctx.model_used,  # the (possibly downgraded) model actually forwarded
            state_mutation_override=mutation_override,
            tokens_before_compression=ctx.tokens_before,
            tokens_after_compression=ctx.tokens_after,
        )
        if ctx.served_from_cache:
            # A cache hit made no upstream call, so it costs ~0 — zero the billed tokens but keep
            # the assistant text so the watchdog still sees the (cached) turn.
            step.served_from_cache = True
            step.prompt_tokens = 0
            step.completion_tokens = 0
        # Watchdog runs in the background task and embeds in a worker thread (Invariant I1).
        result = await ctx.analyzer.analyze(step)
        step.sim_score = result.sim_score
        step.tool_entropy = result.tool_entropy
        step.state_penalty = result.state_penalty
        step.final_score = result.final_score
        step.watchdog_breach = result.breach
        step.watchdog_streak = result.streak
        step.watchdog_tripped = result.tripped

        step.step_index = await ctx.store.append_step(step)
        # Breaker reads the just-persisted trajectory; it sets step.breaker_state/override.
        _snap, label = await ctx.breaker.record(step, result)
        await ctx.store.update_breaker_fields(
            step.session_id, step.step_index, step.breaker_state, step.breaker_override
        )
        await ctx.broadcaster.broadcast(step_event(step, ctx.price_table))
        if ctx.tracer is not None:
            ctx.tracer.record_step(step)
        if ctx.forensics is not None and ctx.original_request is not None:
            # Cache the original request -> observed response for forensic replay/fork (spec 4.7).
            await ctx.forensics.record(
                session_id, step.step_index, ctx.original_request, resp_json, step.model_used
            )
        if ctx.cache is not None and ctx.cache_prompt is not None and not ctx.served_from_cache:
            # Populate the semantic cache on a miss (off the response path — never adds latency).
            await ctx.cache.store(ctx.cache_prompt, resp_json, {"model": req.model})
        log_event(
            logger,
            20,
            "step.captured",
            session=session_id,
            step=step.step_index,
            model=step.model_used,
            tool=step.tool_name or "-",
            sim=step.sim_score,
            ent=step.tool_entropy,
            final=step.final_score,
            tripped=step.watchdog_tripped,
            breaker=step.breaker_state,
            transition=label or "-",
        )
    except Exception as exc:  # analysis must never crash the proxy
        log_event(logger, 40, "step.capture_failed", session=session_id, error=str(exc))
