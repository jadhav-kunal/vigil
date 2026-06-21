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

from .hub import Broadcaster, step_event
from .logging_config import get_logger, log_event, set_level
from .normalize import build_step, normalize_anthropic_request, normalize_openai_request
from .pricing import PriceTable, estimate_cost, load_price_table
from .settings import Settings, get_settings
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    set_level(settings.log_level)
    app.state.settings = settings
    app.state.store = await make_store(settings)
    app.state.broadcaster = Broadcaster()
    app.state.price_table = load_price_table(settings)
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


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    broadcaster: Broadcaster = websocket.app.state.broadcaster
    store: Store = websocket.app.state.store
    table: PriceTable = websocket.app.state.price_table
    await broadcaster.accept(websocket)
    try:
        await websocket.send_json({"type": "hello", "price_table": _table_json(table)})
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
    return {k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in upstream.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}


async def _proxy(request: Request, *, provider: str) -> Response:
    settings: Settings = request.app.state.settings
    http: httpx.AsyncClient = request.app.state.http
    ctx = CaptureCtx(
        store=request.app.state.store,
        broadcaster=request.app.state.broadcaster,
        price_table=request.app.state.price_table,
    )

    body = await request.body()
    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    if provider == "anthropic":
        req = normalize_anthropic_request(parsed)
        base = settings.anthropic_base_url.rstrip("/")
        url = f"{base}/v1/messages"
    else:
        req = normalize_openai_request(parsed)
        base = settings.openai_base_url.rstrip("/")
        url = f"{base}/chat/completions"

    session_id = _session_id(request)
    mutation_override = _state_mutation_override(request)
    headers = _forward_headers(request)

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
            state_mutation_override=mutation_override,
        )
        step.step_index = await ctx.store.append_step(step)
        await ctx.broadcaster.broadcast(step_event(step, ctx.price_table))
        log_event(
            logger,
            20,
            "step.captured",
            session=session_id,
            step=step.step_index,
            model=step.model_used,
            tool=step.tool_name or "-",
            mutated=step.caused_state_mutation,
        )
    except Exception as exc:  # analysis must never crash the proxy
        log_event(logger, 40, "step.capture_failed", session=session_id, error=str(exc))
