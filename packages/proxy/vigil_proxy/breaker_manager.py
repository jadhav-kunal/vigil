"""Stateful circuit breaker: drives the FSM, persists it, runs the goal-judge, builds the OPEN
post-mortem, fires side-effects, and broadcasts transitions to dashboards.

Lives entirely on the analysis path (called from the background capture task). The request hot
path only *reads* the in-memory snapshot via `get()` to decide gating, so intervention takes
effect on the next request without ever blocking the current one (Invariant I1).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import asdict

import httpx

from .breaker import OPEN, BreakerSnapshot, transition
from .hub import Broadcaster
from .judge import LOW_SCORE, Judge
from .logging_config import get_logger, log_event
from .models import Step
from .pricing import PriceTable, estimate_cost
from .settings import Settings

logger = get_logger("breaker")


class BreakerManager:
    def __init__(
        self,
        *,
        store,
        broadcaster: Broadcaster,
        price_table: PriceTable,
        settings: Settings,
        judge: Judge | None = None,
        analyzer=None,
        sentry=None,
    ) -> None:
        self._store = store
        self._broadcaster = broadcaster
        self._price_table = price_table
        self._settings = settings
        self._judge = judge
        self._analyzer = analyzer
        self._sentry = sentry
        self._cache: dict[str, BreakerSnapshot] = {}
        self._goals: dict[str, str] = {}
        self._override_pending: set[str] = set()
        # Per-session lock so concurrent record()/override() can't lose a transition or
        # double-fire _on_open. Only the analysis path and the admin override take it; the
        # hot-path get() stays lock-free so gating never blocks the response (Invariant I1).
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._loaded: set[str] = set()

    def get(self, session_id: str) -> BreakerSnapshot:
        """Hot-path read (in-memory, no I/O)."""
        return self._cache.get(session_id) or BreakerSnapshot(session_id=session_id)

    async def ensure_loaded(self, session_id: str) -> None:
        """Hydrate this session's snapshot from the store once, so an OPEN breaker survives a
        proxy restart instead of silently resuming as CLOSED. Runs at most once per session and
        only on a cache miss (before any record() for that session), so it never contends with
        the record() lock on the hot path."""
        if session_id in self._loaded or session_id in self._cache:
            self._loaded.add(session_id)
            return
        data = await self._store.get_breaker(session_id)
        if data:
            try:
                self._cache[session_id] = BreakerSnapshot(**data)
            except TypeError:  # schema drift -> ignore stale row
                pass
        self._loaded.add(session_id)

    def remember_goal(self, session_id: str, goal: str) -> None:
        if goal:
            self._goals.setdefault(session_id, goal)

    async def record(self, step: Step, result) -> tuple[BreakerSnapshot, str | None]:
        sid = step.session_id
        async with self._locks[sid]:
            snap = self._cache.get(sid) or BreakerSnapshot(session_id=sid)

            force_open = await self._maybe_judge(step, snap)
            new, label = transition(
                snap,
                step_index=step.step_index,
                tripped=result.tripped,
                breach=result.breach,
                recovery_steps=self._settings.recovery_steps,
                force_open=force_open,
            )

            if new.state == OPEN and snap.state != OPEN:
                await self._on_open(new, step)

            if sid in self._override_pending:
                step.breaker_override = True
                self._override_pending.discard(sid)

            step.breaker_state = new.state
            self._cache[sid] = new
            await self._store.set_breaker(sid, asdict(new))

        if label:
            log_event(
                logger,
                40 if new.state == OPEN else 30,
                "breaker.transition",
                session=sid,
                step=step.step_index,
                transition=label,
                state=new.state,
            )
            await self._broadcaster.broadcast(self._event(new, label))
        return new, label

    async def override(self, session_id: str) -> BreakerSnapshot:
        async with self._locks[session_id]:
            fresh = BreakerSnapshot(session_id=session_id)
            self._cache[session_id] = fresh
            self._override_pending.add(session_id)
            if self._analyzer is not None:
                self._analyzer.reset(session_id)
            await self._store.set_breaker(session_id, asdict(fresh))
        log_event(logger, 30, "breaker.override", session=session_id)
        await self._broadcaster.broadcast(self._event(fresh, "OVERRIDE->CLOSED"))
        return fresh

    # ------------------------------------------------------------------ internals

    async def _maybe_judge(self, step: Step, snap: BreakerSnapshot) -> bool:
        """Every G steps, score progress; two consecutive low scores force OPEN."""
        if self._judge is None or snap.state == OPEN:
            return False
        g = self._settings.goal_judge_cadence
        if step.step_index == 0 or step.step_index % g != 0:
            return False
        # get_steps already includes the just-persisted current step (append_step runs first),
        # so do not re-append it.
        recent = (await self._store.get_steps(step.session_id))[-self._settings.window :]
        score = await self._judge.score(self._goals.get(step.session_id, ""), recent)
        if score is None:
            return False
        if score < LOW_SCORE:
            snap.consecutive_low_judge += 1
        else:
            snap.consecutive_low_judge = 0
        log_event(
            logger,
            20,
            "judge.score",
            session=step.session_id,
            score=score,
            low_streak=snap.consecutive_low_judge,
        )
        return snap.consecutive_low_judge >= 2

    async def _on_open(self, snap: BreakerSnapshot, step: Step) -> None:
        steps = await self._store.get_steps(step.session_id)
        series = [
            {
                "step": s.step_index,
                "sim": s.sim_score,
                "ent": s.tool_entropy,
                "final": s.final_score,
            }
            for s in steps
        ]
        trip_at = snap.trip_step_index if snap.trip_step_index is not None else 0
        loop_steps = [s for s in steps if s.step_index >= trip_at] or steps
        # Base the capped-cost estimate on the model the agent ORIGINALLY requested, not the
        # cheap model our own HALF_OPEN mitigation downgraded to — otherwise we understate what
        # the unchecked loop would have cost.
        costs = [
            estimate_cost(
                s.model_requested, s.prompt_tokens, s.completion_tokens, self._price_table
            )
            for s in loop_steps
        ]
        mean_cost = sum(costs) / len(costs) if costs else 0.0
        saved = round(mean_cost * self._settings.breaker_projection_steps, 6)
        snap.saved_estimate_usd = saved
        snap.post_mortem = {
            "state": OPEN,
            "session_id": step.session_id,
            "tripped_at_step": snap.trip_step_index,
            "opened_at_step": step.step_index,
            "reason": "semantic loop: high self-similarity and low tool diversity sustained "
            "through the recovery window",
            "metric_series": series,
            "saved_estimate_usd": saved,
            "projection_steps": self._settings.breaker_projection_steps,
        }
        log_event(
            logger,
            40,
            "breaker.open",
            session=step.session_id,
            tripped_at=snap.trip_step_index,
            opened_at=step.step_index,
            saved_estimate_usd=saved,
        )
        await self._fire_webhook(snap, step)
        if self._sentry is not None:
            self._sentry.capture_breaker_open(step.session_id, snap.post_mortem)

    async def _fire_webhook(self, snap: BreakerSnapshot, step: Step) -> None:
        url = self._settings.orkes_webhook_url
        if not url:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    url,
                    json={
                        "event": "VIGIL_BREAKER_OPEN",
                        "session_id": step.session_id,
                        "post_mortem": snap.post_mortem,
                    },
                )
            log_event(logger, 20, "webhook.fired", session=step.session_id)
        except Exception as exc:  # best-effort
            log_event(logger, 30, "webhook.error", error=str(exc))

    def _event(self, snap: BreakerSnapshot, label: str) -> dict:
        return {
            "type": "breaker",
            "session_id": snap.session_id,
            "state": snap.state,
            "transition": label,
            "trip_step_index": snap.trip_step_index,
            "saved_estimate_usd": snap.saved_estimate_usd,
            "post_mortem": snap.post_mortem,
        }


def make_breaker(settings, store, broadcaster, price_table, judge, analyzer) -> BreakerManager:
    from .integrations.sentry_sink import make_sentry

    return BreakerManager(
        store=store,
        broadcaster=broadcaster,
        price_table=price_table,
        settings=settings,
        judge=judge,
        analyzer=analyzer,
        sentry=make_sentry(settings),
    )
