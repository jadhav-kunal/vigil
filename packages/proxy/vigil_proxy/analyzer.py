"""The watchdog analyzer (spec 4.2): stateful glue around the pure math.

Holds a per-session sliding window of step embeddings + tool names, computes the per-step
quantities, and tracks the consecutive-breach streak that defines a trip. Embedding runs in a
worker thread (asyncio.to_thread) so the event loop — and therefore other requests' hot paths —
is never blocked (Invariant I1); the analyzer itself is already called from the background
capture task.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .embedder import Embedder
from .models import Step
from .settings import Settings
from .watchdog import final_score, is_breach, mean_similarity, shannon_entropy, state_penalty


@dataclass
class WatchdogResult:
    sim_score: float
    tool_entropy: float
    state_penalty: float
    final_score: float
    breach: bool
    streak: int
    tripped: bool


@dataclass
class _WindowEntry:
    embedding: np.ndarray
    tool_name: str | None


class Analyzer:
    def __init__(
        self,
        embedder: Embedder,
        *,
        window: int,
        trip_streak: int,
        theta_sim: float,
        theta_ent: float,
    ) -> None:
        self._embedder = embedder
        self._window = max(1, window)
        self._trip_streak = max(1, trip_streak)
        self._theta_sim = theta_sim
        self._theta_ent = theta_ent
        self._windows: dict[str, deque[_WindowEntry]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        self._streaks: dict[str, int] = defaultdict(int)
        # Per-session lock: captures are scheduled as independent background tasks, so two
        # same-session analyses can overlap across the embed await. Serializing per session
        # keeps the window order and streak count correct; different sessions stay concurrent.
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def analyze(self, step: Step) -> WatchdogResult:
        async with self._locks[step.session_id]:
            emb = await asyncio.to_thread(self._embedder.encode, step.embedding_text())
            win = self._windows[step.session_id]

            # Compare against the W-1 most recent prior steps; the window of size W is those plus
            # the current step (spec: "last W steps").
            prior = list(win)[-(self._window - 1) :] if self._window > 1 else []
            sc = mean_similarity(emb, [e.embedding for e in prior])
            tool_window = [e.tool_name for e in prior] + [step.tool_name]
            h = shannon_entropy(tool_window)
            p = state_penalty(step.caused_state_mutation)
            s = final_score(sc, p)
            breach = is_breach(s, h, self._theta_sim, self._theta_ent)

            streak = self._streaks[step.session_id] + 1 if breach else 0
            self._streaks[step.session_id] = streak
            # `tripped` is a level (in a tripped state), not a rising edge. The breaker FSM
            # (next slice) owns edge handling: it transitions once and won't re-fire while OPEN.
            tripped = streak >= self._trip_streak

            win.append(_WindowEntry(embedding=emb, tool_name=step.tool_name))
        return WatchdogResult(
            sim_score=round(sc, 4),
            tool_entropy=round(h, 4),
            state_penalty=p,
            final_score=round(s, 4),
            breach=breach,
            streak=streak,
            tripped=tripped,
        )

    def reset(self, session_id: str) -> None:
        """Clear a session's window and streak (used by a manual breaker override)."""
        self._windows.pop(session_id, None)
        self._streaks.pop(session_id, None)


def make_analyzer(settings: Settings, embedder: Embedder) -> Analyzer:
    return Analyzer(
        embedder,
        window=settings.window,
        trip_streak=settings.trip_streak,
        theta_sim=settings.theta_sim,
        theta_ent=settings.theta_ent,
    )
