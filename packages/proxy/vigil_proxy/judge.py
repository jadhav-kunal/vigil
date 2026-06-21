"""Goal-judge (spec 4.2): the A-B-A-B-loop catcher that cosine similarity misses.

Every G steps, ask a cheap pluggable LLM (OpenAI-compatible) to score progress toward the
original goal in [0, 1]. Two consecutive scores below 0.4 force the breaker to OPEN. Fully
env-gated: with no judge key configured this is a silent no-op and the watchdog runs on
cosine+entropy+state alone (Invariant I2).
"""

from __future__ import annotations

import re

import httpx

from .logging_config import get_logger, log_event
from .models import Step
from .settings import Settings

logger = get_logger("judge")

LOW_SCORE = 0.4

_PROMPT = (
    "You are monitoring an AI agent for progress. Given the agent's ORIGINAL GOAL and its most "
    "recent steps, rate how much real progress it is making toward completing the goal, from 0.0 "
    "(stuck, looping, or going nowhere) to 1.0 (clearly progressing). Reply with ONLY a number "
    "between 0.0 and 1.0."
)


class GoalJudge:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = client

    async def score(self, goal: str, recent: list[Step]) -> float | None:
        """Return a progress score in [0, 1], or None if the judge call failed."""
        steps_text = "\n".join(
            f"- {s.tool_name or 'reply'}: {s.assistant_text[:200]}" for s in recent
        )
        user = f"ORIGINAL GOAL:\n{goal}\n\nRECENT STEPS:\n{steps_text}"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        }
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=20.0)
        try:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _parse_score(content)
        except Exception as exc:  # judge is best-effort; never break the proxy
            log_event(logger, 30, "judge.error", error=str(exc))
            return None
        finally:
            if owns_client:
                await client.aclose()


def _parse_score(text: str) -> float | None:
    m = re.search(r"[01](?:\.\d+)?|\.\d+", text.strip())
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group())))
    except ValueError:
        return None


def make_judge(settings: Settings) -> GoalJudge | None:
    if not settings.judge_enabled:
        return None
    assert settings.judge_base_url and settings.judge_api_key and settings.judge_model
    log_event(logger, 20, "judge.enabled", model=settings.judge_model)
    return GoalJudge(
        base_url=settings.judge_base_url,
        api_key=settings.judge_api_key,
        model=settings.judge_model,
    )
