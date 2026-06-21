"""WebSocket broadcast hub: fan out step events to every connected dashboard.

Broadcasting happens on the analysis path (after a Step is persisted), never on the request
hot path (Invariant I1). A slow or dead client is dropped, not allowed to back-pressure the
proxy.
"""

from __future__ import annotations

import asyncio
from typing import Any

from starlette.websockets import WebSocket

from .logging_config import get_logger, log_event
from .models import Step
from .pricing import PriceTable, estimate_cost, price_for

logger = get_logger("hub")

# A send that takes longer than this means a wedged/slow client; drop it rather than let it
# back-pressure the capture task or other clients.
_SEND_TIMEOUT_S = 5.0


class Broadcaster:
    def __init__(self) -> None:
        self._conns: set[WebSocket] = set()

    @property
    def count(self) -> int:
        return len(self._conns)

    async def accept(self, ws: WebSocket) -> None:
        await ws.accept()

    def register(self, ws: WebSocket) -> None:
        """Add to the fan-out set. Done only AFTER the initial hello+snapshot have been sent,
        so broadcast() can never send on a socket concurrently with the snapshot loop."""
        self._conns.add(ws)
        log_event(logger, 20, "ws.connect", clients=self.count)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._conns:
            self._conns.discard(ws)
            log_event(logger, 20, "ws.disconnect", clients=self.count)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._conns:
            return

        async def _send(ws: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(ws.send_json(message), timeout=_SEND_TIMEOUT_S)
                return None
            except Exception:  # slow or broken client -> drop it, never block the proxy
                return ws

        # Fan out concurrently so one slow client cannot serialize delivery to the rest.
        results = await asyncio.gather(*(_send(ws) for ws in list(self._conns)))
        for ws in results:
            if ws is not None:
                self.disconnect(ws)


def step_event(step: Step, table: PriceTable) -> dict[str, Any]:
    """Serialize a Step plus its computed cost into a WebSocket event payload."""
    in_rate, out_rate = price_for(step.model_used, table)
    cost = estimate_cost(step.model_used, step.prompt_tokens, step.completion_tokens, table)
    payload = step.model_dump()
    payload["cost_usd"] = cost
    payload["input_rate_per_1k"] = in_rate
    payload["output_rate_per_1k"] = out_rate
    return {"type": "step", "step": payload}
