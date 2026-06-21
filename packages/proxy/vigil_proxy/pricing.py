"""Cost estimation from a configurable price table (spec 4.8).

Prices are USD per 1K tokens (input, output). The table lives in config so it can track current
provider prices without code changes: set `VIGIL_PRICE_TABLE` to a JSON file path of
`{"model-substring": [in_per_1k, out_per_1k], ...}` to override or extend the defaults.

Model matching is by longest substring of the requested model id, so "gpt-4o-mini" wins over
"gpt-4o". Unknown models fall back to a conservative mid-tier rate.
"""

from __future__ import annotations

import json
import os

from .logging_config import get_logger, log_event
from .settings import Settings

logger = get_logger("pricing")

PriceTable = dict[str, tuple[float, float]]

# USD per 1K tokens (input, output). Representative public list prices.
DEFAULT_PRICE_TABLE: PriceTable = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "gpt-4.1": (0.002, 0.008),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o3-mini": (0.0011, 0.0044),
    "o3": (0.002, 0.008),
    "claude-3-5-haiku": (0.0008, 0.004),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-7-sonnet": (0.003, 0.015),
    "claude-3-opus": (0.015, 0.075),
    "claude-haiku-4": (0.001, 0.005),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-opus-4": (0.015, 0.075),
}
DEFAULT_FALLBACK: tuple[float, float] = (0.0010, 0.0030)


def load_price_table(settings: Settings) -> PriceTable:
    """Defaults merged with an optional JSON override file."""
    table = dict(DEFAULT_PRICE_TABLE)
    path = settings.price_table_path
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            for model, pair in raw.items():
                table[model] = (float(pair[0]), float(pair[1]))
            log_event(logger, 20, "pricing.loaded_override", path=path, models=len(raw))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log_event(logger, 40, "pricing.override_failed", path=path, error=str(exc))
    return table


def price_for(model: str, table: PriceTable) -> tuple[float, float]:
    """Longest-substring match; conservative fallback when the model is unknown."""
    best: str | None = None
    for key in table:
        if key in model and (best is None or len(key) > len(best)):
            best = key
    return table[best] if best is not None else DEFAULT_FALLBACK


def estimate_cost(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    table: PriceTable,
) -> float:
    """USD cost for one step. None token counts are treated as 0."""
    in_rate, out_rate = price_for(model, table)
    p = prompt_tokens or 0
    c = completion_tokens or 0
    return round((p / 1000.0) * in_rate + (c / 1000.0) * out_rate, 6)
