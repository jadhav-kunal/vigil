"""Price-table matching and cost estimation."""

from vigil_proxy.pricing import (
    DEFAULT_FALLBACK,
    DEFAULT_PRICE_TABLE,
    estimate_cost,
    load_price_table,
    price_for,
)


def test_longest_substring_match_wins():
    # "gpt-4o-mini" must beat the shorter "gpt-4o" prefix.
    assert price_for("gpt-4o-mini", DEFAULT_PRICE_TABLE) == DEFAULT_PRICE_TABLE["gpt-4o-mini"]
    assert price_for("gpt-4o-2024-08-06", DEFAULT_PRICE_TABLE) == DEFAULT_PRICE_TABLE["gpt-4o"]


def test_unknown_model_uses_fallback():
    assert price_for("some-unknown-model", DEFAULT_PRICE_TABLE) == DEFAULT_FALLBACK


def test_estimate_cost_math():
    # gpt-4o = (0.0025 in, 0.01 out) per 1k. 1000 in + 500 out.
    cost = estimate_cost("gpt-4o", 1000, 500, DEFAULT_PRICE_TABLE)
    assert cost == round(0.0025 + 0.005, 6)


def test_estimate_cost_handles_none_tokens():
    assert estimate_cost("gpt-4o", None, None, DEFAULT_PRICE_TABLE) == 0.0


def test_load_price_table_override(tmp_path):
    import json

    override = tmp_path / "prices.json"
    override.write_text(json.dumps({"my-model": [1.0, 2.0]}))

    class S:
        price_table_path = str(override)

    table = load_price_table(S())  # type: ignore[arg-type]
    assert table["my-model"] == (1.0, 2.0)
    # Defaults are preserved alongside the override.
    assert "gpt-4o" in table
