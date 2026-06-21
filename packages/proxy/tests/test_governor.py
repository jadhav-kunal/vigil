"""Effort governor (spec 4.6) — pure classifier + stateful router. The headline test is the
classification-ordering invariant (EXTRACTION before VERIFICATION); the rest cover routing, the
no-upgrade safety guard, and verification escalation."""

from vigil_proxy.governor import (
    DEFAULT_GOVERNOR_MODELS as MODELS,
)
from vigil_proxy.governor import (
    EXTRACTION,
    PLANNING,
    TOOL_USE,
    VERIFICATION,
    Governor,
    classify,
    is_first_turn,
)
from vigil_proxy.pricing import DEFAULT_PRICE_TABLE


def _msgs(user_last, *, with_assistant=False):
    m = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do the task"}]
    if with_assistant:
        m.append({"role": "assistant", "content": "ok"})
    m.append({"role": "user", "content": user_last})
    return m


# --------------------------------------------------------------------------- classifier


def test_extraction_is_checked_before_verification():
    # The prompt reads as BOTH extraction and verification; the ordering invariant makes it
    # EXTRACTION. Reversing the check order in classify() would flip this to VERIFICATION.
    text = "Extract the line items and verify the totals match."
    assert "verify" in text.lower() and "extract" in text.lower()
    assert classify(_msgs(text, with_assistant=True), None, is_first_turn=False) == EXTRACTION


def test_planning_first_turn_with_no_signal():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "help me out"}]
    assert is_first_turn(msgs) is True
    assert classify(msgs, None, is_first_turn=True) == PLANNING


def test_extraction_keyword():
    assert classify(_msgs("summarize the document"), None, is_first_turn=False) == EXTRACTION


def test_verification_keyword():
    assert classify(_msgs("validate the response is well formed"), None, is_first_turn=False) == (
        VERIFICATION
    )


def test_tool_use_when_tools_present_and_no_signal():
    msgs = _msgs("proceed", with_assistant=True)
    assert classify(msgs, [{"type": "function"}], is_first_turn=False) == TOOL_USE


def test_ambiguous_followup_defaults_to_tool_use_not_cheap():
    # No keyword, no tools, not first turn -> default to the mid tier (don't downgrade blindly).
    assert classify(_msgs("continue", with_assistant=True), None, is_first_turn=False) == TOOL_USE


# --------------------------------------------------------------------------- routing


def _gov():
    return Governor(MODELS, DEFAULT_PRICE_TABLE)


def test_extraction_routes_to_cheaper_model():
    parsed = {"model": "gpt-4o", "messages": _msgs("extract the totals", with_assistant=True)}
    d = _gov().decide("s1", "openai", parsed)
    assert d.effort_class == EXTRACTION
    assert d.routed is True
    assert d.model == "gpt-4o-mini"


def test_no_upgrade_guard_keeps_requested_cheap_model():
    # A PLANNING step would map to gpt-4o, but the user already asked for the cheaper mini -> the
    # governor must NOT upgrade them (routing is strictly cost-reducing).
    parsed = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "design the plan"}]}
    d = _gov().decide("s2", "openai", parsed)
    assert d.effort_class == PLANNING
    assert d.routed is False
    assert d.model == "gpt-4o-mini"


def test_tool_use_does_not_downgrade_a_frontier_request():
    parsed = {
        "model": "gpt-4o",
        "messages": _msgs("go ahead", with_assistant=True),
        "tools": [{"type": "function"}],
    }
    d = _gov().decide("s3", "openai", parsed)
    assert d.effort_class == TOOL_USE
    assert d.routed is False  # TOOL_USE maps to gpt-4o == requested
    assert d.model == "gpt-4o"


def test_repeated_verification_escalates_to_frontier():
    g = _gov()
    parsed = {"model": "gpt-4o", "messages": _msgs("verify the build", with_assistant=True)}
    first = g.decide("s4", "openai", parsed)
    assert first.effort_class == VERIFICATION and first.escalated is False
    assert first.model == "gpt-4o-mini"  # cheap on the first verification
    second = g.decide("s4", "openai", parsed)
    assert second.escalated is True
    assert second.model == "gpt-4o"  # escalated back to the frontier tier


def test_anthropic_routing_uses_anthropic_models():
    parsed = {
        "model": "claude-3-5-sonnet-latest",
        "messages": _msgs("extract the fields", with_assistant=True),
    }
    d = _gov().decide("s5", "anthropic", parsed)
    assert d.model == "claude-3-5-haiku-latest"
    assert d.routed is True
