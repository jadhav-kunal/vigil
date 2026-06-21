"""Layer 1 compressor (spec 4.5) — pure, deterministic. Tests cover the two transforms and the
three hard safety guards: never touch the last message, never touch system messages, and refuse
to compress below the message floor. Copy-on-write is asserted so the pre-compression token
estimate stays honest."""

import copy

from vigil_proxy.compressor import compress_messages

DEFAULTS = dict(min_tool_bytes=4000, floor_messages=4, dedup_min_run=3)


def _cycle(result="still pending", tool="check_status"):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": tool, "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c", "content": result},
    ]


def _loop_conversation(repeats):
    msgs = [
        {"role": "system", "content": "You are a build agent."},
        {"role": "user", "content": "Ship the release."},
    ]
    for _ in range(repeats):
        msgs += _cycle()
    msgs.append({"role": "user", "content": "Status?"})
    return msgs


# --------------------------------------------------------------------------- collapse


def test_collapses_repeated_identical_cycles():
    msgs = _loop_conversation(repeats=6)
    out, stats = compress_messages(msgs, **DEFAULTS)
    assert stats.collapsed_runs == 1
    # first cycle kept (2 msgs) + 1 marker replaces the other 5 cycles (10 msgs)
    assert stats.dropped_messages == 10
    assert stats.changed
    markers = [m for m in out if str(m.get("content", "")).startswith("[vigil-compressed]")]
    assert len(markers) == 1
    assert "5 repeated" in markers[0]["content"]
    assert "check_status" in markers[0]["content"]


def test_below_dedup_min_run_is_not_collapsed():
    msgs = _loop_conversation(repeats=2)  # only 2 identical cycles, min_run=3
    out, stats = compress_messages(msgs, **DEFAULTS)
    assert stats.collapsed_runs == 0
    assert not stats.changed
    assert out is msgs  # identity preserved when nothing changes


def test_different_results_break_the_run():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
    ]
    msgs += _cycle("pending")
    msgs += _cycle("pending")
    msgs += _cycle("DONE")  # different result -> run of identical is only length 2
    msgs.append({"role": "user", "content": "?"})
    _out, stats = compress_messages(msgs, **DEFAULTS)
    assert stats.collapsed_runs == 0


def test_collapsed_request_stays_structurally_valid():
    msgs = _loop_conversation(repeats=6)
    out, _ = compress_messages(msgs, **DEFAULTS)
    # every remaining tool message is immediately preceded by an assistant tool_call
    for i, m in enumerate(out):
        if m.get("role") == "tool":
            prev = out[i - 1]
            assert prev.get("role") == "assistant" and prev.get("tool_calls")


# --------------------------------------------------------------------------- truncate


def test_truncates_oversized_tool_output():
    big = "x" * 9000
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "let me look"},
        {"role": "tool", "tool_call_id": "c", "content": big},
        {"role": "user", "content": "next"},
    ]
    out, stats = compress_messages(msgs, min_tool_bytes=1000, floor_messages=4, dedup_min_run=3)
    assert stats.truncated_outputs == 1
    assert len(out[3]["content"]) < len(big)
    assert "[vigil-compressed] truncated" in out[3]["content"]


def test_truncates_anthropic_tool_result_block():
    big = "y" * 9000
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": big}],
        },
        {"role": "user", "content": "next"},
    ]
    out, stats = compress_messages(msgs, min_tool_bytes=1000, floor_messages=3, dedup_min_run=3)
    assert stats.truncated_outputs == 1
    assert "[vigil-compressed] truncated" in out[2]["content"][0]["content"]


# --------------------------------------------------------------------------- safety guards


def test_floor_refuses_short_conversations():
    msgs = _loop_conversation(repeats=6)[:4]
    out, stats = compress_messages(msgs, min_tool_bytes=4000, floor_messages=4, dedup_min_run=3)
    assert not stats.changed
    assert out is msgs


def test_last_message_is_never_touched():
    # A loop whose final tool output is huge AND is the last message: must be left intact.
    big = "z" * 9000
    msgs = _loop_conversation(repeats=6)
    msgs[-1] = {"role": "tool", "tool_call_id": "c", "content": big}
    out, _ = compress_messages(msgs, min_tool_bytes=1000, floor_messages=4, dedup_min_run=3)
    assert out[-1]["content"] == big  # untouched despite exceeding the truncation threshold


def test_system_message_is_never_collapsed_or_truncated():
    big = "s" * 9000
    msgs = [
        {"role": "system", "content": big},  # oversized system task def
        {"role": "user", "content": "go"},
    ]
    msgs += _cycle() * 1
    msgs += _cycle()
    msgs += _cycle()
    msgs += _cycle()
    msgs.append({"role": "user", "content": "?"})
    out, _ = compress_messages(msgs, min_tool_bytes=1000, floor_messages=4, dedup_min_run=3)
    assert out[0]["content"] == big  # system content preserved verbatim


def test_input_is_never_mutated():
    msgs = _loop_conversation(repeats=6)
    msgs[5]["content"] = "x" * 9000  # an oversized tool output mid-conversation
    snapshot = copy.deepcopy(msgs)
    compress_messages(msgs, min_tool_bytes=1000, floor_messages=4, dedup_min_run=3)
    assert msgs == snapshot  # copy-on-write: caller's dicts untouched
