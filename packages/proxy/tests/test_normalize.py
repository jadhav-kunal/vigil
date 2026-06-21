"""Normalization of OpenAI/Anthropic shapes into the internal Step model."""

from vigil_proxy.normalize import (
    build_step,
    estimate_prompt_tokens,
    estimate_tokens,
    normalize_anthropic_request,
    normalize_openai_request,
)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_estimate_prompt_tokens_from_messages():
    msgs = [
        {"role": "system", "content": "a" * 40},
        {"role": "user", "content": [{"type": "text", "text": "b" * 40}]},
    ]
    # Two 40-char chunks joined by a space -> ~20 tokens; must be > 0, never None.
    assert estimate_prompt_tokens(msgs) > 0


def test_prompt_tokens_estimated_when_usage_absent():
    # Streaming-style response with no usage block: prompt_tokens must be backfilled, not None.
    req = normalize_openai_request(
        {"model": "m", "messages": [{"role": "user", "content": "x" * 80}]}
    )
    resp = {"choices": [{"message": {"content": "ok"}}]}  # no usage
    step = build_step(req=req, response=resp, session_id="s", step_index=0)
    assert step.prompt_tokens is not None and step.prompt_tokens > 0
    assert step.tokens_before_compression == step.prompt_tokens


def test_openai_text_response():
    req = normalize_openai_request(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    )
    resp = {
        "choices": [{"message": {"content": "hello there"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    step = build_step(req=req, response=resp, session_id="s", step_index=0)
    assert step.assistant_text == "hello there"
    assert step.tool_name is None
    assert step.prompt_tokens == 10
    assert step.completion_tokens == 2
    assert step.caused_state_mutation is False


def test_openai_tool_call_response_marks_mutation():
    req = normalize_openai_request({"model": "gpt-4o", "messages": []})
    resp = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "write_file", "arguments": '{"path":"a"}'}}
                    ],
                }
            }
        ]
    }
    step = build_step(req=req, response=resp, session_id="s", step_index=1)
    assert step.tool_name == "write_file"
    assert step.tool_args == {"path": "a"}
    assert step.caused_state_mutation is True


def test_anthropic_response():
    req = normalize_anthropic_request({"model": "claude-3-5-haiku", "messages": []})
    resp = {
        "content": [
            {"type": "text", "text": "let me look"},
            {"type": "tool_use", "name": "get_weather", "input": {"city": "SF"}},
        ],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    step = build_step(req=req, response=resp, session_id="s", step_index=0)
    assert "let me look" in step.assistant_text
    assert step.tool_name == "get_weather"
    assert step.tool_args == {"city": "SF"}
    assert step.prompt_tokens == 7
    assert step.caused_state_mutation is False


def test_embedding_text_shape():
    req = normalize_openai_request({"model": "m", "messages": []})
    resp = {
        "choices": [
            {
                "message": {
                    "content": "x",
                    "tool_calls": [{"function": {"name": "search", "arguments": '{"q":"y"}'}}],
                }
            }
        ]
    }
    step = build_step(req=req, response=resp, session_id="s", step_index=0)
    text = step.embedding_text()
    assert text.startswith("search")
    assert "y" in text and "x" in text
