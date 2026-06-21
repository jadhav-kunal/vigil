"""SSE accumulators reconstruct a full completion from streamed deltas."""

from vigil_proxy.normalize import build_step, normalize_openai_request
from vigil_proxy.streaming import AnthropicStreamAccumulator, OpenAIStreamAccumulator


def _sse(*objs: str) -> list[bytes]:
    return [f"data: {o}\n\n".encode() for o in objs]


def test_openai_text_stream_reconstruction():
    acc = OpenAIStreamAccumulator()
    for chunk in _sse(
        '{"choices":[{"delta":{"content":"Hel"}}]}',
        '{"choices":[{"delta":{"content":"lo"}}]}',
        "[DONE]",
    ):
        acc.feed(chunk)
    resp = acc.to_response()
    assert resp["choices"][0]["message"]["content"] == "Hello"


def test_openai_tool_call_stream_reconstruction():
    acc = OpenAIStreamAccumulator()
    for chunk in _sse(
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"write_file"}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"p\\":"}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}',
    ):
        acc.feed(chunk)
    req = normalize_openai_request({"model": "m", "messages": []})
    step = build_step(req=req, response=acc.to_response(), session_id="s", step_index=0)
    assert step.tool_name == "write_file"
    assert step.tool_args == {"p": 1}
    assert step.caused_state_mutation is True


def test_partial_chunk_boundaries_are_handled():
    # A data: line split across two network chunks must still parse.
    acc = OpenAIStreamAccumulator()
    acc.feed(b'data: {"choices":[{"delta":{"con')
    acc.feed(b'tent":"hi"}}]}\n\n')
    assert acc.to_response()["choices"][0]["message"]["content"] == "hi"


def test_anthropic_stream_reconstruction():
    acc = AnthropicStreamAccumulator()
    events = [
        '{"type":"message_start","message":{"usage":{"input_tokens":5}}}',
        '{"type":"content_block_start","content_block":{"type":"text"}}',
        '{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi "}}',
        '{"type":"content_block_delta","delta":{"type":"text_delta","text":"there"}}',
        '{"type":"message_delta","usage":{"output_tokens":4}}',
    ]
    for chunk in _sse(*events):
        acc.feed(chunk)
    resp = acc.to_response()
    assert resp["content"][0]["text"] == "hi there"
    assert resp["usage"]["input_tokens"] == 5
    assert resp["usage"]["output_tokens"] == 4
