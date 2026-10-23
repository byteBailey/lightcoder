"""Context compression — disabled is a no-op; snip truncates long old tool output."""

from lightcoder.context import ContextManager, estimate_tokens


def _big_tool_msg(n_lines=50):
    return {"role": "tool", "tool_call_id": "x", "content": "\n".join(f"line {i}" for i in range(n_lines))}


def test_disabled_is_noop():
    cm = ContextManager(max_tokens=100, enabled=False)
    msgs = [{"role": "user", "content": "x" * 10_000}]
    before = list(msgs)
    assert cm.maybe_compress(msgs, llm=None) is False
    assert msgs == before


def test_snip_truncates_long_tool_output():
    # tiny budget so the snip layer triggers; long tool message gets head+tail only
    cm = ContextManager(max_tokens=200, enabled=True)
    long_content = "\n".join(f"data row {i} " * 5 for i in range(100))
    msgs = [
        {"role": "user", "content": "do it"},
        {"role": "tool", "tool_call_id": "t", "content": long_content},
    ]
    changed = cm.maybe_compress(msgs, llm=None)
    assert changed is True
    assert "snipped to save context" in msgs[1]["content"]
    assert len(msgs[1]["content"]) < len(long_content)


def test_estimate_tokens_counts_content_and_tool_calls():
    msgs = [
        {"role": "user", "content": "abc" * 10},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
    ]
    assert estimate_tokens(msgs) > 0
