"""Anti-drift 'todo' extra: a persistent goal + running plan, re-injected each turn.

Held on the Agent (not in the transcript), so it survives compaction verbatim and is
re-injected at the END of the prompt once the conversation grows.
"""

from lightcoder import Agent, AgentConfig
from lightcoder.tools.todo import TodoTool


def _agent(extras=()):
    # no LLM needed: we test construction + _full_messages, not the chat loop
    return Agent(None, config=AgentConfig(compression=False, subagent=False, extras=extras))


def test_todo_extra_adds_and_wires_tool():
    a = _agent(extras=("todo",))
    todos = [t for t in a.tools if isinstance(t, TodoTool)]
    assert len(todos) == 1
    assert todos[0]._update is not None                    # wired to a._set_todo
    assert todos[0].run(content="[ ] step 1") == "Plan updated."
    assert a.todo == "[ ] step 1"                           # tool call updated the agent state


def test_no_todo_extra_no_tool():
    assert not any(isinstance(t, TodoTool) for t in _agent().tools)


def test_reminder_injected_only_after_transcript_grows():
    a = _agent(extras=("todo",))
    a.goal = "fix the bug in foo.py"
    a.todo = "[x] found it\n[ ] write the fix"

    # fresh/short transcript -> goal is right there, no reminder yet
    a.messages = [{"role": "user", "content": "fix the bug in foo.py"}]
    assert not any("[Reminder" in (m.get("content") or "") for m in a._full_messages())

    # grown transcript -> reminder appears at the END, carrying goal + current plan
    a.messages = [
        {"role": "user", "content": "fix the bug in foo.py"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "grep output ..."},
        {"role": "tool", "tool_call_id": "2", "content": "file contents ..."},
    ]
    full = a._full_messages()
    last = full[-1]
    assert last["role"] == "user" and "[Reminder" in last["content"]
    assert "fix the bug in foo.py" in last["content"]       # goal re-grounded
    assert "[ ] write the fix" in last["content"]           # current plan included

    # crucially: the reminder is NOT stored in the transcript (no accumulation, compaction-safe)
    assert not any("[Reminder" in (m.get("content") or "") for m in a.messages)


def test_reminder_off_when_extra_disabled():
    a = _agent()                                            # no 'todo' extra
    a.goal = "x"
    a.messages = [{"role": "user", "content": "x"}] * 5
    assert not any("[Reminder" in (m.get("content") or "") for m in a._full_messages())


def test_todo_shows_in_label():
    assert AgentConfig(extras=("todo",)).label() == "+todo"
