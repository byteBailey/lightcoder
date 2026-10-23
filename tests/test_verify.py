"""Execution-guided verification (the 'verify' extra): prompt section + finish-time gate.

The gate is what makes verification *execution-coupled* rather than merely suggested: when
the model tries to finish without having run anything, it can't have verified its fix, so the
agent pushes it once to reproduce-and-run. A scripted fake LLM drives the loop deterministically.
"""

from lightcoder import Agent, AgentConfig
from lightcoder.llm import LLMResponse, ToolCall
from lightcoder.prompt import system_prompt


class FakeLLM:
    """Returns pre-scripted LLMResponses, one per chat() call; counts calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages, tools, on_token=None):
        self.calls += 1
        return self.script.pop(0)


def _finish(text="done"):
    return LLMResponse(content=text, tool_calls=[],
                       message={"role": "assistant", "content": text})


def _bash(cmd="echo hi", cid="1"):
    tc = ToolCall(id=cid, name="run_bash", arguments={"command": cmd})
    return LLMResponse(
        content="", tool_calls=[tc],
        message={"role": "assistant", "content": None,
                 "tool_calls": [{"id": cid, "type": "function",
                                 "function": {"name": "run_bash", "arguments": "{}"}}]},
    )


def _agent(llm, verify=True):
    extras = ("verify",) if verify else ()
    return Agent(llm, config=AgentConfig(compression=False, subagent=False, extras=extras))


# ---------------------------------------------------------------- prompt

def test_verify_section_toggles_with_extra():
    assert "Fix-then-verify workflow" not in system_prompt([], extras=())
    assert "Fix-then-verify workflow" in system_prompt([], extras=("verify",))
    # fix-first ordering: the edit-before-reproduce instruction must be present
    assert "Fix it NOW" in system_prompt([], extras=("verify",))


# ---------------------------------------------------------------- gate

def test_gate_pushes_when_finishing_without_running():
    # finish (no exec) -> gate pushes -> runs bash -> finishes
    llm = FakeLLM([_finish("I edited it"), _bash(), _finish("verified, done")])
    a = _agent(llm, verify=True)
    out = a.chat("fix the bug")
    assert out == "verified, done"
    assert llm.calls == 3                       # the gate forced an extra round
    assert a._verify_nudged and a._ran_bash
    assert any(str(m.get("content", "")).startswith("[Verification required]")
               for m in a.messages)


def test_no_push_when_already_executed():
    llm = FakeLLM([_bash(), _finish("done, ran the repro")])
    a = _agent(llm, verify=True)
    out = a.chat("fix")
    assert out == "done, ran the repro"
    assert llm.calls == 2                        # no extra round
    assert a._ran_bash and not a._verify_nudged


def test_gate_fires_at_most_once():
    # model stubbornly finishes twice without ever executing
    llm = FakeLLM([_finish("a"), _finish("b")])
    a = _agent(llm, verify=True)
    out = a.chat("fix")
    assert out == "b"                            # 2nd finish accepted (gate already spent)
    assert llm.calls == 2
    assert a._verify_nudged and not a._ran_bash


def test_baseline_never_gates():
    llm = FakeLLM([_finish("done")])
    a = _agent(llm, verify=False)
    out = a.chat("fix")
    assert out == "done"
    assert llm.calls == 1 and not a._verify_nudged


def test_flags_reset_at_chat_entry():
    # stale flags from a prior task must not suppress this task's gate
    llm = FakeLLM([_finish("edited"), _bash(), _finish("ok")])
    a = _agent(llm, verify=True)
    a._verify_nudged = True                       # pretend a previous task left these set
    a._ran_bash = True
    out = a.chat("new task")                       # chat() must reset both -> gate re-arms
    assert out == "ok"
    assert llm.calls == 3                          # pushed once, then ran, then finished
