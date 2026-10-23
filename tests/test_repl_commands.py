"""REPL slash commands: /help, /model, /resume.

These change how the session is driven, not what the model can do, so they are handled in
the REPL and never reach the model as tool calls — a plain line must always fall through.
"""

import io

import pytest
from rich.console import Console

from lightcoder import Agent, AgentConfig
from lightcoder.repl import _handle_command
from lightcoder.session import SessionLog
from lightcoder.settings import write_json


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir(parents=True)
    monkeypatch.setenv("LIGHTCODER_HOME", str(h))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(work)
    return h


class _StubLLM:
    def __init__(self, model="stub-model"):
        self.model = model


def _agent():
    a = Agent(None, config=AgentConfig(compression=False, subagent=False))
    a.llm = _StubLLM()
    return a


def _run(agent, line, state=None):
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    handled = _handle_command(agent, console, line, state if state is not None else {"ctx": 1})
    return handled, buf.getvalue()


# ------------------------------------------------------------------ dispatch

def test_plain_text_is_not_a_command(home):
    handled, out = _run(_agent(), "fix the parser bug")
    assert handled is False and out == ""


def test_help_lists_commands(home):
    handled, out = _run(_agent(), "/help")
    assert handled and "/model" in out and "/resume" in out


def test_unknown_command_points_at_help(home):
    handled, out = _run(_agent(), "/nope")
    assert handled and "unknown command" in out and "/help" in out


# ------------------------------------------------------------------ /model

def _models(home):
    write_json(home / "models.json",
               {"models": {"alpha": {"model": "m-a"}, "beta": {"model": "m-b"}}})


def test_model_bare_shows_numbered_picker(home, monkeypatch):
    _models(home)
    monkeypatch.setattr("builtins.input", lambda: "")          # cancel at the prompt
    handled, out = _run(_agent(), "/model")
    assert handled and "alpha" in out and "beta" in out and "select" in out


def test_model_picker_selects_by_number(home, monkeypatch):
    _models(home)
    monkeypatch.setattr("builtins.input", lambda: "2")         # names sort -> 2 is beta
    agent = _agent()
    _run(agent, "/model")
    assert agent.llm.model == "m-b"


def test_model_picker_accepts_a_name_too(home, monkeypatch):
    _models(home)
    monkeypatch.setattr("builtins.input", lambda: "alpha")
    agent = _agent()
    _run(agent, "/model")
    assert agent.llm.model == "m-a"


def test_model_picker_cancel_keeps_current(home, monkeypatch):
    _models(home)
    monkeypatch.setattr("builtins.input", lambda: "")
    agent = _agent()
    _run(agent, "/model")
    assert agent.llm.model == "stub-model"


def test_model_switches_backend(home):
    write_json(home / "models.json", {
        "models": {"beta": {"base_url": "http://b/v1", "model": "m-beta", "api_key": "k"}}
    })
    agent, state = _agent(), {"ctx": 111}
    handled, out = _run(agent, "/model beta", state)

    assert handled and "switched to" in out
    assert agent.llm.model == "m-beta"
    assert agent.llm.base_url == "http://b/v1" if hasattr(agent.llm, "base_url") else True
    assert state["ctx"] != 111        # re-detected for the new backend


def test_model_unconfigured_name_is_rejected(home):
    """Unregistered names have no endpoint or key, so they are refused rather than guessed."""
    write_json(home / "models.json", {"models": {"beta": {"model": "m-beta"}}})
    agent = _agent()
    handled, out = _run(agent, "/model gpt-4o")
    assert handled and "not configured" in out
    assert agent.llm.model == "stub-model"        # unchanged


def test_model_switch_is_recorded_in_transcript(home):
    write_json(home / "models.json", {"models": {"beta": {"model": "m-beta"}}})
    agent = _agent()
    agent.session = SessionLog.create(model="stub-model")
    _run(agent, "/model beta")

    kinds = [l for l in agent.session.path.read_text(encoding="utf-8").splitlines()
             if "model_change" in l]
    assert len(kinds) == 1 and "m-beta" in kinds[0]


# ------------------------------------------------------------------ /resume

def test_resume_bare_lists_sessions(home):
    log = SessionLog.create(model="m")
    log.message({"role": "user", "content": "earlier question"})
    handled, out = _run(_agent(), "/resume")
    assert handled and log.id in out and "earlier question" in out


def test_resume_bare_with_none_says_so(home):
    handled, out = _run(_agent(), "/resume")
    assert handled and "no saved sessions" in out


def test_resume_loads_history_and_rebinds(home):
    log = SessionLog.create(model="m")
    log.message({"role": "user", "content": "one"})
    log.message({"role": "assistant", "content": "two"})

    agent = _agent()
    agent.messages.append({"role": "user", "content": "stale"})
    handled, out = _run(agent, f"/resume {log.id}")

    assert handled and "resumed" in out
    assert [m["content"] for m in agent.messages] == ["one", "two"]   # replaced, not appended
    assert agent.session.path == log.path                             # continues same file
    _, on_disk = SessionLog.read(log.path)
    assert len(on_disk) == 2                                          # not duplicated


def test_resume_unknown_id_errors(home):
    agent = _agent()
    agent.messages.append({"role": "user", "content": "keep"})
    handled, out = _run(agent, "/resume zzzz")
    assert handled and "no session matching" in out
    assert len(agent.messages) == 1        # untouched


# ------------------------------------------------------------------ /mode, shift+tab

def test_mode_lists_and_switches(home):
    agent = _agent()
    handled, out = _run(agent, "/mode")
    assert handled and all(m in out for m in ("plan", "default", "auto", "yolo"))
    _run(agent, "/mode plan")
    assert agent.permissions.mode == "plan"
    handled, out = _run(agent, "/mode nope")
    assert "unknown mode" in out and agent.permissions.mode == "plan"


def test_shift_tab_cycle_skips_yolo_unless_started_in_it(home):
    from lightcoder.repl import _CYCLE, _cycle_mode

    agent = _agent()
    agent.permissions.mode = "yolo"          # AgentConfig's default; not in the cycle
    seen = []
    for _ in range(4):
        _cycle_mode(agent, {"cycle": _CYCLE})
        seen.append(agent.permissions.mode)
    assert seen == ["default", "auto", "plan", "default"]


# ------------------------------------------------------------------ /expand, /clear

def test_expand_reprints_a_full_tool_output(home):
    out_lines = "\n".join(f"row {i}" for i in range(30))
    state = {"ctx": 1, "outputs": [("run_bash", {"command": "seq 30"}, out_lines)]}
    handled, out = _run(_agent(), "/expand", state)
    assert handled and "row 0" in out and "row 29" in out
    handled, out = _run(_agent(), "/expand 5", state)
    assert "no tool call 5" in out


def test_clear_starts_fresh_and_resume_honours_it(home):
    agent = _agent()
    agent.session = SessionLog.create(model="m")
    agent._record({"role": "user", "content": "old"})
    _run(agent, "/clear")
    assert agent.messages == []
    agent._record({"role": "user", "content": "new"})
    _, messages = SessionLog.read(agent.session.path)
    assert [m["content"] for m in messages] == ["new"]
