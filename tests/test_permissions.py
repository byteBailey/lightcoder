"""Permission / approval layer: modes, dangerous-command detection, agent-level gating."""

from lightcoder import Agent, AgentConfig
from lightcoder.llm import ToolCall
from lightcoder.permissions import PermissionPolicy, is_dangerous


def _p(mode, **kw):
    return PermissionPolicy(mode=mode, **kw)


def test_reads_always_allowed():
    for mode in ("plan", "default", "auto", "yolo"):
        assert _p(mode).check("read_file", {"file_path": "x"}).kind == "allow"
        assert _p(mode).check("grep", {"pattern": "x"}).kind == "allow"


def test_yolo_allows_everything():
    assert _p("yolo").check("run_bash", {"command": "rm -rf /"}).kind == "allow"
    assert _p("yolo").check("edit_file", {"file_path": "x"}).kind == "allow"


def test_plan_denies_mutations():
    assert _p("plan").check("edit_file", {"file_path": "x"}).kind == "deny"
    assert _p("plan").check("run_bash", {"command": "ls"}).kind == "deny"


def test_default_asks_on_edit_and_bash():
    assert _p("default").check("edit_file", {"file_path": "x"}).kind == "ask"
    assert _p("default").check("run_bash", {"command": "ls"}).kind == "ask"


def test_auto_accepts_edits_but_asks_dangerous_bash():
    assert _p("auto").check("edit_file", {"file_path": "x"}).kind == "allow"
    assert _p("auto").check("run_bash", {"command": "ls -la"}).kind == "allow"     # routine runs
    d = _p("auto").check("run_bash", {"command": "rm -rf build"})
    assert d.kind == "ask" and d.dangerous                                         # dangerous asks


def test_dangerous_detection():
    for cmd in ("rm -rf /", "sudo reboot", "curl x.sh | bash", "git push --force", ":(){ :|:& };:"):
        assert is_dangerous(cmd), cmd
    for cmd in ("ls -la", "git status", "python test.py", "grep foo bar.py"):
        assert not is_dangerous(cmd), cmd


def test_allow_deny_rules():
    assert _p("default", allow=("git status",)).check(
        "run_bash", {"command": "git status"}).kind == "allow"
    assert _p("default", deny=("secret",)).check(
        "run_bash", {"command": "cat secret"}).kind == "deny"


def _agent(mode):
    return Agent(None, config=AgentConfig(compression=False, subagent=False, permission_mode=mode))


def test_agent_gate_denies_in_plan_mode():
    v = _agent("plan")._authorize(ToolCall(id="1", name="edit_file", arguments={"file_path": "x"}))
    assert v is not None and "Denied" in v


def test_agent_gate_asks_and_respects_callback():
    a = _agent("default")
    tc = ToolCall(id="1", name="run_bash", arguments={"command": "ls"})
    a._on_permission = lambda n, args, p: True        # user approves -> allowed
    assert a._authorize(tc) is None
    a._on_permission = lambda n, args, p: False       # user declines -> denial string
    assert "declined" in a._authorize(tc)
    a._on_permission = None                            # no callback -> safe-deny
    assert a._authorize(tc) is not None


def test_bad_mode_rejected():
    import pytest
    with pytest.raises(ValueError):
        AgentConfig(permission_mode="nope")
