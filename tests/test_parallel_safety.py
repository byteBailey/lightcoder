"""Per-tool parallel safety.

Running every tool call in a turn concurrently corrupts the workspace: edit_file is a
read-modify-write over the whole file, so two of them on one file clobber each other —
measured at 30% loss on a 2 KB source file, and usually while both report success.
Mutating tools therefore run alone; read-only ones still run together.
"""

import threading
import time

from lightcoder import Agent, AgentConfig
from lightcoder.llm import ToolCall
from lightcoder.tools.base import Tool


class _Probe(Tool):
    """Records the highest number of its own invocations running at once."""

    name = "probe"
    description = "test probe"
    parameters = {"type": "object", "properties": {"n": {"type": "string"}}}

    def __init__(self, parallel_safe=True):
        self.parallel_safe = parallel_safe
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run(self, n: str = "") -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return f"ok-{n}"


def _agent(tool):
    return Agent(None, config=AgentConfig(compression=False, subagent=False), tools=[tool])


def _calls(n):
    return [ToolCall(id=str(i), name="probe", arguments={"n": str(i)}) for i in range(n)]


def test_read_only_tools_still_run_concurrently():
    probe = _Probe(parallel_safe=True)
    _agent(probe)._run_tools(_calls(4))
    assert probe.max_active > 1, "parallel-safe tools must still overlap"


def test_mutating_tools_never_overlap():
    probe = _Probe(parallel_safe=False)
    _agent(probe)._run_tools(_calls(4))
    assert probe.max_active == 1, "mutating tools must run one at a time"


def test_results_keep_call_order():
    probe = _Probe(parallel_safe=True)
    results = _agent(probe)._run_tools(_calls(4))
    assert results == ["ok-0", "ok-1", "ok-2", "ok-3"]


def test_mixed_batch_keeps_order_and_serialises_the_mutation():
    """A mutation between two reads must not run while either read is in flight."""
    reads = _Probe(parallel_safe=True)
    writes = _Probe(parallel_safe=False)
    writes.name = "writer"
    agent = Agent(None, config=AgentConfig(compression=False, subagent=False),
                  tools=[reads, writes])
    calls = [
        ToolCall(id="0", name="probe", arguments={"n": "0"}),
        ToolCall(id="1", name="writer", arguments={"n": "1"}),
        ToolCall(id="2", name="probe", arguments={"n": "2"}),
    ]
    assert agent._run_tools(calls) == ["ok-0", "ok-1", "ok-2"]
    assert writes.max_active == 1


def test_concurrent_edits_to_one_file_both_land(tmp_path):
    """The regression this flag exists for."""
    path = tmp_path / "src.py"
    path.write_text("def alpha():\n    return 1\n" + "# pad\n" * 60 +
                    "def beta():\n    return 2\n", encoding="utf-8")
    agent = Agent(None, config=AgentConfig(compression=False, subagent=False))
    agent._run_tools([
        ToolCall(id="1", name="edit_file", arguments={
            "file_path": str(path), "old_string": "return 1", "new_string": "return 111"}),
        ToolCall(id="2", name="edit_file", arguments={
            "file_path": str(path), "old_string": "return 2", "new_string": "return 222"}),
    ])
    text = path.read_text(encoding="utf-8")
    assert "return 111" in text and "return 222" in text


def test_builtin_mutating_tools_are_marked():
    from lightcoder.tools import build_tools

    tools = {t.name: t for t in build_tools(AgentConfig(extras=("todo",)))}
    for name in ("edit_file", "write_file", "run_bash", "delegate", "update_todo"):
        assert tools[name].parallel_safe is False, name
    for name in ("read_file", "grep", "glob"):
        assert tools[name].parallel_safe is True, name
