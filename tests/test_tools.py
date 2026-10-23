"""Tool behavior — no LLM required."""

import pytest

from lightcoder.tools import build_tools, AgentTool
from lightcoder.tools.read import ReadFileTool
from lightcoder.tools.write import WriteFileTool
from lightcoder.tools.edit import EditFileTool
from lightcoder.tools.grep import GrepTool
from lightcoder.tools.glob_tool import GlobTool
from lightcoder.tools.bash import BashTool
from lightcoder.config import AgentConfig


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_write_then_read(workdir):
    out = WriteFileTool().run(file_path="a.txt", content="hello\nworld\n")
    assert "Created a.txt" in out
    read = ReadFileTool().run(file_path="a.txt")
    assert "1\thello" in read and "2\tworld" in read


def test_read_missing(workdir):
    assert "not found" in ReadFileTool().run(file_path="nope.txt")


def test_edit_unique_match(workdir):
    WriteFileTool().run(file_path="m.py", content="x = 1\ny = 2\n")
    out = EditFileTool().run(file_path="m.py", old_string="x = 1", new_string="x = 42")
    assert "Edited m.py" in out
    assert "x = 42" in open("m.py").read()


def test_edit_rejects_non_unique(workdir):
    WriteFileTool().run(file_path="d.py", content="a\na\n")
    out = EditFileTool().run(file_path="d.py", old_string="a", new_string="b")
    assert "must be unique" in out
    # file unchanged
    assert open("d.py").read() == "a\na\n"


def test_edit_rejects_missing_string(workdir):
    WriteFileTool().run(file_path="e.py", content="hello\n")
    out = EditFileTool().run(file_path="e.py", old_string="zzz", new_string="b")
    assert "not found" in out


def test_grep_finds(workdir):
    WriteFileTool().run(file_path="g.py", content="import os\nfoo = 1\n")
    out = GrepTool().run(pattern="foo", path=".")
    assert "foo = 1" in out


def test_glob_lists(workdir):
    WriteFileTool().run(file_path="pkg/x.py", content="1")
    WriteFileTool().run(file_path="pkg/y.txt", content="2")
    out = GlobTool().run(pattern="pkg/**/*.py")
    assert "x.py" in out and "y.txt" not in out


def test_bash_runs(workdir):
    out = BashTool().run(command="echo hi")
    assert "hi" in out and "exit code 0" in out


def test_bash_blocks_dangerous(workdir):
    out = BashTool().run(command="rm -rf /")
    assert "blocked" in out


def test_build_tools_respects_subagent_toggle():
    on = build_tools(AgentConfig(subagent=True))
    off = build_tools(AgentConfig(subagent=False))
    assert any(isinstance(t, AgentTool) for t in on)
    assert not any(isinstance(t, AgentTool) for t in off)
    # core tools present in both
    assert {"read_file", "write_file", "edit_file"} <= {t.name for t in off}


def test_tool_schema_shape():
    schema = ReadFileTool().schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_file"
    assert "file_path" in schema["function"]["parameters"]["properties"]
