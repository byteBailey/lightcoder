"""REPL rendering (lightcoder/ui.py) and the agent hooks it relies on.

ui.py is pure — every block is a rich renderable — so it is tested by rendering to a string.
The agent side covers what the screen reports: execution time per tool (excluding the time a
permission prompt waits on the user), session token totals, and recovery after Ctrl-C.
"""

import io
import time
from types import SimpleNamespace

from rich.console import Console

from lightcoder import Agent, AgentConfig, ui
from lightcoder.llm import LLMResponse, ToolCall, _cached_tokens


def _render(renderable, width=80) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=width).print(renderable)
    return buf.getvalue()


# ------------------------------------------------------------------ tool status

def test_nonzero_exit_is_an_error_even_though_the_tool_ran():
    assert ui.tool_status("run_bash", "No module named x\n(exit code 1)") == "error"
    assert ui.tool_status("run_bash", "ok\n(exit code 0)") == "ok"
    assert ui.tool_status("run_bash", "(no output) (exit code 0)") == "ok"
    assert ui.tool_status("edit_file", "Error: old_string not found.") == "error"
    assert ui.tool_status("edit_file", "[Permission] User declined to run edit_file.") == "denied"


def test_bash_block_previews_the_tail_and_points_at_expand():
    out = "\n".join(f"line {i}" for i in range(12)) + "\n(exit code 0)"
    text = _render(ui.tool_block("run_bash", {"command": "seq"}, out, 0.25, ref=3))
    assert "$ seq" in text
    assert "line 11" in text and "line 0" not in text        # last lines shown
    assert "7 earlier lines · /expand 3" in text
    assert "Took 0.2s" in text or "Took 0.3s" in text
    assert "exit code" not in text                            # only a failure shows it


def test_failed_command_shows_its_exit_code():
    text = _render(ui.tool_block("run_bash", {"command": "false"}, "boom\n(exit code 2)", 0.1))
    assert "boom" in text and "exit code 2" in text


def test_running_block_has_only_the_title():
    text = _render(ui.tool_block("read_file", {"file_path": "a.py", "offset": 10, "limit": 5}))
    assert "read a.py:10-14" in text


def test_read_block_counts_lines():
    assert "3 lines" in _render(ui.tool_block("read_file", {"file_path": "a.py"}, "1\ta\n2\tb\n3\tc"))


# ------------------------------------------------------------------ diffs

def test_edit_result_diff_keeps_real_line_numbers():
    result = ("Edited m.py.\n--- a/m.py\n+++ b/m.py\n@@ -17,3 +17,4 @@\n"
              " import re\n \n+# comment\n MODES = ()\n")
    text = _render(ui.tool_block("edit_file", {"file_path": "m.py"}, result))
    assert "+19 # comment" in text and " 20 MODES = ()" in text
    assert " 0 " not in text                       # no phantom row from the leading newline


def test_permission_preview_diffs_against_the_file(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")
    panel = ui.permission_panel("edit_file",
                                {"file_path": str(f), "old_string": "c\n", "new_string": "C\n"})
    text = _render(panel, width=100)
    assert "Permission required" in text
    assert "-3 c" in text and "+3 C" in text


def test_dangerous_command_is_flagged_in_the_dialog():
    text = _render(ui.permission_panel("run_bash", {"command": "rm -rf build"}, dangerous=True))
    assert "$ rm -rf build" in text and "dangerous" in text


# ------------------------------------------------------------------ blocks and chrome

def test_user_block_keeps_every_wrapped_line():
    msg = " ".join(f"word{i}" for i in range(40))
    text = _render(ui.user_block(msg), width=40)
    assert all(f"word{i}" in text for i in range(40))


def test_footer_shows_mode_usage_cache_and_context():
    text = _render(ui.footer(mode="auto", model="m1", cwd="/tmp", branch="main",
                             prompt_tokens=8900, completion_tokens=541, cached_tokens=8100,
                             ctx_used=3307, ctx_window=128_000), width=100)
    assert "auto mode" in text and "m1" in text and "(main)" in text
    assert "↑8.9k ↓541" in text and "cache 91%" in text and "ctx 2.6% of 128k" in text


def test_fmt_tokens():
    assert [ui.fmt_tokens(n) for n in (950, 3200, 48_400, 1_234_567)] == \
        ["950", "3.2k", "48k", "1.2M"]


def test_git_branch_reads_head(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/feature/x\n")
    sub = tmp_path / "src"
    sub.mkdir()
    assert ui.git_branch(str(sub)) == "feature/x"
    (tmp_path / ".git" / "HEAD").write_text("0123456789abcdef\n")
    assert ui.git_branch(str(tmp_path)) == "0123456"


def test_logo_scales_with_width():
    assert len(ui.logo(100).plain.splitlines()) == 4
    assert len(ui.logo(60).plain.splitlines()) == 2


# ------------------------------------------------------------------ agent hooks

class _ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.model = "stub"

    def chat(self, messages, tools, on_token=None):
        return self.script.pop(0)


def _call(name, args, cid="1"):
    tc = ToolCall(id=cid, name=name, arguments=args)
    return LLMResponse(
        content="", tool_calls=[tc],
        message={"role": "assistant", "content": None,
                 "tool_calls": [{"id": cid, "type": "function",
                                 "function": {"name": name, "arguments": "{}"}}]},
        usage={"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 800},
    )


def _done():
    return LLMResponse(content="ok", tool_calls=[], message={"role": "assistant", "content": "ok"},
                       usage={"prompt_tokens": 1100, "completion_tokens": 10, "cached_tokens": 1000})


def test_tool_time_excludes_the_permission_wait(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = Agent(_ScriptedLLM([_call("run_bash", {"command": "true"}), _done()]),
              config=AgentConfig(compression=False, subagent=False, permission_mode="default"))
    seen = []

    def slow_yes(name, args, prompt):
        time.sleep(0.3)                     # the user takes a while to answer
        return True

    a.chat("go", on_permission=slow_yes, on_result=lambda n, r, s: seen.append((n, s)))
    assert seen and seen[0][0] == "run_bash" and seen[0][1] < 0.25


def test_session_totals_include_cache_hits():
    a = Agent(_ScriptedLLM([_call("read_file", {"file_path": "nope"}), _done()]),
              config=AgentConfig(compression=False, subagent=False))
    a.chat("go")
    assert (a.total_prompt_tokens, a.total_completion_tokens, a.total_cached_tokens) == \
        (2100, 60, 1800)
    assert a.last_prompt_tokens == 1100


def test_cached_tokens_from_either_provider_shape():
    openai_like = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=7))
    deepseek_like = SimpleNamespace(prompt_tokens_details=None, prompt_cache_hit_tokens=9)
    assert _cached_tokens(openai_like) == 7
    assert _cached_tokens(deepseek_like) == 9
    assert _cached_tokens(SimpleNamespace()) == 0


def test_interrupt_leaves_a_valid_transcript():
    a = Agent(None, config=AgentConfig(compression=False, subagent=False))
    a.messages += [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "run_bash", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "run_bash", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "done"},
    ]
    a.settle_interrupted()
    answered = [m["tool_call_id"] for m in a.messages if m["role"] == "tool"]
    assert answered == ["a", "b"]
    a.settle_interrupted()                  # idempotent
    assert len(a.messages) == 4


def test_absolute_paths_inside_the_project_are_shown_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    title = ui.tool_title("read_file", {"file_path": str(tmp_path / "pkg" / "m.py")})
    assert title.plain == "read pkg/m.py"


def test_markdown_list_does_not_open_with_a_blank_line():
    assert _render(ui.markdown("- a\n- b")).splitlines()[0].strip() == "• a"
