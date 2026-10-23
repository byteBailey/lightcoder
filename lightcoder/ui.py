"""Rendering for the inline REPL — pure functions from agent events to rich renderables.

Blocks are told apart by a full-width background rather than by borders: blue-grey for the
user, grey / green / red for a tool call that is running / succeeded / failed, with an output
preview inside each tool block and a two-line footer. Nothing here touches the terminal, so
every block can be tested by rendering it to a string.
"""

from __future__ import annotations

import difflib
import os
import re

from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.segment import Segment
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Palette (dark, truecolor).
TEXT = "#dee0e1"
MUTED = "#9da5a9"
DIM = "#7e888e"
RULE = "#484e52"
ACCENT = "#9776e5"
LINK = "#a798d7"
GREEN = "#68b78d"
RED = "#e48a7a"
YELLOW = "#cd9a22"
BLUE = "#4f8eb3"
USER_BG = "#213b49"
TOOL_BG = {"pending": "#34383a", "ok": "#254131", "error": "#5b282a", "denied": "#5b282a"}

MODE_COLOR = {"plan": BLUE, "default": MUTED, "auto": ACCENT, "yolo": RED}
MODE_HINT = {
    "plan": "read-only",
    "default": "asks before edits and commands",
    "auto": "edits and routine commands run",
    "yolo": "nothing asks",
}

# Markdown in assistant replies, in the palette above (rich's default puts inline code on black).
THEME = Theme({
    "markdown.code": f"bold {ACCENT}",
    "markdown.code_block": GREEN,
    "markdown.h1": f"bold {YELLOW}",
    "markdown.h2": f"bold {YELLOW}",
    "markdown.h3": f"bold {YELLOW}",
    "markdown.h4": YELLOW,
    "markdown.link": BLUE,
    "markdown.link_url": f"underline {MUTED}",
    "markdown.item.bullet": ACCENT,
    "markdown.item.number": ACCENT,
    "markdown.block_quote": MUTED,
    "markdown.hr": RULE,
})

PREVIEW_LINES = 5       # output lines shown inside a finished tool block
DIFF_LINES = 30         # diff lines shown before "… N more"

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# ------------------------------------------------------------------ small helpers

def fmt_tokens(n: int) -> str:
    """Compact token count: 950, 3.2k, 48k, 1.2M."""
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{round(n / 1000)}k"
    return f"{n / 1_000_000:.1f}M"


def fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(int(s), 60)
    return f"{m}m {s}s"


def short_path(path: str) -> str:
    home = os.path.expanduser("~")
    path = os.path.abspath(path)
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def display_path(path: str) -> str:
    """A tool's path argument as the user would type it: relative inside the working
    directory (models often pass absolute paths), ~-relative elsewhere."""
    p = str(path)
    if not os.path.isabs(p):
        return p
    rel = os.path.relpath(p, os.getcwd())
    return rel if not rel.startswith("..") else short_path(p)


def git_branch(cwd: str) -> str:
    """Current branch (or short sha when detached) read from .git/HEAD; '' outside a repo.

    Reads the file instead of running git so the footer never waits on a subprocess.
    """
    d = os.path.abspath(cwd)
    while True:
        dot = os.path.join(d, ".git")
        if os.path.isfile(dot):                      # worktree / submodule: "gitdir: <path>"
            try:
                with open(dot, encoding="utf-8") as f:
                    target = f.read().split(":", 1)[1].strip()
                dot = os.path.join(d, target)
            except (OSError, IndexError):
                return ""
        if os.path.isdir(dot):
            try:
                with open(os.path.join(dot, "HEAD"), encoding="utf-8") as f:
                    head = f.read().strip()
            except OSError:
                return ""
            return head[len("ref: refs/heads/"):] if head.startswith("ref: refs/heads/") \
                else head[:7]
        parent = os.path.dirname(d)
        if parent == d:
            return ""
        d = parent


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ------------------------------------------------------------------ welcome

_LOGO = (
    "█   █ █▀▀ █ █ ▀█▀ █▀▀ █▀█ █▀▄ █▀▀ █▀█",
    "█▄▄ █ █▄█ █▀█  █  █▄▄ █▄█ █▄▀ ██▄ █▀▄",
)
_STOPS = ((0xA7, 0x8B, 0xFA), (0x7A, 0xA2, 0xF7), (0x7D, 0xCF, 0xFF))   # violet -> blue -> cyan
_HALF = {"█": (1, 1), "▀": (1, 0), "▄": (0, 1)}


def _gradient(x: float) -> str:
    seg = min(int(x * (len(_STOPS) - 1)), len(_STOPS) - 2)
    t = x * (len(_STOPS) - 1) - seg
    a, b = _STOPS[seg], _STOPS[seg + 1]
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def logo(width: int = 80) -> Text:
    """The wordmark with a horizontal gradient. At >= 80 columns each half-block pixel of
    the small logo is drawn as a square of two full blocks (4 rows x 74 columns)."""
    small = [ln.ljust(len(_LOGO[0])) for ln in _LOGO]
    if width >= 80:
        pixels = [[_HALF.get(ch, (0, 0))[half] for ch in row] for row in small for half in (0, 1)]
        rows = ["".join("██" if p else "  " for p in prow) for prow in pixels]
    else:
        rows = small
    out = Text()
    span = max(len(rows[0]) - 1, 1)
    for i, row in enumerate(rows):
        for x, ch in enumerate(row):
            out.append(ch, style=f"bold {_gradient(x / span)}" if ch != " " else None)
        if i < len(rows) - 1:
            out.append("\n")
    return out


def welcome(*, version: str, model: str, mode: str, sandbox: str, cwd: str, branch: str,
            width: int = 80) -> Padding:
    head = Text.assemble(("lightcoder", f"bold {TEXT}"), (f" v{version}", DIM),
                         ("   a minimal, hackable coding agent", MUTED))
    where = Text(short_path(cwd), style=MUTED)
    if branch:
        where.append(f" ({branch})", style=DIM)
    info = Text.assemble(("model ", DIM), (model, LINK), ("    mode ", DIM),
                         (mode, MODE_COLOR.get(mode, MUTED)), ("    sandbox ", DIM),
                         (sandbox, MUTED))
    cmds = Table.grid(padding=(0, 2))
    for _ in range(4):
        cmds.add_column()
    cmds.add_row(Text("/help", style=f"bold {TEXT}"), Text("all commands", style=DIM),
                 Text("/model", style=f"bold {TEXT}"), Text("switch model", style=DIM))
    cmds.add_row(Text("/resume", style=f"bold {TEXT}"), Text("resume a session", style=DIM),
                 Text("/mode", style=f"bold {TEXT}"), Text("permission mode", style=DIM))
    keys = Text("shift+tab cycle mode · ctrl+c interrupt · ctrl+d quit", style=DIM)
    return Padding(Group(Text(), logo(width), Text(), head, Text(), where, info, Text(),
                         cmds, Text(), keys, Text()), (0, 2))


# ------------------------------------------------------------------ conversation blocks

class _TrimTop:
    """Drop the blank lines a renderable starts with (rich's Markdown opens lists with one,
    which doubles the gap the REPL already leaves between paragraphs)."""

    def __init__(self, renderable):
        self.renderable = renderable

    def __rich_console__(self, console, options):
        lines = console.render_lines(self.renderable, options, pad=False)
        while lines and not "".join(seg.text for seg in lines[0]).strip():
            lines.pop(0)
        for line in lines:
            yield from line
            yield Segment.line()


def markdown(text: str) -> Padding:
    """Assistant text: Markdown, indented one column like the blocks around it."""
    return Padding(_TrimTop(Markdown(text)), (0, 1))


def user_block(text: str) -> Padding:
    return Padding(Text(text, style=TEXT), (1, 1), style=f"on {USER_BG}", expand=True)


def tool_title(name: str, args: dict | None) -> Text:
    a = args or {}
    bold = f"bold {TEXT}"
    t = Text()
    if name == "run_bash":
        cmd = str(a.get("command", "")).strip()
        lines = cmd.splitlines() or [""]
        shown = "\n".join(lines[:4]) + ("\n…" if len(lines) > 4 else "")
        t.append("$ ", style=bold)
        t.append(shown, style=bold)
        if _int(a.get("timeout")):
            t.append(f" (timeout {_int(a.get('timeout'))}s)", style=DIM)
    elif name == "read_file":
        t.append("read ", style=bold)
        t.append(display_path(a.get("file_path", "")), style=LINK)
        start, limit = _int(a.get("offset")), _int(a.get("limit"))
        if start or limit:
            start = start or 1
            t.append(f":{start}-{start + limit - 1}" if limit else f":{start}-", style=YELLOW)
    elif name in ("edit_file", "write_file"):
        t.append("edit " if name == "edit_file" else "write ", style=bold)
        t.append(display_path(a.get("file_path", "")), style=LINK)
    elif name == "grep":
        t.append("grep ", style=bold)
        t.append(str(a.get("pattern", "")), style=YELLOW)
        if a.get("path"):
            t.append(f" in {display_path(a['path'])}", style=DIM)
        if a.get("glob"):
            t.append(f" ({a['glob']})", style=DIM)
    elif name == "glob":
        t.append("glob ", style=bold)
        t.append(str(a.get("pattern", "")), style=YELLOW)
    elif name == "update_todo":
        t.append("todo", style=bold)
    elif name == "delegate":
        t.append("delegate ", style=bold)
        t.append(_clip(str(a.get("task", "")).splitlines()[0] if a.get("task") else "", 80),
                 style=MUTED)
    else:
        t.append(name, style=bold)
        brief = ", ".join(f"{k}={_clip(str(v), 30)}" for k, v in list(a.items())[:3])
        if brief:
            t.append(f" {brief}", style=DIM)
    return t


_EXIT_RE = re.compile(r"\(exit code (-?\d+)\)\s*$")


def tool_status(name: str, result) -> str:
    """'ok' | 'error' | 'denied'. A shell command that exits non-zero is an error even
    though the tool itself ran fine — that is what the block colour has to show."""
    r = str(result).strip()
    if r.startswith("[Permission]"):
        return "denied"
    if r.startswith("Error"):
        return "error"
    if name == "run_bash":
        m = _EXIT_RE.search(r)
        if m and int(m.group(1)) != 0:
            return "error"
    return "ok"


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def render_diff(diff_text: str, max_lines: int = DIFF_LINES) -> Text:
    """Unified diff -> numbered lines: ' 16 ctx', '+19 added', '-19 removed'."""
    rows: list[tuple[str, int | None, str]] = []
    old = new = 0
    for ln in diff_text.splitlines():
        if ln.startswith(("---", "+++", "\\")):
            continue
        m = _HUNK_RE.match(ln)
        if m:
            if rows:
                rows.append(("…", None, ""))
            old, new = int(m.group(1)), int(m.group(2))
        elif ln.startswith("+"):
            rows.append(("+", new, ln[1:]))
            new += 1
        elif ln.startswith("-"):
            rows.append(("-", old, ln[1:]))
            old += 1
        elif ln.startswith(" "):
            rows.append((" ", new, ln[1:]))
            old += 1
            new += 1
    width = max((len(str(n)) for _, n, _ in rows if n is not None), default=1)
    out = Text()
    for sign, n, body in rows[:max_lines]:
        if sign == "…":
            out.append(" " * (width + 1) + "...\n", style=DIM)
            continue
        style = GREEN if sign == "+" else RED if sign == "-" else MUTED
        out.append(f"{sign}{n:>{width}} {body}\n", style=style)
    if len(rows) > max_lines:
        out.append(f"… {len(rows) - max_lines} more diff lines\n", style=DIM)
    out.rstrip()
    return out


def edit_diff(file_path: str, old_string: str, new_string: str) -> str:
    """The unified diff an edit_file call *would* produce, for the permission dialog.

    Uses the file's real line numbers when old_string matches exactly once (the only case
    edit_file accepts); otherwise falls back to diffing the two strings on their own.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            original = f.read()
    except OSError:
        original = None
    if original is not None and original.count(old_string) == 1:
        before, after = original, original.replace(old_string, new_string, 1)
    else:
        before, after = old_string, new_string
    return "".join(difflib.unified_diff(before.splitlines(keepends=True),
                                        after.splitlines(keepends=True), n=2))


def write_diff(file_path: str, content: str) -> str:
    try:
        with open(file_path, encoding="utf-8") as f:
            before = f.read()
    except OSError:
        before = ""
    return "".join(difflib.unified_diff(before.splitlines(keepends=True),
                                        content.splitlines(keepends=True), n=2))


def _preview(lines: list[str], tail: bool) -> tuple[list[str], int]:
    if len(lines) <= PREVIEW_LINES:
        return lines, 0
    hidden = len(lines) - PREVIEW_LINES
    return (lines[-PREVIEW_LINES:] if tail else lines[:PREVIEW_LINES]), hidden


def _expand_hint(hidden: int, ref: int | None, where: str) -> Text:
    hint = f"… ({hidden} {where} lines"
    hint += f" · /expand {ref})" if ref is not None else ")"
    return Text(hint, style=DIM)


def tool_body(name: str, args: dict | None, result: str, status: str,
              ref: int | None = None) -> list:
    """What goes under the title of a finished tool block (may be empty)."""
    a = args or {}
    result = str(result)
    if status == "denied":
        return [Text(result.replace("[Permission] ", ""), style=RED)]

    if name == "run_bash":
        text = _EXIT_RE.sub("", result.rstrip()).rstrip()
        if text == "(no output)":
            text = ""
        shown, hidden = _preview(text.splitlines(), tail=True)
        body = [_expand_hint(hidden, ref, "earlier")] if hidden else []
        if shown:
            body.append(Text("\n".join(shown), style=MUTED))
        m = _EXIT_RE.search(result.rstrip())
        if m and int(m.group(1)) != 0:
            body.append(Text(f"exit code {m.group(1)}", style=RED))
        return body

    if status == "error":
        shown, hidden = _preview(result.splitlines(), tail=False)
        body = [Text("\n".join(shown), style=TEXT)]
        if hidden:
            body.append(_expand_hint(hidden, ref, "more"))
        return body

    if name == "edit_file":
        diff = result[result.find("\n---") + 1:] if "\n---" in result else ""
        return [render_diff(diff)] if diff else []

    if name == "write_file":
        content = str(a.get("content", ""))
        shown, hidden = _preview(content.splitlines(), tail=False)
        width = len(str(len(shown)))
        body = [Text("\n".join(f"+{i:>{width}} {ln}" for i, ln in enumerate(shown, 1)),
                     style=GREEN)]
        if hidden:
            body.append(Text(f"… ({hidden} more lines)", style=DIM))
        return body

    if name in ("grep", "glob", "delegate", "update_todo"):
        lines = result.splitlines()
        if name == "update_todo":
            lines = str(a.get("content", "")).splitlines()
        shown, hidden = _preview(lines, tail=False)
        body = [Text("\n".join(shown), style=MUTED)] if shown else []
        if hidden:
            body.append(_expand_hint(hidden, ref, "more"))
        return body

    return []          # read_file and unknown tools: the title says enough


def _summary(name: str, result: str, status: str) -> str:
    """A dim count after the title, for tools whose output stays collapsed."""
    if status != "ok":
        return ""
    if name == "read_file":
        n = len(result.splitlines())
        return f"{n} line{'s' if n != 1 else ''}"
    if name == "grep" and result.strip() != "(no matches)":
        n = len(result.splitlines())
        return f"{n} match{'es' if n != 1 else ''}"
    if name == "glob" and result.strip() != "(no files match)":
        n = len(result.splitlines())
        return f"{n} file{'s' if n != 1 else ''}"
    return ""


def tool_block(name: str, args: dict | None, result: str | None = None,
               seconds: float | None = None, ref: int | None = None) -> Padding:
    """A tool call as a full-width block. ``result=None`` draws it as still running."""
    status = "pending" if result is None else tool_status(name, result)
    title = tool_title(name, args)
    parts: list = [title]
    if result is not None:
        summary = _summary(name, result, status)
        if summary:
            title.append(f"  {summary}", style=DIM)
        body = tool_body(name, args, result, status, ref)
        if body:
            parts += [Text()] + body
        if seconds is not None and status != "denied" and name in ("run_bash", "delegate"):
            parts += [Text(), Text(f"Took {fmt_seconds(seconds)}", style=DIM)]
    return Padding(Group(*parts), (1, 1), style=f"on {TOOL_BG[status]}", expand=True)


# ------------------------------------------------------------------ permission dialog

def permission_panel(name: str, args: dict | None, dangerous: bool = False) -> Panel:
    a = args or {}
    body: list = [tool_title(name, a)]
    diff = ""
    if name == "edit_file":
        diff = edit_diff(str(a.get("file_path", "")), str(a.get("old_string", "")),
                         str(a.get("new_string", "")))
    elif name == "write_file":
        diff = write_diff(str(a.get("file_path", "")), str(a.get("content", "")))
    if diff:
        body += [Text(), render_diff(diff, max_lines=20)]
    if dangerous:
        body += [Text(), Text("⚠ matches a dangerous-command pattern", style=f"bold {RED}")]
    color = RED if dangerous else YELLOW
    return Panel(Group(*body), title=Text(" Permission required ", style=f"bold {color}"),
                 title_align="left", border_style=color, box=box.ROUNDED, padding=(1, 2))


# ------------------------------------------------------------------ chrome

class TitledRule:
    """A full-width rule with an optional title on the left: '── <title> ─────────'."""

    def __init__(self, title: Text | None = None, style: str = ACCENT):
        self.title = title
        self.style = style

    def __rich_console__(self, console, options):
        width = options.max_width
        t = Text()
        if self.title:
            t.append("── ", style=self.style)
            t.append_text(self.title)
            t.append(" ")
        t.append("─" * max(width - t.cell_len, 0), style=self.style)
        t.truncate(width)
        yield t


def working_title(elapsed: float) -> Text:
    spin = SPINNER[int(elapsed * 12) % len(SPINNER)]
    return Text.assemble((spin, ACCENT), (f" Working {int(elapsed)}s", MUTED),
                         (" · ctrl+c to interrupt", DIM))


def _line(left: Text, right: Text) -> Table:
    """One footer line: *left* flush left, *right* flush right; the left side gives way."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", no_wrap=True, overflow="ellipsis", ratio=1)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(left, right)
    return grid


def footer(*, mode: str, model: str, cwd: str, branch: str, prompt_tokens: int,
           completion_tokens: int, cached_tokens: int, ctx_used: int,
           ctx_window: int) -> Group:
    """Two lines under the input: mode + model, then location + usage."""
    color = MODE_COLOR.get(mode, MUTED)
    left1 = Text.assemble(("▸▸ ", color), (f"{mode} mode", f"bold {color}"),
                          (" · shift+tab to cycle", DIM))
    right1 = Text(model, style=MUTED)

    where = Text(short_path(cwd), style=DIM)
    if branch:
        where.append(f" ({branch})", style=DIM)
    usage = Text()
    if prompt_tokens or completion_tokens:
        usage.append(f"↑{fmt_tokens(prompt_tokens)} ↓{fmt_tokens(completion_tokens)}", style=DIM)
        if cached_tokens and prompt_tokens:
            usage.append(f" · cache {100 * cached_tokens / prompt_tokens:.0f}%", style=DIM)
        usage.append(" · ", style=DIM)
    frac = ctx_used / max(ctx_window, 1)
    # Colours track the context manager's compression stages (snip at 50%, summarise at 70%).
    ctx_color = DIM if frac < 0.5 else YELLOW if frac < 0.7 else RED
    usage.append(f"ctx {100 * frac:.1f}% of {fmt_tokens(ctx_window)}", style=ctx_color)

    return Group(_line(left1, right1), _line(where, usage))
