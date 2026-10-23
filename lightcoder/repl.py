"""Inline REPL — output stays in the terminal's own scrollback.

    … finished blocks, printed into scrollback …
    ── ⠋ Working 12s · ctrl+c to interrupt ───────────────   live region while a turn runs:
    ›                                                         streaming text, running tools,
    ───────────────────────────────────────────────────────   the input box and the footer
    ▸▸ default mode · shift+tab to cycle        deepseek-v4-pro
    ~/LightCoder (main)      ↑3.2k ↓360 · ctx 2.6% of 128k

Finished blocks are printed above a rich.Live region. While waiting for input, a small
non-fullscreen prompt_toolkit Application draws the same input box and footer, so the bottom
of the screen looks the same idle or busy. Everything visual comes from ui.py; this module
only decides what to draw when.
"""

from __future__ import annotations

import io
import os
import sys
import time

from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.text import Text

from . import __version__, ui
from .permissions import MODES, is_dangerous

# shift+tab order. yolo joins the cycle only when the session started in it, so one stray
# keypress can never switch off every confirmation.
_CYCLE = ("default", "auto", "plan")

_COMMANDS = (
    ("/help", "all commands"),
    ("/model", "switch model"),
    ("/mode", "permission mode"),
    ("/resume", "resume a session"),
    ("/expand", "full output of a tool call"),
    ("/clear", "start a fresh conversation"),
)


def _short_model(model) -> str:
    return os.path.basename(str(model).rstrip("/")) or str(model)


def _to_ansi(renderable, width: int) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor", width=width).print(
        renderable, end=""
    )
    return buf.getvalue()


def _footer(agent, state: dict):
    return ui.footer(
        mode=agent.permissions.mode,
        model=_short_model(agent.llm.model),
        cwd=state.get("cwd", os.getcwd()),
        branch=state.get("branch", ""),
        prompt_tokens=agent.total_prompt_tokens,
        completion_tokens=agent.total_completion_tokens,
        cached_tokens=agent.total_cached_tokens,
        ctx_used=agent.last_prompt_tokens,
        ctx_window=state.get("ctx", 1),
    )


def _cycle_mode(agent, state: dict) -> None:
    cycle = state.get("cycle", _CYCLE)
    cur = agent.permissions.mode
    agent.permissions.mode = cycle[(cycle.index(cur) + 1) % len(cycle)] if cur in cycle \
        else cycle[0]


# ------------------------------------------------------------------ one turn

class _Turn:
    """One agent.chat() call on screen.

    Streaming text and running tools live in a rich.Live region above the input box; each
    finished paragraph or tool block is printed into scrollback. A permission prompt closes
    the Live region and opens a fresh one afterwards — restarting the same Live would redraw
    from its stale height and erase the lines printed in between.
    """

    def __init__(self, agent, console: Console, state: dict, fallback_permission=None):
        self.agent = agent
        self.console = console
        self.state = state
        self.fallback_permission = fallback_permission   # used off a terminal
        self.seg = ""                 # assistant text not yet printed
        self.pending: list[tuple[str, dict]] = []        # announced, unfinished tool calls
        self.t0 = time.time()
        self.live: Live | None = None

    # The Live region. Runs on rich's refresh thread, so it reads snapshots only.
    def __rich_console__(self, console, options):
        seg, pending = self.seg, list(self.pending)
        parts = []
        if seg.strip():
            room = max(console.size.height - 8 - 4 * len(pending), 3)
            parts.append(ui.markdown("\n".join(seg.splitlines()[-room:])))
        for name, args in pending:
            parts += [Text(), ui.tool_block(name, args)]
        parts += [
            Text(),
            ui.TitledRule(ui.working_title(time.time() - self.t0)),
            Text("› ", style=f"bold {ui.ACCENT}"),
            ui.TitledRule(),
            _footer(self.agent, self.state),
        ]
        yield Group(*parts)

    def _open_live(self) -> None:
        self.live = Live(self, console=self.console, refresh_per_second=12, transient=True)
        self.live.start()

    def _close_live(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    def _flush_text(self, final: bool = False) -> None:
        """Print finished paragraphs; keep the one still streaming (and any open code fence)."""
        if final:
            head, self.seg = self.seg, ""
        else:
            cut = self.seg.rfind("\n\n")
            if cut < 0 or self.seg[:cut].count("```") % 2:
                return
            head, self.seg = self.seg[:cut], self.seg[cut + 2:]
        if head.strip():
            self.console.print()
            self.console.print(ui.markdown(head))

    # ------------------------------------------------------------ agent callbacks
    def on_token(self, tok: str) -> None:
        self.seg += tok
        if "\n" in tok:
            self._flush_text()

    def on_tool(self, name: str, args: dict) -> None:
        self._flush_text(final=True)
        self.pending.append((name, args))

    def on_result(self, name: str, result: str, seconds: float | None = None) -> None:
        # Results arrive in call order, so the first pending call of that name is this one.
        i = next((k for k, (n, _) in enumerate(self.pending) if n == name), None)
        args = self.pending.pop(i)[1] if i is not None else {}
        outputs = self.state.setdefault("outputs", [])
        outputs.append((name, args, result))
        self.console.print()
        self.console.print(ui.tool_block(name, args, result, seconds, ref=len(outputs)))

    def on_permission(self, name: str, args: dict, prompt: str) -> bool:
        if not sys.stdin.isatty():
            return bool(self.fallback_permission and
                        self.fallback_permission(name, args, prompt))
        self._close_live()
        try:
            return _ask_permission(self.agent, name, args)
        finally:
            self._open_live()

    # ------------------------------------------------------------ driver
    def run(self, text: str) -> None:
        note = None
        self._open_live()
        try:
            self.agent.chat(text, on_token=self.on_token, on_tool=self.on_tool,
                            on_result=self.on_result, on_permission=self.on_permission)
        except KeyboardInterrupt:
            self.agent.settle_interrupted()
            for name, args in self.pending:          # keep the cut-off calls on screen
                self.console.print()
                self.console.print(ui.tool_block(name, args, "Error: interrupted by the user"))
            self.pending.clear()
            note = Text("✗ Interrupted", style=ui.RED)
        except Exception as e:  # noqa: BLE001
            note = Text(f"✗ {e}", style=ui.RED)
        finally:
            self._flush_text(final=True)
            self._close_live()
        self.console.print()
        self.console.print(note or Text(f"✻ Worked for {ui.fmt_seconds(time.time() - self.t0)}",
                                        style=ui.DIM))


# ------------------------------------------------------------------ prompts

def _make_ask(agent, state: dict, *, input=None, output=None):
    """Return ask() -> str: the input box with the footer under it, hugging the content."""
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
    from prompt_toolkit.key_binding.defaults import load_key_bindings
    from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor

    buf = Buffer(multiline=False)

    def matches():
        t = buf.text
        if not t.startswith("/") or " " in t:
            return []
        return [(c, d) for c, d in _COMMANDS if c.startswith(t)]

    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        event.app.exit(result=buf.text)

    @kb.add("c-c")
    def _(event):
        if buf.text:                  # first ctrl+c clears the line, the next one quits
            buf.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt())

    @kb.add("c-d")
    def _(event):
        if not buf.text:
            event.app.exit(exception=EOFError())

    @kb.add("s-tab")
    def _(event):
        _cycle_mode(agent, state)

    @kb.add("tab")
    def _(event):
        m = matches()
        if m:
            buf.text = m[0][0] + " "
            buf.cursor_position = len(buf.text)

    def width() -> int:
        return get_app().output.get_size().columns

    rule = lambda: Window(char="─", height=1, style=f"fg:{ui.ACCENT}")  # noqa: E731
    prompt = Window(FormattedTextControl([(f"bold fg:{ui.ACCENT}", "› ")]),
                    dont_extend_width=True)
    placeholder = ConditionalProcessor(
        AfterInput("Ask anything…  (/ for commands)", style=f"fg:{ui.DIM}"),
        filter=Condition(lambda: not buf.text),
    )
    inp = Window(BufferControl(buffer=buf, input_processors=[placeholder]), wrap_lines=True,
                 dont_extend_height=True)      # grow with wrapped input, not to the screen

    def hint_lines():
        out = []
        for i, (cmd, desc) in enumerate(matches()):
            out.append((f"bold fg:{ui.TEXT}" if i == 0 else f"fg:{ui.MUTED}", f"  {cmd:<10}"))
            out.append((f"fg:{ui.DIM}", f"{desc}\n"))
        return out

    hints = ConditionalContainer(
        Window(FormattedTextControl(hint_lines), dont_extend_height=True),
        filter=Condition(lambda: bool(matches())),
    )
    footer = Window(
        FormattedTextControl(lambda: ANSI(_to_ansi(_footer(agent, state), width()))),
        height=2,
    )
    app = Application(
        layout=Layout(HSplit([rule(), VSplit([prompt, inp]), rule(), hints, footer]),
                      focused_element=inp),
        key_bindings=merge_key_bindings([load_key_bindings(), kb]),
        full_screen=False,
        erase_when_done=True,
        input=input,
        output=output,
    )

    def ask() -> str:
        buf.reset()
        return app.run()

    return ask


def _select(header, options: list[str], *, input=None, output=None) -> int | None:
    """Arrow-key choice under a rich *header*; returns the index, or None on Esc."""
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    sel = [0]
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _(event):
        sel[0] = (sel[0] - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    def _(event):
        sel[0] = (sel[0] + 1) % len(options)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=sel[0])

    # eager: without it Escape waits to learn whether it began an arrow-key sequence.
    @kb.add("escape", eager=True)
    @kb.add("n")
    def _(event):
        event.app.exit(result=None)

    @kb.add("y")
    def _(event):
        event.app.exit(result=0)

    @kb.add("c-c")
    def _(event):
        event.app.exit(exception=KeyboardInterrupt())

    for i in range(min(len(options), 9)):
        kb.add(str(i + 1))(lambda event, i=i: event.app.exit(result=i))

    def body():
        out = []
        for i, opt in enumerate(options):
            cur = i == sel[0]
            out.append((f"bold fg:{ui.ACCENT}" if cur else f"fg:{ui.MUTED}",
                        f"  {'❯' if cur else ' '} {i + 1}. {opt}\n"))
        out.append((f"fg:{ui.DIM}", "\n  ↑↓ choose · enter confirm · esc deny"))
        return out

    head = Window(FormattedTextControl(
        lambda: ANSI(_to_ansi(header, get_app().output.get_size().columns))))
    app = Application(
        layout=Layout(HSplit([head, Window(FormattedTextControl(body))])),
        key_bindings=kb,
        full_screen=False,
        erase_when_done=True,
        input=input,
        output=output,
    )
    return app.run()


def _ask_permission(agent, name: str, args: dict) -> bool:
    """The permission dialog: what the call will do (a diff for edits), then a choice.

    "Switch to auto mode" is offered only when auto would actually cover this call —
    a dangerous command is asked about in auto mode too.
    """
    dangerous = name == "run_bash" and is_dangerous(str((args or {}).get("command", "")))
    options = ["Yes", "No"] if dangerous or agent.permissions.mode == "auto" else \
        ["Yes", "Yes, and switch to auto mode (stop asking for edits and routine commands)",
         "No"]
    choice = _select(ui.permission_panel(name, args, dangerous), options)
    if choice is None or options[choice] == "No":
        return False
    if choice == 1 and len(options) == 3:
        agent.permissions.mode = "auto"
    return True


def _ask_choice(console: Console, prompt_text: str) -> str | None:
    """Read a picker answer; return None when the user cancels.

    On a terminal this uses prompt_toolkit, because a line-based ``input()`` cannot see a
    bare Escape — it only ever returns on Enter. ``eager`` matters: without it Escape waits
    to learn whether it began an escape sequence (arrow keys), which feels like a stuck key.

    Off a terminal (pipes, tests) prompt_toolkit raises EOFError, which must not be mistaken
    for the user cancelling — so the tty check comes first and those callers get a plain
    line prompt instead.
    """
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import prompt as pt_prompt
            from prompt_toolkit.key_binding import KeyBindings

            kb = KeyBindings()

            @kb.add("escape", eager=True)
            def _(event):
                event.app.exit(result=None)

            return pt_prompt(prompt_text, key_bindings=kb)
        except (EOFError, KeyboardInterrupt):
            return None
        except Exception:  # noqa: BLE001 — prompt_toolkit unusable here; use the fallback
            pass

    try:
        return console.input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        return None


# ------------------------------------------------------------------ slash commands

_HELP = """\
  /help             show this help
  /model            choose a model (numbered picker)
  /model <name>     switch straight to a configured model
  /mode             list permission modes
  /mode <name>      switch mode (shift+tab cycles default → auto → plan)
  /resume           list saved sessions for this directory
  /resume <id>      load one (full id or a unique fragment)
  /expand [n]       full output of tool call n (default: the last one)
  /clear            start a fresh conversation
  exit | quit       leave
"""


def _handle_command(agent, console: Console, line: str, state: dict) -> bool:
    """Run a /slash command. Returns True when *line* was a command (handled or not).

    Kept in the REPL rather than the agent: these change how the session is driven, not
    what the model can do, so they never reach the model as tool calls.
    """
    if not line.startswith("/"):
        return False
    cmd, _, arg = line[1:].partition(" ")
    arg = arg.strip()

    if cmd in ("help", "?"):
        console.print(_HELP, highlight=False)

    elif cmd == "model":
        from .settings import model_names, resolve_model

        names = model_names()
        if not names:
            console.print("  [dim](no models in ~/.lightcoder/models.json)[/dim]")
            return True

        if not arg:                          # numbered picker, Claude-Code style
            current = agent.llm.model
            for i, n in enumerate(names, 1):
                here = " [green](current)[/]" if resolve_model(n).get("model") == current else ""
                console.print(f"  [cyan]{i}[/] {n}{here}")
            choice = _ask_choice(console, f"  select [1-{len(names)}, Esc cancels]: ")
            if choice is None or not choice.strip():
                return True
            choice = choice.strip()
            if choice.isdigit() and 1 <= int(choice) <= len(names):
                arg = names[int(choice) - 1]
            elif choice in names:
                arg = choice                  # typing the name works too
            else:
                console.print(f"  [red]no such model: {choice}[/red]")
                return True
        elif arg not in names:
            console.print(f"  [red]{arg} is not configured[/red] — /model to choose")
            return True

        from .config import LLMConfig
        from .llm import LLM

        agent.llm = LLM(LLMConfig.from_env(model=arg))
        # a different backend usually means a different window
        state["ctx"] = _detect_context_window(agent) or agent.config.max_context_tokens
        console.print(f"  switched to [bold magenta]{_short_model(agent.llm.model)}[/]")
        if agent.session is not None:        # keep the transcript self-describing
            agent.session.write("model_change", model=agent.llm.model)

    elif cmd == "mode":
        if not arg:
            for m in MODES:
                mark = Text(" ❯ " if m == agent.permissions.mode else "   ", style=ui.ACCENT)
                console.print(Text.assemble(mark, (f"{m:<9}", ui.MODE_COLOR[m]),
                                            (ui.MODE_HINT[m], ui.DIM)))
        elif arg not in MODES:
            console.print(f"  [red]unknown mode {arg!r}[/red] — one of {', '.join(MODES)}")
        else:
            agent.permissions.mode = arg
            console.print(Text.assemble("  mode → ", (arg, f"bold {ui.MODE_COLOR[arg]}")))

    elif cmd == "expand":
        outputs = state.get("outputs", [])
        if not outputs:
            console.print("  [dim](no tool output yet)[/dim]")
            return True
        n = int(arg) if arg.isdigit() else len(outputs)
        if not 1 <= n <= len(outputs):
            console.print(f"  [red]no tool call {arg}[/red] — 1 to {len(outputs)}")
            return True
        name, args, result = outputs[n - 1]
        console.print(Padding(Group(ui.tool_title(name, args), Text(),
                                    Text(str(result), style=ui.MUTED)),
                              (1, 1), style=f"on {ui.TOOL_BG['pending']}", expand=True))

    elif cmd == "clear":
        agent.reset()
        agent.last_prompt_tokens = 0
        state["outputs"] = []
        if agent.session is not None:
            agent.session.write("clear")
        console.print("  [dim]started a fresh conversation[/dim]")

    elif cmd == "resume":
        from .session import SessionLog, describe

        if not arg:
            found = SessionLog.list()
            for f in found:
                console.print("  " + describe(f))
            console.print("  [dim]/resume <id> to load one[/dim]" if found
                          else "  [dim](no saved sessions for this directory)[/dim]")
        else:
            path = SessionLog.find(arg)
            if path is None:
                console.print(f"  [red]no session matching {arg!r}[/red]")
                return True
            _, messages = SessionLog.read(path)
            agent.messages[:] = messages     # replace history, then bind the same file
            agent.session = SessionLog.reopen(path)
            console.print(f"  resumed {path.stem} · {len(messages)} messages")

    else:
        console.print(f"  [yellow]unknown command /{cmd}[/yellow] — try /help")
    return True


# ------------------------------------------------------------------ entry point

def run_repl(agent, on_permission=None) -> None:
    """Interactive loop. *on_permission* answers permission prompts when stdin is not a
    terminal (the arrow-key dialog needs one)."""
    console = Console(theme=ui.THEME)
    cwd = os.getcwd()
    state = {
        # Mutable so /model can retarget the context bar at the new backend's window.
        "ctx": _detect_context_window(agent) or agent.config.max_context_tokens,
        "cwd": cwd,
        "branch": ui.git_branch(cwd),
        "outputs": [],
        "cycle": _CYCLE + (("yolo",) if agent.permissions.mode == "yolo" else ()),
    }

    try:
        ask = _make_ask(agent, state)
    except Exception:  # noqa: BLE001 — fall back to a plain prompt
        def ask() -> str:
            return console.input(Text("› ", style=f"bold {ui.ACCENT}"))

    console.print(ui.welcome(version=__version__, model=_short_model(agent.llm.model),
                             mode=agent.permissions.mode, sandbox=agent.config.sandbox,
                             cwd=cwd, branch=state["branch"], width=console.width))
    while True:
        try:
            user = ask().strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user:
            continue
        if user in {"exit", "quit"}:
            break
        if user.startswith("/"):
            console.print(Text(f"› {user}", style=ui.DIM))
            _handle_command(agent, console, user, state)
            continue
        console.print()
        console.print(ui.user_block(user))
        _Turn(agent, console, state, on_permission).run(user)


def _detect_context_window(agent, timeout: float = 3.0) -> int:
    """Ask the endpoint for its context window (vLLM reports ``max_model_len``).

    Bounded and best-effort: this runs at startup *and* on every ``/model`` switch, so a
    slow or unreachable endpoint must never freeze the prompt — the caller falls back to
    the configured budget.
    """
    try:
        client = agent.llm._client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=0)
        for m in client.models.list().data:
            mml = getattr(m, "max_model_len", None)
            if mml:
                return int(mml)
    except Exception:  # noqa: BLE001
        pass
    return 0
