"""Command-line entry point: one-shot (`-p`) or an interactive REPL."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.text import Text

from .config import AgentConfig, LLMConfig, PROMPT_VARIANTS
from .permissions import MODES
from .sandbox import SANDBOX_MODES
from .settings import load_settings, model_names, global_dir
from .llm import LLM
from .agent import Agent

console = Console()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="lightcoder", description="A minimal coding agent.")
    p.add_argument("--base-url", help="OpenAI-compatible base URL.")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY or 'dummy').")
    p.add_argument("-p", "--prompt", help="Run one prompt non-interactively and exit.")
    p.add_argument("--list-models", action="store_true",
                   help="List models configured in ~/.lightcoder/models.json and exit.")
    p.add_argument("--resume", nargs="?", const="", metavar="ID",
                   help="Resume a previous session in this directory "
                        "(bare --resume takes the most recent; or pass an id/prefix).")
    p.add_argument("--list-sessions", action="store_true",
                   help="List saved sessions for this directory and exit.")
    p.add_argument("--no-session", action="store_true",
                   help="Do not write a session transcript for this run.")
    # Everything below defaults to None so settings.json can supply the value; an
    # explicitly passed flag always wins. See _build().
    p.add_argument("--no-parallel", action="store_true", default=None,
                   help="Run tool calls sequentially.")
    p.add_argument("--no-compression", action="store_true", default=None,
                   help="Disable context compression.")
    p.add_argument("--no-subagent", action="store_true", default=None,
                   help="Hide the delegate tool.")
    p.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--extra", action="append", default=[], metavar="NAME",
                   help="Enable an optimization extra (repeatable); maps to AgentConfig.extras.")
    p.add_argument("--mode", choices=MODES, default=None,
                   help="Permission mode: plan (read-only) | default (ask before edits/commands) "
                        "| auto (accept edits and routine commands, ask on dangerous ones) "
                        "| yolo (never ask).")
    p.add_argument("--sandbox", choices=SANDBOX_MODES, default=None,
                   help="Confine run_bash: none (host) | docker (container, --network none, "
                        "workspace bind-mounted).")
    p.add_argument("--sandbox-image", default=None,
                   help="Docker image used when --sandbox docker (needs the project's deps).")
    return p.parse_args(argv)


def _build(args) -> Agent:
    """Assemble the agent, merging: CLI flag > settings.json > built-in default.

    Only the CLI merges settings — the eval harness builds AgentConfig directly, so a
    user's personal settings can never perturb a measured run.
    """
    st = load_settings()

    llm_cfg = LLMConfig.from_env(model=st.get("model"))
    if args.base_url:
        llm_cfg.base_url = args.base_url
    if args.api_key:
        llm_cfg.api_key = args.api_key

    def pick(field, cli_value, default):
        return cli_value if cli_value is not None else st.get(field, default)

    agent_cfg = AgentConfig(
        parallel_tools=pick("parallel_tools", False if args.no_parallel else None, True),
        compression=pick("compression", False if args.no_compression else None, True),
        subagent=pick("subagent", False if args.no_subagent else None, True),
        prompt_variant=pick("prompt_variant", args.prompt_variant, "default"),
        max_rounds=pick("max_rounds", args.max_rounds, 50),
        extras=tuple(args.extra) if args.extra else tuple(st.get("extras", ())),
        permission_mode=pick("permission_mode", args.mode, "default"),
        permission_allow=tuple(st.get("permission_allow", ())),
        permission_deny=tuple(st.get("permission_deny", ())),
        sandbox=pick("sandbox", args.sandbox, "none"),
        sandbox_image=pick("sandbox_image", args.sandbox_image, "python:3.11-slim"),
    )
    return Agent(LLM(llm_cfg), config=agent_cfg)


def _attach_session(agent, args) -> None:
    """Attach an append-only transcript, replaying a previous one when resuming.

    Replay happens *before* the log is attached so restored messages are not written back
    into the file they came from.
    """
    if args.no_session:
        return
    from .session import SessionLog

    if args.resume is not None:
        path = SessionLog.find(args.resume) if args.resume else SessionLog.latest()
        if path is None:
            console.print("[yellow]no session to resume — starting a new one[/yellow]")
        else:
            _, messages = SessionLog.read(path)
            agent.messages.extend(messages)
            agent.session = SessionLog.reopen(path)
            console.print(f"[dim]resumed {path.stem} · {len(messages)} messages[/dim]")
            return

    agent.session = SessionLog.create(
        model=getattr(agent.llm, "model", ""), config=agent.config.label()
    )


def _confirm(name, kwargs, prompt) -> bool:
    """Permission prompt: show the request, read y/N. Deny on EOF/interrupt (safe default)."""
    console.print(Text.assemble(("⚠ Permission ", "yellow"), prompt))
    try:
        # A Text, not a str: as markup, rich would swallow "[y/N]" as a style tag.
        return console.input(Text("  Allow? [y/N] ")).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]  (denied)[/dim]")
        return False


def _on_token(tok):
    print(tok, end="", flush=True)


def _on_tool(name, kwargs):
    brief = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in list(kwargs.items())[:3])
    console.print(f"\n[dim]> {name}({brief})[/dim]")


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.list_sessions:
        from .session import SessionLog, describe

        found = SessionLog.list()
        for f in found:
            console.print("  " + describe(f))
        if not found:
            console.print("  [dim](no sessions for this directory)[/dim]")
        return 0

    if args.list_models:
        names = model_names()
        console.print(f"[dim]{global_dir() / 'models.json'}[/dim]")
        for n in names:
            console.print(f"  {n}")
        if not names:
            console.print("  [dim](no models configured)[/dim]")
        return 0

    try:
        agent = _build(args)
    except ValueError as e:      # e.g. an invalid mode/sandbox in settings
        console.print(f"[red]{e}[/red]")
        return 2

    _attach_session(agent, args)
    try:
        if args.prompt:
            agent.chat(args.prompt, on_token=_on_token, on_tool=_on_tool,
                       on_permission=_confirm)
            print()
            return 0

        # interactive: inline REPL (output stays in the terminal scrollback)
        from .repl import run_repl
        run_repl(agent, on_permission=_confirm)
        return 0
    finally:
        agent.close()  # tear down the sandbox container, if any


if __name__ == "__main__":
    sys.exit(main())
