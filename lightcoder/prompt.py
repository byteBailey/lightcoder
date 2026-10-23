"""System prompt construction.

The prompt is itself an ablation axis: ``AgentConfig.prompt_variant`` selects one
of several instruction styles so we can measure how much the wording matters. All
variants share the same environment header and tool list; they differ in how much
behavioral guidance they give.
"""

from __future__ import annotations

import os
import platform


def _header(tools) -> str:
    uname = platform.uname()
    tool_lines = "\n".join(f"- **{t.name}**: {t.description}" for t in tools)
    return (
        f"# Environment\n"
        f"- Working directory: {os.getcwd()}\n"
        f"- OS: {uname.system} {uname.release} ({uname.machine})\n"
        f"- Python: {platform.python_version()}\n\n"
        f"# Tools\n{tool_lines}\n"
    )


_DEFAULT_RULES = """\
# Rules
1. Read before you edit. Always read a file before modifying it.
2. Prefer `edit_file` for targeted changes; use `write_file` only for new files or full rewrites.
3. `edit_file` needs a unique `old_string` — include enough surrounding context to match exactly once.
4. Verify your work: run the relevant tests or commands after changing code.
5. Be concise. Show code over prose; explain only what matters.
6. Respect the project's existing style and conventions.
7. When the task is done, stop and reply with a short summary (no tool call).
"""

_VERBOSE_RULES = _DEFAULT_RULES + """\
8. Think step by step before acting: state the smallest next action, take it, observe, then continue.
9. If a command or test fails, read the error carefully and fix the root cause rather than guessing.
10. Never fabricate file contents or command output — always use a tool to obtain ground truth.
11. Keep changes minimal and localized; do not refactor unrelated code.
12. If the request is ambiguous and you cannot proceed safely, ask one clarifying question.
"""

_REACT_RULES = """\
# How to work
Operate in an explicit Reason-Act loop. Before each tool call, briefly state your
reasoning (one or two sentences) about what you will do and why; then make the call.
After observing the result, decide the next action. Read files before editing them,
verify changes by running tests, and finish with a short summary when done.
"""

# Appended when the 'verify' extra is on (execution-guided self-verification). This is the
# execution-coupled half of the scaffolding: the agent must *run* its fix, not just inspect
# it. In eval this executes inside the task's container (via the sandbox) so reproductions
# have the project's deps; interactively it runs wherever run_bash runs.
_VERIFY_WORKFLOW = """\
# Fix-then-verify workflow (required)
Make the fix FIRST, then prove it works by executing it. Do NOT sink your budget into
reproduction before you have edited the code.
1. Locate the root cause (read/grep the relevant source).
2. Fix it NOW with `edit_file` — before writing any reproduction. Always leave your edit in
   place; never finish without having edited the source.
3. Verify: write ONE minimal reproduction and run it with `run_bash` to confirm the fix
   works; run the nearest existing test too if it's quick. If it still fails, refine the fix
   (keep an edit in place) — do NOT write many reproduction variants or keep exploring.
4. Finish once verified, or once you've made your best-supported fix. State what you changed
   and what you ran.
"""


def system_prompt(tools, variant: str = "default", extras=()) -> str:
    intro = (
        "You are LightCoder, an AI coding assistant running in the user's terminal. "
        "You help with software engineering: writing code, fixing bugs, refactoring, "
        "running commands, and explaining code."
    )
    header = _header(tools)

    if variant == "minimal":
        # Just identity + environment + tools, no behavioral rules.
        base = f"{intro}\n\n{header}"
    elif variant == "verbose":
        base = f"{intro}\n\n{header}\n{_VERBOSE_RULES}"
    elif variant == "react":
        base = f"{intro}\n\n{header}\n{_REACT_RULES}"
    else:  # default
        base = f"{intro}\n\n{header}\n{_DEFAULT_RULES}"

    if "verify" in extras:
        base = f"{base}\n{_VERIFY_WORKFLOW}"
    return base
