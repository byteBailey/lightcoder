"""Run a shell command, with a small guardrail against obviously destructive ones."""

from __future__ import annotations

import re
import subprocess

from .base import Tool

# Patterns we refuse outright. Not a security boundary — a footgun guard so the
# agent can't trivially wipe the host while iterating.
_BLOCKED = [
    r"\brm\s+-rf\s+/(?:\s|$)",
    r"\brm\s+-rf\s+~(?:/\s*)?$",
    r":\(\)\s*\{\s*:\|:&\s*\}",  # fork bomb
    r"\bmkfs\b",
    r"\bdd\s+if=.*of=/dev/",
    r">\s*/dev/sd[a-z]",
    r"\bshutdown\b|\breboot\b",
    # Package installs mutate the host environment we run in (the agent shares our venv
    # in this eval setup) — block them so a run can't corrupt the interpreter/deps.
    r"\b(?:pip[0-9]?|pip3)\s+install\b",
    r"\bpython[0-9.]*\s+-m\s+pip\s+install\b",
    r"\buv\s+(?:pip\s+install|add)\b",
    r"\b(?:conda|mamba)\s+install\b",
    r"\beasy_install\b",
]
_MAX_OUTPUT = 30_000


class BashTool(Tool):
    parallel_safe = False    # mutates the workspace or agent state
    name = "run_bash"
    description = (
        "Run a shell command in the working directory and return combined "
        "stdout/stderr. Use for building, running tests, git, and file inspection."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)."},
        },
        "required": ["command"],
    }

    # Injected by the parent Agent when an OS-level sandbox is active (see lightcoder/
    # sandbox.py). When set, commands run inside the sandbox instead of on the host; the
    # footgun guard and output formatting below apply to both paths.
    _sandbox = None  # type: ignore[assignment]

    def run(self, command: str, timeout: int = 120) -> str:
        for pat in _BLOCKED:
            if re.search(pat, command):
                return f"Error: command blocked by safety guard (matched {pat!r})."
        try:
            if self._sandbox is not None:
                out, code = self._sandbox.run(command, timeout)
            else:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                out, code = (proc.stdout or "") + (proc.stderr or ""), proc.returncode
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s."
        except Exception as e:  # noqa: BLE001
            return f"Error running command: {e}"

        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + f"\n... (output truncated at {_MAX_OUTPUT} chars)"
        status = f"(exit code {code})"
        return f"{out}\n{status}" if out.strip() else f"(no output) {status}"
