"""Permission / approval layer — gate risky tool calls before they run.

Four modes, cautious -> hands-off (Claude-Code-style):
  - plan     : read-only. Any file write or command execution is denied (for planning).
  - default  : ask before edits and command execution; safe reads/searches run freely.
  - auto     : auto-accept file edits and routine shell commands; still ask for anything
               flagged dangerous.
  - yolo     : run everything without asking (the non-interactive / eval default).

On top of the mode: a dangerous-command detector that forces confirmation even in 'auto',
plus optional user allow/deny substring rules for shell commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MODES = ("plan", "default", "auto", "yolo")

# Tools grouped by risk.
_READ_ONLY = {"read_file", "grep", "glob", "update_todo"}
_EDIT = {"write_file", "edit_file"}
_EXEC = {"run_bash"}

# Shell commands that always require explicit confirmation (destructive / exfiltrating).
_DANGEROUS = [
    r"\brm\s+-[a-z]*[rf]",                  # rm -rf / -r / -f
    r"\bsudo\b", r"\bmkfs\b", r"\bdd\s+if=", r"\bshred\b",
    r":\s*\(\)\s*\{",                       # fork bomb
    r">\s*/dev/(sd|nvme|null|zero)?",       # redirect over devices
    r"\bchmod\s+-R\s+0*777\b",
    r"\bgit\s+push\b.*(--force|-f)\b",
    r"(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh)\b",   # curl ... | sh
    r">\s*/etc/", r"\b(shutdown|reboot|halt|poweroff)\b",
    r"\bchown\s+-R\b",
]
_DANGEROUS_RE = re.compile("|".join(_DANGEROUS), re.IGNORECASE)


def is_dangerous(command: str) -> bool:
    return bool(_DANGEROUS_RE.search(command or ""))


@dataclass
class Decision:
    kind: str            # "allow" | "deny" | "ask"
    reason: str = ""     # deny reason, or the confirmation prompt for "ask"
    dangerous: bool = False


@dataclass
class PermissionPolicy:
    mode: str = "yolo"
    allow: tuple[str, ...] = ()      # shell substrings always allowed (skip the prompt)
    deny: tuple[str, ...] = ()       # shell substrings always denied

    def check(self, tool: str, args: dict) -> Decision:
        if tool in _READ_ONLY:                       # reads/search are always safe
            return Decision("allow")
        if self.mode == "yolo":
            return Decision("allow")
        if self.mode == "plan":
            return Decision("deny", f"plan mode is read-only — {tool} is blocked")

        if tool in _EXEC:
            cmd = str(args.get("command", ""))
            if any(d in cmd for d in self.deny):
                return Decision("deny", "matches a configured deny rule")
            danger = is_dangerous(cmd)
            if not danger and any(a in cmd for a in self.allow):
                return Decision("allow")
            if self.mode == "auto" and not danger:   # auto: routine commands run
                return Decision("allow")
            tag = " [DANGEROUS]" if danger else ""
            return Decision("ask", f"Run shell command{tag}: {cmd[:200]}", dangerous=danger)

        if tool in _EDIT:
            if self.mode == "auto":                  # acceptEdits: edits auto-approved
                return Decision("allow")
            return Decision("ask", f"{tool} {args.get('file_path', '')}")

        return Decision("allow")                      # delegate / other -> allow
