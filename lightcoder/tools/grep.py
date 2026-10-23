"""Search file contents for a regex. Uses ripgrep when available, else pure Python."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from .base import Tool

_MAX_MATCHES = 200


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents for a regular expression and return matching "
        "file:line: text. Optionally restrict to a path and a glob."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "description": "Directory or file to search (default: cwd)."},
            "glob": {"type": "string", "description": "Optional filename glob, e.g. '*.py'."},
            "limit": {"type": "integer", "description": "Max number of matching lines to return."},
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str, path: str = ".", glob: str | None = None, limit=None) -> str:
        cap = _MAX_MATCHES
        if limit is not None:
            try:
                cap = max(1, min(int(limit), _MAX_MATCHES))
            except (TypeError, ValueError):
                cap = _MAX_MATCHES
        if shutil.which("rg"):
            return self._ripgrep(pattern, path, glob, cap)
        return self._python(pattern, path, glob, cap)

    def _ripgrep(self, pattern: str, path: str, glob: str | None, cap: int) -> str:
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", pattern, path]
        if glob:
            cmd[1:1] = ["--glob", glob]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except Exception as e:  # noqa: BLE001
            return f"Error running ripgrep: {e}"
        lines = (proc.stdout or "").splitlines()
        if not lines:
            return "(no matches)"
        if len(lines) > cap:
            extra = len(lines) - cap
            lines = lines[:cap] + [f"... ({extra} more matches)"]
        return "\n".join(lines)

    def _python(self, pattern: str, path: str, glob: str | None, cap: int) -> str:
        import fnmatch

        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"

        results = []
        targets = []
        if os.path.isfile(path):
            targets = [path]
        else:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules"}]
                for fn in files:
                    if glob and not fnmatch.fnmatch(fn, glob):
                        continue
                    targets.append(os.path.join(root, fn))

        for fp in targets:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            results.append(f"{fp}:{i}:{line.rstrip()}")
                            if len(results) >= cap:
                                results.append("... (match limit reached)")
                                return "\n".join(results)
            except (OSError, UnicodeDecodeError):
                continue
        return "\n".join(results) if results else "(no matches)"
