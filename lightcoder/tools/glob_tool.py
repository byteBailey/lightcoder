"""List files matching a glob pattern."""

from __future__ import annotations

import glob as _glob

from .base import Tool

_MAX_FILES = 300


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files by glob pattern (e.g. '**/*.py', 'src/**/test_*.py'). "
        "Returns matching paths."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, '**' allowed."},
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str) -> str:
        try:
            matches = _glob.glob(pattern, recursive=True)
        except Exception as e:  # noqa: BLE001
            return f"Error in glob: {e}"
        matches = [m for m in matches if "/.git/" not in m and "/__pycache__/" not in m]
        matches.sort()
        if not matches:
            return "(no files match)"
        if len(matches) > _MAX_FILES:
            extra = len(matches) - _MAX_FILES
            matches = matches[:_MAX_FILES] + [f"... ({extra} more)"]
        return "\n".join(matches)
