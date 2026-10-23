"""Write (create or overwrite) a file."""

from __future__ import annotations

import os

from .base import Tool


class WriteFileTool(Tool):
    parallel_safe = False    # mutates the workspace or agent state
    name = "write_file"
    description = (
        "Write content to a file, creating it (and parent dirs) or overwriting it. "
        "Use edit_file for small changes to existing files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to write."},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["file_path", "content"],
    }

    def run(self, file_path: str, content: str) -> str:
        try:
            parent = os.path.dirname(os.path.abspath(file_path))
            os.makedirs(parent, exist_ok=True)
            existed = os.path.exists(file_path)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:  # noqa: BLE001
            return f"Error writing {file_path}: {e}"
        n = content.count("\n") + 1
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {file_path} ({n} lines)."
