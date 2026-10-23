"""Search-and-replace edit with a uniqueness guarantee.

The model supplies an ``old_string`` that must appear exactly once in the file;
we replace it with ``new_string`` and return a small unified diff. Requiring a
unique match is what makes blind edits safe.
"""

from __future__ import annotations

import difflib
import os

from .base import Tool


class EditFileTool(Tool):
    parallel_safe = False    # mutates the workspace or agent state
    name = "edit_file"
    description = (
        "Replace an exact, unique substring in a file. old_string must match exactly "
        "once (include surrounding context to disambiguate). Returns a diff."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace (unique)."},
            "new_string": {"type": "string", "description": "Replacement text."},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def run(self, file_path: str, old_string: str, new_string: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: file not found: {file_path}"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original = f.read()
        except Exception as e:  # noqa: BLE001
            return f"Error reading {file_path}: {e}"

        count = original.count(old_string)
        if count == 0:
            return (
                "Error: old_string not found. Read the file and copy the exact text "
                "(including whitespace)."
            )
        if count > 1:
            return (
                f"Error: old_string matched {count} times; it must be unique. Add more "
                "surrounding context."
            )

        updated = original.replace(old_string, new_string, 1)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated)
        except Exception as e:  # noqa: BLE001
            return f"Error writing {file_path}: {e}"

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=2,
        )
        diff_text = "".join(diff)
        if len(diff_text) > 4000:
            diff_text = diff_text[:4000] + "\n... (diff truncated)"
        return f"Edited {file_path}.\n{diff_text}"
