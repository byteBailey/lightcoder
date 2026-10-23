"""Read a file (optionally a line range)."""

from __future__ import annotations

import os

from .base import Tool

_MAX_BYTES = 100_000


def _as_int(v):
    """Coerce a possibly-stringy ('50') or None argument to int (or None)."""
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a text file. Optionally pass offset/limit to read a line range."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file."},
            "offset": {"type": "integer", "description": "1-based first line to read."},
            "limit": {"type": "integer", "description": "Max number of lines to read."},
        },
        "required": ["file_path"],
    }

    def run(self, file_path: str, offset=None, limit=None) -> str:
        offset, limit = _as_int(offset), _as_int(limit)
        if not os.path.exists(file_path):
            return f"Error: file not found: {file_path}"
        if os.path.isdir(file_path):
            return f"Error: {file_path} is a directory, not a file."
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:  # noqa: BLE001
            return f"Error reading {file_path}: {e}"

        start = (offset - 1) if offset else 0
        start = max(start, 0)
        end = (start + limit) if limit else len(lines)
        chunk = lines[start:end]

        out, size = [], 0
        for i, line in enumerate(chunk, start=start + 1):
            out.append(f"{i}\t{line.rstrip(chr(10))}")
            size += len(line)
            if size > _MAX_BYTES:
                out.append(f"... (truncated at {_MAX_BYTES} bytes)")
                break
        return "\n".join(out) if out else "(empty file)"
