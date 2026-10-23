"""Append-only session transcripts, so a conversation survives the process exiting.

One JSONL file per session under ``~/.lightcoder/sessions/<project-slug>/<id>.jsonl``
(see :func:`lightcoder.settings.sessions_dir`). Each line is one record: the first is a
header describing the session, the rest are messages exactly as they entered the
conversation.

Two deliberate choices:

* **Transcripts live in the global home, not the project.** A transcript can contain file
  contents and credentials the agent read during a run, so it must never be committed.
* **The log is never compacted.** Context compression rewrites the *in-memory* history to
  fit the window; the file keeps the full record, and a resumed session simply re-compacts
  as it grows again.

Every record is appended and flushed on its own, so an interrupted run keeps everything up
to the moment it died, and a half-written final line is skipped on read.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path

from .settings import sessions_dir

_ID_TIME_FMT = "%Y%m%d-%H%M%S"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SessionLog:
    """A single transcript file, opened for append."""

    def __init__(self, path: Path, session_id: str):
        self.path = Path(path)
        self.id = session_id

    # ------------------------------------------------------------- creation
    @classmethod
    def create(cls, cwd=None, **meta) -> "SessionLog":
        """Start a new transcript for *cwd* and write its header."""
        directory = sessions_dir(cwd)
        directory.mkdir(parents=True, exist_ok=True)
        sid = f"{datetime.now().strftime(_ID_TIME_FMT)}-{secrets.token_hex(3)}"
        log = cls(directory / f"{sid}.jsonl", sid)
        log.write("session", id=sid, cwd=str(Path(cwd or Path.cwd()).resolve()), **meta)
        return log

    @classmethod
    def reopen(cls, path) -> "SessionLog":
        """Continue an existing transcript (a resumed session keeps one file)."""
        p = Path(path)
        log = cls(p, p.stem)
        log.write("resumed")
        return log

    # -------------------------------------------------------------- lookup
    @classmethod
    def list(cls, cwd=None) -> list[Path]:
        """Transcripts for this project, oldest first (ids sort chronologically)."""
        directory = sessions_dir(cwd)
        return sorted(directory.glob("*.jsonl")) if directory.is_dir() else []

    @classmethod
    def latest(cls, cwd=None) -> Path | None:
        found = cls.list(cwd)
        return found[-1] if found else None

    @classmethod
    def find(cls, ident: str, cwd=None) -> Path | None:
        """Resolve a path, a full session id, or a unique id prefix/suffix."""
        direct = Path(ident)
        if direct.is_file():
            return direct
        candidates = cls.list(cwd)
        for f in candidates:
            if f.stem == ident:
                return f
        partial = [f for f in candidates if ident in f.stem]
        return partial[-1] if partial else None

    @classmethod
    def read(cls, path) -> tuple[dict, list[dict]]:
        """Return ``(header, messages)``; malformed/partial lines are skipped."""
        header: dict = {}
        messages: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a run killed mid-write leaves at most one partial line
                kind = rec.get("type")
                if kind == "session":
                    header = rec
                elif kind == "message" and isinstance(rec.get("message"), dict):
                    messages.append(rec["message"])
                elif kind == "clear":          # /clear: resume from the fresh start
                    messages.clear()
        return header, messages

    # -------------------------------------------------------------- writing
    def write(self, kind: str, **fields) -> None:
        """Append one record. Never raises — logging must not break a run."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"type": kind, "ts": _now(), **fields},
                               ensure_ascii=False, default=str) + "\n"
                )
        except OSError:
            pass

    def message(self, msg: dict) -> None:
        self.write("message", message=msg)


def describe(path) -> str:
    """One-line summary of a transcript for `--list-sessions`."""
    header, messages = SessionLog.read(path)
    first = next((m.get("content") for m in messages
                  if m.get("role") == "user" and m.get("content")), "")
    first = " ".join(str(first).split())[:60]
    return (f"{Path(path).stem}  {len(messages):>4} msgs  "
            f"{header.get('model', '?')[:28]:28s}  {first}")
