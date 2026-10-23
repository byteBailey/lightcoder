"""Two-tier configuration: a global home plus an optional per-project directory.

Layout (the convention pi and Claude Code independently converged on)::

    ~/.lightcoder/                    global, user-private, never committed
      settings.json                   defaults that apply in every project
      models.json     (chmod 600)     per-model endpoint + credentials, keyed by model name
      sessions/<slug>/*.jsonl         transcripts, bucketed by project directory

    <project>/.lightcoder/            project-scoped, safe to commit
      settings.json                   overrides for this repository

Precedence, highest first: **shell environment > project settings > global settings >
built-in defaults**.

Session transcripts deliberately live in the *global* directory: a transcript can
contain file contents and credentials read during a run, so it must never land in a
repository. Project settings, by contrast, are meant to be shared with the team.

Only the CLI merges these layers — ``AgentConfig``/``LLMConfig`` stay plain dataclasses
so the evaluation harness constructs configs that a user's personal settings cannot perturb.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

SETTINGS_FILE = "settings.json"
MODELS_FILE = "models.json"
DIR_NAME = ".lightcoder"

# models.json carries API keys inline, so it is written owner-only.
MODELS_MODE = 0o600


def global_dir() -> Path:
    """The user-level LightCoder home ($LIGHTCODER_HOME, else ~/.lightcoder)."""
    override = os.getenv("LIGHTCODER_HOME")
    return Path(override) if override else Path.home() / DIR_NAME


def project_dir(start: str | os.PathLike | None = None) -> Path | None:
    """Nearest ``.lightcoder/`` walking up from *start* (default: cwd), or None.

    Stops at $HOME so a stray directory above the user's home is never picked up, and
    never returns the global home itself (see the note in the body).
    """
    cur = Path(start or Path.cwd()).resolve()
    home = Path.home().resolve()
    # The global home is usually ~/.lightcoder, which sits on this walk. Skip it, or a
    # session started anywhere under $HOME would apply the global config twice — once as
    # "global" and again as "project".
    glob = global_dir().resolve()
    while True:
        candidate = cur / DIR_NAME
        if candidate.is_dir() and candidate.resolve() != glob:
            return candidate
        if cur == home or cur == cur.parent:
            return None
        cur = cur.parent


def _read_json(path: Path) -> dict:
    """Parse a JSON object, returning {} when absent or malformed (never fatal)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, data: dict, mode: int | None = None) -> None:
    """Write a JSON object, creating parents; ``mode`` (e.g. 0o600) applies to the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if mode is not None:
        os.chmod(path, mode)


def load_settings(start: str | os.PathLike | None = None) -> dict:
    """Global settings merged with (and overridden by) the project's settings."""
    merged = _read_json(global_dir() / SETTINGS_FILE)
    proj = project_dir(start)
    if proj is not None:
        merged.update(_read_json(proj / SETTINGS_FILE))
    return merged


def load_models() -> dict:
    """The global models.json: ``{"default": name, "models": {name: {...}}}``."""
    return _read_json(global_dir() / MODELS_FILE)


def model_names() -> list[str]:
    """Model names configured in models.json."""
    return sorted((load_models().get("models") or {}).keys())


def resolve_model(name: str | None = None) -> dict:
    """Resolve a configured model name to its endpoint configuration.

    Only names present in models.json are accepted: an unregistered name has no endpoint or
    credentials, so honouring it would silently aim at whatever the environment happened to
    supply — a trap rather than a convenience. Unknown names raise; with no name, the
    ``default`` is used.

    An entry's ``model`` field is the id actually sent to the API. Write it in every entry
    so the file says exactly what goes on the wire; it falls back to the key only so a
    hand-written entry without it still works.
    """
    data = load_models()
    models = data.get("models") or {}
    chosen = name or data.get("default")
    if not chosen:
        return {}

    entry = models.get(chosen)
    if entry is None:
        if not name:
            # A stale `default` pointing at a removed entry: fall back to the environment
            # rather than failing a session the user never asked to pin.
            return {}
        known = ", ".join(sorted(models)) or "(none configured)"
        raise ValueError(f"model {chosen!r} is not in models.json; configured: {known}")

    resolved = dict(entry)
    resolved.setdefault("model", chosen)
    resolved["name"] = chosen
    return resolved


def project_slug(cwd: str | os.PathLike | None = None) -> str:
    """Filesystem-safe bucket name for a project directory (used by session storage)."""
    path = str(Path(cwd or Path.cwd()).resolve())
    return re.sub(r"[^A-Za-z0-9]+", "-", path).strip("-") or "root"


def sessions_dir(cwd: str | os.PathLike | None = None) -> Path:
    """Where this project's transcripts live (global home, bucketed by project path)."""
    return global_dir() / "sessions" / project_slug(cwd)
