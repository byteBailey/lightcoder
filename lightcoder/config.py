"""Configuration objects.

Two concerns are deliberately split:

* ``LLMConfig`` — *which model and how to reach it*. Resolved from a shell export, the
  named model in ``~/.lightcoder/models.json``, or a project ``.env`` (in that order).
  Held fixed across an ablation sweep so only the agent design changes between runs.
* ``AgentConfig`` — *how the agent behaves*. Every field here is a knob the
  evaluation harness can flip to ablate a capability or enable an optimization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


def _dotenv_values() -> dict:
    """Values from the project's nearest ``.env``, **without** mutating ``os.environ``.

    Kept because a per-repository ``.env`` is a widely understood way to carry project
    secrets (and is what CI usually mounts). It sits at the bottom of the precedence chain,
    below the configured model, so it can only fill gaps.

    There is deliberately no *global* ``.env``: ``~/.lightcoder/`` (models.json +
    settings.json) is the single global configuration home, and a second one would just be
    a second place to look. See lightcoder/settings.py.

    Reading rather than exporting is what keeps a file-supplied value distinguishable from
    a real shell export, so ``LIGHTCODER_MODEL=... lightcoder`` still outranks both a saved
    configured model and a ``.env`` — including on repeated calls within one process.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}

    env_path = Path(".env")
    if not env_path.exists():
        cur, home = Path.cwd(), Path.home()
        while cur != home and cur != cur.parent:
            candidate = cur / ".env"
            if candidate.exists():
                env_path = candidate
                break
            cur = cur.parent
    if not env_path.exists():
        return {}
    return {k: v for k, v in dotenv_values(env_path).items() if v is not None}


@dataclass
class LLMConfig:
    """Where the model lives and how to sample from it."""

    model: str = "gpt-4o"
    api_key: str = "dummy"
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 4

    @classmethod
    def from_env(cls, model: str | None = None) -> "LLMConfig":
        """Resolve the endpoint, highest precedence first:

        1. a real shell export — ``LIGHTCODER_MODEL=... lightcoder`` always wins,
        2. the named model in ``~/.lightcoder/models.json`` (or its default); an unlisted
           name passes through as a literal model id,
        3. a project ``.env``, then the ``OPENAI_*`` compatibility fallbacks.
        """
        from .settings import resolve_model

        prof = resolve_model(model)
        dotenv = _dotenv_values()

        def pick(key: str, field: str, *fallbacks: str, default=None):
            if key in os.environ:
                return os.environ[key]
            if prof.get(field) not in (None, ""):
                return prof[field]
            if key in dotenv:
                return dotenv[key]
            for fk in fallbacks:
                if fk in os.environ:
                    return os.environ[fk]
                if fk in dotenv:
                    return dotenv[fk]
            return default

        return cls(
            model=pick("LIGHTCODER_MODEL", "model", "OPENAI_MODEL", default="gpt-4o"),
            api_key=pick("LIGHTCODER_API_KEY", "api_key", "OPENAI_API_KEY", default="dummy"),
            base_url=pick("LIGHTCODER_BASE_URL", "base_url", "OPENAI_BASE_URL"),
            temperature=float(pick("LIGHTCODER_TEMPERATURE", "temperature", default="0")),
            max_tokens=int(pick("LIGHTCODER_MAX_TOKENS", "max_tokens", default="4096")),
            timeout=float(pick("LIGHTCODER_TIMEOUT", "timeout", default="120")),
        )


# Prompt variants the ablation can choose between. Implemented in prompt.py.
PROMPT_VARIANTS = ("default", "minimal", "verbose", "react")


@dataclass
class AgentConfig:
    """The agent's behavioral knobs — the surface the testbed ablates over.

    Defaults define the *baseline* agent (everything on, default prompt). An
    ablation flips exactly one field; an optimization adds a name to ``extras``.
    """

    # --- core ablation toggles ---
    parallel_tools: bool = True          # run multiple tool calls in a turn concurrently
    compression: bool = True             # enable layered context compression
    subagent: bool = True                # expose the sub-agent delegation tool
    prompt_variant: str = "default"      # which system-prompt template to use

    # --- loop / context limits ---
    max_rounds: int = 50                 # max LLM<->tool rounds before giving up
    max_context_tokens: int = 128_000    # budget that drives compression thresholds
    max_parallel_tools: int = 8          # thread-pool width when parallel_tools is on

    # --- optimization modules (M4+) ---
    # Names of optional features enabled on top of the baseline. Each maps to a
    # module that registers extra tools or hooks. Empty == pure baseline.
    extras: tuple[str, ...] = ()

    # --- permission / approval (interactive safety) ---
    # Default 'yolo' keeps non-interactive runs (eval, tests) unblocked; the CLI sets
    # 'default' so it asks before edits/commands. See lightcoder/permissions.py.
    permission_mode: str = "yolo"        # plan | default | auto | yolo
    permission_allow: tuple[str, ...] = ()   # shell substrings always allowed
    permission_deny: tuple[str, ...] = ()    # shell substrings always denied

    # --- sandbox (OS-level confinement for run_bash) ---
    # 'none' runs commands on the host (default, incl. eval). 'docker' execs them inside a
    # --network none container with the workspace bind-mounted. See lightcoder/sandbox.py.
    sandbox: str = "none"                # none | docker
    sandbox_image: str = "python:3.11-slim"  # image used when sandbox == 'docker'

    def __post_init__(self) -> None:
        if self.prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"unknown prompt_variant {self.prompt_variant!r}; "
                f"choose from {PROMPT_VARIANTS}"
            )
        from .permissions import MODES
        if self.permission_mode not in MODES:
            raise ValueError(
                f"unknown permission_mode {self.permission_mode!r}; choose from {MODES}"
            )
        from .sandbox import SANDBOX_MODES
        if self.sandbox not in SANDBOX_MODES:
            raise ValueError(
                f"unknown sandbox {self.sandbox!r}; choose from {SANDBOX_MODES}"
            )

    def with_(self, **changes) -> "AgentConfig":
        """Return a copy with fields overridden (handy for building sweeps)."""
        return replace(self, **changes)

    def label(self) -> str:
        """Short, filesystem-safe id describing how this config differs from baseline."""
        base = AgentConfig()
        diffs = []
        for f in ("parallel_tools", "compression", "subagent"):
            if getattr(self, f) != getattr(base, f):
                diffs.append(f"{'' if getattr(self, f) else 'no_'}{f}")
        if self.prompt_variant != base.prompt_variant:
            diffs.append(f"prompt_{self.prompt_variant}")
        for x in self.extras:
            diffs.append(f"+{x}")
        return "baseline" if not diffs else "__".join(diffs)
