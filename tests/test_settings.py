"""Two-tier settings and model profiles: discovery, merge order, and precedence.

The contract under test: shell env > CLI flag > project settings > global settings >
built-in default, with the eval path (plain AgentConfig) untouched by any of it.
"""

import json
import os

import pytest

from lightcoder.settings import (
    global_dir,
    load_settings,
    model_names,
    project_dir,
    project_slug,
    resolve_model,
    sessions_dir,
    write_json,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated global home *and* cwd.

    Both are needed: project discovery walks upward from the working directory, so without
    an isolated cwd (and $HOME) the walk escapes into the developer's real ~/.lightcoder.
    """
    h = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LIGHTCODER_HOME", str(h))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(work)
    return h


# ------------------------------------------------------------------ discovery

def test_global_dir_honours_env(home):
    assert global_dir() == home


def test_project_dir_finds_nearest_and_stops_at_home(tmp_path, monkeypatch):
    proj = tmp_path / "repo" / "sub" / "deep"
    proj.mkdir(parents=True)
    (tmp_path / "repo" / ".lightcoder").mkdir()
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(proj)
    assert project_dir() == tmp_path / "repo" / ".lightcoder"


def test_project_dir_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    assert project_dir() is None


# ------------------------------------------------------------------ merge

def test_project_settings_override_global(home, tmp_path, monkeypatch):
    write_json(home / "settings.json", {"permission_mode": "plan", "max_rounds": 10})
    proj = tmp_path / "repo"
    (proj / ".lightcoder").mkdir(parents=True)
    write_json(proj / ".lightcoder" / "settings.json", {"max_rounds": 99})
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(proj)

    st = load_settings()
    assert st["max_rounds"] == 99         # project wins
    assert st["permission_mode"] == "plan"  # global still fills the rest


def test_malformed_settings_is_not_fatal(home):
    (home).mkdir(parents=True, exist_ok=True)
    (home / "settings.json").write_text("{ not json", encoding="utf-8")
    assert load_settings() == {}


# ------------------------------------------------------------------ models

def test_resolve_named_and_default_model(home):
    write_json(home / "models.json", {
        "default": "local-m",
        "models": {
            "local-m": {"base_url": "http://x/v1", "model": "/path/to/local-m",
                        "api_key": "dummy"},
            "remote-m": {"base_url": "https://y/v1", "api_key": "sk-inline"},
        },
    })
    assert resolve_model()["model"] == "/path/to/local-m"    # falls back to default
    assert resolve_model("remote-m")["model"] == "remote-m"  # key doubles as the id
    assert resolve_model("remote-m")["api_key"] == "sk-inline"  # credentials live inline
    assert model_names() == ["local-m", "remote-m"]


def test_unconfigured_name_is_rejected(home):
    """An unregistered name has no endpoint or key — honouring it would silently aim at
    whatever the environment supplied, so it must fail loudly instead."""
    write_json(home / "models.json", {"models": {"local-m": {"model": "m"}}})
    with pytest.raises(ValueError, match="not in models.json"):
        resolve_model("gpt-4o")


def test_stale_default_does_not_become_a_raw_id(home):
    write_json(home / "models.json", {"default": "removed", "models": {}})
    assert resolve_model() == {}      # fall back to the environment, not a bogus id


def test_no_models_file_resolves_to_nothing(home):
    assert resolve_model() == {}


# ------------------------------------------------------------------ sessions

def test_sessions_bucketed_by_project_path(home, tmp_path):
    slug = project_slug(tmp_path / "a b" / "repo")
    assert "/" not in slug and " " not in slug
    assert sessions_dir(tmp_path).parent == home / "sessions"


# ------------------------------------------------------------------ precedence

def _args(**over):
    from lightcoder.cli import _parse_args
    return _parse_args(over.pop("argv", []))


def test_cli_flag_beats_settings(home, monkeypatch):
    write_json(home / "settings.json", {"permission_mode": "plan", "max_rounds": 10})
    from lightcoder.cli import _build

    a = _build(_args(argv=["--mode", "yolo"]))
    assert a.config.permission_mode == "yolo"   # flag wins
    assert a.config.max_rounds == 10            # settings still supply the rest
    a.close()


def test_settings_beat_builtin_default(home):
    write_json(home / "settings.json", {"sandbox": "docker", "extras": ["todo"]})
    from lightcoder.cli import _build

    a = _build(_args(argv=[]))
    assert a.config.sandbox == "docker"
    assert a.config.extras == ("todo",)
    a.close()


def test_shell_env_beats_configured_model(home, monkeypatch):
    write_json(home / "models.json", {
        "default": "cfg-m",
        "models": {"cfg-m": {"base_url": "http://from-config/v1", "model": "m-config"}},
    })
    from lightcoder.config import LLMConfig

    assert LLMConfig.from_env().model == "m-config"
    monkeypatch.setenv("LIGHTCODER_MODEL", "m-shell")
    assert LLMConfig.from_env().model == "m-shell"    # ad-hoc export still wins


def test_eval_path_ignores_user_settings(home):
    """AgentConfig built directly (the eval harness path) must not read settings."""
    write_json(home / "settings.json", {"permission_mode": "plan", "sandbox": "docker"})
    from lightcoder import AgentConfig

    cfg = AgentConfig()
    assert cfg.permission_mode == "yolo" and cfg.sandbox == "none"


def test_global_home_is_not_treated_as_a_project_dir(tmp_path, monkeypatch):
    """~/.lightcoder sits on the upward walk; applying it as *project* config too would
    double-apply the global settings for anyone working under $HOME."""
    fake_home = tmp_path / "user"
    (fake_home / ".lightcoder").mkdir(parents=True)
    write_json(fake_home / ".lightcoder" / "settings.json", {"permission_mode": "plan"})
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: fake_home))
    monkeypatch.delenv("LIGHTCODER_HOME", raising=False)
    work = fake_home / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    assert global_dir() == fake_home / ".lightcoder"
    assert project_dir() is None                        # not mistaken for a project
    assert load_settings() == {"permission_mode": "plan"}  # applied once, as global
