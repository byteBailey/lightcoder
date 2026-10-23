"""AgentConfig toggle / label behavior."""

import pytest

from lightcoder.config import AgentConfig


def test_baseline_label():
    assert AgentConfig().label() == "baseline"


def test_single_toggle_labels():
    assert AgentConfig(compression=False).label() == "no_compression"
    assert AgentConfig(parallel_tools=False).label() == "no_parallel_tools"
    assert AgentConfig(subagent=False).label() == "no_subagent"
    assert AgentConfig(prompt_variant="minimal").label() == "prompt_minimal"


def test_extras_in_label():
    assert AgentConfig(extras=("repo_map",)).label() == "+repo_map"


def test_with_returns_copy():
    base = AgentConfig()
    other = base.with_(compression=False)
    assert base.compression is True
    assert other.compression is False


def test_invalid_prompt_variant_rejected():
    with pytest.raises(ValueError):
        AgentConfig(prompt_variant="nonsense")
