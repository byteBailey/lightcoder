"""LightCoder — a minimal, hackable coding agent and SWE-bench ablation testbed.

The agent's load-bearing design choices are first-class toggles on ``AgentConfig``
so that ablating a capability or adding an optimization is a one-line config change,
measurable through the same evaluation pipeline.
"""

from .config import AgentConfig, LLMConfig
from .llm import LLM, LLMResponse, ToolCall
from .agent import Agent

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "LLMConfig",
    "LLM",
    "LLMResponse",
    "ToolCall",
    "__version__",
]
