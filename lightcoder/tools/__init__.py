"""Tool registry and per-config tool assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool
from .read import ReadFileTool
from .write import WriteFileTool
from .edit import EditFileTool
from .bash import BashTool
from .grep import GrepTool
from .glob_tool import GlobTool
from .agent import AgentTool
from .todo import TodoTool

if TYPE_CHECKING:
    from ..config import AgentConfig

# Tools every agent gets, regardless of config.
CORE_TOOLS: list[type[Tool]] = [
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    BashTool,
    GrepTool,
    GlobTool,
]


def build_tools(config: "AgentConfig") -> list[Tool]:
    """Instantiate the tool set implied by a config.

    The sub-agent tool is included only when ``config.subagent`` is on — this is
    how the 'subagent' ablation arm removes the capability entirely. Optimization
    modules named in ``config.extras`` can register additional tools here later.
    """
    tools: list[Tool] = [cls() for cls in CORE_TOOLS]
    if config.subagent:
        tools.append(AgentTool())
    if "todo" in config.extras:          # persistent goal/plan anti-drift module
        tools.append(TodoTool())
    return tools


__all__ = [
    "Tool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "BashTool",
    "GrepTool",
    "GlobTool",
    "AgentTool",
    "TodoTool",
    "CORE_TOOLS",
    "build_tools",
]
