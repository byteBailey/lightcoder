"""Sub-agent delegation tool.

Delegating a self-contained sub-task to a fresh agent keeps the parent's context
clean: the child does the exploration/work in its own window and returns only a
short summary. Whether this tool is exposed at all is the ``subagent`` ablation.
"""

from __future__ import annotations

from .base import Tool


class AgentTool(Tool):
    parallel_safe = False    # mutates the workspace or agent state
    name = "delegate"
    description = (
        "Delegate a self-contained sub-task to a fresh sub-agent with its own context "
        "and tools (no nested delegation). Good for: researching part of a codebase, "
        "or an isolated multi-step change. Returns the sub-agent's summary."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "What the sub-agent should accomplish."},
        },
        "required": ["task"],
    }

    # Injected by the parent Agent after construction.
    _spawn = None  # type: ignore[assignment]

    def run(self, task: str) -> str:
        if self._spawn is None:
            return "Error: sub-agent runtime not wired up."
        try:
            result = self._spawn(task)
        except Exception as e:  # noqa: BLE001
            return f"Sub-agent error: {e}"
        if len(result) > 5000:
            result = result[:4500] + "\n... (sub-agent output truncated)"
        return f"[Sub-agent result]\n{result}"
