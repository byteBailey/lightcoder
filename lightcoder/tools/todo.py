"""Persistent plan/TODO tool — an anti-drift aid for long tasks.

On a long task the original goal gets buried deep in the transcript and the agent
drifts. This tool lets the model record and update a running plan/checklist; the Agent
pins that plan (plus the original goal) OUTSIDE the compacted history and re-injects it
at the end of every prompt, so the goal stays salient no matter how long the run gets.
Exposed only when ``"todo"`` is in ``AgentConfig.extras``.
"""

from __future__ import annotations

from .base import Tool


class TodoTool(Tool):
    parallel_safe = False    # mutates the workspace or agent state
    name = "update_todo"
    description = (
        "Record or update your plan and progress for the current task. Overwrites the "
        "previous plan. Use a short checklist, e.g. '[x] located the bug in foo.py', "
        "'[ ] write the fix', '[ ] verify'. This plan is pinned in context so you do not "
        "lose track of the goal on a long task — update it as you make progress."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The full, updated plan/checklist."},
        },
        "required": ["content"],
    }

    # Injected by the parent Agent after construction (mirrors AgentTool._spawn).
    _update = None  # type: ignore[assignment]

    def run(self, content: str) -> str:
        if self._update is None:
            return "Error: todo runtime not wired up."
        self._update(content)
        return "Plan updated."
