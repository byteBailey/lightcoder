"""Tool interface.

A tool is anything the model can call: it has a name, a description, a JSON-Schema
for its arguments, and a ``run`` method returning a text result. Keeping the surface
this small is what lets optimizations add capabilities as plain new tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):

    # Whether this tool may run concurrently with others in the same turn. Tools that
    # mutate the workspace must not: edit_file/write_file are read-modify-write over the
    # whole file, so two of them on one file lose an edit (measured: 30% of the time on a
    # 2 KB source file, and the tools still report success). Read-only tools stay parallel.
    parallel_safe = True
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments object

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return a text result (never raise for normal errors —
        return an explanatory string so the agent loop can react and continue)."""
        ...

    def schema(self) -> dict:
        """OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
