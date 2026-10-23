"""The agent loop.

    user message -> LLM(+tools) --tool calls?--> execute --> feed back --> loop
                                 --plain text?--> done, return to caller

Three behavioral choices in this loop are config-driven so the harness can ablate
them: whether multiple tool calls in one turn run in parallel, whether the context
is compressed, and whether a sub-agent tool is available.
"""

from __future__ import annotations

import concurrent.futures
import os
import time

from .config import AgentConfig
from .llm import LLM, ToolCall
from .tools import build_tools
from .tools.agent import AgentTool
from .tools.bash import BashTool
from .tools.todo import TodoTool
from .prompt import system_prompt
from .context import ContextManager
from .permissions import PermissionPolicy
from .sandbox import make_sandbox


# Pushed back at the model (the 'verify' extra) when it tries to finish without having run
# anything — it can't have verified a fix it never executed. Fires at most once per task.
_VERIFY_PUSH = (
    "[Verification required] You are about to finish, but you have not run anything to check "
    "your change. Make sure you have edited the source to fix the issue, then run a quick "
    "reproduction (or the nearest test) with run_bash to confirm the fix works. Do this now, "
    "then briefly state what you ran and the result and finish."
)


class Agent:
    def __init__(self, llm: LLM, config: AgentConfig | None = None, tools=None):
        self.llm = llm
        self.config = config or AgentConfig()
        # Top-level agents build their own tools (and own a sandbox); a sub-agent is handed
        # the parent's tool instances, so it must NOT build a second sandbox container.
        built_tools = tools is None
        self.tools = build_tools(self.config) if built_tools else tools
        self._by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        # Latest token accounting (for a UI context-usage bar). prompt_tokens of the most
        # recent call ≈ current conversation size sent to the model.
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        # Session totals across every call (for the UI footer): tokens sent, generated, and
        # the share of sent tokens the provider served from its prompt cache.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        # Wall time of each call in the last _run_tools batch (execution only — time spent
        # waiting on a permission prompt is excluded), aligned with its tool_calls.
        self.last_tool_seconds: list[float] = []
        # Persistent anti-drift state (the 'todo' extra): the original goal + a running plan,
        # kept OFF the transcript so compaction can't touch them, and re-injected each turn
        # (see _full_messages) to stay salient on long tasks.
        self.goal = ""
        self.todo = ""
        self.context = ContextManager(
            max_tokens=self.config.max_context_tokens,
            enabled=self.config.compression,
        )
        self._system = system_prompt(
            self.tools, self.config.prompt_variant, extras=self.config.extras
        )

        # OS-level sandbox for run_bash (only the top-level agent owns one — sub-agents
        # share the parent's already-wired BashTool, so re-creating it would spin up a
        # second container). None when sandbox == 'none' (host execution, incl. eval).
        self.sandbox = (
            make_sandbox(self.config.sandbox, image=self.config.sandbox_image,
                         workdir=os.getcwd())
            if built_tools else None
        )

        # Wire callbacks into tools that reach back into the Agent.
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._spawn = self._spawn_subagent
            elif isinstance(t, TodoTool):
                t._update = self._set_todo
            elif isinstance(t, BashTool) and built_tools:
                t._sandbox = self.sandbox

        # Permission / approval gate ('yolo' default -> non-interactive runs unaffected).
        self.permissions = PermissionPolicy(
            mode=self.config.permission_mode,
            allow=self.config.permission_allow,
            deny=self.config.permission_deny,
        )
        self._on_permission = None       # set per chat(): (tool, args, prompt) -> bool

        # Optional append-only transcript (see lightcoder/session.py). Attached by the CLI;
        # the eval harness leaves it None so measured runs write nothing.
        self.session = None

        # Execution-guided verification state (the 'verify' extra); reset each chat().
        self._ran_bash = False           # did the agent execute any shell command this task?
        self._verify_nudged = False      # has the finish-time verify gate already fired?

    # ------------------------------------------------------------------ loop
    def chat(self, user_input: str, on_token=None, on_tool=None, on_result=None,
             on_permission=None) -> str:
        """Process one user message; may span several LLM/tool rounds."""
        self._on_permission = on_permission
        self.goal = user_input          # anchor for the anti-drift reminder (see _full_messages)
        self._ran_bash = False
        self._verify_nudged = False
        self._record({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)

        for _ in range(self.config.max_rounds):
            resp = self.llm.chat(
                messages=self._full_messages(),
                tools=self._schemas(),
                on_token=on_token,
            )
            if resp.usage:
                self.last_prompt_tokens = resp.usage.get("prompt_tokens", self.last_prompt_tokens)
                self.last_completion_tokens = resp.usage.get("completion_tokens", 0)
                self.total_prompt_tokens += resp.usage.get("prompt_tokens", 0)
                self.total_completion_tokens += resp.usage.get("completion_tokens", 0)
                self.total_cached_tokens += resp.usage.get("cached_tokens", 0)

            if not resp.tool_calls:  # model wants to finish
                # Execution-guided verification: if it never ran anything, it can't have
                # verified its fix — push it once to reproduce-and-run before finishing.
                if ("verify" in self.config.extras and not self._verify_nudged
                        and not self._ran_bash):
                    self._verify_nudged = True
                    self._record(resp.message)
                    self._record({"role": "user", "content": _VERIFY_PUSH})
                    continue
                self._record(resp.message)
                return resp.content

            self._record(resp.message)
            results = self._run_tools(resp.tool_calls, on_tool)
            for tc, result, secs in zip(resp.tool_calls, results, self.last_tool_seconds):
                self._record(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
                if tc.name == "run_bash":
                    self._ran_bash = True
                if on_result:
                    on_result(tc.name, result, secs)

            self.context.maybe_compress(self.messages, self.llm)

        return "(reached max rounds without a final answer)"

    # --------------------------------------------------------------- helpers
    def _record(self, message: dict) -> None:
        """Append to the live history and mirror it into the session transcript.

        The transcript is written unconditionally as messages arrive, so an interrupted
        run keeps everything up to the moment it died — unlike the in-memory history,
        which compaction rewrites.
        """
        self.messages.append(message)
        if self.session is not None:
            self.session.message(message)

    def _full_messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self._system}] + self.messages
        # Anti-drift (the 'todo' extra): once the transcript has grown, re-inject the goal +
        # running plan at the END (recency) so a long run doesn't lose track of the objective.
        # Held on self (not in self.messages), so it survives compaction verbatim and never
        # accumulates — one fresh copy per call.
        if "todo" in self.config.extras and self.goal and len(self.messages) > 2:
            plan = self.todo.strip() or "(no plan yet — call update_todo to outline your steps)"
            msgs.append({
                "role": "user",
                "content": (
                    "[Reminder — stay on task]\n"
                    f"Goal: {self.goal}\n"
                    f"Plan / progress:\n{plan}\n"
                    "Do only what the goal requires; don't drift into unrelated changes. "
                    "When the goal is met, stop."
                ),
            })
        return msgs

    def _set_todo(self, content: str) -> None:
        self.todo = content

    def _schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def _run_tools(self, tool_calls: list[ToolCall], on_tool=None) -> list[str]:
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # Permission gate: resolved sequentially up front (a confirmation may prompt the
        # user), so interactive approval never races the parallel execution below.
        denials = [self._authorize(tc) for tc in tool_calls]  # None = allowed, else denial str
        self.last_tool_seconds = [0.0] * len(tool_calls)

        def run(i: int) -> str:
            if denials[i] is not None:
                return denials[i]
            t0 = time.perf_counter()
            try:
                return self._exec(tool_calls[i])
            finally:
                self.last_tool_seconds[i] = time.perf_counter() - t0

        if not self.config.parallel_tools:
            return [run(i) for i in range(len(tool_calls))]

        # Walk the calls in order, running consecutive parallel-safe ones together and any
        # mutating tool on its own. Running everything concurrently loses edits: edit_file
        # is a read-modify-write over the whole file, so two of them on one file clobber
        # each other — measured at 30% on a 2 KB file, usually while both report success.
        results: list[str] = [""] * len(tool_calls)
        batch: list[int] = []

        def flush() -> None:
            if not batch:
                return
            if len(batch) == 1:
                results[batch[0]] = run(batch[0])
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.max_parallel_tools
                ) as pool:
                    for i, result in zip(batch, pool.map(run, batch)):
                        results[i] = result
            batch.clear()

        for i, tc in enumerate(tool_calls):
            tool = self._by_name.get(tc.name)
            if getattr(tool, "parallel_safe", True):
                batch.append(i)
            else:
                flush()                 # queued reads finish before a mutation starts
                results[i] = run(i)
        flush()
        return results

    def _authorize(self, tc: ToolCall) -> "str | None":
        """Apply the permission policy. Returns None to allow, or a denial string."""
        d = self.permissions.check(tc.name, tc.arguments)
        if d.kind == "allow":
            return None
        if d.kind == "deny":
            return f"[Permission] Denied: {d.reason}"
        # 'ask' — needs a confirmation callback; without one, deny (safe default).
        if self._on_permission and self._on_permission(tc.name, tc.arguments, d.reason):
            return None
        return f"[Permission] User declined to run {tc.name}."

    def _exec(self, tc: ToolCall) -> str:
        tool = self._by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        try:
            return tool.run(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Error executing {tc.name}: {e}"

    def _spawn_subagent(self, task: str) -> str:
        # Fresh agent, same model, no nested delegation, tighter round budget.
        sub_cfg = self.config.with_(
            subagent=False,
            max_rounds=min(self.config.max_rounds, 20),
        )
        sub_tools = [t for t in self.tools if not isinstance(t, AgentTool)]
        sub = Agent(self.llm, config=sub_cfg, tools=sub_tools)
        return sub.chat(task)

    def settle_interrupted(self) -> None:
        """Answer tool calls left without a result by a Ctrl-C mid-turn.

        The assistant message carrying tool_calls is recorded before the tools run, so an
        interrupt can leave it unanswered — and the API rejects the next request outright.
        """
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                answered = {m.get("tool_call_id") for m in self.messages[i + 1:]
                            if m.get("role") == "tool"}
                for tc in msg["tool_calls"]:
                    if tc["id"] not in answered:
                        self._record({"role": "tool", "tool_call_id": tc["id"],
                                      "content": "(interrupted by the user before it finished)"})
                return
            if msg.get("role") == "user":
                return

    def reset(self) -> None:
        self.messages.clear()

    def attach_sandbox(self, sandbox) -> None:
        """Replace the sandbox and (re)wire it into BashTool. The eval runner uses this to
        point run_bash at the instance's task container (where the project's deps live) so
        execution-guided verification can actually run reproductions."""
        self.sandbox = sandbox
        for t in self.tools:
            if isinstance(t, BashTool):
                t._sandbox = sandbox

    def close(self) -> None:
        """Release owned resources (tear down the sandbox container). Idempotent."""
        if self.sandbox is not None:
            self.sandbox.close()

