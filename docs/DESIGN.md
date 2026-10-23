# LightCoder — Design

LightCoder is a minimal coding agent that doubles as an **experiment testbed**: the
agent's load-bearing design choices are first-class config toggles, so ablating a
capability or adding an optimization is a one-line change, measured the same way
(SWE-bench resolve rate) every time.

## Architecture

```
                      ┌──────────────── AgentConfig (the knobs) ───────────────┐
                      │ parallel_tools · compression · subagent · prompt_variant │
                      │ max_rounds · max_context_tokens · extras[]               │
                      └──────────────────────────┬───────────────────────────────┘
                                                 │ drives
   user ──▶ Agent.chat ──▶ LLM.chat(messages, tools) ──┬─ plain text ─▶ done
              ▲                                          │
              │                                          └─ tool_calls ─▶ run tools ─▶ feed back
              │                                                              (parallel?│sequential)
              └────────────── ContextManager.maybe_compress (on?│off) ◀──────────────┘
```

| Module | Responsibility | Ablation surface |
|---|---|---|
| `config.py` | `LLMConfig` (where the model is) + `AgentConfig` (how the agent behaves) | **defines all knobs** |
| `llm.py` | OpenAI-compatible client → `LLMResponse{content, tool_calls, message}`; retry w/ backoff | provider via `base_url` |
| `tools/` | `Tool` ABC + read/write/edit/run_bash/grep/glob + `delegate` (sub-agent) | `build_tools(config)` includes `delegate` only if `subagent` |
| `prompt.py` | `system_prompt(tools, variant)` — default/minimal/verbose/react | `prompt_variant` |
| `context.py` | layered compression: snip → summarize → collapse | `compression` on/off |
| `agent.py` | the loop; parallel-vs-sequential tool exec; sub-agent spawning | `parallel_tools`, `max_rounds` |
| `cli.py` | one-shot (`-p`) / REPL; toggles exposed as flags | — |

Separating `LLMConfig` from `AgentConfig` is deliberate: the model is held **fixed**
across a sweep so any change in resolve rate is attributable to the agent design,
not the model.

## The four core ablations

Baseline = everything on, `default` prompt. One-factor-at-a-time (OFAT): flip one knob.

1. **parallel_tools** — when a turn emits ≥2 tool calls, run them concurrently vs serially.
2. **compression** — keep the context-compression layers vs let history grow until the model errors.
3. **subagent** — expose the `delegate` tool (isolated-context sub-tasks) vs not.
4. **prompt_variant** — `default` / `minimal` / `verbose` / `react`.

`AgentConfig.label()` names each run (`baseline`, `no_compression`, `prompt_react`, …)
so results land in `results/<label>/`.

## Optimizations (added later, same machinery)

An optimization is just a new capability behind a toggle in `extras`, e.g. an Aider-style
repo-map, retrieval-based file location, or a verify-and-retry loop. Each is measured as
`baseline` vs `baseline + <extra>` on the identical SWE-bench subset.

## Evaluation (M1+)

Per instance: clone repo @ base_commit → run LightCoder one-shot on the issue text (cwd =
repo) → `git diff` = the model patch → write `predictions.jsonl` → hand to the **official
`swebench` Docker harness** which applies the patch + hidden tests and reports resolved/not.
Same fixed, deterministic subset across every config for a fair comparison.

> Note on expectations: a ~27B open model on SWE-bench resolves a modest fraction, and a
> ~50-instance subset has wide confidence intervals. The project's value is the
> **methodology and the relative deltas between configs**, not an absolute SOTA number.
