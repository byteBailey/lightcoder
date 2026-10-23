# M3 — OFAT ablation

Same 20-instance hermetic subset, Qwen3_5-27B, one knob flipped per arm (`eval/ablation.py`).

| 配置 | 解决率% | 空patch | 错误 | 平均调用 | 平均用时s | 平均patch字符 |
|---|---|---|---|---|---|---|
| baseline | **40.0** (8/20) | 1 | 0 | 34.4 | 302 | 1318 |
| no_compression | 40.0 (8/20) | 0 | 0 | 28.4 | 239 | 1248 |
| no_subagent | 40.0 (8/20) | 2 | 0 | 31.5 | 252 | 1061 |
| prompt_minimal | 40.0 (8/20) | 2 | 0 | 29.6 | 222 | 1068 |
| prompt_react | **30.0** (6/20) | 0 | 0 | 31.4 | 269 | 1648 |

## Findings

1. **The ReAct prompt *hurt* (−10 pts) — the only arm that moved the headline.** Forcing
   "reason-before-every-action" made the model produce larger, messier patches (avg 1648 c
   vs 1318) and solve fewer. For this model on these tasks, direct instruction beats imposed
   chain-of-thought verbosity. Counterintuitive and the most interesting single result.

2. **Compression and sub-agent don't move resolve here — and that's a real finding, not a
   null one.**
   - `no_subagent` resolved the **identical set** to baseline: the model never actually used
     the `delegate` tool on these small tasks, so removing it changed nothing.
   - `no_compression` matched baseline's rate while being **21% faster** (239 vs 302 s, 28.4
     vs 34.4 calls): on tasks this small the context never fills 128 k, so compression is
     **pure overhead**. Its value would only appear on long-horizon tasks.

3. **Equal resolve count ≠ equal resolved set — quantifying the noise floor.** Even arms that
   tied at 8/20 solved *different* instances:
   - `no_compression` vs baseline: 1-for-1 swap (lost seaborn-3407, gained pylint-7114).
   - `prompt_minimal` vs baseline: **2-for-2 swap** (lost flask-4992 + pytest-11148, gained
     flask-4045 + xarray-3364) — same rate, visibly different set.
   So on 20 instances, per-config swings of ~1-2 instances are within run-to-run / prompt
   noise. **Treat sub-2-instance deltas as noise; only the ReAct −4 is clearly signal.** This
   is the honest methodological takeaway: the testbed works, but a tighter read needs a bigger
   subset (or multi-seed) — exactly what the throughput section in M2_NOTES is about.

## Takeaways for M4

- The knobs that *look* important (compression, sub-agent) are inert on this workload;
  **prompt and core model behavior dominate.** So a fruitful optimization targets the agent's
  decision loop, not its scaffolding. The two concrete candidates the runs surfaced:
  - **nudge-to-continue**: the lone empty patch (pytest-5103) came from the loop terminating
    when the model emitted reasoning with no tool call (see M1/M2 notes) — a cheap fix to test.
  - **in-container execution**: let the agent run/verify in the correct env (the recurring
    theme), the highest-leverage change.
