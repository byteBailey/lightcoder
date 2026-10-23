# M4 — fix truncation-induced empty patches, and measure it honestly

## The bug (found by M2/M3, confirmed on both models)

A verbose model can hit `max_tokens` mid-reasoning, emit a cut-off message **with no tool
call**, and the agent loop treats "no tool call" as "done" → it stops mid-thought with no
edit → empty patch. Confirmed:
- 27B `pytest-5103`: last message ≈4096 tokens, ends `"...1. Iterate through"`, no tool call.
- 397B `pytest-11148`: ends `"...affect \`pmx"`.

## The fix

1. `llm.py`: capture `finish_reason`; treat a very large `max_tokens` as "uncapped" (omit
   it so vLLM uses the full remaining context instead of erroring on `prompt+max_tokens>ctx`).
2. `agent.py`: opt-in `continue_on_truncation` — on a no-tool-call message that was cut off
   (`finish_reason == "length"`), nudge the model to continue instead of terminating.
3. `bash.py`: block `pip/conda/uv install` (the agent shares our venv and had been corrupting
   it — a real sandboxing gap surfaced while debugging this).

`max_tokens` note: literal `262144` (unbounded) made the verbose 397B ramble — one instance
didn't finish in 8.5 min. So runs use a **large bounded cap (32768)**: no truncation, no
runaway.

## Result (stable 27B endpoint, same 20-instance subset)

| 配置 | 解决 | 解决率% | 平均patch字符 |
|---|---|---|---|
| baseline (max_tokens=4096) | 8/20 | 40.0 | 1318 |
| +continue_on_truncation (max_tokens=32768) | 6/20 | 30.0 | 1495 |

Per-instance vs baseline: **+flask-4045** (rescued), **−pytest-11148** (endpoint timeout,
infra — not the fix), **−seaborn-3407, −xarray-4094**. `pytest-5103` now emits a 4869-char
patch instead of nothing, but the patch is wrong, so it still fails.

## Honest conclusion

**The fix works at the mechanism level but does not improve the headline metric.**
- It does what it claims: truncated cases now finish and produce a patch (flask-4045 rescued,
  pytest-5103 no longer empty).
- But raising `max_tokens` is **double-edged**: the extra budget helps truncated cases yet
  lets the model *over-think* others (seaborn-3407, xarray-4094 lost, bigger avg patch).
- Net 8→6 (≈7 excluding the unrelated timeout) is within n=20 noise and slightly negative.

This is the mature takeaway: a targeted fix that cleanly addresses one failure mode can trade
it for another, and on a 20-instance subset the aggregate move is within noise. The
**discipline of measuring it** (rather than reporting only "flask-4045 rescued!") is the point.

## What would actually move the needle (future work)

- A **more conservative cap** (e.g. 8192): enough to kill the ~4096 truncation without
  inviting over-thinking. Can't be distinguished from noise at n=20 — needs a bigger subset.
- **In-container execution** (the recurring theme): let the agent run/verify in the correct
  environment so it stops guessing — far higher expected leverage than token-budget tuning.
- pytest-5103 shows truncation was only a *symptom* on hard tasks; the real gap is solution
  quality, which token budget alone doesn't fix.
