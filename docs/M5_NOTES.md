# M5 — bigger subset, execution-guided verification, and an external reference

M5 moves from the 20-instance smoke subset to 44 instances, tries three ways of spending
more effort per task (self-review, best-of-3, execution-guided verification), and calibrates
LightCoder against a real product, nanocoder, on the same models and instances.

## Setup

- **Subset** `eval/subsets/m5_44.json`: 44 SWE-bench Lite instances (pytest 17, sphinx 9,
  pylint 6, xarray 5, seaborn 4, flask 3). `m5_50.json` adds the 6 `psf/requests` instances,
  whose tests need the network; offline the harness errored on 5 of them in each 50-instance
  run, so all comparisons below use the 44.
- **Models**: Qwen3.5-27B on a local vLLM server, and DeepSeek-V4-Pro through its API.
  Temperature 0, one run per configuration.
- **nanocoder** (Nano-Collective, v1.28) runs headless (`--mode yolo --plain run`) in the same
  checkout of each instance; its `git diff` is scored by the same harness
  (`eval/run_nanocoder.py`).

## Results (44 instances)

| 运行 | 模型 | 解决 | 解决率% | 空patch | 平均调用 | 平均用时s |
|---|---|---|---|---|---|---|
| LightCoder baseline | 27B | 16/44 | 36.4 | 3 | 27.5 | 228 |
| + self-review | 27B | 16/44 | 36.4 | 1 | — | — |
| best-of-3 (temp 0.8, majority vote + LLM judge) | 27B | 14/44 | 31.8 | 0 | — | — |
| + verify v1 (reproduce first) | 27B | 13/44 | 29.5 | 7 | — | — |
| + verify v2 (fix first) | 27B | 14/44 | 31.8 | 5 | 36.7 | 241 |
| nanocoder | 27B | 17/44 | 38.6 | 1 | — | 420 |
| LightCoder baseline | DeepSeek-V4-Pro | 21/44 | 47.7 | 0 | 18.6 | 144 |
| + verify v2 (fix first) | DeepSeek-V4-Pro | **24/44** | **54.5** | 1 | 28.5 | 175 |
| nanocoder | DeepSeek-V4-Pro | 18/44 | 40.9 | 2 | — | 265 |

## Execution-guided verification (`--extras verify`)

The baseline edits a host checkout without the project's dependencies, so it can only read
its fix, never run it. `verify` changes three things (details in `eval/README.md`):

1. `run_bash` runs inside the instance's own SWE-bench container (the image the harness scores
   with, `--network none`), with the host checkout bind-mounted at `/testbed`, so a host edit
   is live in a Python that has the real dependencies.
2. The task prompt asks for a fix, then one minimal reproduction to confirm it.
3. A finish-time gate: if the model tries to finish without having run anything, it is told
   to verify first. The gate fires at most once per task.

The agent verifies against its own reproduction and the existing tests, never the hidden
FAIL_TO_PASS tests that do the scoring.

**v1 to v2.** v1 asked for reproduce → fix → re-run. On 27B the model spent its budget on
reproduction and finished without editing: 7 empty patches. v2 is fix-first (edit with
`edit_file` before writing any reproduction, and never finish without an edit), which brought
empty patches down to 5 on 27B and 1 on DeepSeek.

## What changed, instance by instance

- **DeepSeek, baseline → verify: +4 −1.** Gained seaborn-2848, pytest-5692, pytest-7220 and
  pytest-7490; lost pytest-5495.
- **27B, baseline → verify: +2 −4.** Gained seaborn-2848 and pytest-7490; lost xarray-3364,
  pytest-11148, pytest-7168 and sphinx-10325.
- **DeepSeek, nanocoder vs LightCoder baseline.** LightCoder solves 5 instances nanocoder
  misses (seaborn-3190, pylint-7114, pytest-11148, pytest-6116, sphinx-10451). nanocoder
  solves 2 that the baseline misses (seaborn-2848, pytest-5692), and verify recovers both.

## Conclusions

1. **The gain from a scaffold depends on the model.** Verification lifts the stronger model
   from 47.7% to 54.5%. On 27B it costs more tool calls and more empty patches, and the net
   change (−2) is within noise: the weaker model cannot use the extra loop well.
2. **The DeepSeek gain is suggestive, not significant.** Only 5 instances changed outcome (4
   gained, 1 lost); an exact sign test on those gives p ≈ 0.38. With n = 44 and one run per
   arm, +3 cannot be separated from noise. Confirming it needs more instances or several
   seeds per arm, as the 397B run showed (`M_397b_NOTES.md`).
3. **More samples or a second look did not help on 27B.** Self-review left the score at 16;
   best-of-3 dropped it to 14 at three times the cost. Neither is kept in the code.
4. **LightCoder is on par with a real product.** On 27B it is within one instance of
   nanocoder (16 vs 17). On DeepSeek the baseline is ahead (21 vs 18), and with verification
   24 vs 18.
