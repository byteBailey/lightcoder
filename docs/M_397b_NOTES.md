# 397B model run — a confound caught (endpoint nondeterminism)

We re-ran the baseline + 4-arm ablation against a ~397B model on a shared endpoint
(a vLLM server with served-model-name `1`, 256K context) to compare against the 27B.

| 配置 (397B) | 解决率% | 平均用时s | 平均patch字符 |
|---|---|---|---|
| no_compression | 50.0 (10/20) | 86 | 1868 |
| prompt_react | 50.0 (10/20) | 94 | 2505 |
| prompt_minimal | 40.0 (8/20) | 95 | 3491 |
| no_subagent | 35.0 (7/20) | 107 | 3628 |
| baseline | 25.0 (5/20) | 108 | 2438 |

At face value this looks like "the strong model is hindered by scaffolding" (baseline
worst, removing knobs helps). **That interpretation is wrong** — the numbers are
noise-dominated, and we can prove it:

## The proof

`baseline` and `no_compression` differ *only* in the compression flag. We checked every
trajectory in both runs: **compression never fired in either** (the tasks never approached
the 128k threshold — 0/20 instances triggered a snip/summary/collapse). So the two configs
were **operationally identical**, yet scored **5/20 vs 10/20**.

A 25-point gap between two identical configurations can only come from **run-to-run
nondeterminism of the endpoint**: a shared vLLM server under concurrent load does continuous
batching, so even at `temperature=0` the same prompt can yield different tokens on different
calls. On n=20 that shows up as a **±~5-instance swing**.

## Consequence

- **The 397B per-config ablation is not interpretable** — its differences are within the
  noise band, not config effects. `baseline=25%` being worst is an unlucky draw.
- By contrast the **27B endpoint was far more stable** (M3): effectively-identical configs
  differed by ~1 instance, so the 27B ablation *is* readable (and there the one real signal —
  ReAct prompt hurting — held up).
- "397B union across arms = 13/20" is likewise inflated: 5 arms = 5 independent lucky draws,
  not a real capability ceiling.

## What this is actually worth

This is the most important *methodological* result in the project: **before trusting any
model/config comparison, verify the endpoint is deterministic enough for the sample size.**
The discipline of checking "did the knob I flipped even do anything?" (here: did compression
fire?) is what caught a confound that a naive read would have written up as a finding.

To get trustworthy 397B numbers you'd need: a dedicated/low-load endpoint, multiple seeds per
config (report mean±std), and/or a larger subset to shrink the CI below the effect size.

## Honest cross-model statement we *can* make

The 397B did solve several instances the 27B never solved in any run (flask-4045, pytest-5103,
pytest-5221, seaborn-2848, pylint-7114) — consistent with a more capable model — but the exact
resolve-rate ordering across configs is not reliable from this run.
