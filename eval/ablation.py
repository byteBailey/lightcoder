"""Run an OFAT ablation: the same subset under several agent configs, then compare.

Each arm is one `eval.batch` run (generate + harness) on the *same* fixed subset, so
the only thing that varies is the agent design. Arms run serially (they share the LLM
endpoint and the Docker daemon). An arm whose summary.json already exists is skipped, so
this is resumable and the already-computed baseline is reused for free.

    uv run python -m eval.ablation
    uv run python -m eval.ablation --arms baseline no_subagent prompt_react

Final output: a comparison table (resolve% + tool-calls/wall/empty per config) printed
and written to docs/ablation_table.md.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .report import _row, render

# arm name -> extra batch flags. The name matches AgentConfig.label() so results land
# in results/<name>/ and the report groups them correctly.
ARMS = {
    "baseline": [],
    "no_compression": ["--no-compression"],
    "no_subagent": ["--no-subagent"],
    "no_parallel_tools": ["--no-parallel"],
    "prompt_minimal": ["--prompt-variant", "minimal"],
    "prompt_react": ["--prompt-variant", "react"],
    "prompt_verbose": ["--prompt-variant", "verbose"],
}

DEFAULT_ARMS = ["baseline", "no_compression", "no_subagent", "prompt_minimal", "prompt_react"]


def run_arm(name: str, subset: str, gen_workers: int, harness_workers: int,
            max_rounds: int, suffix: str = "") -> int:
    run_id = name + suffix
    if os.path.exists(os.path.join("results", run_id, "summary.json")):
        print(f"\n===== 臂 {run_id}: 已完成,跳过 =====", flush=True)
        return 0
    flags = ARMS[name]
    cmd = [
        sys.executable, "-u", "-m", "eval.batch",
        "--subset", subset, "--run-id", run_id,
        "--gen-workers", str(gen_workers),
        "--harness-workers", str(harness_workers),
        "--max-rounds", str(max_rounds),
        *flags,
    ]
    print(f"\n===== 臂 {run_id}: {' '.join(flags) or '(baseline)'} =====", flush=True)
    return subprocess.run(cmd).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OFAT ablation over agent configs.")
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS,
                    help=f"Subset of: {list(ARMS)}")
    ap.add_argument("--subset", default="eval/subsets/m2_small20.json")
    ap.add_argument("--gen-workers", type=int, default=3)
    ap.add_argument("--harness-workers", type=int, default=6)
    ap.add_argument("--max-rounds", type=int, default=75)
    ap.add_argument("--suffix", default="", help="Appended to each run-id, e.g. _397b, "
                    "to keep a second model's runs separate from the default ones.")
    args = ap.parse_args(argv)

    for name in args.arms:
        if name not in ARMS:
            print(f"未知臂 {name!r},跳过(可选: {list(ARMS)})")
            continue
        rc = run_arm(name, args.subset, args.gen_workers, args.harness_workers,
                     args.max_rounds, args.suffix)
        if rc != 0:
            print(f"[消融] 臂 {name}{args.suffix} 退出码 {rc}(继续后续臂)")

    # final comparison table
    rows = [r for r in (_row(os.path.join("results", n + args.suffix)) for n in args.arms) if r]
    table = render(rows)
    print("\n\n######## 消融对比表 ########")
    print(table)
    out = f"docs/ablation_table{args.suffix}.md"
    model_note = args.suffix.lstrip("_") or "Qwen3_5-27B"
    with open(out, "w") as f:
        f.write(f"# Ablation comparison (same 20-instance subset, model={model_note})\n\n")
        f.write("```\n" + table + "\n```\n")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
