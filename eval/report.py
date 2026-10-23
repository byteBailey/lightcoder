"""Aggregate one or more run directories into a comparison table.

Each run dir (results/<run_id>/) holds records.jsonl (per-instance meta + patch) and
summary.json (harness verdict). This rolls them up into the headline resolve rate plus
the secondary metrics that make an ablation interpretable: tool calls, wall time, empty
and error rates, average patch size.

    uv run python -m eval.report results/baseline results/no_compression ...
    uv run python -m eval.report --glob 'results/*'
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import os

from .predictions import load_records


def _row(run_dir: str) -> dict | None:
    rec_path = os.path.join(run_dir, "records.jsonl")
    summ_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(rec_path):
        return None
    records = load_records(rec_path)
    n = len(records)
    if n == 0:
        return None

    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    metas = [r["meta"] for r in records]
    n_empty = sum(1 for m in metas if m.get("empty_patch"))
    n_err = sum(1 for m in metas if m.get("error"))
    avg_calls = sum(m.get("n_tool_calls", 0) for m in metas) / n
    avg_wall = sum(m.get("wall_time", 0) for m in metas) / n
    avg_patch = sum(len(r.get("model_patch", "")) for r in records) / n
    resolved = summ.get("resolved", 0)
    total = summ.get("total", n)

    return {
        "config": summ.get("config", os.path.basename(run_dir)),
        "n": n,
        "resolved": resolved,
        "resolve_pct": 100 * resolved / max(total, 1),
        "empty": n_empty,
        "errors": n_err,
        "avg_calls": avg_calls,
        "avg_wall": avg_wall,
        "avg_patch": avg_patch,
    }


def render(rows: list[dict]) -> str:
    hdr = ["配置", "样本数", "解决", "解决率%", "空patch", "错误", "平均调用", "平均用时s", "平均patch字符"]
    lines = ["\t".join(hdr)]
    for r in sorted(rows, key=lambda x: -x["resolve_pct"]):
        lines.append("\t".join([
            r["config"], str(r["n"]),
            f"{r['resolved']}/{r['n']}", f"{r['resolve_pct']:.1f}",
            str(r["empty"]), str(r["errors"]),
            f"{r['avg_calls']:.1f}", f"{r['avg_wall']:.0f}", f"{r['avg_patch']:.0f}",
        ]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate run dirs into a comparison table.")
    ap.add_argument("run_dirs", nargs="*", help="results/<run_id> directories.")
    ap.add_argument("--glob", help="Glob for run dirs, e.g. 'results/*'.")
    ap.add_argument("--out", help="Optional path to also write the table (markdown/tsv).")
    args = ap.parse_args(argv)

    dirs = list(args.run_dirs)
    if args.glob:
        dirs += [d for d in globlib.glob(args.glob) if os.path.isdir(d)]
    dirs = sorted(set(dirs))

    rows = [r for r in (_row(d) for d in dirs) if r]
    if not rows:
        print("未找到带 records 的运行目录。")
        return 1

    table = render(rows)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"\n已写入 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
