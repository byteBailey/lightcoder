"""Calibrate LightCoder against nanocoder (external reference agent) on the SAME model + subset.

nanocoder (Nano-Collective, v1.28) is run headless in each instance's repo:

    node dist/cli.js --provider local27b --model <27B> --mode yolo --plain run "<issue>"

with ``~/.config/nanocoder/agents.config.json`` pointing local27b at the same 27B endpoint
LightCoder uses. Its ``git diff`` (excluding nanocoder's own ``.nanocoder/`` scratch dir) is
the predicted patch; the standard SWE-bench harness then scores all of them. This tells us
whether LightCoder's resolve rate is good or bad relative to a real product on the same model.

    uv run python -m eval.run_nanocoder --limit 1 --skip-harness     # validate one instance
    uv run python -m eval.run_nanocoder                              # full m5_50 + harness
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time

from .dataset import load_instances
from .harness import run_harness, summarize
from .predictions import write_predictions, write_records

NODE = os.path.expanduser("~/.nvm/versions/node/v22.22.2/bin/node")
# Machine-specific: a local nanocoder checkout and the model LightCoder is served with.
CLI = os.environ.get("NANOCODER_CLI", os.path.expanduser("~/nanocoder/dist/cli.js"))
MODEL = os.environ.get("LIGHTCODER_MODEL", "Qwen/Qwen3-32B")

TASK = (
    "Resolve the GitHub issue below by editing the repository's source code. Locate the "
    "relevant code with your tools and make the fix with file edits. Do not modify tests "
    "and do not run the test suite.\n\n<issue>\n{ps}\n</issue>"
)


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def run_one(inst: dict, repo_dir: str, timeout: int, provider: str, model: str, wrap: list) -> dict:
    iid = inst["instance_id"]
    _git(["reset", "--hard", "HEAD"], repo_dir)   # pristine base before this attempt
    _git(["clean", "-fdq"], repo_dir)
    t0, err = time.time(), None
    try:
        subprocess.run(
            [*wrap, NODE, CLI, "--provider", provider, "--model", model,
             "--mode", "yolo", "--trust-directory", "--plain", "--context-max", "128k",
             "run", TASK.format(ps=inst["problem_statement"])],
            cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        err = "timeout"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:200]
    # capture the patch, excluding nanocoder's own scratch dir
    _git(["add", "-A"], repo_dir)
    diff = _git(["diff", "--cached", "--no-color", "HEAD", "--", ":(exclude).nanocoder"],
                repo_dir).stdout
    return {
        "instance_id": iid,
        "model_name_or_path": "nanocoder",
        "model_patch": diff,
        "meta": {"wall_time": round(time.time() - t0, 1), "error": err,
                 "empty_patch": not diff.strip()},
        "trajectory": [],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="eval/subsets/m5_50.json")
    ap.add_argument("--repos", default="results/_repos_batch/m5_baseline")
    ap.add_argument("--run-id", default="nano_m5")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N instances (debug).")
    ap.add_argument("--provider", default="local27b", help="nanocoder provider (agents.config.json).")
    ap.add_argument("--model", default=MODEL, help="Model id for that provider.")
    ap.add_argument("--proxychains", default="", help="proxychains conf path: routes node's traffic "
                    "via the proxy (node fetch ignores HTTPS_PROXY; needed for external APIs).")
    ap.add_argument("--skip-harness", action="store_true")
    args = ap.parse_args(argv)

    ids = json.load(open(args.subset))
    if args.limit:
        ids = ids[: args.limit]
    insts = {i["instance_id"]: i for i in load_instances("lite")}
    todo = [insts[i] for i in ids if i in insts]

    out_dir = os.path.join("results", args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    wrap = ([os.path.expanduser("~/.local/bin/proxychains4"), "-q", "-f", args.proxychains]
            if args.proxychains else [])

    def job(inst):
        rd = os.path.abspath(os.path.join(args.repos, inst["instance_id"], "testbed"))
        if not os.path.isdir(rd):
            return {"instance_id": inst["instance_id"], "model_name_or_path": "nanocoder",
                    "model_patch": "", "meta": {"error": "repo missing", "empty_patch": True},
                    "trajectory": []}
        return run_one(inst, rd, args.timeout, args.provider, args.model, wrap)

    print(f"[nano] 跑 {len(todo)} 实例 (并发={args.workers}, provider={args.provider} "
          f"模型={args.model}{' via proxychains' if args.proxychains else ''})")
    records, done = [], 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(job, todo):
            done += 1
            m = rec["meta"]
            flag = "空" if m["empty_patch"] else f"{len(rec['model_patch'])}字符"
            print(f"  [{done}/{len(todo)}] {rec['instance_id']:32s} patch={flag} "
                  f"用时={m.get('wall_time','?')}s err={m.get('error')}")
            records.append(rec)

    preds = write_predictions(records, os.path.join(out_dir, "predictions.jsonl"))
    write_records(records, os.path.join(out_dir, "records.jsonl"))
    nonempty = sum(1 for r in records if not r["meta"]["empty_patch"])
    print(f"\n[nano] 预测 {len(records)} 条, 非空 {nonempty}")
    if args.skip_harness:
        return 0

    ids2 = [r["instance_id"] for r in records]
    res = run_harness(predictions_path=preds, instance_ids=ids2, run_id=args.run_id,
                      report_dir=out_dir, max_workers=4)
    summ = summarize(res["report"])
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(f"\n[nano] === nanocoder on {len(ids2)} 实例: 解决 "
          f"{summ['resolved']}/{summ['total']} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
