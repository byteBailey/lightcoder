"""Derisk driver: run LightCoder on ONE SWE-bench instance, end-to-end.

    uv run python -m eval.run_single --instance psf__requests-2674

Produces under results/<run_id>/:
  predictions.jsonl   minimal predictions for the harness
  records.jsonl       full records (meta + trajectory)
  <model>.<run_id>.json  the harness report
and prints whether the instance was resolved.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

from lightcoder import AgentConfig, LLMConfig
from .dataset import get_instance
from .runner import run_instance
from .predictions import write_predictions, write_records
from .harness import run_harness, summarize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="psf__requests-2674", help="SWE-bench instance id.")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-rounds", type=int, default=30)
    ap.add_argument("--workdir", default=None, help="Where to clone repos (default: temp).")
    ap.add_argument("--skip-harness", action="store_true", help="Only generate the patch.")
    ap.add_argument("--extras", default="", help="Comma-separated optimization features.")
    args = ap.parse_args(argv)

    run_id = args.run_id or f"derisk_{args.instance}"
    out_dir = os.path.join("results", run_id)
    os.makedirs(out_dir, exist_ok=True)
    workdir = args.workdir or tempfile.mkdtemp(prefix="lightcoder_repos_")

    llm_cfg = LLMConfig.from_env()
    extras = tuple(x for x in args.extras.split(",") if x)
    agent_cfg = AgentConfig(max_rounds=args.max_rounds, extras=extras)
    model_name = f"lightcoder__{agent_cfg.label()}"

    print(f"[run_single] instance={args.instance} model={llm_cfg.model}")
    print(f"[run_single] base_url={llm_cfg.base_url} workdir={workdir}")

    instance = get_instance(args.instance, name=args.dataset)
    record = run_instance(instance, llm_cfg, agent_cfg, model_name, workdir, max_rounds=args.max_rounds)

    meta = record["meta"]
    print("\n[run_single] === agent run ===")
    print(f"  tool_calls={meta['n_tool_calls']}  wall={meta['wall_time']}s  "
          f"empty_patch={meta['empty_patch']}  error={meta['error']}")
    print(f"  summary: {meta.get('summary', '')[:300]}")
    print(f"\n[run_single] === model patch ({len(record['model_patch'])} chars) ===")
    print(record["model_patch"][:2000] or "(empty)")

    preds_path = write_predictions([record], os.path.join(out_dir, "predictions.jsonl"))
    write_records([record], os.path.join(out_dir, "records.jsonl"))

    if args.skip_harness or meta["empty_patch"]:
        print("\n[run_single] skipping harness (requested or empty patch).")
        return 0

    print("\n[run_single] === handing to SWE-bench harness (Docker) ===")
    result = run_harness(
        predictions_path=preds_path,
        instance_ids=[args.instance],
        run_id=run_id,
        report_dir=out_dir,
    )
    summ = summarize(result["report"])
    resolved = args.instance in summ["resolved_ids"]
    print("\n[run_single] === VERDICT ===")
    print(f"  resolved={resolved}  ({summ['resolved']}/{summ['total']})")
    print(f"  report: {result['report_path']}")
    print(json.dumps(summ, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
