"""Run a whole subset through one agent config: generate predictions, then evaluate.

Two phases, deliberately separate:
  1. generate — run the agent on every instance (needs only the LLM + git, no Docker).
                 Parallelized across processes because the tools are cwd-based.
  2. evaluate — hand all predictions to the official harness once (Docker, --max_workers).

The agent toggles are exposed as flags, so the M3 ablation sweep is just this command
run once per config (baseline, --no-compression, --prompt-variant react, ...).

    uv run python -m eval.batch --subset eval/subsets/m2_small20.json
    uv run python -m eval.batch --subset eval/subsets/m2_small20.json --no-compression
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os

from lightcoder import AgentConfig, LLMConfig
from .dataset import load_instances, select_subset
from .runner import run_instance
from .predictions import write_predictions, write_records, load_records
from .harness import run_harness, summarize


def _load_existing(out_dir: str) -> dict:
    """Previously-generated good records (successful agent run), keyed by instance_id.

    Lets the batch resume: instances that already ran (no error, >0 tool calls) are
    skipped, so we can re-run as more images finish preloading without redoing work.
    """
    path = os.path.join(out_dir, "records.jsonl")
    if not os.path.exists(path):
        return {}
    keep = {}
    for r in load_records(path):
        m = r.get("meta", {})
        if not m.get("error") and m.get("n_tool_calls", 0) > 0:
            keep[r["instance_id"]] = r
    return keep


def prepare(instances, workdir):
    """Sequentially materialize each repo from its image.

    Docker operations are serialized here (each docker cp is ~1s when the daemon isn't
    busy) so the parallel agent phase never touches Docker — that was the source of the
    daemon contention that made concurrent docker cp hang for minutes.
    Returns (repo_dirs: {id: dir}, missing: [instances without a local image]).
    """
    from .runner import repo_from_image

    repo_dirs, missing = {}, []
    for inst in instances:
        iid = inst["instance_id"]
        try:
            repo_dirs[iid] = repo_from_image(inst, os.path.join(workdir, iid))
        except Exception as e:  # noqa: BLE001 — usually "image not present locally"
            missing.append(inst)
            print(f"  [缺镜像] {iid}: {str(e)[:80]}")
    return repo_dirs, missing


# Module-level so ProcessPoolExecutor can pickle it.
def _gen_one(payload):
    inst, llm_cfg, agent_cfg, model_name, workdir, max_rounds, repo_dir = payload
    return run_instance(inst, llm_cfg, agent_cfg, model_name, workdir,
                        max_rounds=max_rounds, repo_dir=repo_dir)


def generate(instances, repo_dirs, llm_cfg, agent_cfg, model_name, workdir, max_rounds, workers):
    payloads = [
        (inst, llm_cfg, agent_cfg, model_name,
         os.path.join(workdir, inst["instance_id"]), max_rounds,
         repo_dirs[inst["instance_id"]])
        for inst in instances if inst["instance_id"] in repo_dirs
    ]
    records, done = [], 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(_gen_one, payloads):
            done += 1
            m = rec["meta"]
            flag = "空" if m["empty_patch"] else f"{len(rec['model_patch'])}字符"
            print(f"  [{done}/{len(payloads)}] {rec['instance_id']:32s} "
                  f"调用={m['n_tool_calls']:>3} 用时={m['wall_time']:>6}s patch={flag} "
                  f"错误={m['error']}")
            records.append(rec)
    return records


def _missing_record(inst, model_name):
    return {
        "instance_id": inst["instance_id"],
        "model_name_or_path": model_name,
        "model_patch": "",
        "meta": {"repo": inst["repo"], "base_commit": inst["base_commit"],
                 "n_tool_calls": 0, "wall_time": 0.0, "empty_patch": True,
                 "error": "image not present locally"},
        "trajectory": [],
    }


def _agent_config(args) -> AgentConfig:
    extras = tuple(x for x in (args.extras or "").split(",") if x)
    return AgentConfig(
        parallel_tools=not args.no_parallel,
        compression=not args.no_compression,
        subagent=not args.no_subagent,
        prompt_variant=args.prompt_variant,
        max_rounds=args.max_rounds,
        extras=extras,
    )


def _load_subset(args) -> list[dict]:
    instances = load_instances(args.dataset)
    ids = None
    if args.subset:
        ids = json.load(open(args.subset))
    return select_subset(instances, ids=ids, n=args.n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a subset through one agent config.")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--subset", help="Path to a JSON list of instance ids.")
    ap.add_argument("--n", type=int, help="First N instances (if no --subset).")
    ap.add_argument("--run-id", default=None, help="Defaults to the config label.")
    ap.add_argument("--max-rounds", type=int, default=75)
    ap.add_argument("--gen-workers", type=int, default=3, help="Parallel agent runs.")
    ap.add_argument("--harness-workers", type=int, default=4, help="Parallel Docker evals.")
    ap.add_argument("--skip-harness", action="store_true", help="Only generate predictions.")
    # ablation toggles
    ap.add_argument("--no-parallel", action="store_true")
    ap.add_argument("--no-compression", action="store_true")
    ap.add_argument("--no-subagent", action="store_true")
    ap.add_argument("--prompt-variant", default="default")
    ap.add_argument("--extras", default="", help="Comma-separated optimization features "
                    "(maps to AgentConfig.extras).")
    args = ap.parse_args(argv)

    llm_cfg = LLMConfig.from_env()
    agent_cfg = _agent_config(args)
    label = agent_cfg.label()
    run_id = args.run_id or label
    model_name = f"lightcoder__{label}"

    out_dir = os.path.join("results", run_id)
    os.makedirs(out_dir, exist_ok=True)
    workdir = os.path.abspath(os.path.join("results", "_repos_batch", run_id))
    os.makedirs(workdir, exist_ok=True)

    instances = _load_subset(args)
    existing = _load_existing(out_dir)
    todo = [i for i in instances if i["instance_id"] not in existing]
    print(f"[批量] 配置={label} 运行id={run_id} 模型={llm_cfg.model}")
    print(f"[批量] 共 {len(instances)} 实例 | 已完成 {len(existing)} | 待跑 {len(todo)} | 并发={args.gen_workers}\n")

    # Phase 1: extract repos from images, serially (no Docker in the parallel phase).
    print(f"[批量] 阶段1: 从镜像抽取 {len(todo)} 个仓库 ...")
    repo_dirs, missing = prepare(todo, workdir) if todo else ({}, [])
    ready = [i for i in todo if i["instance_id"] in repo_dirs]
    print(f"[批量] 就绪 {len(ready)} | 缺镜像 {len(missing)}\n")

    # Phase 2: run agents in parallel on the pre-extracted repos.
    print(f"[批量] 阶段2: 并行跑 agent ({len(ready)} 个) ...")
    new_records = generate(ready, repo_dirs, llm_cfg, agent_cfg, model_name, workdir,
                           args.max_rounds, args.gen_workers) if ready else []
    new_records += [_missing_record(i, model_name) for i in missing]

    # merge existing + new, preserving subset order
    merged = {**existing, **{r["instance_id"]: r for r in new_records}}
    records = [merged[i["instance_id"]] for i in instances if i["instance_id"] in merged]

    preds_path = write_predictions(records, os.path.join(out_dir, "predictions.jsonl"))
    write_records(records, os.path.join(out_dir, "records.jsonl"))

    n_empty = sum(1 for r in records if r["meta"]["empty_patch"])
    print(f"\n[批量] 预测已写出: {len(records)} 条 (空 patch {n_empty} 条)")

    if args.skip_harness:
        print("[批量] 按要求跳过 harness。")
        return 0

    ids = [r["instance_id"] for r in records]
    print(f"[批量] === 对 {len(ids)} 个实例跑 harness (并发={args.harness_workers}) ===")
    result = run_harness(
        predictions_path=preds_path,
        instance_ids=ids,
        run_id=run_id,
        report_dir=out_dir,
        max_workers=args.harness_workers,
    )
    summ = summarize(result["report"])
    summ["config"] = label
    summ["n_empty_patch"] = n_empty
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    print(f"\n[批量] === {label}: 解决 {summ['resolved']}/{summ['total']} "
          f"({100*summ['resolved']/max(summ['total'],1):.1f}%) ===")
    print(f"[批量] 汇总 -> {os.path.join(out_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
