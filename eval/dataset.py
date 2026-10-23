"""Load a SWE-bench dataset and select a deterministic subset.

The subset is the unit of comparison: every ablation/optimization config is run on
the *same* fixed instance list so differences are attributable to the agent.
"""

from __future__ import annotations

DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


def dataset_name(short: str) -> str:
    if short in DATASETS:
        return DATASETS[short]
    return short  # allow passing a full HF name directly


def load_instances(name: str = "lite", split: str = "test") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name(name), split=split)
    return [dict(r) for r in ds]


def get_instance(instance_id: str, name: str = "lite", split: str = "test") -> dict:
    for inst in load_instances(name, split):
        if inst["instance_id"] == instance_id:
            return inst
    raise KeyError(f"instance {instance_id!r} not found in {name}/{split}")


def select_subset(
    instances: list[dict],
    ids: list[str] | None = None,
    n: int | None = None,
) -> list[dict]:
    """Pick a deterministic subset.

    * ``ids`` — exact instance ids (order preserved).
    * else first ``n`` by sorted instance_id (stable, repeatable).
    * else all.
    """
    if ids:
        by_id = {i["instance_id"]: i for i in instances}
        return [by_id[i] for i in ids if i in by_id]
    ordered = sorted(instances, key=lambda r: r["instance_id"])
    return ordered[:n] if n else ordered
