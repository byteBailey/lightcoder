"""Read/write SWE-bench prediction files and full run records.

The official harness consumes a JSONL of {instance_id, model_name_or_path,
model_patch}. We additionally persist the full records (meta + trajectory) so runs
are auditable and the secondary metrics (tool calls, wall time, empty-patch rate)
survive for the report.
"""

from __future__ import annotations

import json
import os

PRED_FIELDS = ("instance_id", "model_name_or_path", "model_patch")


def write_predictions(records: list[dict], path: str) -> str:
    """Write the minimal prediction JSONL the swebench harness expects."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in PRED_FIELDS}) + "\n")
    return path


def write_records(records: list[dict], path: str) -> str:
    """Persist full records (prediction + meta + trajectory) as JSONL."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_records(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
