"""Thin wrapper around the official SWE-bench evaluation harness.

We shell out to ``swebench.harness.run_evaluation`` (which builds/pulls the per-instance
Docker image, applies the model patch + hidden tests, and runs them), then parse the
report JSON it emits. Reusing the official harness is what makes the resolve numbers
credible rather than home-grown.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys


def run_harness(
    predictions_path: str,
    instance_ids: list[str],
    run_id: str,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    report_dir: str = ".",
    max_workers: int = 1,
    timeout: int = 1800,
    namespace: str | None = "swebench",
    cache_level: str = "env",
) -> dict:
    """Run the harness for the given instances and return the parsed report dict."""
    os.makedirs(report_dir, exist_ok=True)
    report_dir = os.path.abspath(report_dir)

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--split", split,
        "--predictions_path", os.path.abspath(predictions_path),
        "--instance_ids", *instance_ids,
        "--run_id", run_id,
        "--max_workers", str(max_workers),
        "--timeout", str(timeout),
        "--cache_level", cache_level,
        "--report_dir", report_dir,
    ]
    if namespace:
        cmd += ["--namespace", namespace]

    proc = subprocess.run(cmd, cwd=report_dir, text=True)
    report = _find_report(report_dir, run_id)
    return {
        "returncode": proc.returncode,
        "report_path": report[0] if report else None,
        "report": report[1] if report else None,
    }


def _find_report(report_dir: str, run_id: str) -> tuple[str, dict] | None:
    # swebench writes "<model_name_or_path>.<run_id>.json" (slashes sanitized).
    candidates = glob.glob(os.path.join(report_dir, f"*{run_id}.json"))
    if not candidates:
        return None
    path = max(candidates, key=os.path.getmtime)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return path, json.load(f)
    except Exception:  # noqa: BLE001
        return None


def summarize(report: dict | None) -> dict:
    """Pull the headline numbers out of a harness report."""
    if not report:
        return {"resolved": 0, "total": 0, "resolved_ids": [], "unresolved_ids": [], "error_ids": []}
    return {
        "resolved": report.get("resolved_instances", 0),
        "total": report.get("total_instances", 0),
        "submitted": report.get("submitted_instances", 0),
        "completed": report.get("completed_instances", 0),
        "resolved_ids": report.get("resolved_ids", []),
        "unresolved_ids": report.get("unresolved_ids", []),
        "error_ids": report.get("error_ids", []),
        "empty_patch_ids": report.get("empty_patch_ids", []),
    }
