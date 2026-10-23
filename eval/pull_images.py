"""Preload SWE-bench instance Docker images via `crane`.

Why this exists: on hosts where the shell has an HTTP proxy but the Docker *daemon*
has no internet (and you can't sudo to configure its proxy), `docker pull` fails.
`crane` is a userspace binary that honors HTTPS_PROXY and writes a `docker load`-able
tarball, so we can fetch the official `swebench/...` images anyway.

    uv run python -m eval.pull_images --dataset lite --ids pallets__flask-4045
    uv run python -m eval.pull_images --dataset lite --n 50

Alternative (if you have sudo): give the daemon the proxy once and the official
harness pulls natively — then you don't need this at all. See docs/M1_NOTES.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

from .dataset import load_instances, select_subset


def _crane() -> str:
    return shutil.which("crane") or os.path.expanduser("~/.local/bin/crane")


def image_key(instance: dict, namespace: str = "swebench") -> str:
    from swebench.harness.test_spec.test_spec import make_test_spec

    return make_test_spec(instance, namespace=namespace).instance_image_key


def have_image(tag: str) -> bool:
    out = subprocess.run(["docker", "images", "-q", tag], capture_output=True, text=True)
    return bool(out.stdout.strip())


def pull_load(tag: str, crane: str, workdir: str, retries: int = 5, timeout: int = 5400) -> bool:
    """crane pull docker.io/<tag> -> docker load. Retries on the flaky-proxy EOFs.

    The proxy drops large transfers ("unexpected EOF"); retrying usually gets through.
    A per-attempt timeout keeps a stalled download from hanging the whole run.
    """
    tar = os.path.join(workdir, tag.replace("/", "__").replace(":", "_") + ".tar")
    try:
        for attempt in range(1, retries + 1):
            try:
                cp = subprocess.run([crane, "pull", f"docker.io/{tag}", tar], timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"      attempt {attempt}/{retries}: timed out after {timeout}s", flush=True)
                continue
            if cp.returncode == 0:
                ld = subprocess.run(["docker", "load", "-i", tar])
                return ld.returncode == 0
            print(f"      attempt {attempt}/{retries}: crane rc={cp.returncode}", flush=True)
        return False
    finally:
        if os.path.exists(tar):
            os.remove(tar)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Preload SWE-bench instance images via crane.")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--ids", nargs="*", help="Explicit instance ids.")
    ap.add_argument("--n", type=int, help="First N by sorted instance_id.")
    ap.add_argument("--namespace", default="swebench")
    args = ap.parse_args(argv)

    crane = _crane()
    if not os.path.exists(crane) and not shutil.which("crane"):
        print(f"ERROR: crane not found ({crane}). Install it first (see docs/M1_NOTES.md).")
        return 2

    instances = select_subset(load_instances(args.dataset), ids=args.ids, n=args.n)
    workdir = tempfile.mkdtemp(prefix="lightcoder_imgs_")
    print(f"Preloading {len(instances)} image(s) via crane ({crane})")

    ok = skipped = failed = 0
    for inst in instances:
        tag = image_key(inst, args.namespace)
        if have_image(tag):
            skipped += 1
            print(f"  [skip] {tag}")
            continue
        print(f"  [pull] {tag}")
        if pull_load(tag, crane, workdir):
            ok += 1
        else:
            failed += 1
            print(f"  [FAIL] {tag}")

    shutil.rmtree(workdir, ignore_errors=True)
    print(f"\nDone: {ok} pulled, {skipped} already present, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
