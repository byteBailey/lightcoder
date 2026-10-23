# M1 — single-instance pipeline: findings

M1 wired the SWE-bench loop end-to-end and ran it on real instances. The pipeline is
validated: **clone @ base_commit → LightCoder one-shot → `git diff` → predictions.jsonl
→ official swebench Docker harness → resolved/not verdict.** Several non-obvious issues
surfaced and shaped the design.

## 1. Docker daemon had no internet (proxy mismatch)

The shell has an HTTP proxy (`HTTPS_PROXY=http://127.0.0.1:27890`) but the Docker daemon
was not configured to use it, so every `docker pull` timed out while `curl` to the same
registry worked.

- **No-sudo fix (what we use):** [`crane`](https://github.com/google/go-containerregistry)
  honors `HTTPS_PROXY` and writes a `docker load`-able tarball. `eval/pull_images.py`
  preloads the official `swebench/sweb.eval.*` images this way. Validated: a 2.34 GB
  requests image and a 2.57 GB flask image loaded and were reused by the harness
  (`Found 1 existing instance images. Will reuse them.`).
- **With-sudo fix (cleaner for scale):** give the daemon the proxy once via
  `/etc/systemd/system/docker.service.d/http-proxy.conf` + `systemctl restart docker`,
  then the official harness pulls natively and `pull_images.py` is unnecessary.

## 2. Some repos' tests need network (`psf/requests`)

`psf__requests-2674` hung for ~14 min: inside the container `pytest test_requests.py`
makes live calls to `httpbin.org`, which the offline container can't reach.
**Consequence for M2: select hermetic repos** (django, sympy, scikit-learn, sphinx,
flask, …) and avoid `requests`. Most of SWE-bench Lite is hermetic.

## 3. The agent burned its whole budget fighting the environment

With a generic "fix it and verify" prompt, the agent:
- on flask-4045, spent **75/75 rounds** on `pip install -e .`, import/`sys.path`
  debugging, and re-running pytest in our py3.11 venv (wrong env) — **0 edits, empty patch**;
- on requests-2674, made the correct `adapters.py` fix but also 6 files of unrelated
  `collections.abc` compatibility churn (again from running old code in py3.11).

Root cause: the agent runs tools in our venv, not the instance's prepared environment.

**Fix applied (cheap, big effect):** a sharper SWE task prompt (`eval/runner.runner`) that
forbids installing packages / running tests / unrelated cleanups and forces `edit_file`.
Result on flask-4045: **3 tool calls, 8.1 s, a clean 1-line patch.**

**Proper fix (M4 candidate):** run the agent's tools *inside the instance container* so it
sees the correct environment — the SWE-agent approach. Tracked as an optimization to
measure, not a prerequisite.

## 4. First clean verdict

`pallets__flask-4045`, baseline config, Qwen3_5-27B: the model added
`assert "." not in name, ...` in `Blueprint.__init__`. The test expects a **`ValueError`**,
so `AssertionError` fails it → **resolved=False**. A plausible-but-wrong fix — the precise
kind of gap the ablation/optimization sweep is built to move.

## Operational defaults going into M2

- Subset: hermetic repos only; fixed, deterministic instance list shared across configs.
- `max_rounds`: 75 baseline.
- Preload images with `eval/pull_images.py` (or configure the daemon proxy).
- Run harness with `--namespace swebench` (reuses local images, no rebuild).
