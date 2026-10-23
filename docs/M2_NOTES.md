# M2 — batch baseline: findings

M2 turns the single-instance pipeline into a batch that runs a whole subset through one
agent config and aggregates the results.

## Baseline (full 20-instance subset)

| 配置 | 样本 | 解决 | 解决率% | 空patch | 错误 | 平均调用 | 平均用时s | 平均patch字符 |
|---|---|---|---|---|---|---|---|---|
| baseline | 20 | 8/20 | 40.0 | 1 | 0 | 34.4 | 302 | 1318 |

Subset: 20 small hermetic instances (4 seaborn, 3 flask, 4 xarray, 4 pylint, 4 pytest,
1 sphinx), Qwen3_5-27B, baseline config (`eval/subsets/m2_small20.json`).
Resolved 8: seaborn-3010, seaborn-3407, flask-4992, xarray-4094, pylint-5859,
pytest-11143, pytest-11148, sphinx-10325. 1 empty patch (pytest-5103), 0 errors.

40% on a 27B open model is a believable, honest baseline; the point of the project is the
*deltas* the ablation/optimization sweep produces against it, not the absolute number.
(Earlier 6-instance partial run: 3/6 — superseded by the full 20.)

## Bugs found and fixed (the batch surfaced what single-instance hid)

1. **GitHub clone through the proxy is unusable.** 3 concurrent clones over the
   HTTP/2 proxy → `curl 92 ... stream not closed cleanly: CANCEL`, `early EOF`; a single
   filtered pytest clone also timed out at 120s. **Fix: get the repo from the instance's
   Docker image** (`docker cp /testbed`) — the image has it checked out at base_commit
   with a clean tree, costs no extra download, and is rock-solid. (`runner.repo_from_image`)

2. **Docker daemon contention.** Running the crane image-preload (`docker load`, 2.5 GB
   each) at the same time as the batch (`docker cp`) made the daemon serialize: a 1-second
   `docker cp` blocked for 400+ seconds and failed. **Rule: never preload images and run a
   batch at the same time.** Pull first, then run.

3. **Leaked containers.** Failed `docker create` calls under contention left 80 stopped
   containers (124 GB). `docker container prune -f` reclaims them; do it between runs.

4. **The relative-cwd × chdir bug (mine).** `run_instance` does `os.chdir(repo_dir)` and
   later passes `repo_dir` as a subprocess cwd. With a *relative* `repo_dir` the second use
   resolves against the already-changed directory → `FileNotFoundError`, an empty patch,
   and a wasted multi-minute agent run. M1's single-instance path only worked because it
   was handed an absolute `--workdir`. **Fix: `os.path.abspath(repo_dir)` before chdir.**

## Architecture that resulted

`eval/batch.py` runs two explicit phases:
- **Phase 1 (serial):** extract each repo from its image (`docker cp`, ~1 s, no contention).
- **Phase 2 (parallel):** run agents (`ProcessPoolExecutor`) on the pre-extracted repos —
  **no Docker at all** in this phase, so concurrency is safe.

Plus **resume** (`_load_existing`): instances with a good prior record are skipped, so the
batch can be re-run to fill in instances as their images finish preloading.

## Throughput reality

Images are ~2.5 GB each (download ~1 GB compressed) and the proxy is the bottleneck. Its
throughput is *variable*: it collapsed to ~0.2 MB/s for a spell (a 1 GB image would take
~80 min) then recovered to ~3 MB/s (~5 min/image). crane can't resume a dropped transfer,
so `eval/pull_images.py` was hardened with **retries (5×) + a per-attempt timeout +
unbuffered output**; with that, all 14 remaining images pulled with 0 failures despite
repeated `TLS handshake timeout` drops. Layer sharing keeps on-disk cost modest (~10 GB
for 20 images) even though nominal sizes sum to ~50 GB.

Two rules the batch enforces: **(1)** preload all images *before* batching (never pull and
run a harness at the same time — daemon contention), and **(2)** the repo itself comes from
the image (`docker cp`), so no GitHub access is needed at run time.
