"""Run LightCoder on one SWE-bench instance and produce a prediction.

Pipeline for a single instance:
  1. clone the repo and checkout base_commit into a scratch dir,
  2. run the agent one-shot on the issue text with cwd = repo dir,
  3. capture `git diff` (incl. new files) as the model patch.

cwd is changed for the duration because the tools operate relative to the process
working directory. That makes this runner single-instance at a time; batch runs use
process-level isolation (see eval/batch via subprocess) rather than threads.
"""

from __future__ import annotations

import os
import subprocess
import time

from lightcoder import Agent, LLM, AgentConfig, LLMConfig
from lightcoder.sandbox import DockerSandbox


GITHUB = "https://github.com/{repo}.git"

TASK_TEMPLATE = """\
You are working inside the `{repo}` repository, already checked out at the correct commit.
Resolve the GitHub issue below by editing the repository's **source code**.

Constraints (important):
- Locate the relevant code with `read_file`, `grep`, and `glob`, then make the fix with
  the `edit_file` tool. Perform edits via `edit_file`, not by echo/sed through the shell.
- Do NOT install packages, repair the environment, or run the test suite. A separate
  harness runs the tests in a correctly-prepared environment — you cannot and need not
  run them here. Do not spend steps on pip, imports, sys.path, or virtualenvs.
- Change ONLY what the issue requires. Do not edit tests and do not make unrelated
  cleanups (e.g. Python version-compatibility tweaks).

When you have made the edit(s) that fix the issue, stop and give a one-line summary.

<issue>
{problem_statement}
</issue>
"""

# Execution-guided variant (the 'verify' extra): run_bash executes inside the instance's task
# container (deps present), so — unlike the baseline template — the agent is told to reproduce
# and run its fix rather than to avoid the environment.
VERIFY_TASK_TEMPLATE = """\
You are working inside the `{repo}` repository, already checked out at the correct commit.
Resolve the GitHub issue below by editing the repository's **source code**, and verify your
fix by actually running it.

How to work:
- Locate the relevant code with `read_file`, `grep`, and `glob`; make edits with `edit_file`
  (not via echo/sed through the shell).
- Fix first: make the fix with `edit_file` BEFORE writing any reproduction. Always leave your
  edit in place — never finish without having edited the source.
- Then verify: write ONE minimal reproduction and run it with `run_bash` to confirm the fix
  works (your shell runs inside the project's prepared environment — imports and dependencies
  are available, no network). Run the nearest existing test if it's quick. If it still fails,
  refine the fix — do NOT write many reproduction variants or keep exploring.
- Change ONLY what the issue requires. Do NOT edit tests to make them pass, and do not make
  unrelated cleanups.

When your fix is verified (or you've made your best-supported fix), stop and give a one-line
summary of what you changed and what you ran.

<issue>
{problem_statement}
</issue>
"""


def _run_git(args: list[str], cwd: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def _image_key(instance: dict, namespace: str = "swebench") -> str:
    from swebench.harness.test_spec.test_spec import make_test_spec

    return make_test_spec(instance, namespace=namespace).instance_image_key


def repo_from_image(instance: dict, dest: str, namespace: str = "swebench") -> str:
    """Materialize the repo from the instance's Docker image (no GitHub access).

    The swebench image has the repo checked out at base_commit in /testbed with a clean
    working tree. We `docker cp` it out, which is far more reliable than cloning through
    a slow/HTTP2-flaky proxy and reuses the image we need for evaluation anyway.
    """
    img = _image_key(instance, namespace)
    have = subprocess.run(["docker", "images", "-q", img], capture_output=True, text=True)
    if not have.stdout.strip():
        raise RuntimeError(f"image not present locally: {img} (preload it first)")

    repo_dir = os.path.join(dest, "testbed")
    if os.path.exists(repo_dir):
        subprocess.run(["rm", "-rf", repo_dir])
    os.makedirs(dest, exist_ok=True)

    cid = subprocess.run(["docker", "create", img], capture_output=True, text=True).stdout.strip()
    try:
        cp = subprocess.run(["docker", "cp", f"{cid}:/testbed", repo_dir],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"docker cp failed: {cp.stderr[-300:]}")
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)

    # pristine base_commit working tree (defensive; usually already clean)
    _run_git(["reset", "--hard", "HEAD"], cwd=repo_dir)
    _run_git(["clean", "-fdq"], cwd=repo_dir)
    return repo_dir


def clone_checkout(repo: str, base_commit: str, dest: str) -> str:
    """[fallback] Clone repo into dest/<name> and checkout base_commit. Returns the repo dir."""
    repo_dir = os.path.join(dest, repo.replace("/", "__"))
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        os.makedirs(dest, exist_ok=True)
        cp = _run_git(["clone", GITHUB.format(repo=repo), repo_dir], cwd=dest, timeout=900)
        if cp.returncode != 0:
            raise RuntimeError(f"git clone failed: {cp.stderr[-500:]}")
    # clean any prior state, then checkout the base commit
    _run_git(["reset", "--hard"], cwd=repo_dir)
    _run_git(["clean", "-fdx"], cwd=repo_dir)
    cp = _run_git(["checkout", "-f", base_commit], cwd=repo_dir)
    if cp.returncode != 0:
        # fetch the specific commit if it isn't present, then retry
        _run_git(["fetch", "--all", "--tags"], cwd=repo_dir, timeout=900)
        cp = _run_git(["checkout", "-f", base_commit], cwd=repo_dir)
        if cp.returncode != 0:
            raise RuntimeError(f"git checkout {base_commit} failed: {cp.stderr[-500:]}")
    return repo_dir


def extract_patch(repo_dir: str, include_new: bool = True) -> str:
    """Diff the working tree against the base commit (HEAD).

    include_new=True stages everything so newly-created files are captured — the default,
    used by baseline runs. include_new=False diffs tracked files only (excludes untracked
    files); the verify runner uses this so the agent's reproduction scratch files don't leak
    into the model patch — where a stray ``test_*.py`` could break test collection at grading
    time. SWE-bench Lite fixes edit existing files, so nothing real is lost.
    """
    if include_new:
        _run_git(["add", "-A"], cwd=repo_dir)
        cp = _run_git(["diff", "--cached", "--no-color", "HEAD"], cwd=repo_dir)
    else:
        cp = _run_git(["diff", "--no-color", "HEAD"], cwd=repo_dir)
    return cp.stdout


def run_instance(
    instance: dict,
    llm_cfg: LLMConfig,
    agent_cfg: AgentConfig,
    model_name: str,
    workdir: str,
    max_rounds: int | None = None,
    repo_dir: str | None = None,
) -> dict:
    """Run the agent on one instance and return a rich record (prediction + meta).

    If ``repo_dir`` is given it is used as-is (the caller already materialized the repo,
    e.g. by extracting it from the image sequentially to avoid Docker contention);
    otherwise the repo is extracted from the instance's image here.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base = instance["base_commit"]

    record = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": "",
        "meta": {
            "repo": repo,
            "base_commit": base,
            "config": agent_cfg.label(),
            "n_tool_calls": 0,
            "wall_time": 0.0,
            "empty_patch": True,
            "error": None,
        },
        "trajectory": [],
    }

    verify = "verify" in agent_cfg.extras
    record["meta"]["verify"] = verify

    t0 = time.time()
    old_cwd = os.getcwd()
    agent = None
    try:
        if repo_dir is None:
            repo_dir = repo_from_image(instance, workdir)
        # Absolute path is essential: we chdir into repo_dir, and extract_patch later
        # passes repo_dir as a subprocess cwd. A relative path would resolve against the
        # already-changed cwd and break.
        repo_dir = os.path.abspath(repo_dir)
        os.chdir(repo_dir)

        cfg = agent_cfg if max_rounds is None else agent_cfg.with_(max_rounds=max_rounds)
        agent = Agent(LLM(llm_cfg), config=cfg)

        # Execution-guided verification: route run_bash into the instance's task container,
        # with the (host) repo bind-mounted at /testbed so edits are live and deps are present.
        if verify:
            agent.attach_sandbox(DockerSandbox(
                _image_key(instance), workdir=repo_dir,
                container_dir="/testbed", network="none",
            ))
        template = VERIFY_TASK_TEMPLATE if verify else TASK_TEMPLATE

        n_calls = {"n": 0}

        def on_tool(name, kwargs):
            n_calls["n"] += 1

        task = template.format(repo=repo, problem_statement=instance["problem_statement"])
        summary = agent.chat(task, on_tool=on_tool)

        patch = extract_patch(repo_dir, include_new=not verify)
        record["model_patch"] = patch
        record["trajectory"] = agent.messages
        record["meta"].update(
            n_tool_calls=n_calls["n"],
            empty_patch=(not patch.strip()),
            summary=summary[:1000],
        )
    except Exception as e:  # noqa: BLE001 — never let one instance kill a batch
        record["meta"]["error"] = f"{type(e).__name__}: {e}"
    finally:
        if agent is not None:
            try:
                agent.close()          # tear down the verification container, if any
            except Exception:          # noqa: BLE001
                pass
        os.chdir(old_cwd)
        record["meta"]["wall_time"] = round(time.time() - t0, 1)

    return record
