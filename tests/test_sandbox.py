"""Docker sandbox: argv construction, lazy start/reuse, teardown, and BashTool routing.

No real Docker is needed — a fake runner records the argv and returns canned results, so the
container lifecycle is pinned deterministically. One live end-to-end smoke test runs only when
LIGHTCODER_DOCKER_TEST=1 and docker is present (it needs to pull/run an image).
"""

import os
import shutil

import pytest

from lightcoder.sandbox import DockerSandbox, make_sandbox, SANDBOX_MODES
from lightcoder.tools.bash import BashTool


class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class _FakeRunner:
    """Stand-in for subprocess.run: records every argv, answers docker run/exec."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            return _Proc(stdout="container123\n")
        if argv[:2] == ["docker", "exec"]:
            return _Proc(stdout="hello\n", returncode=0)
        return _Proc()


# --------------------------------------------------------------- factory / config

def test_make_sandbox_none_is_host():
    assert make_sandbox("none") is None
    assert make_sandbox("bogus") is None
    assert isinstance(make_sandbox("docker", workdir="/tmp"), DockerSandbox)


def test_sandbox_modes_advertised():
    assert SANDBOX_MODES == ("none", "docker")


# ----------------------------------------------------------------- argv shape

def test_docker_run_has_isolation_flags():
    r = _FakeRunner()
    DockerSandbox("img:1", "/proj", runner=r).run("echo hi")
    run_argv = r.calls[0]
    assert run_argv[:4] == ["docker", "run", "-d", "--rm"]
    assert "--network=none" in run_argv
    assert "-v" in run_argv and f"/proj:{DockerSandbox.WORKDIR}" in run_argv
    assert "-w" in run_argv and DockerSandbox.WORKDIR in run_argv
    assert run_argv[-3:] == ["img:1", "sleep", "infinity"]


def test_exec_targets_the_started_container():
    r = _FakeRunner()
    DockerSandbox("img", "/proj", runner=r).run("echo a")
    exec_argv = next(c for c in r.calls if c[:2] == ["docker", "exec"])
    assert exec_argv == ["docker", "exec", "container123", "bash", "-lc", "echo a"]


def test_network_override():
    r = _FakeRunner()
    DockerSandbox("img", "/proj", network="host", runner=r).run("x")
    assert "--network=host" in r.calls[0]


# ------------------------------------------------------------- lifecycle

def test_container_started_once_then_reused():
    r = _FakeRunner()
    sb = DockerSandbox("img", "/proj", runner=r)
    sb.run("echo a")
    sb.run("echo b")
    kinds = [c[:2] for c in r.calls]
    assert kinds.count(["docker", "run"]) == 1      # single lazy start
    assert kinds.count(["docker", "exec"]) == 2      # both commands reuse it


def test_run_returns_output_and_exit_code():
    out, code = DockerSandbox("img", "/proj", runner=_FakeRunner()).run("echo hi")
    assert out == "hello\n" and code == 0


def test_close_removes_container_and_is_idempotent():
    r = _FakeRunner()
    sb = DockerSandbox("img", "/proj", runner=r)
    sb.run("echo a")
    sb.close()
    assert r.calls[-1][:3] == ["docker", "rm", "-f"]
    assert r.calls[-1][3] == "container123"
    before = len(r.calls)
    sb.close()                       # second close is a no-op (nothing to remove)
    assert len(r.calls) == before


def test_start_failure_raises():
    class Boom(_FakeRunner):
        def __call__(self, argv, **kw):
            self.calls.append(argv)
            return _Proc(stderr="Cannot connect to the Docker daemon", returncode=1)

    with pytest.raises(RuntimeError, match="could not start sandbox"):
        DockerSandbox("img", "/proj", runner=Boom()).run("echo a")


# ------------------------------------------------------- BashTool integration

def test_bashtool_routes_through_sandbox_when_wired():
    class Stub:
        def __init__(self):
            self.got = None

        def run(self, cmd, timeout):
            self.got = (cmd, timeout)
            return ("out\n", 0)

    t = BashTool()
    t._sandbox = Stub()
    result = t.run("ls -la", timeout=99)
    assert t._sandbox.got == ("ls -la", 99)
    assert "out" in result and "exit code 0" in result


def test_bashtool_host_path_without_sandbox():
    result = BashTool().run("printf hello")     # real host execution
    assert "hello" in result and "exit code 0" in result


def test_footgun_guard_fires_before_sandbox():
    class Boom:
        def run(self, cmd, timeout):
            raise AssertionError("guard should short-circuit before the sandbox")

    t = BashTool()
    t._sandbox = Boom()
    assert "blocked by safety guard" in t.run("rm -rf /")


# --------------------------------------------------------------- live smoke (opt-in)

@pytest.mark.skipif(
    not os.environ.get("LIGHTCODER_DOCKER_TEST") or shutil.which("docker") is None,
    reason="set LIGHTCODER_DOCKER_TEST=1 (docker + image present) to run the live sandbox",
)
def test_live_docker_sandbox_offline_and_isolated():
    image = os.environ.get("LIGHTCODER_SANDBOX_IMAGE", "python:3.11-slim")
    sb = make_sandbox("docker", image=image, workdir="/tmp")
    try:
        out, code = sb.run("echo sandboxed && whoami")
        assert "sandboxed" in out and code == 0
        # --network none: any outbound connection must fail.
        out2, code2 = sb.run(
            "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\""
        )
        assert code2 != 0
    finally:
        sb.close()
