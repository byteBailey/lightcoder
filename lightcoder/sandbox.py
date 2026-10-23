"""Sandboxed command execution — confine the agent's shell to a container.

Bubblewrap/Seatbelt aren't available on this host, so the OS-level sandbox is Docker: a
long-lived container with the project directory bind-mounted and ``--network none``. File
tools still edit on the *host*, but because the same directory is mounted, the container
sees those edits; shell commands run *inside* the container, so a bad — or approved-but-
malicious — command can't reach the network or touch anything outside the workspace. This
is also the execution substrate for run-the-tests verification in the eval (Phase 3).

Semantics mirror the "workspace-write + no-network" sandbox that Codex and current Claude
Code use:

  * writes under ``/workspace`` round-trip to the host cwd (intended — the agent edits the
    project it was pointed at),
  * writes elsewhere in the container are ephemeral and isolated,
  * no network.

Caveat: file tools use paths relative to the host cwd, which maps to ``/workspace`` in the
container (shell cwd), so relative paths stay consistent across both. An *absolute* host
path won't resolve inside the container — but the agent works in relative paths and the
shell runs in ``/workspace``, so this doesn't bite in practice.
"""

from __future__ import annotations

import os
import subprocess
import threading

SANDBOX_MODES = ("none", "docker")


class DockerSandbox:
    """A persistent Docker container that shell commands are ``docker exec``'d into.

    The container starts lazily on the first command, so constructing an agent stays free
    and a missing Docker / un-pullable image surfaces as a tool error instead of crashing
    startup. ``runner`` is a seam for tests — it defaults to :func:`subprocess.run`.
    """

    WORKDIR = "/workspace"  # default bind-mount point for the host cwd (also the container cwd)

    def __init__(self, image, workdir, network="none", container_dir=None,
                 runner=subprocess.run):
        self.image = image
        self.host_dir = os.path.abspath(workdir)
        self.network = network
        # Where the host dir is mounted inside the container. Defaults to /workspace; the eval
        # runner overrides it to /testbed so a swebench image's editable install resolves to
        # the (edited) mounted source while deps come from the image.
        self.container_dir = container_dir or self.WORKDIR
        self._run = runner
        self._cid = None
        self._lock = threading.Lock()  # parallel tool calls may race the lazy start

    def _ensure(self) -> str:
        with self._lock:
            if self._cid:
                return self._cid
            proc = self._run(
                [
                    "docker", "run", "-d", "--rm",
                    f"--network={self.network}",
                    "-v", f"{self.host_dir}:{self.container_dir}",
                    "-w", self.container_dir,
                    self.image, "sleep", "infinity",
                ],
                capture_output=True, text=True, timeout=180,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"could not start sandbox ({self.image}): "
                    f"{(proc.stderr or '').strip()[-300:]}"
                )
            self._cid = proc.stdout.strip()
            return self._cid

    def run(self, command: str, timeout: int = 120):
        """Exec ``command`` in the container; return ``(combined_output, exit_code)``.

        Lets :class:`subprocess.TimeoutExpired` propagate so the caller (BashTool) reports
        timeouts the same way for host and sandbox execution.
        """
        cid = self._ensure()
        proc = self._run(
            ["docker", "exec", cid, "bash", "-lc", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return (proc.stdout or "") + (proc.stderr or ""), proc.returncode

    def close(self) -> None:
        """Force-remove the container. Safe to call more than once."""
        with self._lock:
            if self._cid:
                self._run(
                    ["docker", "rm", "-f", self._cid],
                    capture_output=True, text=True,
                )
                self._cid = None


def make_sandbox(mode, image="python:3.11-slim", workdir=None, network="none"):
    """Return a sandbox for ``mode``, or ``None`` for 'none'/unknown (host execution)."""
    if mode == "docker":
        return DockerSandbox(image, workdir or os.getcwd(), network=network)
    return None
