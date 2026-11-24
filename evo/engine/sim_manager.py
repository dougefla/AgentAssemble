"""Sim-worker lifecycle manager — one fresh Python worker per agent call.

Successor to the Blender lifecycle manager: each agent that needs the
simulation backend wraps its work in ``with SimWorkerManager(port=...,
runtime_globals={...}):``. On enter a ``python -m evo.engine.sim_worker``
subprocess starts serving the NUL-terminated JSON execute protocol on the
given port; on exit it is terminated. One worker per loop worker, each on
its own port, so parallel workers don't collide.

The immutable runtime globals (ASSEMBLY_SDK_DIR / SKILL_DIR / INPUT_DIR /
OUTPUT_DIR) are handed to the worker via the EVO_SIM_GLOBALS env var and
re-injected before every execution — both agent backends get identical
injection semantics.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from evo.engine.sim_worker import GLOBALS_ENV, PARENT_PID_ENV

log = logging.getLogger(__name__)

SIM_PORT = 9876
STARTUP_TIMEOUT = 30                 # seconds to wait for the worker socket
SHUTDOWN_TIMEOUT = 10                # seconds for graceful terminate


def memory_kill_notice(port: int | str) -> str | None:
    """If the worker on ``port`` killed itself over the memory cap, consume its
    marker and return the message the executor should see instead of a bare
    socket error (which the loop would otherwise retry as a transient blip)."""
    from evo.engine.sim_worker import memkill_marker_path
    marker = memkill_marker_path(port)
    if not marker.exists():
        return None
    try:
        info = json.loads(marker.read_text())
    except (OSError, ValueError):
        info = {}
    try:
        marker.unlink()
    except OSError:
        pass
    rss_gb = float(info.get("rss_mb", 0)) / 1024
    limit_gb = float(info.get("limit_mb", 0)) / 1024
    return (
        f"MemoryError: the sim worker for this session was killed: it stayed at "
        f"{rss_gb:.1f} GB (resident + compressed), above the {limit_gb:.1f} GB per-session memory cap, for "
        f"{info.get('grace_sec', '?')} s while running your snippet and could not be "
        f"interrupted. The Python session and all its variables are gone; further sim "
        f"calls will fail."
    )


class SimStartupError(RuntimeError):
    """Raised when the sim worker fails to start or its socket never opens."""


def send_to_sim_worker(code: str, port: int, timeout: float = 600.0) -> dict:
    """Execute Python source in the sim worker on `port`; return its reply.

    Wire protocol (shared with the legacy Blender add-on): one NUL-terminated
    JSON request {"type": "execute", "code": ...}, one NUL-terminated JSON
    reply {"status", "result", "message", "stdout", "stderr"}.
    """
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
        conn.sendall(
            json.dumps({"type": "execute", "code": code}).encode("utf-8") + b"\0"
        )
        buffer = b""
        while not buffer.endswith(b"\0"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            buffer += chunk
    if not buffer:
        raise ConnectionError(f"sim worker on port {port} closed without replying")
    return json.loads(buffer.rstrip(b"\0").decode("utf-8"))


class SimWorkerManager:
    """Context manager for a single sim-worker session.

    Usage:
        with SimWorkerManager(port=9876, runtime_globals={...}):
            # agent runs here, driving mcp__sim__execute_python_code
            ...
        # worker killed on exit

    ``headless`` is accepted for call-site compatibility; the worker never
    has a GUI.
    """

    def __init__(
        self,
        port: int = SIM_PORT,
        startup_timeout: int = STARTUP_TIMEOUT,
        headless: bool = True,
        runtime_globals: dict[str, object] | None = None,
    ):
        self.port = port
        self.startup_timeout = startup_timeout
        self.headless = headless
        self.runtime_globals = {
            str(k): str(v) for k, v in (runtime_globals or {}).items()
        }
        self.process: subprocess.Popen | None = None

    # ---- lifecycle ----

    def start(self) -> None:
        self._kill_stale()
        env = os.environ.copy()
        env[GLOBALS_ENV] = json.dumps(self.runtime_globals)
        # A context manager cannot run __exit__ if the whole training process
        # is terminated.  Give the worker an explicit parent identity so its
        # watchdog can terminate it (and therefore close LightPhyx's watched
        # stdin pipe) instead of leaving an orphaned simulation stack.
        env[PARENT_PID_ENV] = str(os.getpid())
        from evo.engine.sim_worker import memkill_marker_path
        try:
            memkill_marker_path(self.port).unlink()
        except OSError:
            pass
        project_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = (
            project_root + os.pathsep + env["PYTHONPATH"]
            if env.get("PYTHONPATH") else project_root
        )
        cmd = [sys.executable, "-m", "evo.engine.sim_worker", "--port", str(self.port)]
        log.info(f"Launching sim worker on port {self.port}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        try:
            self._wait_for_socket()
        except SimStartupError:
            self.stop()
            raise
        log.info(f"Sim worker ready on port {self.port} (pid={self.process.pid})")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            pid = self.process.pid
            log.info(f"Stopping sim worker (pid={pid})")
            self.process.terminate()
            try:
                self.process.wait(timeout=SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                log.warning(f"Sim worker pid={pid} did not terminate; killing")
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    # ---- helpers ----

    def _kill_stale(self) -> None:
        """Free OUR specific port from stale *sim-worker* processes only.

        Port-scoped (parallel workers never touch each other's ports) and
        command-checked: a foreign process holding the port is a config
        error worth surfacing, not something to silently kill.
        """
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{self.port}"],
                capture_output=True, text=True, check=False,
            )
        except FileNotFoundError:
            log.warning(f"lsof not available; skipping port-{self.port} cleanup")
            return
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        stale: list[str] = []
        for pid in pids:
            args = subprocess.run(
                ["ps", "-p", pid, "-o", "command="],
                capture_output=True, text=True, check=False,
            ).stdout.strip().lower()
            if "sim_worker" in args or "blender" in args:
                stale.append(pid)
            else:
                log.warning(
                    f"Port {self.port} is held by a foreign process pid={pid} "
                    f"({args[:80] or 'unknown'}); NOT killing it — startup will "
                    f"fail. Free the port or change --sim-base-port."
                )
        if not stale:
            return
        for pid in stale:
            subprocess.run(["kill", pid], check=False, capture_output=True)
        deadline = time.time() + 3
        remaining = stale
        while remaining and time.time() < deadline:
            time.sleep(0.2)
            remaining = [
                p for p in remaining
                if subprocess.run(["kill", "-0", p], check=False,
                                  capture_output=True).returncode == 0
            ]
        for pid in remaining:
            log.warning(f"Stale worker pid={pid} ignored SIGTERM; sending SIGKILL")
            subprocess.run(["kill", "-9", pid], check=False, capture_output=True)
        time.sleep(0.2)  # let the OS release the port

    def _wait_for_socket(self) -> None:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise SimStartupError(
                    f"sim worker exited during startup "
                    f"(exit code {self.process.returncode}); check that the "
                    f"evo package imports cleanly in {sys.executable}"
                )
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    return
            except OSError:
                time.sleep(0.1)
        raise SimStartupError(
            f"sim worker socket (port {self.port}) not ready after {self.startup_timeout}s"
        )

    # ---- context manager ----

    def __enter__(self) -> "SimWorkerManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
