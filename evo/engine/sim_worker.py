"""Persistent Python execution worker — the sim-backend successor to the
headless-Blender MCP add-on.

One worker per agent session, launched by ``SimWorkerManager``. It serves the
same NUL-terminated JSON protocol the official Blender add-on spoke, so the
transport layer (``_send_to_sim_worker``, the Codex stdio bridge) carries over
unchanged:

    request:  {"type": "execute", "code": "<python source>"}\\0
    reply:    {"status": "ok"|"error", "result": ..., "message": ...,
               "stdout": "...", "stderr": "..."}\\0

Unlike Blender's add-on, the namespace is PERSISTENT across calls: variables,
imports and the preloaded ``assembly`` object survive. The immutable runtime
globals (ASSEMBLY_SDK_DIR / SKILL_DIR / INPUT_DIR / OUTPUT_DIR, passed as the
EVO_SIM_GLOBALS JSON env var) are re-injected before every execution, so an
agent reassigning one only affects its own call, never the session.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback

from evo.assembly_sdk.core import AssemblyError

GLOBALS_ENV = "EVO_SIM_GLOBALS"
PARENT_PID_ENV = "EVO_SIM_PARENT_PID"
PORT_ENV = "EVO_SIM_WORKER_PORT"
# Per-session resident-memory cap. Run 18 (164 GB) and run 23 (112 GB) each
# had one worker eat the machine: trimesh.load(..., force="mesh") decodes the
# 8192² textures the ArtVIP .mtl files reference (≈0.8 GB per part) and
# mesh.split() duplicates them per component. The guard interrupts the
# snippet, frees the session, and tells the executor how to load geometry
# only; a snippet it cannot interrupt gets the worker killed after a grace
# period and a marker file explains the death to the tool wrapper.
MEMORY_LIMIT_ENV = "EVO_SIM_WORKER_MAX_RSS_MB"
MEMORY_GRACE_ENV = "EVO_SIM_WORKER_MEMORY_GRACE_SEC"
DEFAULT_MAX_RSS_MB = 6144
DEFAULT_MEMORY_GRACE_SEC = 10.0
MEMORY_POLL_SEC = 0.1



def memkill_marker_path(port: int | str) -> "Path":
    """Where a memory-killed worker leaves its last words for the tool wrapper."""
    import tempfile
    from pathlib import Path
    return Path(tempfile.gettempdir()) / f"evo-sim-worker-{port}.memkill"


class SimMemoryLimitExceeded(MemoryError):
    """Raised inside the running snippet when the session passes the RSS cap."""


def _current_rss_bytes() -> int:
    """Memory footprint of this process (resident + compressed/swapped on
    macOS, resident on Linux), without psutil."""
    if sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util
            libproc = _current_rss_bytes._libproc  # type: ignore[attr-defined]
        except AttributeError:
            try:
                libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
            except OSError:
                libproc = None
            _current_rss_bytes._libproc = libproc  # type: ignore[attr-defined]
        if libproc is not None:
            class _RUsageInfoV2(ctypes.Structure):
                _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
                    (name, ctypes.c_uint64) for name in (
                        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
                        "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
                        "ri_resident_size", "ri_phys_footprint",
                        "ri_proc_start_abstime", "ri_proc_exit_abstime",
                        "ri_child_user_time", "ri_child_system_time",
                        "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
                        "ri_child_pageins", "ri_child_elapsed_abstime",
                        "ri_diskio_bytesread", "ri_diskio_byteswritten",
                    )
                ]
            buf = _RUsageInfoV2()
            if libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) == 0:
                # phys_footprint counts pages the compressor or swap took away;
                # resident_size does not. Run 25's 100 GB worker showed 0.8 GB
                # resident with 111 GB compressed, so the guard never fired.
                return int(max(buf.ri_resident_size, buf.ri_phys_footprint))
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class _MemoryGuard:
    """Watchdog thread: interrupt a snippet that crosses the RSS cap; kill the
    worker if the snippet keeps running above the cap for ``grace_sec``."""

    def __init__(self, limit_mb: int, grace_sec: float, port: int | str | None) -> None:
        self.limit_bytes = int(limit_mb) * 2 ** 20
        self.grace_sec = float(grace_sec)
        self.port = port
        self.executing = False
        self.tripped_rss = 0
        self._main_ident = threading.main_thread().ident
        self._signalled = False
        self._over_since: float | None = None

    def start(self) -> None:
        signal.signal(signal.SIGUSR1, self._on_signal)
        threading.Thread(target=self._loop, name="sim-worker-memory-guard", daemon=True).start()

    def _on_signal(self, *_: object) -> None:
        if self.executing:
            raise SimMemoryLimitExceeded(
                f"session memory footprint {self.tripped_rss / 2 ** 30:.2f} GB exceeds the "
                f"{self.limit_bytes / 2 ** 30:.1f} GB per-session memory cap"
            )

    def begin(self) -> None:
        self.executing = True
        self._signalled = False

    def end(self) -> None:
        self.executing = False
        self._signalled = False

    def _loop(self) -> None:
        while True:
            time.sleep(MEMORY_POLL_SEC)
            rss = _current_rss_bytes()
            if rss <= self.limit_bytes or not self.executing:
                self._over_since = None
                continue
            now = time.monotonic()
            if self._over_since is None:
                self._over_since = now
            if not self._signalled and self._main_ident is not None:
                self.tripped_rss = rss
                self._signalled = True
                signal.pthread_kill(self._main_ident, signal.SIGUSR1)
            if now - self._over_since > self.grace_sec:
                self._die(rss)

    def _die(self, rss: int) -> None:
        try:
            memkill_marker_path(self.port if self.port is not None else os.getpid()).write_text(json.dumps({
                "pid": os.getpid(), "port": self.port, "rss_mb": rss // 2 ** 20,
                "limit_mb": self.limit_bytes // 2 ** 20, "grace_sec": self.grace_sec,
                "time": time.time(),
            }))
        except OSError:
            pass
        os._exit(3)


_GUARD: _MemoryGuard | None = None


def _start_memory_guard(port: int | str | None) -> _MemoryGuard | None:
    global _GUARD
    try:
        limit_mb = int(os.environ.get(MEMORY_LIMIT_ENV, DEFAULT_MAX_RSS_MB))
        grace = float(os.environ.get(MEMORY_GRACE_ENV, DEFAULT_MEMORY_GRACE_SEC))
    except ValueError:
        limit_mb, grace = DEFAULT_MAX_RSS_MB, DEFAULT_MEMORY_GRACE_SEC
    if limit_mb <= 0:
        return None
    _GUARD = _MemoryGuard(limit_mb, grace, port)
    _GUARD.start()
    return _GUARD


def _watch_parent(parent_pid: int, poll_seconds: float = 0.5) -> None:
    """Terminate this worker after its creating training process disappears.

    ``SimWorkerManager.__exit__`` handles normal completion.  This watchdog
    covers abrupt SIGTERM/SIGKILL of the parent, when Python never unwinds the
    manager context and the worker would otherwise be reparented to PID 1.
    """
    while os.getppid() == parent_pid:
        time.sleep(poll_seconds)
    os.kill(os.getpid(), signal.SIGTERM)


def _start_parent_watchdog() -> None:
    raw = os.environ.get(PARENT_PID_ENV, "")
    try:
        parent_pid = int(raw)
    except (TypeError, ValueError):
        return
    if parent_pid <= 1:
        return
    threading.Thread(
        target=_watch_parent,
        args=(parent_pid,),
        name="sim-worker-parent-watchdog",
        daemon=True,
    ).start()


def _runtime_globals() -> dict[str, str]:
    raw = os.environ.get(GLOBALS_ENV, "")
    if not raw:
        return {}
    values = json.loads(raw)
    return {str(k): str(v) for k, v in values.items() if v is not None}


def _make_namespace(runtime_globals: dict[str, str]) -> dict:
    namespace: dict = {"__name__": "__evo_sim_session__", "__builtins__": __builtins__}
    namespace.update(runtime_globals)
    for key, value in runtime_globals.items():
        os.environ[key] = value
    return namespace


def _preload_assembly(namespace: dict) -> None:
    """Best-effort preload of the Assembly SDK so `assembly` is ready."""
    sdk_dir = namespace.get("ASSEMBLY_SDK_DIR")
    if not sdk_dir:
        return
    runtime = os.path.join(str(sdk_dir), "runtime.py")
    try:
        with open(runtime) as stream:
            exec(compile(stream.read(), runtime, "exec"), namespace)
    except Exception:
        print("[sim_worker] assembly preload failed:", file=sys.stderr)
        traceback.print_exc()


def _parse_repairing_stray_indent(code: str, max_repairs: int = 8) -> tuple[ast.Module, list[str]]:
    """ast.parse that tolerates a lone single-space indent on a statement line.

    gpt-6-astra (run 12, 2026-09-29) emits one stray leading space on roughly
    one line in six of some snippets; Python raises IndentationError and the
    training loop then restarts the whole executor session (29 restarts in
    five rounds). A statement line indented by exactly one space after a
    column-0 line that opens no block can never be valid Python, so stripping
    that space is unambiguous. Anything else is re-raised untouched.
    """
    lines = code.splitlines(keepends=True)
    notes: list[str] = []
    for _ in range(max_repairs):
        try:
            return ast.parse(code, mode="exec"), notes
        except IndentationError as exc:
            lineno = exc.lineno or 0
            if exc.msg != "unexpected indent" or not 1 <= lineno <= len(lines):
                raise
            line = lines[lineno - 1]
            if not (line.startswith(" ") and not line.startswith("  ")):
                raise
            lines[lineno - 1] = line[1:]
            code = "".join(lines)
            notes.append(f"[sim_worker] stripped a stray leading space on line {lineno}")
    return ast.parse(code, mode="exec"), notes


def _execute(code: str, namespace: dict, runtime_globals: dict[str, str]) -> dict:
    # Immutable globals win over any reassignment a previous call attempted.
    namespace.update(runtime_globals)
    for key, value in runtime_globals.items():
        os.environ[key] = value

    stdout, stderr = io.StringIO(), io.StringIO()
    result = None
    try:
        tree, repair_notes = _parse_repairing_stray_indent(code)
        for note in repair_notes:
            stderr.write(note + "\n")
        trailing_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            trailing_expr = ast.Expression(tree.body.pop(-1).value)
        # Add-on compatibility: code that assigns a variable named `result`
        # gets that variable back; otherwise the trailing expression wins.
        assigns_result = any(
            isinstance(target, ast.Name) and target.id == "result"
            for node in ast.walk(tree)
            for target in (
                node.targets if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign))
                else []
            )
        )
        guard = _GUARD
        if guard is not None:
            guard.begin()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(tree, "<agent_code>", "exec"), namespace)
                if trailing_expr is not None:
                    result = eval(compile(trailing_expr, "<agent_code>", "eval"), namespace)
                elif assigns_result:
                    result = namespace.get("result")
        finally:
            if guard is not None:
                guard.end()
    except SimMemoryLimitExceeded as exc:
        tb = traceback.format_exc(limit=16)
        cleared = _recover_memory(namespace, runtime_globals, guard)
        return {
            "status": "error",
            "message": (
                f"MemoryError: this snippet pushed the sim session to "
                f"{guard.tripped_rss / 2 ** 30:.2f} GB (resident + compressed), over the "
                f"{guard.limit_bytes / 2 ** 30:.1f} GB per-session memory cap, and was interrupted; "
                f"its result is discarded. "
                + ("All session variables were cleared to free memory."
                   if cleared else "Session variables were kept.")
                + "\n" + tb
            ),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
    except AssemblyError as exc:
        # A deterministic SDK contract rejection is an expected result of
        # agent-authored code, not a simulation-engine failure.  Preserve the
        # structured error and keep the MCP/tool call successful so the agent
        # can inspect, correct, and retry it.  Transport/runtime failures still
        # take the generic error path below and remain blocking.
        return {
            "status": "ok",
            "result": exc.as_dict(),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
    except Exception:
        return {
            "status": "error",
            "message": traceback.format_exc(limit=16),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        result = repr(result)
    return {
        "status": "ok",
        "result": result,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
    }


def _recover_memory(namespace: dict, runtime_globals: dict[str, str], guard: "_MemoryGuard") -> bool:
    """After an interrupted snippet: collect garbage; if the session is still
    above three quarters of the cap, drop every user variable (keeping the
    runtime globals and the preloaded SDK). Returns whether it cleared."""
    import gc
    gc.collect()
    if _current_rss_bytes() <= guard.limit_bytes * 0.75:
        return False
    keep = set(runtime_globals) | {"__builtins__", "__name__", "__doc__", "assembly"}
    for key in [k for k in namespace if k not in keep]:
        namespace.pop(key, None)
    gc.collect()
    try:
        _preload_assembly(namespace)
    except Exception:  # noqa: BLE001 — a failed re-preload must not mask the memory error
        pass
    return True


def _serve_connection(conn: socket.socket, namespace: dict,
                      runtime_globals: dict[str, str]) -> None:
    buffer = b""
    while True:
        try:
            chunk = conn.recv(65536)
        except OSError:
            return
        if not chunk:
            return
        buffer += chunk
        while b"\0" in buffer:
            frame, buffer = buffer.split(b"\0", 1)
            try:
                request = json.loads(frame.decode("utf-8"))
            except Exception as exc:
                reply = {"status": "error", "message": f"malformed request: {exc}"}
            else:
                if request.get("type") != "execute":
                    reply = {"status": "error",
                             "message": f"unsupported request type: {request.get('type')!r}"}
                else:
                    reply = _execute(str(request.get("code", "")), namespace, runtime_globals)
            try:
                conn.sendall(json.dumps(reply).encode("utf-8") + b"\0")
            except OSError:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    # SimWorkerManager stops us with SIGTERM, which bypasses atexit by
    # default — convert it to a clean SystemExit so session resources
    # (notably the lightphyx render server the SDK may have launched)
    # shut down instead of leaking one orphaned node process per session.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    _start_parent_watchdog()
    _start_memory_guard(args.port)

    runtime_globals = _runtime_globals()
    namespace = _make_namespace(runtime_globals)
    _preload_assembly(namespace)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(4)
    print(f"[sim_worker] serving on {args.host}:{args.port}", flush=True)
    while True:
        conn, _ = server.accept()
        with conn:
            _serve_connection(conn, namespace, runtime_globals)


if __name__ == "__main__":
    main()
