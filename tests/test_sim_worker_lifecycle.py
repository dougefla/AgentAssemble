"""Lifecycle regressions for the persistent simulation worker."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evo.assembly_sdk.core import ErrorCode, fail
from evo.engine.sim_worker import _execute


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _process_is_live(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    # A dead orphan can briefly remain as a zombie until PID 1 reaps it; it
    # no longer owns sockets or child processes and counts as terminated.
    return stat.read_text().split()[2] != "Z"


def test_sdk_contract_error_is_a_structured_recoverable_result() -> None:
    namespace = {"fail": fail, "ErrorCode": ErrorCode}
    response = _execute(
        "fail(ErrorCode.UNKNOWN_HANDLE, 'set_part_scale', "
        "'unknown part handle', handle='stale')",
        namespace,
        {},
    )

    assert response["status"] == "ok"
    assert response["result"] == {
        "ok": False,
        "error": {
            "code": "unknown_handle",
            "operation": "set_part_scale",
            "message": "unknown part handle",
            "details": {"handle": "stale"},
        },
    }
    assert "Traceback" not in response.get("message", "")


def test_unexpected_agent_python_error_remains_a_tool_error() -> None:
    response = _execute("raise RuntimeError('boom')", {}, {})
    assert response["status"] == "error"
    assert "RuntimeError: boom" in response["message"]


@pytest.mark.skipif(sys.platform != "linux", reason="uses /proc lifecycle probe")
def test_worker_exits_when_manager_process_dies(tmp_path: Path) -> None:
    """SIGKILL-like parent loss must not orphan a port-owning sim worker."""
    pid_file = tmp_path / "worker.pid"
    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import os\n"
        "from pathlib import Path\n"
        "from evo.engine.sim_manager import SimWorkerManager\n"
        f"manager = SimWorkerManager(port={_free_port()})\n"
        "manager.start()\n"
        f"Path({str(pid_file)!r}).write_text(str(manager.process.pid))\n"
        "os._exit(0)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30)
    pid = int(pid_file.read_text())
    try:
        deadline = time.monotonic() + 5
        while _process_is_live(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _process_is_live(pid), f"orphan sim-worker still live: pid={pid}"
    finally:
        if _process_is_live(pid):
            os.kill(pid, 9)


def test_stray_single_space_indent_is_repaired_not_fatal() -> None:
    # gpt-6-astra's habit: one statement line with a lone leading space.
    response = _execute("import os\n x = 1\ny = x + 1\n z = y * 2\nz", {}, {})
    assert response["status"] == "ok" and response["result"] == 4
    assert response["stderr"].count("stripped a stray leading space") == 2
    # Real indentation mistakes (two spaces, or an indent after a block opener) still fail.
    assert _execute("import os\n  x = 1\n", {}, {})["status"] == "error"
    assert "IndentationError" in _execute("if True:\nx = 1\n", {}, {})["message"]

