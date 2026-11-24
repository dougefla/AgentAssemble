"""The sim worker's per-session memory cap (run 18: 164 GB, run 23: 112 GB)."""
from __future__ import annotations

import socket
import time

import pytest

from evo.engine.sim_manager import SimWorkerManager, memory_kill_notice, send_to_sim_worker
from evo.engine.sim_worker import MEMORY_GRACE_ENV, MEMORY_LIMIT_ENV, memkill_marker_path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_memory_guard_interrupts_a_runaway_snippet_and_keeps_the_session(monkeypatch):
    monkeypatch.setenv(MEMORY_LIMIT_ENV, "600")
    monkeypatch.setenv(MEMORY_GRACE_ENV, "30")
    port = _free_port()
    with SimWorkerManager(port=port, runtime_globals={"INPUT_DIR": "/tmp"}):
        assert send_to_sim_worker("small = 1", port)["status"] == "ok"
        rep = send_to_sim_worker(
            "keep = []\nwhile True:\n    keep.append(bytearray(50 * 2 ** 20))\n", port, timeout=60,
        )
        assert rep["status"] == "error"
        msg = rep["message"]
        assert msg.startswith("MemoryError: this snippet pushed the sim session to")
        assert "per-session memory cap" in msg and "trimesh" not in msg  # facts only, no advice
        assert 'File "<agent_code>"' in msg  # the loop classifies it as an executor code error
        assert "All session variables were cleared" in msg
        # The worker survives, answers, and the runtime globals are intact.
        after = send_to_sim_worker("print(INPUT_DIR); 'keep' in globals() or 'small' in globals()", port)
        assert after["status"] == "ok" and after["result"] is False and "/tmp" in after["stdout"]
    assert not memkill_marker_path(port).exists()


def test_memory_guard_kills_a_snippet_that_swallows_the_interrupt(monkeypatch):
    monkeypatch.setenv(MEMORY_LIMIT_ENV, "600")
    monkeypatch.setenv(MEMORY_GRACE_ENV, "1.5")
    port = _free_port()
    with SimWorkerManager(port=port) as mgr:
        pid = mgr.process.pid
        t0 = time.time()
        with pytest.raises(Exception):
            send_to_sim_worker(
                "import time\nkeep = [bytearray(50 * 2 ** 20) for _ in range(16)]\n"
                "while True:\n    try:\n        time.sleep(0.01)\n    except BaseException:\n        pass\n",
                port, timeout=30,
            )
        assert time.time() - t0 < 20
        deadline = time.time() + 5
        while mgr.process.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        assert mgr.process.poll() == 3, f"worker pid {pid} should have exited with code 3"
        notice = memory_kill_notice(port)
        assert notice and notice.startswith("MemoryError: the sim worker for this session was killed")
        assert "0.6 GB per-session memory cap" in notice and "trimesh" not in notice
        assert memory_kill_notice(port) is None  # consumed
