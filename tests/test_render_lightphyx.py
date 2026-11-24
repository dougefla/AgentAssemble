"""Unit tests for the lightphyx oracle-render backend (no server needed).

Covers the pure-math pieces that must stay in lockstep with the legacy
BLENDER_SCRIPT: rpy decomposition, OBJ AABB parsing, the flattened-URDF
writer, and the EVO_RENDER_BACKEND switch. Rendering itself is exercised
by the A/B comparison flow, not unit tests.
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from evo.oracle.render_lightphyx import (
    _launch_local,
    _local_aabb,
    _mat_to_xyz_rpy,
    _mesh_entries,
    _split_uniform_scale,
    _write_flat_urdf,
    _world_aabb,
    render_scene,
)
from evo.oracle.render_motion import render_backend


def _rpy_to_mat(rpy, xyz=(0.0, 0.0, 0.0)):
    """Reference composition — same formula as render_motion.build_pose."""
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
        [-sp, cp * sr, cp * cr, xyz[2]],
        [0, 0, 0, 1],
    ]


def _mat_close(a, b, tol=1e-9):
    return all(abs(a[i][j] - b[i][j]) < tol for i in range(4) for j in range(4))


def test_launch_local_uses_recovery_controls_on_new_sdk():
    calls = []

    class ModernSDK:
        @staticmethod
        def launch(*, rpc_timeout=120.0, startup_attempts=1):
            calls.append((rpc_timeout, startup_attempts))
            return "modern"

    assert _launch_local(ModernSDK) == "modern"
    assert calls == [(45.0, 3)]


def test_launch_local_falls_back_for_legacy_sdk_signature():
    calls = []

    class LegacySDK:
        @staticmethod
        def launch(headless=True, out_dir=None, pool=None):
            calls.append((headless, out_dir, pool))
            return "legacy"

    assert _launch_local(LegacySDK) == "legacy"
    assert calls == [(True, None, None)]


@pytest.mark.parametrize("rpy", [
    (0.0, 0.0, 0.0),
    (0.3, -0.7, 2.1),
    (-3.0, 1.2, -0.4),
    (0.1, math.pi / 2, 0.2),        # gimbal lock, pitch = +90°
    (0.5, -math.pi / 2, -1.0),      # gimbal lock, pitch = -90°
])
def test_mat_to_xyz_rpy_roundtrip(rpy):
    xyz = (1.5, -2.0, 0.25)
    M = _rpy_to_mat(rpy, xyz)
    got_xyz, got_rpy = _mat_to_xyz_rpy(M)
    assert got_xyz == pytest.approx(xyz)
    # rpy itself may differ (gimbal degeneracy) — the matrix must match.
    assert _mat_close(_rpy_to_mat(got_rpy, got_xyz), M, tol=1e-9)


def test_local_aabb_parses_vertices(tmp_path):
    obj = tmp_path / "tri.obj"
    obj.write_text("v -1 0 2\nv 3 -4 0\nv 0 5 -6\nf 1 2 3\n")
    mn, mx = _local_aabb(str(obj))
    assert mn == (-1.0, -4.0, -6.0)
    assert mx == (3.0, 5.0, 2.0)


def test_world_aabb_uses_transformed_corners(tmp_path):
    # Unit cube [0,1]^3 rotated 90° about Z then shifted +10 in x:
    # corners land in x∈[9,10], y∈[0,1], z∈[0,1].
    obj = tmp_path / "cube.obj"
    lines = [f"v {x} {y} {z}" for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    obj.write_text("\n".join(lines) + "\n")
    M = _rpy_to_mat((0.0, 0.0, math.pi / 2), (10.0, 0.0, 0.0))
    mn, mx = _world_aabb([(M, str(obj))])
    assert mn == pytest.approx((9.0, 0.0, 0.0))
    assert mx == pytest.approx((10.0, 1.0, 1.0))


def test_mesh_entries_prefers_parts_map(tmp_path):
    obj_a = tmp_path / "a.obj"
    obj_b = tmp_path / "b.obj"
    for p in (obj_a, obj_b):
        p.write_text("v 0 0 0\n")
    links = {
        "root": {"visual_xyz": (0, 0, 0), "visual_rpy": (0, 0, 0),
                 "mesh_path": "a.obj"},
        "door": {"visual_xyz": (1, 0, 0), "visual_rpy": (0, 0, 0),
                 "mesh_path": None},
    }
    poses = {"root": _rpy_to_mat((0, 0, 0)), "door": _rpy_to_mat((0, 0, 0))}
    entries = _mesh_entries(links, poses, tmp_path,
                            {"door": [str(obj_b)]})
    paths = sorted(p for _, p in entries)
    assert paths == sorted([str(obj_a), str(obj_b)])
    # door's visual_xyz folded into its world matrix
    door_M = next(M for M, p in entries if p == str(obj_b))
    assert door_M[0][3] == pytest.approx(1.0)


def test_mesh_entries_reads_parts_map_v2_scale(tmp_path):
    obj = tmp_path / "part.obj"
    obj.write_text("v -0.5 0 0\nv 0.5 0 0\n")
    links = {
        "root": {
            "visual_xyz": (2, 0, 0),
            "visual_rpy": (0, 0, math.pi / 2),
            "mesh_path": None,
        },
    }
    poses = {"root": _rpy_to_mat((0, 0, 0))}
    parts_map = {
        "version": 2,
        "links": {"root": [{"path": str(obj), "scale": 0.25}]},
    }
    entries = _mesh_entries(links, poses, tmp_path, parts_map)
    assert len(entries) == 1
    matrix, path = entries[0]
    rigid, scale = _split_uniform_scale(matrix)
    assert path == str(obj)
    assert scale == pytest.approx(0.25)
    assert rigid[0][3] == pytest.approx(2.0)
    mn, mx = _world_aabb(entries)
    assert mx[1] - mn[1] == pytest.approx(0.25)


def test_write_flat_urdf_bakes_world_pose(tmp_path):
    mesh = tmp_path / "meshes" / "part.obj"
    mesh.parent.mkdir()
    mesh.write_text("v 0 0 0\n")
    urdf = tmp_path / "out" / "scene.urdf"
    urdf.parent.mkdir()
    M = _rpy_to_mat((0.1, 0.2, 0.3), (4.0, 5.0, 6.0))
    _write_flat_urdf(urdf, [(M, str(mesh))])

    root = ET.parse(urdf).getroot()
    assert [l.attrib["name"] for l in root.findall("link")] == ["base", "m0"]
    joint = root.find("joint")
    assert joint.attrib["type"] == "fixed"
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.attrib["xyz"].split()]
    rpy = [float(v) for v in origin.attrib["rpy"].split()]
    assert xyz == pytest.approx([4.0, 5.0, 6.0])
    assert _mat_close(_rpy_to_mat(rpy, xyz), M, tol=1e-6)
    # mesh path is URDF-relative and resolves back to the real file
    rel = root.find("link[@name='m0']/visual/geometry/mesh").attrib["filename"]
    assert not os.path.isabs(rel)
    assert (urdf.parent / rel).resolve() == mesh.resolve()


def test_write_flat_urdf_preserves_uniform_mesh_scale(tmp_path):
    mesh = tmp_path / "part.obj"
    mesh.write_text("v -0.5 0 0\nv 0.5 0 0\n")
    urdf = tmp_path / "scene.urdf"
    matrix = _rpy_to_mat((0.1, -0.2, 0.3), (1.0, 2.0, 3.0))
    for row in range(3):
        for column in range(3):
            matrix[row][column] *= 0.375

    _write_flat_urdf(urdf, [(matrix, str(mesh))])

    root = ET.parse(urdf).getroot()
    visual = root.find("link[@name='m0']/visual/geometry/mesh")
    assert [float(value) for value in visual.attrib["scale"].split()] == pytest.approx(
        [0.375, 0.375, 0.375]
    )
    origin = root.find("joint/origin")
    xyz = [float(value) for value in origin.attrib["xyz"].split()]
    rpy = [float(value) for value in origin.attrib["rpy"].split()]
    assert _mat_close(
        _rpy_to_mat(rpy, xyz),
        _rpy_to_mat((0.1, -0.2, 0.3), (1.0, 2.0, 3.0)),
        tol=1e-6,
    )


@pytest.fixture
def fresh_pool(monkeypatch):
    """Isolate the module-level Sim pool; returns the module."""
    import threading

    from evo.oracle import render_lightphyx as rl

    monkeypatch.setattr(rl, "_POOL", threading.Condition())
    monkeypatch.setattr(rl, "_IDLE", [])
    monkeypatch.setattr(rl, "_LIVE", 0)
    monkeypatch.setattr(rl, "_MAX_SIMS", 1)
    monkeypatch.setattr(rl, "_GENERATION", 0)
    monkeypatch.setattr(rl, "_ATEXIT_REGISTERED", True)
    return rl


class _FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeSim:
    """Records the calls render_scene makes; `capture` is injectable."""

    def __init__(self, capture=None, fail_with=None):
        import numpy as np

        self.removed = []
        self.closed = False
        self._client = _FakeClient()
        self._fail_with = fail_with
        self._capture = capture or (lambda passes, outline: {
            "rgb": np.full((8, 8, 3), 7, dtype=np.uint8),
            "seg": np.ones((8, 8), dtype=np.int32),
        })

    def load_urdf(self, path, name):
        assert Path(path).is_file()
        if self._fail_with is not None:
            raise self._fail_with
        return type("Robot", (), {"name": name})()

    def camera(self, **kwargs):
        sim = self

        class Cam:
            def capture(self, passes, outline=False, **extra):
                # Nothing may be saved server-side any more (item: hub PNG work).
                assert not extra, extra
                assert tuple(passes) == ("rgb", "seg")
                assert isinstance(outline, bool)
                return sim._capture(passes, outline)

        return Cam()

    def remove_robot(self, name):
        self.removed.append(name)

    def close(self):
        self.closed = True


def _render(tmp_path, out_name="preview.png"):
    mesh = tmp_path / "part.obj"
    mesh.write_text("v 0 0 0\n")
    out = tmp_path / out_name
    render_scene(
        [(_rpy_to_mat((0.0, 0.0, 0.0)), str(mesh))],
        out,
        cam_pos=(2.0, -2.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
        resolution=8,
    )
    return out


def test_render_scene_rejects_background_only_segmentation(tmp_path, fresh_pool, monkeypatch):
    """A successful RPC and non-empty RGB are insufficient render evidence."""
    import numpy as np

    fake = _FakeSim(capture=lambda passes, outline: {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "seg": np.zeros((8, 8), dtype=np.int32),
    })
    monkeypatch.setattr(fresh_pool, "_new_sim", lambda: fake)

    with pytest.raises(RuntimeError, match="no foreground geometry"):
        _render(tmp_path)

    assert fake.removed and fake.removed[0].startswith("__lp_")
    assert not list(tmp_path.glob("__lp_*"))
    assert not (tmp_path / "preview.png").exists()
    # A render-level failure leaves the connection usable.
    assert fresh_pool._IDLE == [fake] and not fake.closed


def test_render_scene_writes_rgb_png_client_side(tmp_path, fresh_pool, monkeypatch):
    from PIL import Image

    fake = _FakeSim()
    monkeypatch.setattr(fresh_pool, "_new_sim", lambda: fake)
    out = _render(tmp_path)
    with Image.open(out) as img:
        assert img.mode == "RGB" and img.size == (8, 8)
        assert img.getpixel((3, 3)) == (7, 7, 7)
    # Only the PNG is left behind: no temp URDF, seg PNG or labels JSON.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["part.obj", "preview.png"]


def test_transport_failure_drops_socket_without_destroy_and_retries(tmp_path, fresh_pool, monkeypatch):
    broken = _FakeSim(fail_with=ConnectionError("page disconnected"))
    good = _FakeSim()
    sims = iter([broken, good])
    monkeypatch.setattr(fresh_pool, "_new_sim", lambda: next(sims))

    _render(tmp_path)

    # The broken Sim's socket is closed BEFORE close(): its destroy_envs then
    # fails fast instead of waiting out the RPC deadline on a dead page.
    assert broken._client.closed and broken.closed
    assert fresh_pool._IDLE == [good] and fresh_pool._LIVE == 1


def test_two_transport_failures_raise_unavailable(tmp_path, fresh_pool, monkeypatch):
    from evo.oracle.render_lightphyx import LightphyxUnavailable

    monkeypatch.setattr(
        fresh_pool, "_new_sim", lambda: _FakeSim(fail_with=ConnectionError("down")),
    )
    with pytest.raises(LightphyxUnavailable, match="failed twice"):
        _render(tmp_path)
    assert fresh_pool._LIVE == 0 and fresh_pool._IDLE == []


def test_pool_renders_concurrently_up_to_its_size(tmp_path, fresh_pool, monkeypatch):
    import threading
    import time

    import numpy as np

    active = 0
    peak = 0
    lock = threading.Lock()

    def slow_capture(passes, outline):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"rgb": np.zeros((8, 8, 3), np.uint8), "seg": np.ones((8, 8), np.int32)}

    made = []

    def new_sim():
        sim = _FakeSim(capture=slow_capture)
        made.append(sim)
        return sim

    monkeypatch.setattr(fresh_pool, "_new_sim", new_sim)
    fresh_pool.configure_render_pool(3)
    threads = [
        threading.Thread(target=_render, args=(tmp_path, f"f{i}.png")) for i in range(9)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak == 3            # parallel, but never above the cap
    assert len(made) == 3       # Sims are reused, not reconnected per frame
    assert len(list(tmp_path.glob("f*.png"))) == 9


def test_shutdown_retires_leased_sims_on_return(fresh_pool):
    sim = _FakeSim()
    fresh_pool._IDLE.append(sim)
    fresh_pool._LIVE = 1
    leased, generation = fresh_pool._lease()
    fresh_pool._shutdown()               # e.g. the pool is being recycled
    fresh_pool._give_back(leased, generation, healthy=True)
    assert leased.closed and fresh_pool._IDLE == [] and fresh_pool._LIVE == 0


def test_shared_connect_gets_a_deadline(fresh_pool, monkeypatch):
    import sys
    from types import SimpleNamespace

    seen = {}

    def connect(url, *, physics=True, rpc_timeout=None):
        seen.update(url=url, physics=physics, rpc_timeout=rpc_timeout)
        return "sim"

    monkeypatch.setitem(sys.modules, "lightphyx", SimpleNamespace(connect=connect))
    monkeypatch.setenv("EVO_LIGHTPHYX_URL", "ws://shared/ws")
    assert fresh_pool._new_sim() == "sim"
    assert seen == {"url": "ws://shared/ws", "physics": False, "rpc_timeout": 45.0}


def test_render_backend_switch(monkeypatch):
    monkeypatch.delenv("EVO_RENDER_BACKEND", raising=False)
    assert render_backend() == "lightphyx"
    monkeypatch.setenv("EVO_RENDER_BACKEND", "blender")
    assert render_backend() == "blender"
    monkeypatch.setenv("EVO_RENDER_BACKEND", "cycles")
    with pytest.raises(ValueError):
        render_backend()


def test_unreachable_server_raises_unavailable(tmp_path, fresh_pool, monkeypatch):
    from evo.oracle.render_lightphyx import LightphyxUnavailable

    def refuse():
        raise ConnectionRefusedError("server gone")

    monkeypatch.setattr(fresh_pool, "_new_sim", refuse)
    with pytest.raises(LightphyxUnavailable, match="failed twice"):
        _render(tmp_path)
    assert fresh_pool._LIVE == 0
