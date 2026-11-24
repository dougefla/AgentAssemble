"""Oracle 3: the pose metric is a surface distance (see evo/oracle/surface_distance.py)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from evo.oracle.check_urdf import Thresholds, run_oracle  # noqa: E402


def _cylinder(radius: float, height: float, segments: int = 24, jitter: float = 0.37) -> str:
    """A closed cylinder about z through the origin; the angular sampling is
    irregular (jitter), like a real tessellation, so vertex moments are not
    invariant under a spin about z even though the solid is."""
    ring = []
    for i in range(segments):
        a = 2 * math.pi * (i + (jitter if i < 5 else 0.0)) / segments
        ring.append((radius * math.cos(a), radius * math.sin(a)))
    v = [(x, y, 0.0) for x, y in ring] + [(x, y, height) for x, y in ring] + [(0, 0, 0.0), (0, 0, height)]
    f = []
    n = segments
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
    return "".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v) + "".join(
        f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f)


def _box(lo, hi) -> str:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    quads = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    return "".join(f"v {x} {y} {z}\n" for x, y, z in v) + "".join("f %d %d %d %d\n" % q for q in quads)


def _urdf(child_rpy=(0, 0, 0), child_scale=1.0, child_xyz=(0, 0, 0.1), jtype="continuous") -> str:
    limit = "" if jtype == "continuous" else '<limit lower="0" upper="1.2" effort="1" velocity="1"/>'
    return f"""<?xml version="1.0"?>
<robot name="t">
  <link name="link_0"><visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="base.obj"/></geometry></visual></link>
  <link name="link_1"><visual><origin xyz="0 0 0" rpy="{child_rpy[0]} {child_rpy[1]} {child_rpy[2]}"/>
    <geometry><mesh filename="top.obj" scale="{child_scale} {child_scale} {child_scale}"/></geometry></visual></link>
  <joint name="joint_0" type="{jtype}"><parent link="link_0"/><child link="link_1"/>
    <origin xyz="{child_xyz[0]} {child_xyz[1]} {child_xyz[2]}" rpy="0 0 0"/><axis xyz="0 0 1"/>{limit}</joint>
</robot>
"""


def _case(tmp_path: Path, top: str, **pred) -> tuple[Path, Path]:
    gt = tmp_path / "gt"; gt.mkdir()
    pr = tmp_path / "pred"; pr.mkdir()
    for d in (gt, pr):
        (d / "base.obj").write_text(_box((-0.3, -0.3, 0.0), (0.3, 0.3, 0.1)))
        (d / "top.obj").write_text(top)
    (gt / "scene.urdf").write_text(_urdf(jtype=pred.get("jtype", "continuous")))
    (pr / "furniture.urdf").write_text(_urdf(**pred))
    return gt, pr


def _check(result, name):
    return next(c for c in result.checks if c.name == name)


def test_spin_about_the_symmetry_axis_is_the_same_solid(tmp_path: Path, monkeypatch) -> None:
    gt, pr = _case(tmp_path, _cylinder(0.25, 0.04), child_rpy=(0, 0, 1.9))
    result = run_oracle(gt, pr, Thresholds())
    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.passed]
    # the 2.0 moment metric rejects the same answer
    monkeypatch.setenv("EVO_ORACLE_POSE_METRIC", "moments")
    assert not _check(run_oracle(gt, pr, Thresholds()), "rest_pose_alignment").passed


def test_a_tilted_disc_is_still_wrong_and_blamed_on_orientation(tmp_path: Path) -> None:
    gt, pr = _case(tmp_path, _cylinder(0.25, 0.04), child_rpy=(0.6, 0, 0))
    result = run_oracle(gt, pr, Thresholds())
    assert not _check(result, "rest_pose_alignment").passed
    rep = result.joints["joint_0"]
    assert rep.pose_cause == "orientation" and rep.rest_drift_m > rep.pose_tol_m
    assert rep.rot_err_deg is not None


def test_scale_error_is_linear_and_does_not_charge_the_root(tmp_path: Path) -> None:
    # 3 % too large: inside the 5 % size tolerance, and ~1 cm of surface motion
    # on a 0.35 m part, far below the 3.5 cm pose tolerance of this object.
    gt, pr = _case(tmp_path, _box((0.0, -0.2, 0.0), (0.35, 0.2, 0.03)), child_scale=1.03, jtype="revolute")
    result = run_oracle(gt, pr, Thresholds())
    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.passed]
    rep = result.joints["joint_0"]
    assert 0.005 < rep.rest_drift_m < 0.02
    # 20 % too large fails, the root link is not among the mis-posed links, and
    # the diagnosis names the size
    gt2, pr2 = _case(tmp_path / "b", _box((0.0, -0.2, 0.0), (0.35, 0.2, 0.03)), child_scale=1.2, jtype="revolute") \
        if (tmp_path / "b").mkdir() is None else (None, None)
    bad = run_oracle(gt2, pr2, Thresholds())
    rest = _check(bad, "rest_pose_alignment")
    assert not rest.passed and "link_0" not in rest.detail
    assert bad.joints["joint_0"].pose_cause == "scale"


def test_wrong_pivot_is_blamed_on_position(tmp_path: Path) -> None:
    gt, pr = _case(tmp_path, _box((0.0, -0.2, 0.0), (0.35, 0.2, 0.03)), child_xyz=(0.15, 0, 0.1), jtype="revolute")
    result = run_oracle(gt, pr, Thresholds())
    assert not result.passed
    rep = result.joints["joint_0"]
    assert rep.pose_cause == "position" and rep.pos_err_m == pytest.approx(0.15, abs=1e-6)
