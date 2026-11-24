"""Oracle canonicalization regression tests.

Physically-equivalent URDF representations must be judged identical by the
oracle:
  (1) axis sign flip + limits negate-swap
  (2) CONTINUOUS ↔ REVOLUTE[0, 2π]
  (3) lower-limit shift compensated by joint_origin

The unit tests exercise pure functions and always run. The end-to-end test
needs ground-truth URDFs (gitignored / absent on a fresh checkout) and skips
when they are not present — point EVO_GT_ROOT at a populated gt/ to run it.
"""
from __future__ import annotations

import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from evo.engine.paths import gt_root
from evo.oracle.check_urdf import (
    _canonicalize_axis_limits,
    _canonicalize_joint,
    _mat4_rotate_axis_angle,
    _mat4_to_rpy,
    parse_urdf,
    run_oracle,
)


def approx(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))
    return abs(a - b) < tol


# --------------------------------------------------------------------------
# Unit: _canonicalize_axis_limits
# --------------------------------------------------------------------------

AXIS_LIMIT_CASES = [
    # (axis, lo, hi, expected_axis, expected_lo, expected_hi)
    ((1.0, 0.0, 0.0), 0.0, 1.57, (1.0, 0.0, 0.0), 0.0, 1.57),       # already canonical
    ((-1.0, 0.0, 0.0), -1.57, 0.0, (1.0, 0.0, 0.0), 0.0, 1.57),     # flip X
    ((0.0, 0.0, -1.0), -1.57, 0.0, (0.0, 0.0, 1.0), 0.0, 1.57),     # flip Z
    ((0.0, 0.0, -1.0), -3.14, 3.14, (0.0, 0.0, 1.0), -3.14, 3.14),  # symmetric
    ((0.0, -1.0, 0.0), -0.3, 0.0, (0.0, 1.0, 0.0), 0.0, 0.3),       # prismatic flip
    ((0.0, 1.0, 0.0), 0.0, 0.3, (0.0, 1.0, 0.0), 0.0, 0.3),         # canonical prismatic
    ((-0.866, 0.5, 0.0), -1.0, 0.0, (0.866, -0.5, 0.0), 0.0, 1.0),  # off-axis: flip dominant comp
    ((1.0, -1e-7, -1e-7), 0.0, 1.0, (1.0, -1e-7, -1e-7), 0.0, 1.0), # noisy +X stays
    ((-1.0, 1e-7, 1e-7), 0.0, 1.0, (1.0, -1e-7, -1e-7), -1.0, 0.0), # noisy -X flips
    (None, None, None, None, None, None),                          # no axis
    ((1.0, 0.0, 0.0), None, None, (1.0, 0.0, 0.0), None, None),     # axis only
]


@pytest.mark.parametrize("ax,lo,hi,exp_ax,exp_lo,exp_hi", AXIS_LIMIT_CASES)
def test_canonicalize_axis_limits(ax, lo, hi, exp_ax, exp_lo, exp_hi):
    out_ax, out_lo, out_hi = _canonicalize_axis_limits(ax, lo, hi)
    assert approx(out_ax, exp_ax), f"axis {out_ax} != {exp_ax}"
    assert approx(out_lo, exp_lo), f"lo {out_lo} != {exp_lo}"
    assert approx(out_hi, exp_hi), f"hi {out_hi} != {exp_hi}"


# --------------------------------------------------------------------------
# Unit: _canonicalize_joint
# --------------------------------------------------------------------------

def test_continuous_normalizes_to_revolute_0_2pi():
    out = _canonicalize_joint(
        "CONTINUOUS", (0.0, 1.0, 0.0), None, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    assert out[0] == "REVOLUTE"
    assert approx(out[1], (0.0, 1.0, 0.0))
    assert approx(out[2], 0.0)
    assert approx(out[3], 2 * math.pi)


def test_prismatic_lower_shift_compensated_by_origin():
    # axis=(0,1,0), limits [-0.05, 0], origin (1,2,3):
    # canonicalize composes origin × T(axis, lo) → new origin (1, 1.95, 3),
    # limits shift to [0, 0.05].
    out = _canonicalize_joint(
        "PRISMATIC", (0.0, 1.0, 0.0), -0.05, 0.0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0)
    )
    assert out[0] == "PRISMATIC"
    assert approx(out[1], (0.0, 1.0, 0.0))
    assert approx(out[2], 0.0)
    assert approx(out[3], 0.05)
    assert approx(out[4], (1.0, 1.95, 3.0))
    assert approx(out[5], (0.0, 0.0, 0.0))


def test_revolute_axis_angle_roundtrips_through_rpy():
    # The end-to-end limits-shift test relies on axis-angle → rpy → matrix
    # agreeing with the oracle's own canonicalization. Sanity-check the pair.
    R = _mat4_rotate_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
    rpy = _mat4_to_rpy(R)
    assert approx(rpy[2], math.pi / 2, tol=1e-6)


# --------------------------------------------------------------------------
# End-to-end: needs GT data (skips when absent)
# --------------------------------------------------------------------------

def _movable(joint_type: str) -> bool:
    return joint_type.lower() in ("revolute", "prismatic", "continuous")


def _make_double_flipped(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy a URDF dir, flip every movable joint's axis sign and negate-swap
    its limits — a physically identical model in a different representation."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    urdf_path = dst_dir / "furniture.urdf"
    (dst_dir / "scene.urdf").rename(urdf_path)
    tree = ET.parse(urdf_path)
    flipped = []
    for joint in tree.getroot().findall("joint"):
        if not _movable(joint.attrib.get("type", "fixed")):
            continue
        axis_el = joint.find("axis")
        if axis_el is None:
            continue
        ax = axis_el.attrib.get("xyz", "0 0 0").split()
        axis_el.set("xyz", " ".join(f"{-float(x):.9f}" for x in ax))
        limit_el = joint.find("limit")
        if limit_el is not None:
            lo = float(limit_el.attrib.get("lower", 0))
            hi = float(limit_el.attrib.get("upper", 0))
            limit_el.set("lower", f"{-hi:.9f}")
            limit_el.set("upper", f"{-lo:.9f}")
        flipped.append(joint.attrib["name"])
    tree.write(urdf_path)
    return flipped


def _find_gt_instance_with_movable_joint() -> Path | None:
    root = gt_root()
    if not root.exists():
        return None
    for urdf in sorted(root.glob("*/*/scene.urdf")):
        try:
            tree = ET.parse(urdf)
        except ET.ParseError:
            continue
        if any(_movable(j.attrib.get("type", "fixed")) for j in tree.getroot().findall("joint")):
            return urdf.parent
    return None


def test_double_flip_is_canonically_equivalent():
    gt_dir = _find_gt_instance_with_movable_joint()
    if gt_dir is None:
        pytest.skip(
            f"no GT instance with a movable joint under {gt_root()} "
            "(set EVO_GT_ROOT to a populated gt/ to run this)"
        )
    tmp = Path(tempfile.mkdtemp(prefix="canon_flip_"))
    try:
        flipped_dir = tmp / "out"
        flipped = _make_double_flipped(gt_dir, flipped_dir)
        assert flipped, "expected at least one movable joint to flip"
        # canonicalized forms of GT and double-flipped pred must match
        gt_model = parse_urdf(gt_dir / "scene.urdf")
        fp_model = parse_urdf(flipped_dir / "furniture.urdf")
        for name, gj in gt_model.joints.items():
            fj = fp_model.joints[name]
            assert approx(gj.axis, fj.axis), f"{name}: axis {gj.axis} != {fj.axis}"
        # and the oracle must accept the pred
        result = run_oracle(gt_dir, flipped_dir)
        assert result.passed, (
            "double-flipped GT should pass the oracle; failed checks: "
            + ", ".join(c.name for c in result.checks if not c.passed)
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# (4) joint frame rotation vs axis sign: an untraced joint must sit at q=0
# ---------------------------------------------------------------------------

def _box_obj(path: Path, center, half=0.05) -> None:
    cx, cy, cz = center
    verts = [(cx + sx * half, cy + sy * half, cz + sz * half)
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    faces = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    path.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in verts)
                    + "".join(f"f {a} {b} {c} {d}\n" for a, b, c, d in faces))


def _two_drawer_scene(root: Path, *, rpy: str, axis: str) -> Path:
    """Body + two drawers sliding along world -X by 0.2, both drawers at rest at q=0.

    Meshes are world-baked (rest pose); the visual origin of each drawer is
    the inverse of its joint frame so the mesh stays put at q=0.
    """
    root.mkdir(parents=True, exist_ok=True)
    _box_obj(root / "body.obj", (0.0, 0.0, 0.0), half=0.3)
    _box_obj(root / "d0.obj", (0.0, 0.0, 0.15))
    _box_obj(root / "d1.obj", (0.0, 0.0, -0.15))
    pivots = {"d0": (0.0, 0.0, 0.15), "d1": (0.0, 0.0, -0.15)}
    links = ['<link name="body"><visual><origin xyz="0 0 0" rpy="0 0 0"/>'
             '<geometry><mesh filename="body.obj"/></geometry></visual></link>']
    joints = []
    for name, (px, py, pz) in pivots.items():
        # visual origin = inverse of the joint frame T(p)·R(rpy); the two rpy
        # values used here are 0 and π about z, whose inverse is R(-rpy)·(-p).
        if rpy == "0 0 0":
            vis_xyz, vis_rpy = f"{-px} {-py} {-pz}", "0 0 0"
        else:
            vis_xyz, vis_rpy = f"{px} {py} {-pz}", "0 0 -3.141592653589793"
        links.append(f'<link name="{name}"><visual><origin xyz="{vis_xyz}" rpy="{vis_rpy}"/>'
                     f'<geometry><mesh filename="{name}.obj"/></geometry></visual></link>')
        joints.append(f'<joint name="j_{name}" type="prismatic"><parent link="body"/><child link="{name}"/>'
                      f'<origin xyz="{px} {py} {pz}" rpy="{rpy}"/><axis xyz="{axis}"/>'
                      f'<limit lower="0" upper="0.2" effort="100" velocity="1"/></joint>')
    urdf = root / "scene.urdf"
    urdf.write_text('<?xml version="1.0"?><robot name="t">' + "".join(links) + "".join(joints) + "</robot>")
    return urdf


def test_untraced_joints_rest_at_reference_not_canonical_lower(tmp_path: Path):
    # GT expresses "slide along world -X" as rpy=(0,0,π) with axis +X; the
    # prediction expresses the same motion as rpy=0 with axis -X. Parsing
    # flips the pred axis and moves its reference q to `upper`; holding the
    # untraced joint at the canonical lower limit would leave it fully
    # extended on the pred side only and fail motion by the whole travel.
    gt = tmp_path / "gt"
    pred = tmp_path / "pred"
    _two_drawer_scene(gt, rpy="0 0 3.141592653589793", axis="1 0 0")
    _two_drawer_scene(pred, rpy="0 0 0", axis="-1 0 0")
    result = run_oracle(gt, pred)
    by_name = {c.name: c for c in result.checks}
    assert by_name["joint_axes"].passed, by_name["joint_axes"].detail
    assert by_name["kinematic_equivalence"].passed, by_name["kinematic_equivalence"].detail
    assert all(rep.motion_ok for rep in result.joints.values())
