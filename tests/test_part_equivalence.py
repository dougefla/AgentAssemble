"""Interchangeable parts: classes in GT, closest assignment wins in the oracle."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evo.oracle import check_urdf
from evo.oracle.check_urdf import Thresholds, parse_urdf, run_oracle
from evo.oracle.part_equivalence import (
    EQUIVALENCE_NAME, candidate_matchings, equivalence_classes, load_part_equivalence,
    write_part_equivalence,
)


def _box(center, half, *, nudge=0.0) -> str:
    """A box with each corner repeated 6x (48 vertices) so a one-vertex nudge is a
    ~1% shape change: "nearly" but not exactly the same mesh."""
    cx, cy, cz = center
    corners = [(cx + sx * half[0], cy + sy * half[1], cz + sz * half[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    pts = corners * 6
    if nudge:
        x, y, z = pts[-1]
        pts[-1] = (x, y + nudge, z)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts]
    lines += ["f 1 2 4 3", "f 5 7 8 6", "f 1 5 6 2", "f 3 4 8 7", "f 1 3 7 5", "f 2 6 8 4"]
    return "\n".join(lines) + "\n"


def _urdf(d0, d1) -> str:
    """Chest with two drawers; d0/d1 are the (origin xyz, mesh) of the two drawer links."""
    links = ['<link name="body"><visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="body.obj"/></geometry></visual></link>']
    joints = []
    for name, (oz, mesh) in (("link_1", d0), ("link_2", d1)):
        links.append(f'<link name="{name}"><visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh}"/></geometry></visual></link>')
        joints.append(f'<joint name="j_{name}" type="prismatic"><parent link="body"/><child link="{name}"/>'
                      f'<origin xyz="0 0 {oz}" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="0" upper="0.2" effort="100" velocity="1"/></joint>')
    return '<?xml version="1.0"?><robot name="chest">' + "".join(links) + "".join(joints) + "</robot>"


@pytest.fixture
def chest(tmp_path: Path) -> dict:
    gt = tmp_path / "gt"; gt.mkdir()
    (gt / "body.obj").write_text(_box((0, 0, 0), (0.3, 0.3, 0.5)))
    (gt / "d_a.obj").write_text(_box((0, 0, 0), (0.25, 0.25, 0.08)))
    (gt / "d_b.obj").write_text(_box((0, 0, 0), (0.25, 0.25, 0.08), nudge=0.01))   # nearly identical
    (gt / "scene.urdf").write_text(_urdf((0.3, "d_a.obj"), (-0.3, "d_b.obj")))
    # the prediction uses the same drawer meshes but puts each in the other slot
    swapped = tmp_path / "swapped"; swapped.mkdir()
    for n in ("body.obj", "d_a.obj", "d_b.obj"):
        (swapped / n).write_text((gt / n).read_text())
    (swapped / "furniture.urdf").write_text(_urdf((-0.3, "d_a.obj"), (0.3, "d_b.obj")))
    return {"gt": gt, "swapped": swapped}


def test_classes_group_nearly_identical_parts_only(chest: dict) -> None:
    gt = parse_urdf(chest["gt"] / "scene.urdf")
    assert equivalence_classes(gt) == [["link_1", "link_2"]]
    classes = write_part_equivalence(chest["gt"])
    assert classes == [["link_1", "link_2"]]
    payload = json.loads((chest["gt"] / EQUIVALENCE_NAME).read_text())
    assert payload["classes"][0]["parts"] == ["d_a.obj", "d_b.obj"] and payload["classes"][0]["joint_type"] == "PRISMATIC"
    assert load_part_equivalence(chest["gt"]) == [["link_1", "link_2"]]
    assert load_part_equivalence(chest["swapped"]) is None


def test_swapped_interchangeable_parts_pass_and_are_reported(chest: dict) -> None:
    write_part_equivalence(chest["gt"])
    result = run_oracle(chest["gt"], chest["swapped"], Thresholds())
    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.passed]
    link_matching = next(c for c in result.checks if c.name == "link_matching")
    assert "closest assignment" in link_matching.detail
    assert result.matching == {"body": "body", "link_1": "link_2", "link_2": "link_1"}
    assert all(rep.motion_ok for rep in result.joints.values())


def test_without_classes_the_swap_fails(chest: dict, monkeypatch) -> None:
    # The matcher's geometry term (x1000) pins each nearly-identical drawer to
    # its own name; only the equivalence classes let the oracle try the swap.
    monkeypatch.setattr(check_urdf, "load_part_equivalence", lambda _d: [])
    result = run_oracle(chest["gt"], chest["swapped"], Thresholds())
    assert not result.passed
    assert not next(c for c in result.checks if c.name == "rest_pose_alignment").passed


def test_candidate_matchings_permute_within_classes_and_cap() -> None:
    m = {"a": "pa", "b": "pb", "c": "pc", "d": "pd"}
    cands = candidate_matchings(m, [["a", "b"], ["c", "d"]])
    assert cands[0] == m and len(cands) == 4
    assert {"a": "pb", "b": "pa", "c": "pd", "d": "pc"} in cands
    assert candidate_matchings(m, [["a", "zzz"]]) == [m]          # unmatched members are ignored
    many = {f"l{i}": f"p{i}" for i in range(8)}
    assert len(candidate_matchings(many, [[f"l{i}" for i in range(8)]], limit=50)) == 50


def test_differently_sampled_congruent_parts_need_the_recorded_transform(tmp_path: Path) -> None:
    """Two drawers with the same surface but different vertex sampling: the moment
    descriptors differ, so a swap only passes when the GT carries the
    registration transform and the oracle compares through the swapped cloud."""
    pytest.importorskip("numpy")
    from evo.oracle.part_equivalence import derive_part_equivalence
    gt = tmp_path / "gt"; gt.mkdir()
    # the body gets a distinct vertex count so the (scale-free) matcher never
    # confuses it with a drawer; only the two drawers are near-identical
    body = _box((0, 0, 0), (0.3, 0.3, 0.5)).replace("f 1 2 4 3", "v 0 0 0\nv 0 0 0\nv 0 0 0\nv 0 0 0\nf 1 2 4 3")
    (gt / "body.obj").write_text(body)
    (gt / "d_a.obj").write_text(_box((0, 0, 0), (0.25, 0.25, 0.08)))
    # same box, corners repeated 6x plus an edge midpoint twice: same surface, other moments
    cx, cy, cz, hx, hy, hz = 0, 0, 0, 0.25, 0.25, 0.08
    corners = [(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    pts = corners * 6 + [(cx, cy + hy, cz + hz)] * 2
    (gt / "d_b.obj").write_text("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in pts)
                                + "f 1 2 4 3\nf 5 7 8 6\nf 1 5 6 2\nf 3 4 8 7\nf 1 3 7 5\nf 2 6 8 4\n")
    (gt / "scene.urdf").write_text(_urdf((0.3, "d_a.obj"), (-0.3, "d_b.obj")))
    swapped = tmp_path / "swapped"; swapped.mkdir()
    for n in ("body.obj", "d_a.obj", "d_b.obj"):
        (swapped / n).write_text((gt / n).read_text())
    (swapped / "furniture.urdf").write_text(_urdf((-0.3, "d_a.obj"), (0.3, "d_b.obj")))
    classes, transforms = derive_part_equivalence(gt, max_distance=0.5, max_scale_diff=0.5)
    assert classes == [["link_1", "link_2"]] and ("link_1", "link_2") in transforms
    assert transforms[("link_1", "link_2")]["residual_m"] < 1e-6
    # without the file: classes are derived on the fly but no transform -> moments disagree -> fail
    assert not run_oracle(gt, swapped, Thresholds()).passed
    write_part_equivalence(gt, max_distance=0.5, max_scale_diff=0.5)
    result = run_oracle(gt, swapped, Thresholds())
    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.matching == {"body": "body", "link_1": "link_2", "link_2": "link_1"}
