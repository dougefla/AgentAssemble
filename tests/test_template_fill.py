"""Joint-fill templates: build, leak scan, GT round trip, finalize, scored dims."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evo.oracle.check_urdf import JointReport, Thresholds, parse_urdf, run_oracle
from evo.oracle.template_fill import (
    PLACEHOLDER, build_template, fill_template, finalize_filled_urdf,
    finalize_pred_dir, gt_fill_values, leaked_numbers, needs_finalize,
    unfilled_blanks,
)


def _cube_obj(center, half) -> str:
    cx, cy, cz = center
    corners = [
        (cx + sx * half[0], cy + sy * half[1], cz + sz * half[2])
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in corners]
    faces = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    lines += [f"f {a} {b} {c} {d}" for a, b, c, d in faces]
    return "\n".join(lines) + "\n"


HINGE = (0.5, 0.0, 0.0)          # door hinge in world = joint origin (parent is root)
GT_URDF = f"""<?xml version="1.0" ?>
<robot name="t">
  <link name="link_0"><visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="assets/part_0.obj"/></geometry></visual></link>
  <link name="link_1"><visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="assets/part_1.obj"/></geometry></visual></link>
  <joint name="joint_0" type="revolute">
    <origin xyz="{HINGE[0]} {HINGE[1]} {HINGE[2]}" rpy="0 0 0"/>
    <parent link="link_0"/><child link="link_1"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="1.2" effort="100" velocity="1"/>
  </joint>
</robot>
"""


@pytest.fixture
def dataset(tmp_path: Path) -> dict:
    """GT in link frames (door mesh around its hinge) + world-baked task parts."""
    gt = tmp_path / "gt" / "task_00" / "0"
    (gt / "assets").mkdir(parents=True)
    (gt / "scene.urdf").write_text(GT_URDF)
    body = _cube_obj((0.0, 0.0, 0.5), (0.5, 0.2, 0.5))
    door_local = _cube_obj((0.2, 0.0, 0.5), (0.2, 0.02, 0.5))       # hinge at local origin
    door_world = _cube_obj((0.2 + HINGE[0], 0.0, 0.5), (0.2, 0.02, 0.5))
    (gt / "assets" / "part_0.obj").write_text(body)
    (gt / "assets" / "part_1.obj").write_text(door_local)
    parts = tmp_path / "tasks" / "task_00" / "0" / "parts"
    parts.mkdir(parents=True)
    (parts / "part_0.obj").write_text(body)
    (parts / "part_1.obj").write_text(door_world)
    (tmp_path / "dataset_variant.json").write_text(json.dumps({
        "dataset": "test", "task_mesh_transforms": ["world_bake_q0"],
        "scored_joint_dims": ["axis_ok", "pivot_ok", "motion_ok"],
    }))
    template = build_template(gt / "scene.urdf")
    (parts / "template.urdf").write_text(template)
    return {"gt": gt, "parts": parts, "template": template, "root": tmp_path}


def test_template_blanks_and_no_leak(dataset: dict, tmp_path: Path) -> None:
    tpl = dataset["template"]
    assert tpl.split("<robot", 1)[1].count(PLACEHOLDER) == 2   # origin + axis of the one joint
    body = tpl.split("<robot", 1)[1]
    assert 'rpy="0 0 0"' in body and "<origin" not in body.split("<joint")[0]  # no visual origins
    assert leaked_numbers(tpl, dataset["gt"] / "scene.urdf") == []
    t = tmp_path / "t.urdf"
    t.write_text(tpl)
    assert unfilled_blanks(t) == ["joint_0/origin", "joint_0/axis"]
    with pytest.raises(ValueError):
        finalize_filled_urdf(t, tmp_path / "out", mesh_dir=dataset["parts"])


def test_gt_refill_round_trip_passes_oracle(dataset: dict, tmp_path: Path) -> None:
    filled = tmp_path / "filled.urdf"
    filled.write_text(fill_template(dataset["template"], gt_fill_values(dataset["gt"] / "scene.urdf")))
    assert needs_finalize(filled)
    out = finalize_filled_urdf(filled, tmp_path / "pred", mesh_dir=dataset["parts"])
    model = parse_urdf(out)
    # world-baked door mesh must be pulled back by the hinge offset
    assert model.links["link_1"].visual_xyz == pytest.approx((-HINGE[0], 0.0, 0.0), abs=1e-9)
    assert (tmp_path / "pred" / "meshes" / "part_1.obj").exists()
    result = run_oracle(dataset["gt"], tmp_path / "pred", Thresholds())
    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.passed]


def test_wrong_hinge_fails_only_geometry_dims(dataset: dict, tmp_path: Path) -> None:
    wrong = {"joint_0": {"origin_xyz": "0.9 0 0", "origin_rpy": "0 0 0", "axis": "0 0 1"}}
    filled = tmp_path / "wrong.urdf"
    filled.write_text(fill_template(dataset["template"], wrong))
    finalize_filled_urdf(filled, tmp_path / "pred", mesh_dir=dataset["parts"])
    result = run_oracle(dataset["gt"], tmp_path / "pred", Thresholds())
    rep = result.joints["joint_0"]
    assert rep.type_ok and rep.limits_ok and rep.axis_ok
    assert rep.pivot_ok is False and rep.motion_ok is False
    old = JointReport.DIMS
    try:
        assert rep.score() == pytest.approx(3 / 5)
        JointReport.configure_dims(["axis_ok", "pivot_ok", "motion_ok"])
        assert rep.score() == pytest.approx(1 / 3)
        with pytest.raises(ValueError):
            JointReport.configure_dims(["axis_ok", "bogus"])
    finally:
        JointReport.DIMS = old


def test_finalize_pred_dir_is_a_no_op_for_sdk_exports(dataset: dict, tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    filled = pred / "furniture.urdf"
    pred.mkdir()
    filled.write_text(fill_template(dataset["template"], gt_fill_values(dataset["gt"] / "scene.urdf")))
    out = finalize_pred_dir(pred, mesh_dir=dataset["parts"])
    assert out == filled and (pred / "furniture.filled.urdf").exists()
    assert not needs_finalize(filled)
    assert finalize_pred_dir(pred, mesh_dir=dataset["parts"]) is None   # already finalized
    assert run_oracle(dataset["gt"], pred, Thresholds()).passed


def test_apply_dataset_variant_narrows_dims(dataset: dict) -> None:
    from evo.engine.loop import _apply_dataset_variant
    old = JointReport.DIMS
    try:
        variant = _apply_dataset_variant(dataset["root"] / "tasks")
        assert variant["dataset"] == "test"
        assert JointReport.DIMS == ("axis_ok", "pivot_ok", "motion_ok")
    finally:
        JointReport.DIMS = old


# ---------------------------------------------------------------------------
# pose-fill view: centered meshes, visual origins blanked too
# ---------------------------------------------------------------------------

def _builder():
    """scripts/dataset/build_pose_fill_view.py as a module (scripts/ is not a package)."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "dataset" / "build_pose_fill_view.py"
    spec = importlib.util.spec_from_file_location("build_pose_fill_view", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def posefill(dataset: dict, tmp_path: Path) -> dict:
    """Centered task parts + offsets + a template with blank visual origins."""
    from evo.oracle.template_fill import posefill_values
    b = _builder()
    import shutil
    parts = tmp_path / "pose" / "parts"
    parts.mkdir(parents=True)
    root_part, transforms = b.plan_transforms(dataset["gt"] / "scene.urdf", dataset["parts"], rotate=True, seed=7, tag="t")
    assert root_part == "part_0.obj" and transforms["part_0.obj"]["anchored"]
    for name, tf in transforms.items():
        if tf["anchored"]:
            shutil.copy2(dataset["parts"] / name, parts / name)
        else:
            b.center_obj(dataset["parts"] / name, parts / name, tf["center"], tf["rotation"])
    template = build_template(dataset["gt"] / "scene.urdf", blank_visual_origins=True)
    (parts / "template.urdf").write_text(template)
    joints, visuals = posefill_values(dataset["gt"] / "scene.urdf", transforms)
    return {"parts": parts, "template": template, "transforms": transforms, "joints": joints, "visuals": visuals}


def test_posefill_template_blanks_visuals_and_centers_parts(posefill: dict, tmp_path: Path) -> None:
    body = posefill["template"].split("<robot", 1)[1]
    assert body.count(PLACEHOLDER) == 4            # door: visual xyz + rpy; joint origin + axis
    assert '<origin xyz="0 0 0" rpy="0 0 0" />' in body.split("<joint")[0]   # root visual given
    t = tmp_path / "t.urdf"
    t.write_text(posefill["template"])
    assert unfilled_blanks(t) == ["link_1/visual_xyz", "link_1/visual_rpy", "joint_0/origin", "joint_0/axis"]
    b = _builder()
    assert b.aabb_center(posefill["parts"] / "part_1.obj") == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert b.aabb_center(posefill["parts"] / "part_0.obj") == pytest.approx([0.0, 0.0, 0.5])   # base untouched
    # the door's world center was (0.7, 0, 0.5); link_1's origin is the hinge (0.5, 0, 0)
    assert posefill["transforms"]["part_1.obj"]["center"] == pytest.approx([0.7, 0.0, 0.5])
    assert posefill["visuals"]["link_1"]["xyz"] == pytest.approx([0.2, 0.0, 0.5])
    assert posefill["visuals"]["link_0"]["xyz"] == pytest.approx([0.0, 0.0, 0.0])
    assert posefill["visuals"]["link_0"]["rpy"] == pytest.approx([0.0, 0.0, 0.0])
    assert posefill["joints"]["joint_0"]["origin_xyz"] == pytest.approx([0.5, 0.0, 0.0])
    # the visual rpy undoes the applied rotation: R(rpy) @ R == I
    from evo.oracle.check_urdf import _mat4_rotate_rpy
    R = posefill["transforms"]["part_1.obj"]["rotation"]
    M = _mat4_rotate_rpy(tuple(posefill["visuals"]["link_1"]["rpy"]))
    prod = [sum(M[i][k] * R[k][j] for k in range(3)) for i in range(3) for j in range(3)]
    assert prod == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0, 1], abs=1e-9)
    assert any(abs(v) > 0.1 for v in posefill["visuals"]["link_1"]["rpy"])   # a real rotation was applied


def test_posefill_gt_refill_passes_and_zero_fill_fails(dataset: dict, posefill: dict, tmp_path: Path) -> None:
    from evo.oracle.template_fill import stage_filled_urdf
    filled = tmp_path / "filled.urdf"
    filled.write_text(fill_template(posefill["template"], posefill["joints"], posefill["visuals"]))
    assert not needs_finalize(filled)
    out = stage_filled_urdf(filled, tmp_path / "pred", mesh_dir=posefill["parts"])
    assert (tmp_path / "pred" / "meshes" / "part_1.obj").exists()
    assert parse_urdf(out).links["link_1"].visual_xyz == pytest.approx((0.2, 0.0, 0.5), abs=1e-6)
    result = run_oracle(dataset["gt"], tmp_path / "pred", Thresholds())
    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.passed]
    zero = tmp_path / "zero.urdf"
    zero.write_text(fill_template(
        posefill["template"],
        {"joint_0": {"origin_xyz": "0 0 0", "origin_rpy": "0 0 0", "axis": "0 0 1"}},
        {"link_0": {"xyz": "0 0 0", "rpy": "0 0 0"}, "link_1": {"xyz": "0 0 0", "rpy": "0 0 0"}}))
    stage_filled_urdf(zero, tmp_path / "zero_pred", mesh_dir=posefill["parts"])
    assert not run_oracle(dataset["gt"], tmp_path / "zero_pred", Thresholds()).passed


def test_finalize_pred_dir_stages_a_posefill_template(dataset: dict, posefill: dict, tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    (pred / "furniture.urdf").write_text(fill_template(posefill["template"], posefill["joints"], posefill["visuals"]))
    out = finalize_pred_dir(pred, mesh_dir=posefill["parts"])
    assert out == pred / "furniture.urdf" and (pred / "furniture.filled.urdf").exists()
    assert finalize_pred_dir(pred, mesh_dir=posefill["parts"]) is None      # meshes now resolve
    assert run_oracle(dataset["gt"], pred, Thresholds()).passed


def test_scored_checks_whitelist_changes_score_and_pass(dataset: dict, tmp_path: Path) -> None:
    import evo.oracle.check_urdf as cu
    wrong = {"joint_0": {"origin_xyz": "0.9 0 0", "origin_rpy": "0 0 0", "axis": "0 0 1"}}
    filled = tmp_path / "wrong.urdf"
    filled.write_text(fill_template(dataset["template"], wrong))
    finalize_filled_urdf(filled, tmp_path / "pred", mesh_dir=dataset["parts"])
    full = run_oracle(dataset["gt"], tmp_path / "pred", Thresholds())
    assert {c.name for c in full.checks} == set(cu.ALL_CHECKS)
    n_fail = sum(1 for c in full.checks if not c.passed)
    assert full.score == pytest.approx((12 - n_fail) / 12)
    old = cu.SCORED_CHECKS
    try:
        chosen = cu.configure_scored_checks(["joint_axes", "joint_limits", "rest_pose_alignment",
                                             "joint_pivot_alignment", "kinematic_equivalence"])
        assert chosen == ("joint_axes", "joint_limits", "rest_pose_alignment",
                          "joint_pivot_alignment", "kinematic_equivalence")
        narrow = run_oracle(dataset["gt"], tmp_path / "pred", Thresholds())
        by = {c.name: c for c in narrow.checks}
        assert all(not by[n].scored for n in cu.ALL_CHECKS if n not in chosen)
        assert all(by[n].scored for n in chosen)
        n_fail_narrow = sum(1 for n in chosen if not by[n].passed)
        assert n_fail_narrow == n_fail  # only geometry checks failed in the first place
        assert narrow.score == pytest.approx((5 - n_fail) / 5)
        assert narrow.passed is False
        with pytest.raises(ValueError):
            cu.configure_scored_checks(["joint_axes", "bogus"])
        # a dataset variant wires it through the loop
        from evo.engine.loop import _apply_dataset_variant
        marker = dataset["root"] / "dataset_variant.json"
        v = json.loads(marker.read_text()); v["scored_checks"] = ["kinematic_equivalence"]
        marker.write_text(json.dumps(v))
        _apply_dataset_variant(dataset["root"] / "tasks")
        assert cu.SCORED_CHECKS == ("kinematic_equivalence",)
    finally:
        cu.SCORED_CHECKS = old
        JointReport.DIMS = JointReport.ALL_DIMS


# ---------------------------------------------------------------------------
# Link-frame joint-fill view (artic100_simple): meshes are in their link frames,
# the template keeps zero visual origins, and only the joint blanks are filled.
# Nothing must be derived at finalize time — a filled template is staged as-is.
# ---------------------------------------------------------------------------

@pytest.fixture
def linkframe(tmp_path: Path) -> dict:
    """artic100_simple-shaped root: link-frame task parts, zero visual origins, no bake marker."""
    gt = tmp_path / "gt" / "task_000" / "0"
    (gt / "assets").mkdir(parents=True)
    (gt / "scene.urdf").write_text(GT_URDF)
    body = _cube_obj((0.0, 0.0, 0.5), (0.5, 0.2, 0.5))
    door_local = _cube_obj((0.2, 0.0, 0.5), (0.2, 0.02, 0.5))       # hinge at local origin
    (gt / "assets" / "part_0.obj").write_text(body)
    (gt / "assets" / "part_1.obj").write_text(door_local)
    parts = tmp_path / "tasks" / "task_000" / "0" / "parts"
    parts.mkdir(parents=True)
    (parts / "part_0.obj").write_text(body)
    (parts / "part_1.obj").write_text(door_local)                     # verbatim link-frame copy
    (tmp_path / "dataset_variant.json").write_text(json.dumps({
        "dataset": "artic100_test", "task_mesh_transforms": [],
        "template_kind": "jointfill",
        "scored_joint_dims": ["axis_ok", "pivot_ok", "motion_ok"],
    }))
    template = GT_URDF.replace("assets/part_", "part_")
    template = template.replace(f'<origin xyz="{HINGE[0]} {HINGE[1]} {HINGE[2]}" rpy="0 0 0"/>',
                                f'<origin xyz="{PLACEHOLDER}" rpy="0 0 0"/>')
    template = template.replace('<axis xyz="0 0 1"/>', f'<axis xyz="{PLACEHOLDER}"/>')
    (parts / "template.urdf").write_text(template)
    return {"gt": gt, "parts": parts, "template": template, "root": tmp_path}


def test_linkframe_template_keeps_visual_origins_and_blanks_only_joints(linkframe: dict) -> None:
    body = linkframe["template"].split("<robot", 1)[1]
    assert body.count(PLACEHOLDER) == 2
    assert 'origin xyz="0 0 0"' in body            # visual origins are given, not blanks
    assert not leaked_numbers(linkframe["template"], linkframe["gt"] / "scene.urdf")


def test_linkframe_gt_refill_is_staged_not_finalized_and_passes(linkframe: dict, tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    filled = fill_template(linkframe["template"], gt_fill_values(linkframe["gt"] / "scene.urdf"))
    (pred / "furniture.urdf").write_text(filled)
    assert not needs_finalize(pred / "furniture.urdf")  # visual origins present → no FK derivation
    out = finalize_pred_dir(pred, mesh_dir=linkframe["parts"])
    assert out == pred / "furniture.urdf" and (pred / "meshes" / "part_1.obj").exists()
    model = parse_urdf(out)
    assert model.links["link_1"].visual_xyz == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert model.joints["joint_0"].origin_xyz == pytest.approx(HINGE, abs=1e-9)
    result = run_oracle(linkframe["gt"], pred, Thresholds())
    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.passed]


def test_linkframe_zero_fill_fails(linkframe: dict, tmp_path: Path) -> None:
    pred = tmp_path / "zero_pred"
    pred.mkdir()
    (pred / "furniture.urdf").write_text(fill_template(
        linkframe["template"],
        {"joint_0": {"origin_xyz": "0 0 0", "origin_rpy": "0 0 0", "axis": "0 0 1"}}))
    finalize_pred_dir(pred, mesh_dir=linkframe["parts"])
    result = run_oracle(linkframe["gt"], pred, Thresholds())
    assert not result.passed
    assert "rest_pose_alignment" in {c.name for c in result.checks if not c.passed}


# ---------------------------------------------------------------------------
# Unit-scale view (artic100_rotscale): a non-root task part is scaled to a unit
# cube, the template's <mesh scale> is the blank, and staging bakes the filled
# scale into the mesh copy because the oracle reads no scale attribute.
# ---------------------------------------------------------------------------

@pytest.fixture
def unitscale(linkframe: dict) -> dict:
    """linkframe root, but the door part is shrunk to a unit cube (longest side 1 m) in tasks/."""
    parts = linkframe["parts"]
    door_local = _cube_obj((0.2, 0.0, 0.5), (0.2, 0.02, 0.5))       # longest side = 1.0 m already (z: 0..1)
    # shrink to half so the restoring factor is 2 — a real, non-trivial answer
    half = "\n".join((f"v {float(x)/2:.6f} {float(y)/2:.6f} {float(z)/2:.6f}" if l.startswith("v ") else l)
                     for l in door_local.splitlines() for x, y, z in [l.split()[1:4] if l.startswith("v ") else ("0", "0", "0")]) + "\n"
    (parts / "part_1.obj").write_text(half)
    template = linkframe["template"].replace('<mesh filename="part_1.obj"/>', '<mesh filename="part_1.obj" scale="? ? ?"/>')
    template = template.replace('<mesh filename="part_0.obj"/>', '<mesh filename="part_0.obj" scale="1 1 1"/>')
    (parts / "template.urdf").write_text(template)
    return {**linkframe, "template": template, "scale_answer": 2.0}


def test_unitscale_blank_is_detected_and_filled(unitscale: dict, tmp_path: Path) -> None:
    t = tmp_path / "t.urdf"; t.write_text(unitscale["template"])
    assert "link_1/visual_scale" in unfilled_blanks(t)
    filled = fill_template(unitscale["template"], gt_fill_values(unitscale["gt"] / "scene.urdf"),
                           {"link_1": {"xyz": "0 0 0", "rpy": "0 0 0", "scale": [2.0, 2.0, 2.0]}})
    f = tmp_path / "f.urdf"; f.write_text(filled)
    assert unfilled_blanks(f) == []
    assert [float(x) for x in re.search(r'part_1.obj" scale="([^"]+)"', filled).group(1).split()] == [2.0, 2.0, 2.0]


def test_unitscale_staging_bakes_scale_and_passes(unitscale: dict, tmp_path: Path) -> None:
    from evo.oracle.template_fill import stage_filled_urdf
    filled = tmp_path / "filled.urdf"
    filled.write_text(fill_template(unitscale["template"], gt_fill_values(unitscale["gt"] / "scene.urdf"),
                                    {"link_1": {"xyz": "0 0 0", "rpy": "0 0 0", "scale": "2 2 2"}}))
    out = stage_filled_urdf(filled, tmp_path / "pred", mesh_dir=unitscale["parts"])
    staged = (tmp_path / "pred" / "meshes" / "part_1.obj").read_text()
    assert "scale=" not in out.read_text()                     # attribute consumed
    zs = [float(l.split()[3]) for l in staged.splitlines() if l.startswith("v ")]
    assert max(zs) == pytest.approx(1.0, abs=1e-6)             # 0.5 m unit-cube copy restored to 1.0 m
    assert run_oracle(unitscale["gt"], tmp_path / "pred", Thresholds()).passed


def test_unitscale_wrong_scale_fails(unitscale: dict, tmp_path: Path) -> None:
    from evo.oracle.template_fill import stage_filled_urdf
    filled = tmp_path / "wrong.urdf"
    filled.write_text(fill_template(unitscale["template"], gt_fill_values(unitscale["gt"] / "scene.urdf"),
                                    {"link_1": {"xyz": "0 0 0", "rpy": "0 0 0", "scale": "1 1 1"}}))
    stage_filled_urdf(filled, tmp_path / "pred", mesh_dir=unitscale["parts"])
    assert not run_oracle(unitscale["gt"], tmp_path / "pred", Thresholds()).passed


def test_unitscale_link_scale_check_and_scale_ok_dimension(unitscale: dict, tmp_path: Path) -> None:
    """Wrong scale: link_scale is reported (diagnostic by default), scale_ok False with the ratio; scored on opt-in."""
    import evo.oracle.check_urdf as cu
    from evo.oracle.template_fill import stage_filled_urdf
    old_dims, old_checks = JointReport.DIMS, cu.SCORED_CHECKS
    try:
        # default view: the dimension and the check exist but are not scored
        JointReport.configure_dims(["axis_ok", "pivot_ok", "motion_ok"])
        cu.configure_scored_checks(list(cu.DEFAULT_SCORED_CHECKS))
        filled = tmp_path / "wrong.urdf"
        filled.write_text(fill_template(unitscale["template"], gt_fill_values(unitscale["gt"] / "scene.urdf"),
                                        {"link_1": {"xyz": "0 0 0", "rpy": "0 0 0", "scale": "1 1 1"}}))   # should be 2
        stage_filled_urdf(filled, tmp_path / "pred", mesh_dir=unitscale["parts"])
        res = cu.run_oracle(unitscale["gt"], tmp_path / "pred", Thresholds())
        by = {c.name: c for c in res.checks}
        assert "link_scale" in by and not by["link_scale"].passed and not by["link_scale"].scored
        rep = res.joints["joint_0"]
        assert rep.scale_ok is False and rep.scale_ratio == pytest.approx(0.5, abs=0.02)
        assert "scale_ok" not in rep.flags()
        # unit-scale view: opt in through the same hooks dataset_variant.json drives
        JointReport.configure_dims(["axis_ok", "pivot_ok", "motion_ok", "scale_ok"])
        cu.configure_scored_checks(list(cu.ALL_CHECKS))
        res2 = cu.run_oracle(unitscale["gt"], tmp_path / "pred", Thresholds())
        by2 = {c.name: c for c in res2.checks}
        assert by2["link_scale"].scored and not by2["link_scale"].passed
        assert "scale_ok" in res2.joints["joint_0"].flags() and res2.joints["joint_0"].score() < 1.0
        # the right scale passes everything, including the new check
        good = tmp_path / "good.urdf"
        good.write_text(fill_template(unitscale["template"], gt_fill_values(unitscale["gt"] / "scene.urdf"),
                                      {"link_1": {"xyz": "0 0 0", "rpy": "0 0 0", "scale": "2 2 2"}}))
        stage_filled_urdf(good, tmp_path / "pred_good", mesh_dir=unitscale["parts"])
        res3 = cu.run_oracle(unitscale["gt"], tmp_path / "pred_good", Thresholds())
        assert res3.passed and res3.joints["joint_0"].scale_ok is True
    finally:
        JointReport.DIMS = old_dims; cu.SCORED_CHECKS = old_checks
