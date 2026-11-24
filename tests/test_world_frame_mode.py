"""Assembly SDK `world` mesh-frame mode (the artvip_beginner contract).

Parts are imported in their rest-pose world coordinates and must never move:
joint operations re-solve each link's visual transform, mesh-moving APIs are
refused, and `audit` reports drift. Legacy `link` mode is untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evo.assembly_sdk.core import AssemblyError, ErrorCode
from evo.assembly_sdk.host_runtime import AssemblySDK, detect_mesh_frame
from evo.oracle.check_urdf import (
    _joint_positions_at_reference, compute_world_poses, parse_urdf,
)


def _cube_obj(center, half=0.1) -> str:
    cx, cy, cz = center
    corners = [
        (cx + sx * half, cy + sy * half, cz + sz * half)
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in corners]
    faces = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    lines += [f"f {a} {b} {c} {d}" for a, b, c, d in faces]
    return "\n".join(lines) + "\n"


@pytest.fixture
def world_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal artvip_beginner-shaped root: marker + one task with two parts."""
    root = tmp_path / "dataset"
    parts = root / "tasks" / "task_00" / "0" / "parts"
    parts.mkdir(parents=True)
    (root / "dataset_variant.json").write_text(json.dumps({
        "dataset": "test", "task_mesh_transforms": ["world_bake_q0"],
    }))
    (parts / "body.obj").write_text(_cube_obj((0.0, 0.0, 0.5), half=0.5))
    (parts / "door.obj").write_text(_cube_obj((0.55, 0.0, 0.5), half=0.05))
    out = tmp_path / "out"
    out.mkdir()
    return parts, out


def _aabb(sdk: AssemblySDK, label: str) -> tuple[list[float], list[float]]:
    for entry in sdk.inspect_parts():
        if entry["label"] == label:
            return entry["world_aabb"]["min"], entry["world_aabb"]["max"]
    raise AssertionError(label)


def _assert_in_place(sdk: AssemblySDK, label: str) -> None:
    lo, hi = _aabb(sdk, label)
    entry = next(e for e in sdk.inspect_parts() if e["label"] == label)
    assert lo == pytest.approx(entry["local_aabb"]["min"], abs=1e-9)
    assert hi == pytest.approx(entry["local_aabb"]["max"], abs=1e-9)


def test_marker_selects_world_mode_and_env_overrides(world_dataset, monkeypatch) -> None:
    parts, _ = world_dataset
    assert detect_mesh_frame(parts) == "world"
    monkeypatch.setenv("ASSEMBLY_MESH_FRAME", "link")
    assert detect_mesh_frame(parts) == "link"
    monkeypatch.setenv("ASSEMBLY_MESH_FRAME", "bogus")
    with pytest.raises(AssemblyError) as caught:
        detect_mesh_frame(parts)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT


def test_unmarked_directory_stays_in_link_mode(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    parts.mkdir()
    (parts / "body.obj").write_text(_cube_obj((0, 0, 0)))
    sdk = AssemblySDK(tmp_path / "state.json")
    sdk.import_parts(parts)
    assert sdk.state()["mesh_frame"] == "link"
    sdk.create_link("body", "part:body.obj")
    # Legacy placement API still works in link mode.
    sdk.set_link_visual_pose("body", [1.0, 0.0, 0.0])
    lo, _ = _aabb(sdk, "body")
    assert lo[0] == pytest.approx(0.9)


def test_joints_never_move_meshes_in_world_mode(world_dataset) -> None:
    parts, out = world_dataset
    sdk = AssemblySDK(out / "state.json")
    sdk.import_parts(parts)
    assert sdk.state()["mesh_frame"] == "world"
    sdk.create_link("body", "part:body.obj")
    sdk.create_link("door", "part:door.obj")
    _assert_in_place(sdk, "door")

    # A hinge at the door's inner edge: the mesh must stay where it was imported.
    sdk.create_joint("hinge", "REVOLUTE", "body", "door", origin_xyz=[0.5, 0.0, 0.5],
                     axis="Z", limits=[0.0, 1.57])
    _assert_in_place(sdk, "door")
    _assert_in_place(sdk, "body")
    link = sdk.state()["links"]["door"]
    assert link["visual_xyz"] == pytest.approx([-0.5, 0.0, -0.5])

    # Moving the joint frame, changing its parameters, or re-parenting: still in place.
    sdk.set_joint_world_frame("hinge", [0.5, 0.0, 0.0], [0, 0, 1])
    _assert_in_place(sdk, "door")
    sdk.update_joint("hinge", origin_xyz=[0.5, 0.1, 0.2], origin_rpy=[0.0, 0.0, 1.0])
    _assert_in_place(sdk, "door")

    # Joint motion still moves the child — the mode constrains q=0 only.
    sdk.set_joint_positions({"hinge": 1.0})
    lo_moved, _ = _aabb(sdk, "door")
    lo_rest = next(e for e in sdk.inspect_parts() if e["label"] == "door")["local_aabb"]["min"]
    assert lo_moved != pytest.approx(lo_rest, abs=1e-6)
    sdk.reset_joints()
    _assert_in_place(sdk, "door")


def test_mesh_moving_apis_are_refused_in_world_mode(world_dataset) -> None:
    parts, out = world_dataset
    sdk = AssemblySDK(out / "state.json")
    sdk.import_parts(parts)
    sdk.create_link("body", "part:body.obj")
    sdk.create_link("door", "part:door.obj")
    sdk.create_joint("hinge", "REVOLUTE", "body", "door", origin_xyz=[0.5, 0, 0.5], axis="Z", limits=[0, 1])
    for call in (
        lambda: sdk.set_link_visual_pose("door", [0.0, 0.0, 0.0]),
        lambda: sdk.set_link_pose("door", [0.0, 0.0, 0.0]),
        lambda: sdk.bake_current_pose_as_rest("door"),
        lambda: sdk.ground(),
        lambda: sdk.set_part_scale("part:door.obj", 2.0),
    ):
        with pytest.raises(AssemblyError) as caught:
            call()
        assert caught.value.code == ErrorCode.INVARIANT_VIOLATION
        assert caught.value.details.get("mesh_frame") == "world"
    with pytest.raises(AssemblyError) as caught:
        sdk.create_link("bogus", [], visual_xyz=[1.0, 0.0, 0.0])
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT
    _assert_in_place(sdk, "door")
    # safe_call surfaces the same refusal as a structured result, not a traceback.
    result = sdk.safe_call("ground")
    assert result["ok"] is False and result["error"]["code"] == "invariant_violation"


def test_audit_and_export_preserve_world_pose(world_dataset) -> None:
    parts, out = world_dataset
    sdk = AssemblySDK(out / "state.json")
    sdk.import_parts(parts)
    sdk.create_link("body", "part:body.obj")
    sdk.create_link("door", "part:door.obj")
    sdk.create_joint("hinge", "REVOLUTE", "body", "door", origin_xyz=[0.5, 0.0, 0.5],
                     origin_rpy=[0.0, 0.0, 0.3], axis="Z", limits=[0.0, 1.57])
    audit = sdk.audit()
    assert audit["ok"], audit["issues"]
    assert audit["mesh_frame"] == "world"

    sdk.export_urdf(out)
    model = parse_urdf(out / "furniture.urdf")
    poses = compute_world_poses(model, _joint_positions_at_reference(model))
    # world = chain @ visual must be the identity for every link, so the exported
    # (source-coordinate) meshes land exactly where they were imported.
    from evo.assembly_sdk.host_runtime import _matrix, _mat_mul
    for name, link in model.visual_links.items():
        world = _mat_mul(poses[name], _matrix(link.visual_xyz, getattr(link, "visual_rpy", (0, 0, 0))))
        for i in range(3):
            for j in range(4):
                assert world[i][j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-6), (name, i, j)

    # Tampering with the persisted visual transform is caught by audit.
    state = json.loads((out / "state.json").read_text())
    state["links"]["door"]["visual_xyz"][0] += 0.05
    (out / "state.json").write_text(json.dumps(state))
    issues = sdk.audit()["issues"]
    assert any(i["code"] == "world_frame_drift" for i in issues)
