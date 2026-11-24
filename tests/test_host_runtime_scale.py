from __future__ import annotations

import json
from pathlib import Path

import pytest

from evo.assembly_sdk.host_runtime import AssemblySDK


_CUBE_OBJ = """v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 2 3 4
f 5 8 7 6
"""


def _extent(bounds: dict, axis: int) -> float:
    return float(bounds["max"][axis]) - float(bounds["min"][axis])


def test_host_runtime_persists_scale_through_lazy_restore_and_export(
    tmp_path: Path,
) -> None:
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    source = parts_dir / "body.obj"
    source.write_text(_CUBE_OBJ)

    sdk = AssemblySDK(tmp_path / "state.json")
    imported = sdk.import_parts(parts_dir)
    handle = imported[0]["handle"]
    sdk.create_link("body", handle)
    assert sdk.set_part_scale(handle, 0.25) == {
        "handle": handle,
        "scale": 0.25,
    }

    first = sdk.inspect_parts()[0]
    assert first["scale"] == pytest.approx(0.25)
    assert [_extent(first["world_aabb"], axis) for axis in range(3)] == pytest.approx(
        [0.25, 0.25, 0.25]
    )

    # Simulate a fresh worker process: durable state remains, mesh cache does not.
    sdk._registry.clear()
    restored = sdk.inspect_parts()[0]
    assert restored["scale"] == pytest.approx(0.25)
    assert [_extent(restored["world_aabb"], axis) for axis in range(3)] == pytest.approx(
        [0.25, 0.25, 0.25]
    )
    assert sdk.audit()["ok"] is True

    exported = sdk.export_urdf(tmp_path / "export")
    parts_map = json.loads(Path(exported["parts_map"]).read_text())
    assert parts_map == {
        "links": {"body": [{"path": str(source), "scale": 0.25}]},
        "version": 2,
    }

    vertices = [
        tuple(float(value) for value in line.split()[1:4])
        for line in (tmp_path / "export/meshes/body_visual.obj").read_text().splitlines()
        if line.startswith("v ")
    ]
    assert max(v[0] for v in vertices) - min(v[0] for v in vertices) == pytest.approx(
        0.25
    )


def test_host_runtime_rejects_non_positive_scale(tmp_path: Path) -> None:
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    (parts_dir / "body.obj").write_text(_CUBE_OBJ)
    sdk = AssemblySDK(tmp_path / "state.json")
    handle = sdk.import_parts(parts_dir)[0]["handle"]

    result = sdk.safe_call("set_part_scale", handle=handle, scale=0.0)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_argument"


def test_host_runtime_confines_artifacts_to_output_dir(tmp_path: Path) -> None:
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    (parts_dir / "body.obj").write_text(_CUBE_OBJ)
    output_dir = tmp_path / "output"
    sdk = AssemblySDK(output_dir / ".assembly_state.json")
    handle = sdk.import_parts(parts_dir)[0]["handle"]
    sdk.create_link("body", handle)

    exported = sdk.export_urdf("relative_export")
    assert Path(exported["urdf"]) == output_dir / "relative_export/furniture.urdf"

    escaped = sdk.safe_call("export_urdf", output_dir=tmp_path / "outside")
    assert escaped["ok"] is False
    assert escaped["error"]["code"] == "invalid_argument"
    assert "inside OUTPUT_DIR" in escaped["error"]["message"]
