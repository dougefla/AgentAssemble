"""Pure filesystem tests for collision-safe MTL and texture export."""
from __future__ import annotations

from pathlib import Path
import re

from evo.assembly_sdk.materials import (
    preserve_materials,
    resolve_source_material,
)


def _source(
    root: Path,
    directory: str,
    stem: str,
    material: str,
    texture_bytes: bytes | None,
) -> Path:
    source_dir = root / directory
    texture_dir = source_dir / "textures"
    texture_dir.mkdir(parents=True)
    obj = source_dir / f"{stem}.obj"
    obj.write_text(f"mtllib {stem}.mtl\nusemtl {material}\n")
    (source_dir / f"{stem}.mtl").write_text(
        f"newmtl {material}\nmap_Kd textures/tex.png\n"
    )
    if texture_bytes is not None:
        (texture_dir / "tex.png").write_bytes(texture_bytes)
    return obj


def test_cross_source_materials_and_textures_are_namespaced(tmp_path: Path) -> None:
    left = _source(tmp_path, "sub_a", "part", "mat_shared", b"LEFT")
    right = _source(tmp_path, "sub_b", "part", "mat_shared", b"RIGHT")
    meshes = tmp_path / "out" / "meshes"
    bundle = preserve_materials(
        "body",
        [("part:sub_a/part.obj", left), ("part:sub_b/part.obj", right)],
        meshes,
    )
    assert bundle.mtl_filename == "body_visual.mtl"
    left_name = bundle.material_names["part:sub_a/part.obj"]["mat_shared"]
    right_name = bundle.material_names["part:sub_b/part.obj"]["mat_shared"]
    assert left_name != right_name

    mtl = (meshes / bundle.mtl_filename).read_text()
    assert f"newmtl {left_name}" in mtl
    assert f"newmtl {right_name}" in mtl
    texture_refs = re.findall(r"map_Kd\s+(\S+)", mtl)
    assert len(texture_refs) == 2
    assert texture_refs[0] != texture_refs[1]
    assert (meshes / texture_refs[0]).read_bytes() == b"LEFT"
    assert (meshes / texture_refs[1]).read_bytes() == b"RIGHT"
    assert not bundle.warnings


def test_literal_numeric_suffix_wins_over_blender_collision_suffix() -> None:
    declared = {"wood.001", "wood"}
    assert resolve_source_material("wood.001", declared) == "wood.001"
    assert resolve_source_material("wood.001.001", declared) == "wood.001"
    assert resolve_source_material("wood.002", declared) == "wood"
    assert resolve_source_material("unknown.002", declared) is None


def test_missing_texture_produces_actionable_warning(tmp_path: Path) -> None:
    source = _source(tmp_path, "source", "part", "wood", None)
    meshes = tmp_path / "out" / "meshes"
    bundle = preserve_materials(
        "body", [("part:source/part.obj", source)], meshes
    )
    assert len(bundle.warnings) == 1
    assert "missing texture" in bundle.warnings[0]
    mtl = (meshes / bundle.mtl_filename).read_text()
    ref = re.search(r"map_Kd\s+(\S+)", mtl).group(1)
    assert not (meshes / ref).exists()


def test_material_namespace_is_independent_of_absolute_root(tmp_path: Path) -> None:
    first = _source(tmp_path / "one", "src", "part", "wood", b"A")
    second = _source(tmp_path / "two", "src", "part", "wood", b"A")
    one = preserve_materials(
        "body", [("part:src/part.obj", first)], tmp_path / "out_one"
    )
    two = preserve_materials(
        "body", [("part:src/part.obj", second)], tmp_path / "out_two"
    )
    assert one.material_names == two.material_names
