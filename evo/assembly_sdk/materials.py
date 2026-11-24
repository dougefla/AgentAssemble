"""Pure, collision-safe material/texture export helpers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shlex
import shutil


_TEXTURE_DIRECTIVES = frozenset({
    "map_ka", "map_kd", "map_ks", "map_ke", "map_ns", "map_d",
    "map_bump", "bump", "disp", "decal", "norm", "refl",
})
_BLENDER_SUFFIX_RE = re.compile(r"^(.*)\.\d{3,}$")


@dataclass(frozen=True)
class MaterialBundle:
    """Namespaced MTL output and source-to-export identity mapping."""

    mtl_filename: str | None
    material_names: dict[str, dict[str, str]]
    warnings: tuple[str, ...]


def _safe_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return token or fallback


def source_namespace(source_id: str, source_obj: Path) -> str:
    """Stable across machines while remaining readable in exported files."""
    stem = _safe_token(source_obj.stem, "part")
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{digest}"


def exported_material_name(namespace: str, original: str) -> str:
    token = _safe_token(original, "material")
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
    return f"{namespace}__{token}__{digest}"


def resolve_source_material(slot_name: str, declared: set[str]) -> str | None:
    """Map a Blender slot back to one source MTL name without guessing.

    A literal source name wins. Blender collision suffixes are removed only
    when the full name is not declared and the stripped candidate is.
    """
    if slot_name in declared:
        return slot_name
    candidate = slot_name
    while True:
        match = _BLENDER_SUFFIX_RE.fullmatch(candidate)
        if match is None:
            return None
        candidate = match.group(1)
        if candidate in declared:
            return candidate


def _source_mtls(source_obj: Path) -> list[Path]:
    references: list[Path] = []
    if source_obj.is_file():
        for line in source_obj.read_text(errors="replace").splitlines():
            if not line.lower().startswith("mtllib "):
                continue
            try:
                names = shlex.split(line)[1:]
            except ValueError:
                names = line.split()[1:]
            references.extend(source_obj.parent / name for name in names)
    fallback = source_obj.with_suffix(".mtl")
    if not references and fallback.is_file():
        references.append(fallback)
    # Preserve declaration order but remove duplicates.
    seen: set[Path] = set()
    result = []
    for path in references:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _material_names(mtl_path: Path) -> list[str]:
    if not mtl_path.is_file():
        return []
    names = []
    for line in mtl_path.read_text(errors="replace").splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2 and fields[0].lower() == "newmtl" and fields[1].strip():
            names.append(fields[1].strip())
    return names


def _safe_texture_relative(reference: str) -> Path:
    raw = Path(reference.replace("\\", "/"))
    parts = [part for part in raw.parts if part not in {"", ".", "..", "/"}]
    return Path(*parts) if parts else Path("texture")


def _rewrite_mtl(
    mtl_path: Path,
    namespace: str,
    material_map: dict[str, str],
    meshes_dir: Path,
    warnings: list[str],
) -> list[str]:
    if not mtl_path.is_file():
        warnings.append(f"missing MTL: {mtl_path}")
        return []
    output = [f"# Source MTL: {mtl_path.name}"]
    for line in mtl_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue
        head = stripped.split(None, 1)[0].lower()
        if head == "newmtl":
            fields = stripped.split(None, 1)
            if len(fields) != 2 or fields[1].strip() not in material_map:
                warnings.append(f"malformed newmtl in {mtl_path.name}: {line}")
                output.append(line)
                continue
            original = fields[1].strip()
            output.append(f"newmtl {material_map[original]}")
            continue
        if head in _TEXTURE_DIRECTIVES:
            try:
                tokens = shlex.split(stripped)
            except ValueError:
                tokens = stripped.split()
            if len(tokens) < 2:
                output.append(line)
                continue
            reference = tokens[-1]
            source_texture = (mtl_path.parent / reference).resolve()
            relative = Path("textures") / namespace / _safe_texture_relative(reference)
            destination = meshes_dir / relative
            if source_texture.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_texture, destination)
            else:
                warnings.append(
                    f"missing texture referenced by {mtl_path.name}: {reference}"
                )
            tokens[-1] = relative.as_posix()
            output.append(" ".join(tokens))
            continue
        output.append(line)
    return output


def preserve_materials(
    link_name: str,
    sources: list[tuple[str, Path]],
    meshes_dir: Path,
) -> MaterialBundle:
    """Namespace each source's materials and texture paths.

    ``sources`` contains ``(stable_part_handle, source_obj_path)`` pairs. The
    stable handle, not an absolute filesystem path or Blender-generated name,
    defines the namespace, so output is deterministic across machines.
    """
    warnings: list[str] = []
    mappings: dict[str, dict[str, str]] = {}
    rewritten: list[str] = []
    any_mtl = False
    for source_id, source_obj_value in sources:
        source_obj = Path(source_obj_value)
        namespace = source_namespace(source_id, source_obj)
        mtls = _source_mtls(source_obj)
        if not mtls:
            mappings[source_id] = {}
            continue
        declared: list[str] = []
        for mtl in mtls:
            if not mtl.is_file():
                warnings.append(f"missing MTL referenced by {source_obj.name}: {mtl}")
                continue
            declared.extend(_material_names(mtl))
        duplicates = sorted({name for name in declared if declared.count(name) > 1})
        if duplicates:
            warnings.append(
                f"duplicate material declarations in {source_obj.name}: {duplicates}"
            )
        material_map = {
            name: exported_material_name(namespace, name)
            for name in dict.fromkeys(declared)
        }
        mappings[source_id] = material_map
        for mtl in mtls:
            if not mtl.is_file():
                continue
            any_mtl = True
            rewritten.extend(
                _rewrite_mtl(mtl, namespace, material_map, meshes_dir, warnings)
            )
            rewritten.append("")

    if not any_mtl:
        return MaterialBundle(None, mappings, tuple(warnings))
    meshes_dir.mkdir(parents=True, exist_ok=True)
    destination_name = f"{link_name}_visual.mtl"
    (meshes_dir / destination_name).write_text("\n".join(rewritten).rstrip() + "\n")
    return MaterialBundle(destination_name, mappings, tuple(warnings))
