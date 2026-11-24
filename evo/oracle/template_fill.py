"""Joint-fill task templates: build, fill, and finalize.

The joint-fill dataset view (``artvip_jointfill``) hands the solver the GT
URDF with every joint's ``<origin xyz>`` and ``<axis xyz>`` replaced by a
placeholder. Topology, joint types, limits and the link <-> part mapping are
given; the solver's only job is to write the pivot and axis of every joint.

Task meshes are the ``artvip_beginner`` world-baked OBJs: every part already
sits in its q=0 rest pose in world coordinates. A URDF that references such
meshes is self-consistent only when every link's visual transform is the
inverse of that link's q=0 world frame, and that frame depends on the joint
origins the solver fills in. So the template carries no visual origins at
all, and :func:`finalize_filled_urdf` derives them by FK once the blanks are
filled. The result is a standard URDF the unchanged oracle can score.

The pose-fill view (``artvip_posefill``) goes one step further: every task
mesh is translated so its AABB center sits at the origin, and the template
also blanks each link's visual ``<origin xyz>``. The solver places the parts
(visual origin per link) *and* fills the joints; nothing is derived at
finalize time, the filled template only needs its meshes staged next to it
(:func:`stage_filled_urdf`). ``gt/<task>/<inst>/part_offsets.json`` records
the subtracted centers so the reference answer can be reconstructed
(:func:`posefill_values`).

Stdlib only. Shares the FK helpers of :mod:`evo.oracle.check_urdf`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from evo.oracle.check_urdf import (
    Mat4,
    _joint_positions_at_reference,
    _mat4_to_rpy,
    compute_world_poses,
    parse_urdf,
)

PLACEHOLDER = "? ? ?"
TEMPLATE_NAME = "template.urdf"
MOVABLE = ("revolute", "prismatic", "continuous")

_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

POSEFILL_HEADER = """\
<!--
  Pose-fill task. Topology, joint types, travel limits and the link <-> mesh
  mapping are given. The root link's part is the base of the object and is
  given in place: its mesh is in world coordinates and its visual origin is
  fixed at zero. Every other mesh has been translated so that its
  axis-aligned bounding-box center sits at the origin and then rotated by an
  unknown rotation about that origin. The blanks (three question marks, three
  numbers each) are:

    * every non-root link's visual <origin xyz> and <origin rpy>: a pose that
      puts the part onto the base in its assembled position and orientation
      (any orientation that yields the assembled geometry is correct);
    * every joint's <origin xyz> and <axis xyz>.

  Frames: the root link frame is the world frame, and <origin rpy> is "0 0 0"
  on every joint (you may change it, but you never need to), so every link
  frame is a pure translation of the world frame. A joint's <origin xyz> is
  the pivot expressed in the parent link's frame (world coordinates for a
  child of the root); <axis xyz> is the axis direction, which with zero joint
  rpy is the same in every frame. A link's visual <origin xyz> is the part
  center minus the link's own origin, both in world coordinates; its visual
  <origin rpy> is the part's orientation in the world.

  The positive axis direction is the direction of motion from limit "lower"
  towards "upper". A revolute pivot may be any point on the hinge line; a
  prismatic origin is not scored (only its direction is). Placement is scored
  through the rest pose and the motion of every part.
-->
"""

TEMPLATE_HEADER = """\
<!--
  Joint-fill task. Topology, joint types, travel limits and the link <-> mesh
  mapping are given. Every joint's <origin xyz> and <axis xyz> is left blank
  (three question marks): fill in three numbers each.

  Frames: every mesh is already in its q=0 rest pose in WORLD coordinates.
  The root link frame is the world frame, <origin rpy> is "0 0 0" on every
  joint (you may change it, but you never need to), so for a joint whose
  parent is the root, <origin xyz> is the pivot's world position and
  <axis xyz> the axis direction in world coordinates. For a nested joint,
  both are expressed in the parent link's frame (parent pivot as origin).
  Visual transforms are derived from your joint origins at export time; do
  not add them.

  The positive axis direction is the direction of motion from limit "lower"
  towards "upper". A revolute pivot may be any point on the hinge line; a
  prismatic origin is not scored (only its direction is).
-->
"""


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------

def _fmt(values) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def template_root_link(gt_urdf: Path) -> str:
    """The link that is no joint's child (the base of the object)."""
    root = ET.parse(gt_urdf).getroot()
    children = {j.find("child").attrib["link"] for j in root.findall("joint")}
    for link_el in root.findall("link"):
        if link_el.attrib["name"] not in children:
            return link_el.attrib["name"]
    raise ValueError(f"{gt_urdf}: no root link")


def build_template(gt_urdf: Path, *, mesh_prefix: str = "", blank_visual_origins: bool = False,
                   header: bool = True) -> str:
    """Return the joint-fill (or, with ``blank_visual_origins``, pose-fill) template text.

    Links keep their names and reference ``<mesh_prefix><basename>`` (the
    task-side copy of the same part). Joints keep name, type, parent, child
    and limits; origin xyz and axis xyz become :data:`PLACEHOLDER`; origin rpy
    is reset to zero so the source orientation never leaks. Reversed limits
    are sorted because the oracle repairs them for the reference only.

    With ``blank_visual_origins`` every non-root visual gets ``<origin
    xyz="? ? ?" rpy="? ? ?"/>`` as well (pose-fill view: the solver places
    and orients the parts); the root link's visual origin is written as zero,
    its part being the base that anchors the world frame. ``header=False``
    omits the explanatory XML comment, leaving a bare URDF with blanks.
    """
    src = ET.parse(gt_urdf).getroot()
    robot = ET.Element("robot", {"name": src.attrib.get("name", "robot")})
    root_link = template_root_link(gt_urdf)
    for link_el in src.findall("link"):
        link = ET.SubElement(robot, "link", {"name": link_el.attrib["name"]})
        mesh_el = link_el.find("visual/geometry/mesh")
        if mesh_el is None:
            continue
        visual = ET.SubElement(link, "visual")
        if blank_visual_origins:
            blank = link_el.attrib["name"] != root_link
            ET.SubElement(visual, "origin", {"xyz": PLACEHOLDER if blank else "0 0 0",
                                             "rpy": PLACEHOLDER if blank else "0 0 0"})
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "mesh", {
            "filename": mesh_prefix + Path(mesh_el.attrib["filename"]).name,
        })
    for joint_el in src.findall("joint"):
        jtype = joint_el.attrib["type"].lower()
        joint = ET.SubElement(robot, "joint", {
            "name": joint_el.attrib["name"], "type": jtype,
        })
        ET.SubElement(joint, "parent", {"link": joint_el.find("parent").attrib["link"]})
        ET.SubElement(joint, "child", {"link": joint_el.find("child").attrib["link"]})
        ET.SubElement(joint, "origin", {"xyz": PLACEHOLDER, "rpy": "0 0 0"})
        if jtype in MOVABLE:
            ET.SubElement(joint, "axis", {"xyz": PLACEHOLDER})
        limit_el = joint_el.find("limit")
        if limit_el is not None and jtype in ("revolute", "prismatic"):
            lo = float(limit_el.attrib.get("lower", 0.0))
            hi = float(limit_el.attrib.get("upper", 0.0))
            lo, hi = min(lo, hi), max(lo, hi)
            ET.SubElement(joint, "limit", {
                "lower": f"{lo:.6f}", "upper": f"{hi:.6f}",
                "effort": limit_el.attrib.get("effort", "100"),
                "velocity": limit_el.attrib.get("velocity", "1"),
            })
    ET.indent(robot, space="  ")
    comment = (POSEFILL_HEADER if blank_visual_origins else TEMPLATE_HEADER) if header else ""
    return '<?xml version="1.0" ?>\n' + comment + ET.tostring(robot, encoding="unicode") + "\n"


def posefill_values(gt_urdf: Path, transforms: dict[str, dict | list[float]]) -> tuple[dict[str, dict], dict[str, dict]]:
    """The pose-fill reference answer: ``(joint_values, visual_by_link)``.

    ``transforms`` maps each part basename (``part_<i>.obj``) to the transform
    applied when the task mesh was made: ``{"center": c, "rotation": R}`` with
    task vertices ``v' = R (v - c)`` (a bare list is a center with no
    rotation). In the template convention every link frame is a pure
    translation of the world frame, so a joint's origin is the child's world
    origin minus the parent's, its axis is the world axis, and a link's visual
    origin is ``xyz = c - link_world_origin`` with ``rpy`` of ``R^T`` (the
    rotation that puts the mesh back).
    """
    model = parse_urdf(gt_urdf)
    world = compute_world_poses(model, _joint_positions_at_reference(model))
    pos = {name: [T[i][3] for i in range(3)] for name, T in world.items()}
    joints: dict[str, dict] = {}
    for j in model.joints.values():
        Tc = world[j.child]
        origin = [pos[j.child][i] - pos[j.parent][i] for i in range(3)]
        axis = None
        if j.axis is not None:
            axis = [sum(Tc[i][k] * j.axis[k] for k in range(3)) for i in range(3)]
        joints[j.name] = {"origin_xyz": origin, "origin_rpy": "0 0 0", "axis": axis}
    visuals: dict[str, dict] = {}
    for link_el in ET.parse(gt_urdf).getroot().findall("link"):
        mesh_el = link_el.find("visual/geometry/mesh")
        if mesh_el is None:
            continue
        name = link_el.attrib["name"]
        tf = transforms[Path(mesh_el.attrib["filename"]).name]
        center = tf["center"] if isinstance(tf, dict) else tf
        R = (tf.get("rotation") if isinstance(tf, dict) else None) or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # visual rotation = R^T, as a 4x4 for the shared rpy extraction
        inv = tuple(tuple(float(R[j][i]) for j in range(3)) + (0.0,) for i in range(3)) + ((0.0, 0.0, 0.0, 1.0),)
        visuals[name] = {"xyz": [center[i] - pos[name][i] for i in range(3)], "rpy": list(_mat4_to_rpy(inv))}
    return joints, visuals


def gt_fill_values(gt_urdf: Path) -> dict[str, dict]:
    """The reference answer: ``{joint: {origin_xyz, origin_rpy, axis}}``."""
    values: dict[str, dict] = {}
    for joint_el in ET.parse(gt_urdf).getroot().findall("joint"):
        origin = joint_el.find("origin")
        axis = joint_el.find("axis")
        values[joint_el.attrib["name"]] = {
            "origin_xyz": origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            "origin_rpy": origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0",
            "axis": axis.attrib.get("xyz", "0 0 1") if axis is not None else None,
        }
    return values


def leaked_numbers(template_text: str, gt_urdf: Path) -> list[str]:
    """Numbers from the GT joint origins/axes that appear verbatim in the template.

    Zero, one and the limit values are expected; anything else is a leak.
    """
    allowed = {"0", "0.0", "0.000000", "1", "1.0", "1.000000", "-1", "-1.0", "-1.000000"}
    for joint_el in ET.parse(gt_urdf).getroot().findall("joint"):
        limit_el = joint_el.find("limit")
        if limit_el is not None:
            for key in ("lower", "upper"):
                allowed.add(limit_el.attrib.get(key, ""))
                allowed.add(f"{float(limit_el.attrib.get(key, 0.0)):.6f}")
    present = set(_NUMBER_RE.findall(template_text))
    leaks: list[str] = []
    for name, v in gt_fill_values(gt_urdf).items():
        tokens = v["origin_xyz"].split() + v["origin_rpy"].split() + (v["axis"] or "").split()
        for tok in tokens:
            if abs(float(tok)) < 1e-9 or abs(abs(float(tok)) - 1.0) < 1e-9:
                continue
            if tok in present or tok in allowed:
                leaks.append(f"{name}: {tok}")
    return leaks


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------

def fill_template(template_text: str, values: dict[str, dict],
                  visual_xyz: dict[str, list[float] | str] | None = None) -> str:
    """Write ``values`` (see :func:`gt_fill_values`) into a template's blanks.

    ``visual_xyz`` (pose-fill view) maps link names to the visual origin to
    write: ``{"xyz": [...], "rpy": [...]}``, or just the xyz as a list/string
    (rpy then becomes zero when blank); an optional ``"scale"`` entry is
    written to the visual's ``<mesh scale>``; links not listed keep whatever
    the template holds.
    """
    root = ET.fromstring(template_text)
    for link_el in root.findall("link"):
        v = (visual_xyz or {}).get(link_el.attrib["name"])
        if v is None:
            continue
        for vis in link_el.findall("visual"):
            origin = vis.find("origin")
            if origin is None:
                origin = ET.Element("origin", {"rpy": "0 0 0"})
                vis.insert(0, origin)
            xyz = v.get("xyz") if isinstance(v, dict) else v
            rpy = v.get("rpy") if isinstance(v, dict) else None
            origin.set("xyz", _fmt(xyz.split()) if isinstance(xyz, str) else _fmt(xyz))
            if rpy is not None:
                origin.set("rpy", _fmt(rpy.split()) if isinstance(rpy, str) else _fmt(rpy))
            elif "?" in origin.attrib.get("rpy", ""):
                origin.set("rpy", "0 0 0")
            scale = v.get("scale") if isinstance(v, dict) else None
            if scale is not None:
                for mesh_el in vis.findall("geometry/mesh"):
                    mesh_el.set("scale", _fmt(scale.split()) if isinstance(scale, str) else _fmt(scale))
    for joint_el in root.findall("joint"):
        v = values.get(joint_el.attrib["name"])
        if v is None:
            continue
        origin = joint_el.find("origin")
        if origin is not None:
            origin.set("xyz", _fmt(str(v["origin_xyz"]).split()) if isinstance(v["origin_xyz"], str) else _fmt(v["origin_xyz"]))
            rpy = v.get("origin_rpy", "0 0 0")
            origin.set("rpy", _fmt(rpy.split()) if isinstance(rpy, str) else _fmt(rpy))
        axis = joint_el.find("axis")
        if axis is not None and v.get("axis") is not None:
            a = v["axis"]
            axis.set("xyz", _fmt(a.split()) if isinstance(a, str) else _fmt(a))
    ET.indent(root, space="  ")
    return '<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def unfilled_blanks(urdf_path: Path) -> list[str]:
    """``joint/attribute`` entries that still hold a placeholder or no number."""
    root = ET.parse(urdf_path).getroot()
    missing: list[str] = []
    for link_el in root.findall("link"):
        for vis in link_el.findall("visual"):
            origin = vis.find("origin")
            if origin is None:
                continue
            for attr in ("xyz", "rpy"):
                value = origin.attrib.get(attr, "")
                if "?" in value or (attr == "xyz" and len(_NUMBER_RE.findall(value)) != 3):
                    missing.append(f"{link_el.attrib.get('name', '?')}/visual_{attr}")
        for mesh_el in link_el.findall("visual/geometry/mesh"):
            value = mesh_el.attrib.get("scale")
            if value is not None and ("?" in value or len(_NUMBER_RE.findall(value)) != 3):
                missing.append(f"{link_el.attrib.get('name', '?')}/visual_scale")
    for joint_el in root.findall("joint"):
        name = joint_el.attrib.get("name", "?")
        for tag in ("origin", "axis"):
            el = joint_el.find(tag)
            if el is None:
                continue
            value = el.attrib.get("xyz", "")
            if "?" in value or len(_NUMBER_RE.findall(value)) != 3:
                missing.append(f"{name}/{tag}")
    return missing


def needs_finalize(urdf_path: Path) -> bool:
    """True when a URDF is a filled template: at least one visual without an origin."""
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError:
        return False
    for link_el in root.findall("link"):
        for vis in link_el.findall("visual"):
            if vis.find("geometry/mesh") is not None and vis.find("origin") is None:
                return True
    return False


# ---------------------------------------------------------------------------
# Finalize: derive visual transforms so world-baked meshes stay in place
# ---------------------------------------------------------------------------

def _invert_rigid(m: Mat4) -> Mat4:
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    t = [m[i][3] for i in range(3)]
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    nt = [-sum(rt[i][k] * t[k] for k in range(3)) for i in range(3)]
    return tuple(
        tuple(rt[i] + [nt[i]]) for i in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def _resolve_mesh(ref: str, urdf_dir: Path, mesh_dir: Path | None) -> Path:
    candidates = [urdf_dir / ref]
    if mesh_dir is not None:
        candidates.append(mesh_dir / Path(ref).name)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"mesh {ref!r} not found next to the URDF" + (f" or in {mesh_dir}" if mesh_dir else ""))


def finalize_filled_urdf(
    filled_urdf: Path,
    out_dir: Path,
    *,
    mesh_dir: Path | None = None,
    copy_meshes: bool = False,
    output_name: str = "furniture.urdf",
) -> Path:
    """Turn a filled template into a standard URDF at ``out_dir/output_name``.

    Every visual/collision origin is set to the inverse of the link's q=0
    world frame so the world-baked meshes stay exactly where they are. Meshes
    are linked (or copied with ``copy_meshes``) into ``out_dir/meshes/`` and
    referenced as ``meshes/<basename>``. Joint values are written back
    verbatim. Raises ``ValueError`` when blanks remain.
    """
    filled_urdf = Path(filled_urdf)
    out_dir = Path(out_dir)
    blanks = unfilled_blanks(filled_urdf)
    if blanks:
        raise ValueError(f"template still has unfilled blanks: {blanks}")

    model = parse_urdf(filled_urdf)
    world = compute_world_poses(model, _joint_positions_at_reference(model))
    root = ET.parse(filled_urdf).getroot()

    meshes_dir = out_dir / "meshes"
    out_dir.mkdir(parents=True, exist_ok=True)
    if meshes_dir.is_symlink() or meshes_dir.is_file():
        meshes_dir.unlink()
    meshes_dir.mkdir(exist_ok=True)

    urdf_dir = filled_urdf.resolve().parent
    staged: dict[str, tuple | None] = {}
    for link_el in root.findall("link"):
        name = link_el.attrib["name"]
        T = world.get(name)
        if T is None:
            raise ValueError(f"link {name!r} is not reachable from the root")
        inv = _invert_rigid(T)
        xyz = (inv[0][3], inv[1][3], inv[2][3])
        rpy = _mat4_to_rpy(inv)
        for tag in ("visual", "collision"):
            for el in link_el.findall(tag):
                mesh_el = el.find("geometry/mesh")
                if mesh_el is None:
                    continue
                src = _resolve_mesh(mesh_el.attrib["filename"], urdf_dir, mesh_dir)
                _stage_mesh(src, meshes_dir, staged, copy_meshes, scale=_mesh_scale(mesh_el))
                mesh_el.set("filename", f"meshes/{src.name}")
                mesh_el.attrib.pop("scale", None)
                origin = el.find("origin")
                if origin is None:
                    origin = ET.Element("origin")
                    el.insert(0, origin)
                origin.set("xyz", " ".join(f"{v:.9g}" for v in xyz))
                origin.set("rpy", " ".join(f"{v:.9g}" for v in rpy))
    ET.indent(root, space="  ")
    out = out_dir / output_name
    out.write_text('<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n")
    return out


def _mesh_scale(mesh_el: ET.Element) -> tuple[float, float, float] | None:
    """The ``<mesh scale>`` of a filled template as three floats, ``None`` when absent or unit."""
    value = mesh_el.attrib.get("scale")
    if value is None:
        return None
    nums = [float(x) for x in _NUMBER_RE.findall(value)]
    if len(nums) != 3:
        raise ValueError(f"mesh scale {value!r} is not three numbers")
    if all(abs(s - 1.0) < 1e-12 for s in nums):
        return None
    return (nums[0], nums[1], nums[2])


def _write_scaled_obj(src: Path, dst: Path, scale: tuple[float, float, float]) -> None:
    """Copy an OBJ with every vertex multiplied by ``scale`` (normals are left alone)."""
    with open(src, errors="replace") as fi, open(dst, "w") as fo:
        for line in fi:
            if line.startswith("v "):
                parts = line.split()
                xyz = [float(c) * s for c, s in zip(parts[1:4], scale)]
                fo.write("v %.9g %.9g %.9g" % tuple(xyz) + (" " + " ".join(parts[4:]) if len(parts) > 4 else "") + "\n")
            else:
                fo.write(line)


def _stage_mesh(src: Path, meshes_dir: Path, staged: dict[str, tuple | None] | set, copy: bool,
                scale: tuple[float, float, float] | None = None) -> None:
    """Link/copy an OBJ plus its MTL and textures into ``meshes_dir`` once.

    With ``scale`` (a filled template's ``<mesh scale>``) the staged OBJ is a scaled
    copy, so the oracle, which reads no scale attribute, sees the intended size.
    """
    if isinstance(staged, set):  # legacy callers
        if src.name in staged:
            return
        staged.add(src.name)
    else:
        if src.name in staged:
            if staged[src.name] != scale:
                raise ValueError(f"mesh {src.name} is referenced with two different scales")
            return
        staged[src.name] = scale
    targets = [src]
    mtl = src.with_suffix(".mtl")
    if mtl.is_file():
        targets.append(mtl)
    for t in targets:
        dst = meshes_dir / t.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if scale is not None and t is src:
            _write_scaled_obj(t, dst, scale)
        elif copy:
            shutil.copy2(t, dst)
        else:
            os.symlink(t.resolve(), dst)
    textures = src.parent / "textures"
    tex_dst = meshes_dir / "textures"
    if textures.is_dir() and not tex_dst.exists():
        if copy:
            shutil.copytree(textures, tex_dst)
        else:
            os.symlink(textures.resolve(), tex_dst)


def unresolved_meshes(urdf_path: Path) -> list[str]:
    """Mesh references of a URDF that do not resolve next to it."""
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError:
        return []
    base = Path(urdf_path).resolve().parent
    return sorted({m.attrib["filename"] for m in root.iter("mesh")
                   if "filename" in m.attrib and not (base / m.attrib["filename"]).is_file()})


def stage_filled_urdf(
    filled_urdf: Path,
    out_dir: Path,
    *,
    mesh_dir: Path | None = None,
    copy_meshes: bool = False,
    output_name: str = "furniture.urdf",
) -> Path:
    """Pose-fill counterpart of :func:`finalize_filled_urdf`: keep every origin as
    written, only stage the referenced meshes into ``out_dir/meshes/`` and
    rewrite the references. Raises ``ValueError`` when blanks remain.
    """
    filled_urdf = Path(filled_urdf)
    out_dir = Path(out_dir)
    blanks = unfilled_blanks(filled_urdf)
    if blanks:
        raise ValueError(f"template still has unfilled blanks: {blanks}")
    root = ET.parse(filled_urdf).getroot()
    meshes_dir = out_dir / "meshes"
    out_dir.mkdir(parents=True, exist_ok=True)
    if meshes_dir.is_symlink() or meshes_dir.is_file():
        meshes_dir.unlink()
    meshes_dir.mkdir(exist_ok=True)
    urdf_dir = filled_urdf.resolve().parent
    staged: dict[str, tuple | None] = {}
    for mesh_el in root.iter("mesh"):
        src = _resolve_mesh(mesh_el.attrib["filename"], urdf_dir, mesh_dir)
        _stage_mesh(src, meshes_dir, staged, copy_meshes, scale=_mesh_scale(mesh_el))
        mesh_el.set("filename", f"meshes/{src.name}")
        mesh_el.attrib.pop("scale", None)
    ET.indent(root, space="  ")
    out = out_dir / output_name
    out.write_text('<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n")
    return out


def finalize_pred_dir(pred_dir: Path, *, mesh_dir: Path | None = None) -> Path | None:
    """Loop hook: make ``pred_dir/furniture.urdf`` scorable when it is a filled template.

    - joint-fill view (visuals without an origin): visual origins are derived
      by FK and the meshes staged (:func:`finalize_filled_urdf`);
    - pose-fill view (origins present but the template's ``part_<i>.obj``
      references do not resolve next to the URDF): meshes are staged from
      ``mesh_dir`` and nothing else changes (:func:`stage_filled_urdf`).

    A URDF that already carries visual origins and resolvable meshes (the
    SDK's ``export_urdf`` output, or an already finalized file) is left
    untouched and ``None`` is returned. Otherwise the raw file is kept as
    ``furniture.filled.urdf`` and the finalized URDF replaces it.
    """
    pred_dir = Path(pred_dir)
    urdf = pred_dir / "furniture.urdf"
    if not urdf.is_file():
        return None
    if needs_finalize(urdf):
        raw = pred_dir / "furniture.filled.urdf"
        shutil.copy2(urdf, raw)
        return finalize_filled_urdf(raw, pred_dir, mesh_dir=mesh_dir, copy_meshes=True)
    if mesh_dir is not None and unresolved_meshes(urdf):
        raw = pred_dir / "furniture.filled.urdf"
        shutil.copy2(urdf, raw)
        return stage_filled_urdf(raw, pred_dir, mesh_dir=mesh_dir, copy_meshes=True)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the template for one GT scene.urdf")
    b.add_argument("gt_urdf", type=Path)
    b.add_argument("out", type=Path)
    f = sub.add_parser("finalize", help="filled template -> standard URDF with derived visual origins")
    f.add_argument("filled_urdf", type=Path)
    f.add_argument("out_dir", type=Path)
    f.add_argument("--mesh-dir", type=Path, default=None)
    f.add_argument("--copy", action="store_true", help="copy meshes instead of symlinking")
    c = sub.add_parser("check", help="list unfilled blanks")
    c.add_argument("urdf", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        args.out.write_text(build_template(args.gt_urdf))
        print(args.out)
    elif args.cmd == "finalize":
        print(finalize_filled_urdf(args.filled_urdf, args.out_dir, mesh_dir=args.mesh_dir, copy_meshes=args.copy))
    else:
        blanks = unfilled_blanks(args.urdf)
        print(json.dumps({"unfilled": blanks}))
        return 1 if blanks else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
