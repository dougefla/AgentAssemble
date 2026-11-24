#!/usr/bin/env python3
"""URDF comparison oracle for the /assemble skill.

Compares a predicted URDF against a ground-truth URDF and returns a
structured verdict. Useful as a 1-bit oracle in a co-evolution loop,
or standalone for batch evaluation.

Zero external dependencies — uses only Python stdlib.

Usage:
    python check_urdf.py <gt_dir> <pred_dir> [--verbose] [--thresholds T]

    gt_dir:   contains scene.urdf + mesh files (outputs/...)
    pred_dir: contains furniture.urdf (or *.urdf) + meshes/

Exit code: 0 = pass, 1 = fail, 2 = error.
"""
from __future__ import annotations

from evo.oracle.part_equivalence import (
    candidate_matchings, equivalence_classes, load_part_equivalence,
    load_part_equivalence_transforms,
)

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

# Recorded in every OracleResult.to_dict() as "oracle_version" so an oracle.json
# says which oracle build scored it. CHANGELOG:
#   1.x                      — mesh geometry read from the OBJ as written; a
#                              URDF <mesh scale> was honoured only when the
#                              loop's finalisation step baked it into staged
#                              vertex copies (skipped for URDFs that already had
#                              visual origins and resolvable meshes).
#   2.0-scale-unconditional  — <mesh scale> applied to the vertices at mesh
#                              load time for pred and GT alike, regardless of
#                              staging (2026-11-02).
#   3.0-surface              — pose metric (rest_pose_alignment and
#                              kinematic_equivalence) is the two-sided
#                              point-to-surface distance between the posed
#                              meshes instead of vertex moments: invariant to
#                              a part's own symmetries and linear in the error.
#                              The global similarity is taken from the anchor
#                              (root) link alone, so one link's size error no
#                              longer leaks into every other link. Per-joint
#                              reports gain rest_drift_m and a pose_cause
#                              diagnosis (2026-11-12). EVO_ORACLE_POSE_METRIC=
#                              moments restores the 2.0 metric.
ORACLE_VERSION = "3.0-surface"

UNIT_SCALE: tuple[float, float, float] = (1.0, 1.0, 1.0)

from evo.oracle import surface_distance as _sd  # noqa: E402


def _surface_metric_enabled() -> bool:
    """Oracle 3 pose metric unless numpy/scipy are missing or it is switched off."""
    return _sd.AVAILABLE and os.environ.get("EVO_ORACLE_POSE_METRIC", "surface").strip().lower() != "moments"


# ---------------------------------------------------------------------------
# Data model (lightweight, no LinkForge dependency)
# ---------------------------------------------------------------------------

@dataclass
class LinkInfo:
    name: str
    mesh_paths: list[str] = field(default_factory=list)
    has_visual: bool = False
    # First visual element's <origin> in link frame. Used to position the
    # mesh inside the link. Both pred and GT can use this freely (e.g.,
    # GT centers the mesh and pushes the offset into visual_origin; pred
    # often bakes the offset into mesh vertices and leaves visual_origin
    # at zero). The pose checks transform mesh features through this
    # origin so both conventions land at the same world position.
    visual_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visual_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # ``<mesh scale="sx sy sz">`` of each entry in ``mesh_paths`` (parallel
    # list, unit when the attribute is absent). Applied to the OBJ vertices at
    # load time (``link_signature``) so the oracle sees the intended part size
    # whether or not the loop's staging step baked the scale into a copy.
    mesh_scales: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def mesh_scale(self) -> tuple[float, float, float]:
        """Scale of the first visual mesh (the one every geometry check reads)."""
        return self.mesh_scales[0] if self.mesh_scales else UNIT_SCALE

@dataclass
class JointInfo:
    name: str
    type: str  # FIXED, REVOLUTE, PRISMATIC, CONTINUOUS, FLOATING, PLANAR
    parent: str
    child: str
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] | None = None
    limit_lower: float | None = None
    limit_upper: float | None = None
    # Joint value in the canonicalized representation that corresponds to
    # q=0 in the source URDF.  The reference render is authored at q=0; after
    # shifting lower limits to zero this is not generally the lower endpoint.
    reference_position: float = 0.0

@dataclass
class URDFModel:
    name: str
    links: dict[str, LinkInfo] = field(default_factory=dict)
    joints: dict[str, JointInfo] = field(default_factory=dict)
    base_dir: Path = field(default_factory=lambda: Path("."))
    # GT only, set per candidate link assignment by run_oracle: when the
    # matched prediction link holds a *different* interchangeable part X than
    # this link Y, the pose metrics compare against X's own vertex cloud moved
    # into Y's slot: {Y: (LinkInfo of X, 4x4 in Y's link frame)}.
    link_substitutes: dict = field(default_factory=dict)

    @property
    def visual_links(self) -> dict[str, LinkInfo]:
        """Links that have at least one visual mesh (i.e., real parts)."""
        return {n: l for n, l in self.links.items() if l.has_visual}

    @property
    def movable_joints(self) -> dict[str, JointInfo]:
        """Non-FIXED joints."""
        return {n: j for n, j in self.joints.items() if j.type != "FIXED"}

    def parent_of(self, link_name: str) -> JointInfo | None:
        """Find the joint whose child is link_name."""
        for j in self.joints.values():
            if j.child == link_name:
                return j
        return None

    def children_of(self, link_name: str) -> list[JointInfo]:
        """Find joints whose parent is link_name."""
        return [j for j in self.joints.values() if j.parent == link_name]

    def root_link(self) -> str | None:
        """Find the root link (no joint has it as child)."""
        children = {j.child for j in self.joints.values()}
        roots = [n for n in self.links if n not in children]
        return roots[0] if len(roots) == 1 else None


# ---------------------------------------------------------------------------
# URDF parsing (stdlib XML)
# ---------------------------------------------------------------------------

def _parse_scale(text: str | None) -> tuple[float, float, float]:
    """``<mesh scale>`` as three floats; absent -> unit. One number scales uniformly."""
    if text is None or not text.strip():
        return UNIT_SCALE
    parts = text.split()
    if len(parts) == 1:
        v = float(parts[0])
        return (v, v, v)
    if len(parts) != 3:
        raise ValueError(f"mesh scale {text!r} is not three numbers")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_xyz(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    parts = text.strip().split()
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _canonicalize_axis_limits(
    axis: tuple[float, float, float] | None,
    lo: float | None,
    hi: float | None,
) -> tuple[tuple[float, float, float] | None, float | None, float | None]:
    """Flip axis sign + negate-and-swap limits so the dominant axis component
    is non-negative. (Step 1 of joint canonicalization — does NOT touch the
    joint_origin; see `_canonicalize_joint` for the full pipeline.)

    Two URDFs that differ only by this convention are physically identical
    (same rest pose at q=0, same set of reachable poses, same FK output for
    every reachable physical configuration).

    Examples:
        (axis=(-1,0,0), lo=-1.57, hi=0)    → (axis=(1,0,0),  lo=0,    hi=1.57)
        (axis=(0,0,-1), lo=-3.14, hi=3.14) → (axis=(0,0,1),  lo=-3.14, hi=3.14)  (symmetric, lo/hi swap)
        (axis=(0,1,0),  lo=0,     hi=0.3)  → unchanged (already canonical)

    No-op when axis is None or all-zero.
    """
    if axis is None:
        return axis, lo, hi
    primary = max(range(3), key=lambda i: abs(axis[i]))
    if axis[primary] >= 0:
        return axis, lo, hi
    flipped_axis = (-axis[0], -axis[1], -axis[2])
    if lo is None or hi is None:
        return flipped_axis, lo, hi
    return flipped_axis, -hi, -lo


def _mat4_to_rpy(m: "Mat4") -> tuple[float, float, float]:
    """Extract URDF (extrinsic XYZ) rpy from a rotation matrix. Inverse of
    `_mat4_rotate_rpy`. Handles gimbal lock at pitch=±π/2 by zeroing roll
    (any decomposition is valid; we pick the canonical one)."""
    # Recall _mat4_rotate_rpy returns Rz(y)·Ry(p)·Rx(r) so:
    #   m[2][0] = -sin(p)
    #   m[2][1] = cos(p)·sin(r)
    #   m[2][2] = cos(p)·cos(r)
    #   m[1][0] = sin(y)·cos(p)
    #   m[0][0] = cos(y)·cos(p)
    sp = -m[2][0]
    sp = max(-1.0, min(1.0, sp))  # clamp for acos/asin numerical safety
    pitch = math.asin(sp)
    cp = math.cos(pitch)
    if abs(cp) > 1e-7:
        roll = math.atan2(m[2][1], m[2][2])
        yaw = math.atan2(m[1][0], m[0][0])
    else:
        # Gimbal lock: pitch ≈ ±π/2. Convention: roll = 0, fold rotation
        # into yaw. m[0][1] and m[0][2] then encode yaw.
        roll = 0.0
        yaw = math.atan2(-m[0][1], m[1][1])
    return (roll, pitch, yaw)


def _canonicalize_joint(
    jtype: str,
    axis: tuple[float, float, float] | None,
    lo: float | None,
    hi: float | None,
    origin_xyz: tuple[float, float, float],
    origin_rpy: tuple[float, float, float],
) -> tuple[
    str,                                            # canonical joint type
    tuple[float, float, float] | None,              # canonical axis
    float | None, float | None,                     # canonical limits
    tuple[float, float, float],                     # canonical origin xyz
    tuple[float, float, float],                     # canonical origin rpy
]:
    """Canonicalize a parsed joint to a unique form within its physical
    equivalence class. Composes three normalizations:

      1. CONTINUOUS → REVOLUTE with limits=[0, 2π]. Same physical free
         rotation, same FK trace under t∈[0,1] sweep. Lets joint_types and
         joint_limits checks compare cleanly without special-casing.

      2. Axis sign canonicalization (see `_canonicalize_axis_limits`):
         flip axis + negate-swap limits when axis dominant component is negative.

      3. Limits-shift canonicalization: shift `lo` to 0 by composing
         joint_origin with M(axis, lo), where M=R for revolute and M=T for
         prismatic. Sliding `lo` along the parameter axis is gauge for both
         joint types because translation along axis commutes with T(axis, q),
         and rotation around axis commutes with R(axis, q).

    All three are physical-equivalence preserving: world poses of the child
    link at every q ∈ [original lo, original hi] are unchanged (after
    reparameterizing q → q - shift).
    """
    # (1) CONTINUOUS → REVOLUTE [0, 2π]
    if jtype == "CONTINUOUS":
        jtype = "REVOLUTE"
        if lo is None or hi is None:
            lo, hi = 0.0, 2.0 * math.pi

    # (2) Axis sign flip + limits negate/swap
    axis, lo, hi = _canonicalize_axis_limits(axis, lo, hi)

    # (3) Shift lo → 0 by composing joint_origin with M(axis, lo)
    if (
        axis is not None
        and lo is not None
        and hi is not None
        and abs(lo) > 1e-9
        and jtype in ("REVOLUTE", "PRISMATIC")
    ):
        T_orig = _mat4_mul(_mat4_translate(origin_xyz),
                           _mat4_rotate_rpy(origin_rpy))
        if jtype == "REVOLUTE":
            M = _mat4_rotate_axis_angle(axis, lo)
        else:  # PRISMATIC: translate by lo along normalized axis
            mag = math.sqrt(sum(a * a for a in axis))
            if mag > 1e-9:
                shift = (axis[0] / mag * lo, axis[1] / mag * lo, axis[2] / mag * lo)
            else:
                shift = (0.0, 0.0, 0.0)
            M = _mat4_translate(shift)
        T_new = _mat4_mul(T_orig, M)
        origin_xyz = (T_new[0][3], T_new[1][3], T_new[2][3])
        origin_rpy = _mat4_to_rpy(T_new)
        hi = hi - lo
        lo = 0.0

    return jtype, axis, lo, hi, origin_xyz, origin_rpy


def _canonical_reference_position(
    jtype: str,
    axis: tuple[float, float, float] | None,
    lo: float | None,
    hi: float | None,
) -> float:
    """Map source-URDF q=0 into `_canonicalize_joint` coordinates."""
    if jtype == "CONTINUOUS" and (lo is None or hi is None):
        lo, hi = 0.0, 2.0 * math.pi
    if axis is not None:
        primary = max(range(3), key=lambda i: abs(axis[i]))
        if axis[primary] < 0 and lo is not None and hi is not None:
            lo, hi = -hi, -lo
    if jtype in ("REVOLUTE", "PRISMATIC", "CONTINUOUS") and lo is not None:
        return -lo
    return 0.0


def parse_urdf(
    urdf_path: Path, *, allow_duplicate_joint_names: bool = False,
    repair_reversed_limits: bool = False,
) -> URDFModel:
    """Parse a URDF file into our lightweight model.

    A handful of ArtVIP reference URDFs contain duplicate ``joint`` names.
    Joint names are identifiers in URDF, so predictions with duplicates are
    invalid and are rejected.  Reference files are parsed with
    ``allow_duplicate_joint_names=True`` and receive deterministic internal
    suffixes; topology and FK are then evaluated without silently dropping an
    edge from the dictionary.  One ArtVIP reference joint has its finite limit
    endpoints reversed.  ``repair_reversed_limits`` sorts reference endpoints
    before canonicalization; independently authored predictions never receive
    this dataset-repair exception and fail ``model_integrity`` instead.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    model = URDFModel(
        name=root.attrib.get("name", "unknown"),
        base_dir=urdf_path.parent,
    )

    for link_el in root.findall("link"):
        name = link_el.attrib["name"]
        info = LinkInfo(name=name)
        first_visual_origin: tuple | None = None
        for vis in link_el.findall("visual"):
            geom = vis.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None:
                    info.mesh_paths.append(mesh.attrib.get("filename", ""))
                    info.mesh_scales.append(_parse_scale(mesh.attrib.get("scale")))
                    if not info.has_visual:
                        # Capture origin of the first visual that has a mesh.
                        origin_el = vis.find("origin")
                        if origin_el is not None:
                            info.visual_xyz = _parse_xyz(origin_el.attrib.get("xyz"))
                            info.visual_rpy = _parse_xyz(origin_el.attrib.get("rpy"))
                    info.has_visual = True
        model.links[name] = info

    for joint_el in root.findall("joint"):
        source_name = joint_el.attrib["name"]
        name = source_name
        if name in model.joints:
            if not allow_duplicate_joint_names:
                raise ValueError(f"duplicate joint name: {source_name!r}")
            suffix = 2
            while f"{source_name}__duplicate_{suffix}" in model.joints:
                suffix += 1
            name = f"{source_name}__duplicate_{suffix}"
        jtype = joint_el.attrib.get("type", "fixed").upper()

        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        parent = parent_el.attrib["link"] if parent_el is not None else ""
        child = child_el.attrib["link"] if child_el is not None else ""

        origin_el = joint_el.find("origin")
        origin_xyz = _parse_xyz(origin_el.attrib.get("xyz")) if origin_el is not None else (0, 0, 0)
        origin_rpy = _parse_xyz(origin_el.attrib.get("rpy")) if origin_el is not None else (0, 0, 0)

        axis_el = joint_el.find("axis")
        axis = _parse_xyz(axis_el.attrib.get("xyz")) if axis_el is not None else None

        limit_el = joint_el.find("limit")
        limit_lower = float(limit_el.attrib.get("lower", 0)) if limit_el is not None else None
        limit_upper = float(limit_el.attrib.get("upper", 0)) if limit_el is not None else None
        if (
            repair_reversed_limits
            and limit_lower is not None
            and limit_upper is not None
            and limit_lower > limit_upper
        ):
            limit_lower, limit_upper = limit_upper, limit_lower
        reference_position = _canonical_reference_position(
            jtype, axis, limit_lower, limit_upper,
        )

        # Canonicalize the joint to a unique form within its physical
        # equivalence class — collapses gauge freedoms (axis sign,
        # CONTINUOUS↔REVOLUTE[0,2π], joint_origin shift along axis with
        # limits offset) so URDFs that describe the same articulation
        # compare as identical. See `_canonicalize_joint` for details.
        (jtype, axis, limit_lower, limit_upper,
         origin_xyz, origin_rpy) = _canonicalize_joint(
            jtype, axis, limit_lower, limit_upper, origin_xyz, origin_rpy,
        )

        model.joints[name] = JointInfo(
            name=name, type=jtype, parent=parent, child=child,
            origin_xyz=origin_xyz, origin_rpy=origin_rpy,
            axis=axis, limit_lower=limit_lower, limit_upper=limit_upper,
            reference_position=reference_position,
        )

    return model


def find_urdf(directory: Path) -> Path:
    """Find the URDF file in a directory."""
    candidates = list(directory.glob("*.urdf"))
    if not candidates:
        raise FileNotFoundError(f"No .urdf file found in {directory}")
    # Prefer scene.urdf (GT) or furniture.urdf (predicted)
    for name in ("scene.urdf", "furniture.urdf"):
        for c in candidates:
            if c.name == name:
                return c
    return candidates[0]


# ---------------------------------------------------------------------------
# Mesh signature for link matching
# ---------------------------------------------------------------------------

def count_obj_vertices(obj_path: Path) -> int:
    """Count vertices in a .obj file (lines starting with 'v ')."""
    count = 0
    try:
        with open(obj_path) as f:
            for line in f:
                if line.startswith("v "):
                    count += 1
    except (OSError, UnicodeDecodeError):
        return -1
    return count


def resolve_mesh_path(mesh_ref: str, base_dir: Path) -> Path:
    """Resolve a mesh filename from URDF to an absolute path."""
    # Strip package:// prefix if present
    if mesh_ref.startswith("package://"):
        mesh_ref = mesh_ref.split("//", 1)[1]
        # Remove package name segment
        parts = mesh_ref.split("/", 1)
        mesh_ref = parts[1] if len(parts) > 1 else parts[0]
    return base_dir / mesh_ref


@dataclass
class MeshSignature:
    vertex_count: int
    file_size: int  # fallback tiebreaker
    # Geometric features in mesh-local frame, used by the pose checks
    # (rest_pose_alignment, kinematic_equivalence) to detect 180° flips and
    # asymmetry shifts that bbox-only comparisons miss.
    bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    centroid: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Rigid-transform-invariant geometry fingerprint. ``radial_quantiles``
    # is retained as the serialized field name for compatibility, but stores
    # normalized radial moments (mean, stddev, max, cubic mean).  Moments are
    # linear-time and remain stable when Blender bakes a rigid transform into
    # exported OBJ vertices, unlike an axis-aligned bbox.
    rms_radius: float = 0.0
    radial_quantiles: tuple[float, ...] = ()
    # Deterministic sparse vertex cloud used for pose/FK comparison.  Nearest-
    # neighbour set distance makes this robust to rigid transforms baked into
    # a re-exported mesh, unlike pairing local AABB corners by index.
    sample_points: tuple[tuple[float, float, float], ...] = ()
    covariance: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
    )
    # Flattened symmetric third central-moment tensor, index i*9+j*3+k.
    # Unlike covariance, it changes under a 180-degree asymmetric flip.
    third_moment: tuple[float, ...] = (0.0,) * 27

    def distance(self, other: MeshSignature) -> float:
        """Rigid-transform-invariant geometry distance; lower is better.

        Vertex count alone is not a safe identity signal: repeated furniture
        parts frequently have equal counts, and unrelated parts can be close.
        Combining relative count, physical scale, and radial shape makes the
        cost useful for global assignment while keeping this oracle stdlib-only.
        """
        if self.vertex_count < 0 or other.vertex_count < 0:
            return float("inf")
        count_ref = max(self.vertex_count, other.vertex_count, 1)
        count_error = abs(self.vertex_count - other.vertex_count) / count_ref

        if self.radial_quantiles and other.radial_quantiles:
            n = min(len(self.radial_quantiles), len(other.radial_quantiles))
            shape_error = sum(
                abs(self.radial_quantiles[i] - other.radial_quantiles[i])
                for i in range(n)
            ) / n
        else:
            shape_error = 1.0

        # Anonymous task meshes intentionally erase per-part scale. Vertex
        # count and normalized radial shape remain comparable; physical scale
        # is evaluated only after the agent reconstructs the assembly and the
        # oracle applies one global similarity normalization.
        return 4.0 * count_error + shape_error


def _scan_obj_geometry(
    path: Path,
    scale: tuple[float, float, float] | None = None,
) -> tuple[
    int, tuple, tuple, tuple, float, tuple[float, ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...], tuple[float, ...],
]:
    """Single pass over an OBJ file: return
    (vertex_count, bbox_min, bbox_max, centroid, rms_radius,
    normalized_radial_moments, sample_points, covariance, third_moment).

    ``scale`` (a URDF ``<mesh scale>``) multiplies every vertex component-wise
    before any statistic is taken, exactly as a renderer would apply it."""
    n = 0
    sx = sy = sz = 0.0
    minx = miny = minz = float("inf")
    maxx = maxy = maxz = float("-inf")
    vertices: list[tuple[float, float, float]] = []
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                except ValueError:
                    continue
                if scale is not None:
                    x *= scale[0]; y *= scale[1]; z *= scale[2]
                n += 1
                vertices.append((x, y, z))
                sx += x; sy += y; sz += z
                if x < minx: minx = x
                if x > maxx: maxx = x
                if y < miny: miny = y
                if y > maxy: maxy = y
                if z < minz: minz = z
                if z > maxz: maxz = z
    except OSError:
        return (
            -1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), 0.0, (), (),
            ((0.0, 0.0, 0.0),) * 3, (0.0,) * 27,
        )
    if n == 0:
        return (
            0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), 0.0, (), (),
            ((0.0, 0.0, 0.0),) * 3, (0.0,) * 27,
        )
    centroid = (sx / n, sy / n, sz / n)
    radii = [math.sqrt(
        (x - centroid[0]) ** 2
        + (y - centroid[1]) ** 2
        + (z - centroid[2]) ** 2
    ) for x, y, z in vertices]
    rms_radius = math.sqrt(sum(r * r for r in radii) / n)
    if rms_radius > 1e-12:
        mean = sum(radii) / (n * rms_radius)
        stddev = math.sqrt(sum(
            (r / rms_radius - mean) ** 2 for r in radii
        ) / n)
        radial_quantiles = (
            mean,
            stddev,
            max(radii) / rms_radius,
            sum((r / rms_radius) ** 3 for r in radii) / n,
        )
    else:
        radial_quantiles = (0.0, 0.0, 0.0, 0.0)
    sample_count = min(n, 64)
    if sample_count == 1:
        sample_points = (vertices[0],)
    else:
        sample_points = tuple(
            vertices[round(i * (n - 1) / (sample_count - 1))]
            for i in range(sample_count)
        )

    cov = [[0.0] * 3 for _ in range(3)]
    third = [0.0] * 27
    for vertex in vertices:
        d = tuple(vertex[axis] - centroid[axis] for axis in range(3))
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]
                for k in range(3):
                    third[i * 9 + j * 3 + k] += d[i] * d[j] * d[k]
    covariance = tuple(tuple(value / n for value in row) for row in cov)
    third_moment = tuple(value / n for value in third)
    return (
        n,
        (minx, miny, minz),
        (maxx, maxy, maxz),
        centroid,
        rms_radius,
        radial_quantiles,
        sample_points,
        covariance,
        third_moment,
    )


_SIGNATURE_CACHE: dict[tuple[str, int, int], MeshSignature] = {}


def compute_signature(
    mesh_ref: str, base_dir: Path,
    scale: tuple[float, float, float] | None = None,
) -> MeshSignature:
    """Compute a geometry signature for a mesh file.

    ``scale`` is the URDF ``<mesh scale>`` of the visual that references the
    file; unit / ``None`` reads the OBJ as written. Prefer :func:`link_signature`,
    which takes the scale from the parsed link."""
    if scale is not None and all(abs(s - 1.0) < 1e-12 for s in scale):
        scale = None
    path = resolve_mesh_path(mesh_ref, base_dir)
    try:
        stat = path.stat()
        cache_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, scale)
    except OSError:
        cache_key = (str(path.resolve()), -1, -1, scale)
    cached = _SIGNATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    (vcount, bb_min, bb_max, centroid,
     rms_radius, radial_quantiles, sample_points,
     covariance, third_moment) = _scan_obj_geometry(path, scale)
    try:
        fsize = path.stat().st_size
    except OSError:
        fsize = 0
    signature = MeshSignature(
        vertex_count=vcount, file_size=fsize,
        bbox_min=bb_min, bbox_max=bb_max, centroid=centroid,
        rms_radius=rms_radius, radial_quantiles=radial_quantiles,
        sample_points=sample_points,
        covariance=covariance, third_moment=third_moment,
    )
    # Bound long-running training processes without sacrificing reuse inside
    # a round. Entries are tiny, but datasets may contain many thousands of
    # unique meshes across repeated evaluations.
    if len(_SIGNATURE_CACHE) >= 4096:
        _SIGNATURE_CACHE.clear()
    _SIGNATURE_CACHE[cache_key] = signature
    return signature


def link_signature(link: LinkInfo, base_dir: Path) -> MeshSignature:
    """Signature of a link's first visual mesh with its ``<mesh scale>`` applied.

    The single entry point every geometry check uses to read link geometry, so
    the scale attribute is honoured unconditionally (oracle 2.0)."""
    return compute_signature(link.mesh_paths[0], base_dir, link.mesh_scale)


# ---------------------------------------------------------------------------
# Link matching (global assignment, symmetry-aware pose tie-breaking)
# ---------------------------------------------------------------------------

@dataclass
class LinkMatchResult:
    matching: dict[str, str]
    geometry_costs: dict[str, float]
    rejected: list[tuple[str, str, float]]


def _minimum_cost_assignment(costs: list[list[float]]) -> list[int]:
    """Return the globally minimum-cost column for each row.

    This is the O(n^3) Hungarian algorithm with square padding, implemented
    locally to preserve the oracle's stdlib-only contract.  Returned columns
    may point into padding when the matrix is rectangular.
    """
    if not costs:
        return []
    rows = len(costs)
    cols = max((len(row) for row in costs), default=0)
    if cols == 0:
        return [-1] * rows
    size = max(rows, cols)
    huge = 1e12
    matrix = [
        [
            (costs[i][j] if i < rows and j < len(costs[i]) else huge)
            for j in range(size)
        ]
        for i in range(size)
    ]
    matrix = [
        [value if math.isfinite(value) else huge for value in row]
        for row in matrix
    ]

    # Potentials u/v and the augmenting-path bookkeeping use one-based
    # indexing, following the standard shortest augmenting path formulation.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * rows
    for j in range(1, size + 1):
        row = p[j] - 1
        if 0 <= row < rows:
            assignment[row] = j - 1
    return assignment


def _relative_rest_centroids(
    model: URDFModel,
) -> tuple[dict[str, tuple[float, float, float]], float]:
    """Mesh centroids relative to the assembly centroid at canonical rest.

    Relative positions remove global translation and provide a deterministic
    tie-breaker inside groups of repeated or symmetric meshes.  They never
    override geometry because matching weights geometry three orders of
    magnitude more strongly.
    """
    world = compute_world_poses(model, _joint_positions_at_reference(model))
    sig_cache: dict[str, MeshSignature] = {}
    centers: dict[str, tuple[float, float, float]] = {}
    for name, link in model.visual_links.items():
        if name not in world:
            continue
        descriptor = _link_world_descriptor(
            link, world[name], model.base_dir, sig_cache,
        )
        if descriptor:
            centers[name] = descriptor.centroid
    if not centers:
        return {}, 1.0
    n = len(centers)
    mean = tuple(sum(p[i] for p in centers.values()) / n for i in range(3))
    relative = {
        name: tuple(point[i] - mean[i] for i in range(3))
        for name, point in centers.items()
    }
    scale = max(
        (math.sqrt(sum(x * x for x in point)) for point in relative.values()),
        default=1.0,
    )
    return relative, max(scale, 1e-6)


def _match_links_detailed(
    gt: URDFModel,
    pred: URDFModel,
    *,
    max_geometry_cost: float,
    verbose: bool = False,
) -> LinkMatchResult:
    """Globally match visual links and reject geometrically implausible pairs."""
    gt_vis = gt.visual_links
    pred_vis = pred.visual_links

    # Compute signatures
    gt_sigs: dict[str, MeshSignature] = {}
    for name, link in gt_vis.items():
        if link.mesh_paths:
            gt_sigs[name] = link_signature(link, gt.base_dir)

    pred_sigs: dict[str, MeshSignature] = {}
    for name, link in pred_vis.items():
        if link.mesh_paths:
            pred_sigs[name] = link_signature(link, pred.base_dir)

    if verbose:
        print(f"  GT visual links: {len(gt_sigs)}, Pred visual links: {len(pred_sigs)}")

    gt_names = sorted(gt_sigs)
    pred_names = sorted(pred_sigs)
    if not gt_names or not pred_names:
        return LinkMatchResult({}, {}, [])

    gt_centers, gt_scale = _relative_rest_centroids(gt)
    pred_centers, pred_scale = _relative_rest_centroids(pred)

    geometry_matrix: list[list[float]] = []
    total_matrix: list[list[float]] = []
    for gi, gt_name in enumerate(gt_names):
        geometry_row: list[float] = []
        total_row: list[float] = []
        for pi, pred_name in enumerate(pred_names):
            geometry_cost = gt_sigs[gt_name].distance(pred_sigs[pred_name])
            geometry_row.append(geometry_cost)
            if gt_name in gt_centers and pred_name in pred_centers:
                pose_cost = math.sqrt(sum(
                    (
                        gt_centers[gt_name][axis] / gt_scale
                        - pred_centers[pred_name][axis] / pred_scale
                    ) ** 2
                    for axis in range(3)
                ))
            else:
                pose_cost = 0.0
            # Geometry is primary. Pose only resolves exact/near-exact
            # symmetric candidates; the tiny index term makes ties stable.
            total_row.append(1000.0 * geometry_cost + pose_cost + pi * 1e-9)
        geometry_matrix.append(geometry_row)
        total_matrix.append(total_row)

    assignment = _minimum_cost_assignment(total_matrix)
    matching: dict[str, str] = {}
    geometry_costs: dict[str, float] = {}
    rejected: list[tuple[str, str, float]] = []
    for gi, pi in enumerate(assignment):
        if pi < 0 or pi >= len(pred_names):
            continue
        gt_name = gt_names[gi]
        pred_name = pred_names[pi]
        geometry_cost = geometry_matrix[gi][pi]
        if math.isfinite(geometry_cost) and geometry_cost <= max_geometry_cost:
            matching[gt_name] = pred_name
            geometry_costs[gt_name] = geometry_cost
        else:
            rejected.append((gt_name, pred_name, geometry_cost))

    if verbose:
        for gt_name, pred_name in matching.items():
            cost = geometry_costs[gt_name]
            print(
                f"  Global match: GT '{gt_name}' -> Pred '{pred_name}' "
                f"(geometry_cost={cost:.6f})"
            )
        for gt_name, pred_name, cost in rejected:
            print(
                f"  Rejected match: GT '{gt_name}' -> Pred '{pred_name}' "
                f"(geometry_cost={cost:.6f}, max={max_geometry_cost})"
            )
    return LinkMatchResult(matching, geometry_costs, rejected)


def match_links(
    gt: URDFModel,
    pred: URDFModel,
    verbose: bool = False,
    max_geometry_cost: float = 0.08,
) -> dict[str, str] | None:
    """Compatibility wrapper returning ``gt_link -> pred_link`` mapping."""
    result = _match_links_detailed(
        gt, pred, max_geometry_cost=max_geometry_cost, verbose=verbose,
    )
    return result.matching or None


# ---------------------------------------------------------------------------
# Comparison checks
# ---------------------------------------------------------------------------

# Object-level checks in report order. A dataset variant may narrow the SCORED
# subset (see configure_scored_checks); the others still run and are reported
# as diagnostics (scored=False) but count for neither `score` nor `passed`.
ALL_CHECKS: tuple[str, ...] = (
    "model_integrity", "valid_tree", "link_count", "joint_type_distribution",
    "link_matching", "topology", "joint_types", "joint_axes", "joint_limits",
    "rest_pose_alignment", "joint_pivot_alignment", "kinematic_equivalence",
    "link_scale",
)
# ``link_scale`` (per-link size vs GT) is diagnostic unless the dataset lists it
# in ``scored_checks``: only views that hide part scale (unit-cube task parts)
# have anything to score there, and older views keep their 12-check score.
DEFAULT_SCORED_CHECKS: tuple[str, ...] = tuple(c for c in ALL_CHECKS if c != "link_scale")
SCORED_CHECKS: tuple[str, ...] = DEFAULT_SCORED_CHECKS


def configure_scored_checks(names) -> tuple[str, ...]:
    """Restrict the scored object-level checks to ``names`` (ALL_CHECKS order)."""
    global SCORED_CHECKS
    chosen = tuple(c for c in ALL_CHECKS if c in set(names))
    unknown = sorted(set(names) - set(ALL_CHECKS))
    if unknown or not chosen:
        raise ValueError(f"scored_checks must be a non-empty subset of {ALL_CHECKS}, got {list(names)}")
    SCORED_CHECKS = chosen
    return chosen


def _apply_scored_checks(checks: list["CheckResult"]) -> list["CheckResult"]:
    """Demote checks outside SCORED_CHECKS to diagnostics (and promote the ones
    inside). parse_* failures are not in ALL_CHECKS and stay scored: an
    unreadable URDF can never pass."""
    for c in checks:
        if c.name in ALL_CHECKS:
            c.scored = c.name in SCORED_CHECKS
    return checks


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    scored: bool = True

    def __str__(self):
        if self.scored:
            status = "PASS" if self.passed else "FAIL"
        else:
            status = (
                "DIAG-PASS" if self.passed
                else "BLOCKED" if self.detail.startswith("BLOCKED:")
                else "DIAG-FAIL"
            )
        return f"  [{status}] {self.name}: {self.detail}"


@dataclass
class JointReport:
    """Per-GT-joint outcome across the joint-level checks.

    The checks keep their one-line pass/fail contract; this record is the
    structured side: which GT joint, which pred joint it resolved to, and the
    error magnitude on every dimension. It is the pairing unit for the A/B
    gate (a 2.3-joint object gives ~2.3 samples instead of 1) and the feedback
    unit for the Generator. GT-side values are redacted before either sees it.
    """
    gt_joint: str
    gt_child: str
    gt_type: str
    matched: bool = False                 # GT child link matched a pred link
    pred_joint: str | None = None
    pred_child: str | None = None
    pred_type: str | None = None
    type_ok: bool | None = None
    axis_angle_deg: float | None = None
    axis_ok: bool | None = None
    pivot_distance_m: float | None = None
    pivot_ok: bool | None = None          # None: not applicable / not comparable
    limit_ratio: float | None = None      # predicted range / GT range
    limits_ok: bool | None = None
    trace_drift_m: float | None = None
    motion_ok: bool | None = None
    scale_ratio: float | None = None      # predicted child-link size / GT size (rms radius)
    scale_ok: bool | None = None
    # Oracle 3 diagnostics of the child link at rest (not scored dimensions):
    # surface distance to the GT link, the pose tolerance it is judged against,
    # and, when it exceeds the tolerance, which single correction of the link
    # (its size, its orientation about the pivot, or the pivot position) would
    # bring it within tolerance: "scale" / "orientation" / "position", a "+"
    # joined pair, "scale+orientation+position", or "unknown".
    rest_drift_m: float | None = None
    pose_tol_m: float | None = None
    pose_cause: str | None = None
    rot_err_deg: float | None = None      # rotation of the best rigid fit pred -> GT
    pos_err_m: float | None = None        # distance between the predicted and GT pivot
    # Set when K independent executor sessions were aggregated: the boolean
    # flags above are then majority votes, `score_mean` the mean of the K
    # per-session scores (what score() returns), `score_spread` max - min,
    # and `dim_pass_frac` the pass fraction per dimension.
    reps: int = 1
    score_mean: float | None = None
    score_spread: float | None = None
    dim_pass_frac: dict | None = None

    ALL_DIMS = ("type_ok", "axis_ok", "pivot_ok", "limits_ok", "motion_ok", "scale_ok")
    # Dimensions that count towards score() / the A/B gate. A dataset view
    # that hands the solver some of these (the joint-fill view gives type and
    # limits) narrows the list via configure_dims(); see dataset_variant.json
    # "scored_joint_dims". ``scale_ok`` is scored only by views that hide part
    # size. Process-wide by design: one dataset per run.
    DIMS = ("type_ok", "axis_ok", "pivot_ok", "limits_ok", "motion_ok")

    @classmethod
    def configure_dims(cls, dims) -> tuple[str, ...]:
        """Restrict the scored dimensions to ``dims`` (ALL_DIMS order); returns the new tuple."""
        chosen = tuple(d for d in cls.ALL_DIMS if d in set(dims))
        unknown = sorted(set(dims) - set(cls.ALL_DIMS))
        if unknown or not chosen:
            raise ValueError(f"scored_joint_dims must be a non-empty subset of {cls.ALL_DIMS}, got {list(dims)}")
        cls.DIMS = chosen
        return chosen

    def flags(self) -> dict[str, bool]:
        return {d: bool(getattr(self, d)) for d in self.DIMS if getattr(self, d) is not None}

    def score(self) -> float:
        """Fraction of comparable dimensions that passed; 0 when nothing resolved."""
        if self.score_mean is not None:
            return self.score_mean
        f = self.flags()
        if not self.matched or self.pred_joint is None or not f:
            return 0.0
        return sum(f.values()) / len(f)

    def to_dict(self, *, redact_gt: bool = False) -> dict:
        d = {
            "gt_joint": self.gt_joint, "matched": self.matched,
            "pred_joint": self.pred_joint, "pred_child": self.pred_child,
            "pred_type": self.pred_type, "type_ok": self.type_ok,
            "axis_angle_deg": None if self.axis_angle_deg is None else round(self.axis_angle_deg, 1),
            "axis_ok": self.axis_ok,
            "pivot_distance_m": None if self.pivot_distance_m is None else round(self.pivot_distance_m, 4),
            "pivot_ok": self.pivot_ok,
            "limit_ratio": None if self.limit_ratio is None else round(self.limit_ratio, 3),
            "limits_ok": self.limits_ok,
            "trace_drift_m": None if self.trace_drift_m is None else round(self.trace_drift_m, 4),
            "motion_ok": self.motion_ok,
            "scale_ratio": None if self.scale_ratio is None else round(self.scale_ratio, 4),
            "scale_ok": self.scale_ok,
            "rest_drift_m": None if self.rest_drift_m is None else round(self.rest_drift_m, 4),
            "pose_tol_m": None if self.pose_tol_m is None else round(self.pose_tol_m, 4),
            "pose_cause": self.pose_cause,
            "rot_err_deg": None if self.rot_err_deg is None else round(self.rot_err_deg, 1),
            "pos_err_m": None if self.pos_err_m is None else round(self.pos_err_m, 4),
            "score": round(self.score(), 3),
        }
        if self.reps > 1:
            d["reps"] = self.reps
            d["score_spread"] = None if self.score_spread is None else round(self.score_spread, 3)
            d["dim_pass_frac"] = self.dim_pass_frac
        if not redact_gt:
            d["gt_child"] = self.gt_child
            d["gt_type"] = self.gt_type
        return d



@dataclass
class Thresholds:
    # 10 degrees, matching the commonly used OPD articulation threshold.
    axis_cos_min: float = math.cos(math.radians(10.0))
    limit_rel_tol: float = 0.15
    limit_abs_tol: float = 0.05
    # Per-link size: |pred rms radius / GT rms radius - 1| (rotation- and
    # translation-invariant, so a mis-oriented part is judged on size alone).
    scale_rel_tol: float = 0.05
    link_count_tolerance: int = 0
    # Pose/FK tolerance is scale-normalized. ``kinematic_tol`` remains as an
    # explicit backwards-compatible override for callers that need one.
    kinematic_tol: float | None = None
    kinematic_abs_tol: float = 0.02
    kinematic_rel_tol: float = 0.04
    kinematic_samples: int = 5
    jtd_per_type_tol: int = 0
    link_matching_min_pct: float = 1.00
    link_match_max_geometry_cost: float = 0.08
    # Pivot-line position is scored in both task views. Callers may switch it
    # to diagnostic-only, except for a pre-placed dataset (the artvip_beginner
    # view: gt/ carries bake_transforms.json) where run_oracle forces it on —
    # with placement removed, the pivot line is the only place a wrong joint
    # origin can still show.
    score_pivot_alignment: bool = True


def check_link_count(gt: URDFModel, pred: URDFModel, th: Thresholds) -> CheckResult:
    gt_count = len(gt.visual_links)
    pred_count = len(pred.visual_links)
    diff = abs(gt_count - pred_count)
    passed = diff <= th.link_count_tolerance
    return CheckResult(
        "link_count",
        passed,
        f"GT={gt_count}, Pred={pred_count}, diff={diff} (tol={th.link_count_tolerance})",
    )


def check_joint_type_distribution(
    gt: URDFModel, pred: URDFModel, th: Thresholds | None = None,
) -> CheckResult:
    """Count each joint type in both URDFs; pass if every type's count
    differs by at most `th.jtd_per_type_tol`."""
    th = th or Thresholds()

    def dist(model: URDFModel) -> dict[str, int]:
        d: dict[str, int] = {}
        for j in model.movable_joints.values():
            d[j.type] = d.get(j.type, 0) + 1
        return d

    gt_dist = dist(gt)
    pred_dist = dist(pred)
    all_types = sorted(set(gt_dist) | set(pred_dist))
    per_type_diff = {t: abs(gt_dist.get(t, 0) - pred_dist.get(t, 0)) for t in all_types}
    max_diff = max(per_type_diff.values()) if per_type_diff else 0
    passed = max_diff <= th.jtd_per_type_tol
    return CheckResult(
        "joint_type_distribution",
        passed,
        f"GT={gt_dist}, Pred={pred_dist} (max_per_type_diff={max_diff}, tol={th.jtd_per_type_tol})",
    )


def check_topology(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str],
) -> CheckResult:
    """Compare the undirected edge set under the link matching.

    For each GT joint connecting two matched links, check that the
    predicted URDF has some joint on the corresponding matched pair
    (either direction counts).
    """
    errors = []

    for jname, joint in gt.joints.items():
        if joint.parent not in matching or joint.child not in matching:
            continue  # skip unmatched (abstract/world) links
        pred_parent = matching[joint.parent]
        pred_child = matching[joint.child]
        target_edge = frozenset({pred_parent, pred_child})

        found = False
        for pj in pred.joints.values():
            if frozenset({pj.parent, pj.child}) == target_edge:
                found = True
                break
        if not found:
            errors.append(
                f"{{{joint.parent}, {joint.child}}} "
                f"(expected undirected edge {{{pred_parent}, {pred_child}}} in pred)"
            )

    passed = len(errors) == 0
    detail = (
        f"{len(errors)} topology mismatches (undirected)"
        + (f": {errors[:3]}" if errors else "")
    )
    return CheckResult("topology", passed, detail)


def check_joint_types(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str],
    reports: dict[str, JointReport] | None = None,
) -> CheckResult:
    """For each matched link pair connected by a joint, check joint type matches."""
    errors = []
    checked = 0

    for jname, gj in gt.joints.items():
        if gj.child not in matching:
            continue
        pred_child = matching[gj.child]
        # Find the pred joint whose child is pred_child
        pj = pred.parent_of(pred_child)
        checked += 1
        rep = (reports or {}).get(jname)
        if pj is None:
            errors.append(f"{gj.child}: missing predicted parent joint")
            if rep is not None:
                rep.type_ok = False
            continue
        if rep is not None:
            rep.pred_joint, rep.pred_child, rep.pred_type = pj.name, pred_child, pj.type
            rotational_ = {"REVOLUTE", "CONTINUOUS"}
            rep.type_ok = gj.type == pj.type or {gj.type, pj.type} <= rotational_
        if gj.type != pj.type:
            # Allow CONTINUOUS ↔ REVOLUTE mismatch (common, both rotational)
            rotational = {"REVOLUTE", "CONTINUOUS"}
            if {gj.type, pj.type} <= rotational:
                continue
            errors.append(f"{gj.child}: GT={gj.type}, Pred={pj.type}")

    passed = len(errors) == 0
    detail = f"checked {checked} joints, {len(errors)} type mismatches"
    if errors:
        detail += f": {errors[:5]}"
    return CheckResult("joint_types", passed, detail)


def _cos_angle(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """|cos(angle)| between two vectors — i.e. treats +v and -v as equal."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    return abs(dot / (mag_a * mag_b))


def check_joint_axes(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str], th: Thresholds,
    reports: dict[str, JointReport] | None = None,
) -> CheckResult:
    """Compare physical joint-axis directions in world space at source q=0."""
    errors = []
    checked = 0
    gt_world = compute_world_poses(gt, _joint_positions_at_reference(gt))
    pred_world = compute_world_poses(pred, _joint_positions_at_reference(pred))

    for jname, gj in gt.movable_joints.items():
        if gj.child not in matching:
            continue
        checked += 1
        rep = (reports or {}).get(jname)
        if rep is not None:
            rep.axis_ok = False
        if gj.axis is None:
            errors.append(f"{gj.child}: reference movable joint has no axis")
            continue
        pred_child = matching[gj.child]
        pj = pred.parent_of(pred_child)
        if pj is None or pj.axis is None or pj.type == "FIXED":
            errors.append(f"{gj.child}: missing predicted movable axis")
            continue
        if gj.parent not in gt_world or pj.parent not in pred_world:
            errors.append(f"{gj.child}: parent world pose unavailable")
            continue
        gt_line = _joint_world_axis_line(gt_world[gj.parent], gj)
        pred_line = _joint_world_axis_line(pred_world[pj.parent], pj)
        if gt_line is None or pred_line is None:
            errors.append(f"{gj.child}: invalid physical axis line")
            continue
        gt_axis = gt_line[1]
        pred_axis = pred_line[1]
        cos = _cos_angle(gt_axis, pred_axis)
        if rep is not None:
            rep.axis_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            rep.axis_ok = cos >= th.axis_cos_min
        if cos < th.axis_cos_min:
            errors.append(
                f"{gj.child}: world-axis cos={cos:.3f} "
                f"(GT={gt_axis}, Pred={pred_axis})"
            )

    passed = len(errors) == 0
    detail = f"checked {checked} world axes, {len(errors)} misaligned"
    if errors:
        detail += f": {errors[:3]}"
    return CheckResult("joint_axes", passed, detail)


def _limits_close(
    gt_lo, gt_hi, pred_lo, pred_hi, th: Thresholds, *, pred_scale: float = 1.0,
) -> bool:
    """Check if joint limits are close enough."""
    for gv, pv in [(gt_lo, pred_lo), (gt_hi, pred_hi)]:
        if gv is None and pv is None:
            continue
        if gv is None or pv is None:
            return False
        pv = pv * pred_scale
        diff = abs(gv - pv)
        ref = max(abs(gv), abs(pv), 1e-6)
        if diff > th.limit_abs_tol and diff / ref > th.limit_rel_tol:
            return False
    return True


def check_joint_limits(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str], th: Thresholds,
    reports: dict[str, JointReport] | None = None,
) -> CheckResult:
    """Check joint limits for matched movable joints."""
    errors = []
    checked = 0
    similarity = _similarity_alignment(gt, pred, matching)

    for jname, gj in gt.movable_joints.items():
        if gj.child not in matching:
            continue
        checked += 1
        pred_child = matching[gj.child]
        pj = pred.parent_of(pred_child)
        rep = (reports or {}).get(jname)
        if rep is not None:
            rep.limits_ok = False
        if pj is None or pj.type in ("FIXED", "CONTINUOUS"):
            errors.append(f"{gj.child}: missing predicted movable limits")
            continue
        linear_scale = similarity.scale if gj.type == "PRISMATIC" else 1.0
        close = _limits_close(
            gj.limit_lower, gj.limit_upper,
            pj.limit_lower, pj.limit_upper, th,
            pred_scale=linear_scale,
        )
        if rep is not None:
            rep.limits_ok = close
            try:
                gt_range = abs(float(gj.limit_upper) - float(gj.limit_lower))
                pred_range = abs(float(pj.limit_upper) - float(pj.limit_lower)) * linear_scale
                rep.limit_ratio = pred_range / gt_range if gt_range > 1e-9 else None
            except (TypeError, ValueError):
                rep.limit_ratio = None
        if not close:
            errors.append(
                f"{gj.child}: GT=[{gj.limit_lower}, {gj.limit_upper}], "
                f"Pred=[{pj.limit_lower}, {pj.limit_upper}], "
                f"linear_scale={linear_scale:.6g}"
            )

    passed = len(errors) == 0
    detail = f"checked {checked} limit pairs, {len(errors)} out of tolerance"
    if errors:
        detail += f": {errors[:3]}"
    return CheckResult("joint_limits", passed, detail)


def _gt_link_descriptor(
    gt: URDFModel, gt_link: str, link_world_pose: "Mat4", sig_cache: dict[str, MeshSignature],
) -> "PoseDescriptor | None":
    """GT-side descriptor honouring ``gt.link_substitutes`` (interchangeable parts)."""
    sub = gt.link_substitutes.get(gt_link)
    if sub is None:
        return _link_world_descriptor(gt.links[gt_link], link_world_pose, gt.base_dir, sig_cache)
    link_x, local = sub
    return _link_world_descriptor(link_x, _mat4_mul(link_world_pose, local), gt.base_dir, sig_cache)


def _substitute_variants(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str], classes: list[list[str]],
    transforms: dict[tuple[str, str], list[list[list[float]]]],
    gt_sigs: dict[str, MeshSignature], pred_sigs: dict[str, MeshSignature],
    *, limit: int = 64,
) -> list[dict]:
    """The GT-side substitute maps worth scoring for one candidate assignment.

    For every class member Y matched to prediction link P, the members whose
    vertex cloud P may hold are those with P's vertex count and a
    rigid-invariant signature distance ~0. The mesh file name pins the choice
    when it names exactly one member (template-fill outputs keep
    ``part_<i>.obj``); otherwise every plausible identity is tried, because
    mirror-image parts share the invariant signature and only the full pose
    comparison can tell them apart. Returns at least ``[{}]``.
    """
    per_link: list[tuple[str, list]] = []
    if transforms:
        for cls in classes:
            for y in cls:
                p = matching.get(y)
                if p is None or p not in pred_sigs or y not in gt_sigs:
                    continue
                ps = pred_sigs[p]
                plausible = [x for x in cls if x in gt_sigs and gt_sigs[x].vertex_count == ps.vertex_count
                             and gt_sigs[x].distance(ps) <= 1e-4]
                pred_name = Path(pred.links[p].mesh_paths[0]).name if pred.links[p].mesh_paths else ""
                named = [x for x in plausible if gt.links[x].mesh_paths and Path(gt.links[x].mesh_paths[0]).name == pred_name]
                if len(named) == 1:
                    plausible = named
                options: list = []
                for x in plausible:
                    if x == y:
                        options.append(None)
                    elif (x, y) in transforms:
                        for L in transforms[(x, y)]:       # best registration first, then alternatives
                            options.append((gt.links[x], tuple(tuple(float(v) for v in row) for row in L)))
                if options and options != [None]:
                    per_link.append((y, options))
    variants: list[dict] = [{}]
    for y, options in per_link:
        grown: list[dict] = []
        for base in variants:
            for opt in options:
                d = dict(base)
                if opt is not None:
                    d[y] = opt
                grown.append(d)
                if len(grown) >= limit:
                    break
            if len(grown) >= limit:
                break
        variants = grown
    return variants

def _reference_pose_descriptors(
    gt: URDFModel,
    pred: URDFModel,
    matching: dict[str, str],
) -> tuple[
    dict[str, "Mat4"],
    dict[str, "Mat4"],
    dict[str, "PoseDescriptor"],
    dict[str, "PoseDescriptor"],
    "SimilarityAlignment",
]:
    """Resolve q=0 link descriptors plus the shared global translation."""
    gt_world = compute_world_poses(gt, _joint_positions_at_reference(gt))
    pred_world = compute_world_poses(pred, _joint_positions_at_reference(pred))
    gt_sig_cache: dict[str, MeshSignature] = {}
    pred_sig_cache: dict[str, MeshSignature] = {}
    gt_desc: dict[str, PoseDescriptor] = {}
    pred_desc: dict[str, PoseDescriptor] = {}
    for gt_link, pred_link in matching.items():
        if gt_link not in gt_world or pred_link not in pred_world:
            continue
        gli = gt.links.get(gt_link)
        pli = pred.links.get(pred_link)
        if gli is None or pli is None:
            continue
        gd = _gt_link_descriptor(gt, gt_link, gt_world[gt_link], gt_sig_cache)
        pd = _link_world_descriptor(
            pli, pred_world[pred_link], pred.base_dir, pred_sig_cache,
        )
        if gd is None or pd is None:
            continue
        gt_desc[gt_link] = gd
        pred_desc[gt_link] = pd
    align = _descriptor_similarity_alignment(
        gt, pred, matching, gt_desc, pred_desc,
    ) if gt_desc else SimilarityAlignment()
    return gt_world, pred_world, gt_desc, pred_desc, align


def _diagnose_pose_cause(
    gd: "PoseDescriptor", pd: "PoseDescriptor", align: "SimilarityAlignment", tol: float,
) -> tuple[str, float | None, float | None]:
    """(cause, rotation error in degrees, pivot distance in metres) for one link.

    Tries the three single corrections of the predicted link, each about its
    own pivot (resize it, re-orient it, move the pivot), then the pairs, and
    names the smallest set that brings the surface within ``tol`` of the GT
    link. Needs index-paired vertices (the same source mesh on both sides) for
    the similarity fit; "unknown" otherwise.
    """
    np = _sd.np
    if gd.field is None or pd.field is None or gd.link_origin is None or pd.link_origin is None:
        return "unknown", None, None
    Tg = np.asarray(gd.transform, dtype=float)
    Tp = np.asarray(pd.transform, dtype=float)
    s_al = align.scale
    t_al = np.asarray(align.translation, dtype=float)
    Mp = np.eye(4)                       # pred mesh frame -> aligned world
    Mp[:3, :3] = s_al * Tp[:3, :3]
    Mp[:3, 3] = s_al * Tp[:3, 3] + t_al
    p0 = s_al * np.asarray(pd.link_origin, dtype=float) + t_al
    g0 = np.asarray(gd.link_origin, dtype=float)
    pos_err = float(np.linalg.norm(g0 - p0))
    vg, vp = gd.field.vertices, pd.field.vertices
    if len(vg) != len(vp) or len(vg) < 3:
        return "unknown", None, pos_err
    wg = vg @ Tg[:3, :3].T + Tg[:3, 3]
    wp = vp @ Mp[:3, :3].T + Mp[:3, 3]
    s_fit, R_fit, _t = _sd.similarity_fit(wp, wg)
    if not (math.isfinite(s_fit) and s_fit > 1e-9):
        return "unknown", None, pos_err
    rot_err = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R_fit)) - 1.0) / 2.0))))

    def fixed(scale: bool, orient: bool, position: bool) -> float:
        L = np.eye(3)
        if scale:
            L = s_fit * L
        if orient:
            L = R_fit @ L
        M = np.eye(4)
        M[:3, :3] = L @ Mp[:3, :3]
        M[:3, 3] = p0 + L @ (Mp[:3, 3] - p0) + ((g0 - p0) if position else 0.0)
        return _sd.posed_surface_distance(gd.field, Tg, pd.field, M)[0]

    singles = {"scale": (True, False, False), "orientation": (False, True, False),
               "position": (False, False, True)}
    hits = {name: fixed(*flags) for name, flags in singles.items()}
    ok = [name for name, d in hits.items() if d <= tol]
    if ok:
        return min(ok, key=lambda n: hits[n]), rot_err, pos_err
    pairs = {"scale+orientation": (True, True, False), "scale+position": (True, False, True),
             "orientation+position": (False, True, True)}
    hits2 = {name: fixed(*flags) for name, flags in pairs.items()}
    ok2 = [name for name, d in hits2.items() if d <= tol]
    if ok2:
        return min(ok2, key=lambda n: hits2[n]), rot_err, pos_err
    return "scale+orientation+position", rot_err, pos_err


def check_rest_pose_alignment(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str], th: Thresholds,
    reports: dict[str, JointReport] | None = None,
) -> CheckResult:
    """Rest-pose mesh feature-point check.

    For each matched link pair, transform a sparse mesh point cloud through
    `link_world × visual_origin` at the source URDF reference configuration
    (q=0 before canonicalization), and compare GT vs pred world coordinates
    using symmetric nearest-neighbour set distance.

    Catches structural problems that pure URDF-text checks miss:
      - a wrong joint/visual transform product at q=0
      - mesh internally flipped 180° (vertex centroid moves to the
        opposite side of the bbox; bbox corners alone wouldn't catch
        a symmetric flip)

    This intentionally does *not* claim to locate a hinge pivot: a wrong
    joint origin can be cancelled by an opposite visual offset at q=0. The
    separate joint_pivot_alignment check compares the physical axis line.
    """
    if not matching:
        return CheckResult(
            "rest_pose_alignment", False,
            "no matched link pairs — upstream link_matching must pass first",
        )

    _, _, gt_desc, pred_desc, align = _reference_pose_descriptors(
        gt, pred, matching,
    )
    if not gt_desc:
        return CheckResult(
            "rest_pose_alignment", False,
            "no matched links produced pose descriptors",
        )

    drifts = {
        gt_link: _descriptor_distance(descriptor, pred_desc[gt_link], align)
        for gt_link, descriptor in gt_desc.items()
    }

    tol = _kinematic_tolerance(gt, th)
    over = [(k, v) for k, v in drifts.items() if v > tol]
    max_drift = max(drifts.values())
    passed = len(over) == 0
    surface = _surface_metric_enabled()
    for rep in (reports or {}).values():
        if rep.gt_child not in drifts:
            continue
        rep.rest_drift_m = drifts[rep.gt_child]
        rep.pose_tol_m = tol
        if surface and drifts[rep.gt_child] > tol:
            try:
                rep.pose_cause, rep.rot_err_deg, rep.pos_err_m = _diagnose_pose_cause(
                    gt_desc[rep.gt_child], pred_desc[rep.gt_child], align, tol,
                )
            except Exception:  # a diagnosis must never change a verdict
                rep.pose_cause = "unknown"
    if passed:
        detail = (
            f"max feature drift {max_drift:.3f}m at rest over "
            f"{len(drifts)} matched link(s) (tol={tol}m, "
            + ("two-sided surface distance)" if surface
               else "centroid + second/third geometry moments)")
        )
    else:
        worst = sorted(over, key=lambda x: -x[1])[:3]
        worst_str = ", ".join(f"{k}={d:.3f}m" for k, d in worst)
        detail = (
            f"{len(over)} matched link(s) mis-posed at rest by > {tol}m "
            f"(max={max_drift:.3f}m): {worst_str}"
        )
    return CheckResult("rest_pose_alignment", passed, detail)


def _joint_world_axis_line(
    parent_world: "Mat4", joint: JointInfo,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return the physical revolute axis line as (point, unit direction)."""
    if joint.axis is None:
        return None
    origin = _mat4_mul(
        parent_world,
        _mat4_mul(
            _mat4_translate(joint.origin_xyz),
            _mat4_rotate_rpy(joint.origin_rpy),
        ),
    )
    point = (origin[0][3], origin[1][3], origin[2][3])
    direction = tuple(
        sum(origin[row][column] * joint.axis[column] for column in range(3))
        for row in range(3)
    )
    magnitude = math.sqrt(sum(value * value for value in direction))
    if magnitude < 1e-9:
        return None
    return point, tuple(value / magnitude for value in direction)


def _axis_line_distance(
    first_point: tuple[float, float, float],
    first_axis: tuple[float, float, float],
    second_point: tuple[float, float, float],
    second_axis: tuple[float, float, float],
) -> float:
    """Shortest distance between two infinite 3-D lines."""
    delta = tuple(second_point[i] - first_point[i] for i in range(3))
    cross_axes = (
        first_axis[1] * second_axis[2] - first_axis[2] * second_axis[1],
        first_axis[2] * second_axis[0] - first_axis[0] * second_axis[2],
        first_axis[0] * second_axis[1] - first_axis[1] * second_axis[0],
    )
    cross_magnitude = math.sqrt(sum(value * value for value in cross_axes))
    if cross_magnitude < 1e-8:
        cross_delta = (
            delta[1] * first_axis[2] - delta[2] * first_axis[1],
            delta[2] * first_axis[0] - delta[0] * first_axis[2],
            delta[0] * first_axis[1] - delta[1] * first_axis[0],
        )
        return math.sqrt(sum(value * value for value in cross_delta))
    return abs(sum(delta[i] * cross_axes[i] for i in range(3))) / cross_magnitude


def check_joint_pivot_alignment(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str], th: Thresholds,
    reports: dict[str, JointReport] | None = None,
) -> CheckResult:
    """Compare physical revolute/continuous axis lines at source q=0.

    Translation along an axis is ignored because it describes the same
    infinite hinge line. Prismatic joints have no unique pivot and are not
    applicable; their direction remains covered by joint_axes and their q=0
    placement by rest_pose_alignment.
    """
    applicable = [
        joint for joint in gt.movable_joints.values()
        if joint.type == "REVOLUTE"
    ]
    if not applicable:
        # No hinge to compare: a diagnostic pass, never a free scored point
        # (prismatic-only objects are scored on the remaining checks).
        return CheckResult(
            "joint_pivot_alignment", True,
            "not applicable: no revolute/continuous reference joints",
            scored=False,
        )
    if not matching:
        return CheckResult(
            "joint_pivot_alignment", False,
            "no matched link pairs — upstream link_matching must pass first",
        )

    gt_world, pred_world, gt_desc, _, align = _reference_pose_descriptors(
        gt, pred, matching,
    )
    if not gt_desc:
        return CheckResult(
            "joint_pivot_alignment", False,
            "no matched links available for global alignment",
        )

    tol = _kinematic_tolerance(gt, th)
    checked = 0
    max_distance = 0.0
    errors: list[str] = []
    blocked: list[str] = []
    for gj in applicable:
        rep = (reports or {}).get(gj.name)
        if gj.child not in matching:
            errors.append(f"{gj.child}: unmatched reference child")
            if rep is not None:
                rep.pivot_ok = False
            continue
        pj = pred.parent_of(matching[gj.child])
        if pj is None or pj.type != "REVOLUTE":
            errors.append(f"{gj.child}: missing comparable revolute pivot")
            if rep is not None:
                rep.pivot_ok = False
            continue
        if gj.parent not in gt_world or pj.parent not in pred_world:
            errors.append(f"{gj.child}: parent world pose unavailable")
            continue
        gt_line = _joint_world_axis_line(gt_world[gj.parent], gj)
        pred_line = _joint_world_axis_line(pred_world[pj.parent], pj)
        if gt_line is None or pred_line is None:
            errors.append(f"{gj.child}: invalid physical axis line")
            continue
        gt_point, gt_axis = gt_line
        pred_point, pred_axis = pred_line
        pred_point = tuple(
            align.scale * pred_point[i] + align.translation[i]
            for i in range(3)
        )
        cosine = _cos_angle(gt_axis, pred_axis)
        if cosine < th.axis_cos_min:
            blocked.append(
                f"{gj.child}: world axis mismatch (cos={cosine:.3f})"
            )
            continue
        checked += 1
        distance = _axis_line_distance(
            gt_point, gt_axis, pred_point, pred_axis,
        )
        max_distance = max(max_distance, distance)
        if rep is not None:
            rep.pivot_distance_m = distance
            rep.pivot_ok = distance <= tol
        if distance > tol:
            errors.append(f"{gj.child}: line_distance={distance:.3f}m")

    if blocked and not errors:
        detail = (
            "BLOCKED: pivot position is not comparable until joint_axes "
            f"passes; {len(blocked)} joint(s): {blocked[:3]}"
        )
        return CheckResult(
            "joint_pivot_alignment", False, detail, scored=False,
        )

    passed = not errors and not blocked and checked == len(applicable)
    detail = (
        f"checked {checked}/{len(applicable)} revolute pivot line(s); "
        f"max line distance={max_distance:.3f}m (tol={tol:.3f}m)"
    )
    if errors:
        detail += f": {errors[:3]}"
    if blocked:
        detail += f"; BLOCKED by world-axis mismatch: {blocked[:3]}"
    return CheckResult("joint_pivot_alignment", passed, detail, scored=th.score_pivot_alignment)


# ---------------------------------------------------------------------------
# Forward kinematics — pure-stdlib 4×4 transform math.
#
# Why: an audit showed all 3 oracle false-positive
# samples passed every structural check (link_count, link_matching,
# topology, joint_types/axes/limits) yet the assembly was clearly wrong in
# absolute world position (drawers shifted down, buttons misplaced, plates
# in wrong spot). No pairwise-matching check catches "right parts in wrong
# place". `check_kinematic_equivalence` below computes world poses under
# matched joint sweeps and fails if any matched link drifts > kinematic_tol.
#
# Stdlib-only to preserve the oracle's "no external deps" contract.
# ---------------------------------------------------------------------------

Mat4 = tuple[tuple[float, ...], ...]


def _mat4_identity() -> Mat4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mat4_translate(xyz: tuple[float, float, float]) -> Mat4:
    x, y, z = xyz
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mat4_rotate_rpy(rpy: tuple[float, float, float]) -> Mat4:
    """Extrinsic XYZ (URDF standard). R = Rz * Ry * Rx."""
    r, p, y = rpy
    cx, sx = math.cos(r), math.sin(r)
    cy, sy = math.cos(p), math.sin(p)
    cz, sz = math.cos(y), math.sin(y)
    # Combined: Rz(y) @ Ry(p) @ Rx(r)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx, 0.0),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx, 0.0),
        (-sy, cy * sx, cy * cx, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mat4_rotate_axis_angle(axis: tuple[float, float, float], angle: float) -> Mat4:
    """Rodrigues' rotation around unit `axis` by `angle` radians."""
    mag = math.sqrt(sum(a * a for a in axis))
    if mag < 1e-9:
        return _mat4_identity()
    ux, uy, uz = axis[0] / mag, axis[1] / mag, axis[2] / mag
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return (
        (c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s, 0.0),
        (uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s, 0.0),
        (uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def _mat4_translation(m: Mat4) -> tuple[float, float, float]:
    return (m[0][3], m[1][3], m[2][3])


def _mat4_apply(m: Mat4, p: tuple[float, float, float]) -> tuple[float, float, float]:
    """Apply a 4×4 to a 3D point (treats p as homogeneous w=1)."""
    return (
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
    )


def _bbox_corners(
    bbmin: tuple[float, float, float], bbmax: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """8 corners of an axis-aligned bbox, in canonical sign-extremum order."""
    return [
        (bbmin[0], bbmin[1], bbmin[2]),
        (bbmax[0], bbmin[1], bbmin[2]),
        (bbmin[0], bbmax[1], bbmin[2]),
        (bbmin[0], bbmin[1], bbmax[2]),
        (bbmax[0], bbmax[1], bbmin[2]),
        (bbmax[0], bbmin[1], bbmax[2]),
        (bbmin[0], bbmax[1], bbmax[2]),
        (bbmax[0], bbmax[1], bbmax[2]),
    ]


def _link_world_features(
    link: LinkInfo,
    link_world_pose: Mat4,
    base_dir: Path,
    sig_cache: dict[tuple, MeshSignature],
) -> list[tuple[float, float, float]] | None:
    """Return a sparse mesh point cloud in world frame: vertex centroid
    plus up to 64 deterministic mesh vertices. Each is sampled from the first
    visual mesh (in mesh-local frame), then transformed through
    link_world × visual_origin.

    Callers compare these as unordered sets. This is robust when an executor
    exports the same source mesh with a rigid transform baked into its local
    vertices, a representation change that invalidates indexed bbox corners.
    """
    if not link.has_visual or not link.mesh_paths:
        return None
    cache_key = (link.mesh_paths[0], link.mesh_scale)
    sig = sig_cache.get(cache_key)
    if sig is None:
        sig = link_signature(link, base_dir)
        sig_cache[cache_key] = sig
    if sig.vertex_count <= 0:
        return None

    # Mesh-local frame → link frame (visual_origin) → world frame
    T_visual = _mat4_mul(
        _mat4_translate(link.visual_xyz),
        _mat4_rotate_rpy(link.visual_rpy),
    )
    T_link_to_world = _mat4_mul(link_world_pose, T_visual)

    local_pts = [sig.centroid] + list(sig.sample_points)
    return [_mat4_apply(T_link_to_world, p) for p in local_pts]


@dataclass(frozen=True)
class PoseDescriptor:
    centroid: tuple[float, float, float]
    covariance: tuple[tuple[float, float, float], ...]
    third_moment: tuple[float, ...]
    scale: float
    # Oracle 3: the mesh itself (surface_distance.MeshField), its mesh-frame ->
    # world transform, and the world position of the link frame origin (the
    # joint pivot in the fill views). None on the legacy moment path.
    field: object = dataclasses.field(default=None, compare=False, repr=False)
    transform: tuple | None = dataclasses.field(default=None, compare=False, repr=False)
    link_origin: tuple[float, float, float] | None = dataclasses.field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class SimilarityAlignment:
    """Map prediction coordinates into GT coordinates: p_gt=s*p_pred+t."""
    scale: float = 1.0
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _link_world_descriptor(
    link: LinkInfo,
    link_world_pose: Mat4,
    base_dir: Path,
    sig_cache: dict[tuple, MeshSignature],
) -> PoseDescriptor | None:
    """Transform mesh central moments into world coordinates."""
    if not link.has_visual or not link.mesh_paths:
        return None
    cache_key = (link.mesh_paths[0], link.mesh_scale)
    sig = sig_cache.get(cache_key)
    if sig is None:
        sig = link_signature(link, base_dir)
        sig_cache[cache_key] = sig
    if sig.vertex_count <= 0:
        return None
    T_visual = _mat4_mul(
        _mat4_translate(link.visual_xyz),
        _mat4_rotate_rpy(link.visual_rpy),
    )
    transform = _mat4_mul(link_world_pose, T_visual)
    rotation = tuple(tuple(transform[i][j] for j in range(3)) for i in range(3))
    centroid = _mat4_apply(transform, sig.centroid)
    if _surface_metric_enabled():
        field_key = ("field", link.mesh_paths[0], link.mesh_scale)
        if field_key not in sig_cache:
            scale = link.mesh_scale
            if all(abs(v - 1.0) < 1e-12 for v in scale):
                scale = None
            sig_cache[field_key] = _sd.mesh_field(resolve_mesh_path(link.mesh_paths[0], base_dir), scale)
        mesh_field = sig_cache[field_key]
        if mesh_field is not None:
            # The moments are not used by the surface metric; only the centroid
            # (anchor translation) and the size (anchor scale) are.
            return PoseDescriptor(
                centroid, sig.covariance, sig.third_moment, sig.rms_radius,
                field=mesh_field, transform=transform,
                link_origin=_mat4_translation(link_world_pose),
            )

    covariance = tuple(tuple(
        sum(
            rotation[i][a] * sig.covariance[a][b] * rotation[j][b]
            for a in range(3) for b in range(3)
        )
        for j in range(3)
    ) for i in range(3))

    third = [0.0] * 27
    for i in range(3):
        for j in range(3):
            for k in range(3):
                third[i * 9 + j * 3 + k] = sum(
                    rotation[i][a] * rotation[j][b] * rotation[k][c]
                    * sig.third_moment[a * 9 + b * 3 + c]
                    for a in range(3) for b in range(3) for c in range(3)
                )
    return PoseDescriptor(centroid, covariance, tuple(third), sig.rms_radius)


def _descriptor_distance(
    gt: PoseDescriptor,
    pred: PoseDescriptor,
    alignment: SimilarityAlignment | tuple[float, float, float],
) -> float:
    """Pose discrepancy expressed in metres from central moments."""
    if isinstance(alignment, tuple):
        alignment = SimilarityAlignment(1.0, alignment)
    scale = alignment.scale
    translation = alignment.translation
    if gt.field is not None and pred.field is not None:
        # Oracle 3: worst two-sided point-to-surface distance, in GT metres.
        return _sd.posed_surface_distance(
            gt.field, gt.transform, pred.field, pred.transform, scale, translation,
        )[0]
    centroid_error = math.sqrt(sum(
        (
            gt.centroid[axis]
            - (scale * pred.centroid[axis] + translation[axis])
        ) ** 2
        for axis in range(3)
    ))
    # Covariance has units m² and third moment m³. Convert their Frobenius
    # discrepancies back to a length scale before taking the worst signal.
    covariance_diff = math.sqrt(sum(
        (gt.covariance[i][j] - scale ** 2 * pred.covariance[i][j]) ** 2
        for i in range(3) for j in range(3)
    ))
    covariance_ref = max(
        math.sqrt(sum(value * value for row in gt.covariance for value in row)),
        scale ** 2 * math.sqrt(sum(
            value * value for row in pred.covariance for value in row
        )),
        gt.scale ** 2, (scale * pred.scale) ** 2, 1e-12,
    )
    covariance_error = (
        0.0 if covariance_diff <= covariance_ref * 1e-10
        else math.sqrt(covariance_diff)
    )
    third_diff = math.sqrt(sum(
        (a - scale ** 3 * b) ** 2
        for a, b in zip(gt.third_moment, pred.third_moment)
    ))
    third_ref = max(
        math.sqrt(sum(value * value for value in gt.third_moment)),
        scale ** 3 * math.sqrt(sum(value * value for value in pred.third_moment)),
        gt.scale ** 3, (scale * pred.scale) ** 3, 1e-12,
    )
    third_error = (
        0.0 if third_diff <= third_ref * 1e-10
        else third_diff ** (1.0 / 3.0)
    )
    return max(centroid_error, covariance_error, third_error)


def _descriptor_similarity_alignment(
    gt: URDFModel,
    pred: URDFModel,
    matching: dict[str, str],
    gt_desc: dict[str, PoseDescriptor],
    pred_desc: dict[str, PoseDescriptor],
) -> SimilarityAlignment:
    candidates = [name for name in gt_desc if name in pred_desc]
    if not candidates:
        return SimilarityAlignment()
    gt_root = gt.root_link()
    pred_root = pred.root_link()
    if (
        gt_root in candidates
        and gt_root in matching
        and matching[gt_root] == pred_root
    ):
        anchor = gt_root
    else:
        anchor = max(candidates, key=lambda name: gt_desc[name].scale)
    if _surface_metric_enabled():
        # Oracle 3: one global scale, read from the anchor alone. The median
        # over all links (2.0) let a single mis-sized child rescale the whole
        # prediction, which charged its error to correctly placed links too
        # (with two links the median is the mean of the two ratios).
        scale_candidates = (
            [gt_desc[anchor].scale / pred_desc[anchor].scale]
            if gt_desc[anchor].scale > 1e-12 and pred_desc[anchor].scale > 1e-12 else []
        )
    else:
        scale_candidates = [
            gt_desc[name].scale / pred_desc[name].scale
            for name in candidates
            if gt_desc[name].scale > 1e-12 and pred_desc[name].scale > 1e-12
        ]
    if scale_candidates:
        ordered_scales = sorted(scale_candidates)
        middle = len(ordered_scales) // 2
        scale = (
            ordered_scales[middle]
            if len(ordered_scales) % 2
            else 0.5 * (ordered_scales[middle - 1] + ordered_scales[middle])
        )
    else:
        gt_diagonal = _model_feature_diagonal(gt)
        pred_diagonal = _model_feature_diagonal(pred)
        scale = (
            gt_diagonal / pred_diagonal
            if gt_diagonal > 1e-12 and pred_diagonal > 1e-12
            else 1.0
        )
    translation = tuple(
        gt_desc[anchor].centroid[axis]
        - scale * pred_desc[anchor].centroid[axis]
        for axis in range(3)
    )
    return SimilarityAlignment(scale, translation)


def _descriptor_alignment_translation(
    gt: URDFModel,
    pred: URDFModel,
    matching: dict[str, str],
    gt_desc: dict[str, PoseDescriptor],
    pred_desc: dict[str, PoseDescriptor],
) -> tuple[float, float, float]:
    """Legacy translation-only helper retained for external callers."""
    return _descriptor_similarity_alignment(
        gt, pred, matching, gt_desc, pred_desc,
    ).translation


def _similarity_alignment(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str],
) -> SimilarityAlignment:
    """Resolve the q=0 global scale+translation without using GT metadata."""
    _, _, gt_desc, pred_desc, alignment = _reference_pose_descriptors(
        gt, pred, matching,
    )
    return alignment if gt_desc and pred_desc else SimilarityAlignment()


def _robust_point_set_distance(
    first: list[tuple[float, float, float]],
    second: list[tuple[float, float, float]],
) -> float:
    """Symmetric 95th-percentile nearest-neighbour distance.

    A percentile rather than a raw maximum prevents one differently sampled
    vertex from dominating, while still requiring nearly the whole sparse
    mesh to agree.  Point counts are capped at 65, so a stdlib O(n²) search is
    both deterministic and cheap.
    """
    if not first or not second:
        return float("inf")

    def directed(source, target) -> float:
        distances = sorted(
            min(
                math.sqrt(sum((a[axis] - b[axis]) ** 2 for axis in range(3)))
                for b in target
            )
            for a in source
        )
        index = min(len(distances) - 1, math.ceil(0.95 * len(distances)) - 1)
        return distances[index]

    return max(directed(first, second), directed(second, first))


def _feature_alignment_translation(
    gt: URDFModel,
    pred: URDFModel,
    matching: dict[str, str],
    gt_feats: dict[str, list[tuple[float, float, float]]],
    pred_feats: dict[str, list[tuple[float, float, float]]],
) -> tuple[float, float, float]:
    """Align one stable anchor link, avoiding movable-part error leakage.

    Aligning the centroid of the entire assembly lets a badly placed door
    shift the cabinet body in the opposite direction, falsely charging both
    links. Prefer a matched visual root; otherwise use the spatially largest
    matched link as the stable anchor.
    """
    candidates = [name for name in gt_feats if name in pred_feats]
    if not candidates:
        return (0.0, 0.0, 0.0)
    gt_root = gt.root_link()
    pred_root = pred.root_link()
    if (
        gt_root in candidates
        and gt_root in matching
        and matching[gt_root] == pred_root
    ):
        anchor = gt_root
    else:
        def spread(name: str) -> float:
            points = gt_feats[name]
            center = tuple(
                sum(p[axis] for p in points) / len(points) for axis in range(3)
            )
            return max(
                math.sqrt(sum((p[axis] - center[axis]) ** 2 for axis in range(3)))
                for p in points
            )
        anchor = max(candidates, key=spread)

    gt_points = gt_feats[anchor]
    pred_points = pred_feats[anchor]
    gt_mean = tuple(
        sum(p[axis] for p in gt_points) / len(gt_points) for axis in range(3)
    )
    pred_mean = tuple(
        sum(p[axis] for p in pred_points) / len(pred_points) for axis in range(3)
    )
    return tuple(gt_mean[axis] - pred_mean[axis] for axis in range(3))


def compute_world_poses(
    model: URDFModel, joint_positions: dict[str, float],
) -> dict[str, Mat4]:
    """BFS the kinematic tree from the root, accumulating 4×4 transforms.

    Each joint contributes: origin (xyz+rpy) then motion (axis×position for
    movable types). URDF convention: the joint frame lives in the parent
    link's frame, and the child link adopts the joint's frame as its rest
    pose. Unrooted links get identity (they won't move, and the check will
    flag that mismatch on its own via link_matching / topology).
    """
    poses: dict[str, Mat4] = {}
    root = model.root_link()
    if root is None:
        return poses
    poses[root] = _mat4_identity()
    queue = [root]
    while queue:
        parent = queue.pop(0)
        for j in model.children_of(parent):
            T_origin = _mat4_mul(
                _mat4_translate(j.origin_xyz),
                _mat4_rotate_rpy(j.origin_rpy),
            )
            if j.type in ("REVOLUTE", "CONTINUOUS") and j.axis is not None:
                pos = joint_positions.get(j.name, 0.0)
                T_motion = _mat4_rotate_axis_angle(j.axis, pos)
            elif j.type == "PRISMATIC" and j.axis is not None:
                pos = joint_positions.get(j.name, 0.0)
                # Normalize axis, then translate by pos along it
                mag = math.sqrt(sum(a * a for a in j.axis))
                if mag > 1e-9:
                    T_motion = _mat4_translate(
                        (j.axis[0] / mag * pos, j.axis[1] / mag * pos, j.axis[2] / mag * pos)
                    )
                else:
                    T_motion = _mat4_identity()
            else:  # FIXED / FLOATING / PLANAR / missing axis — no motion contribution
                T_motion = _mat4_identity()
            poses[j.child] = _mat4_mul(_mat4_mul(poses[parent], T_origin), T_motion)
            queue.append(j.child)
    return poses


def _model_feature_diagonal(model: URDFModel) -> float:
    """World-space diagonal of all rest-pose mesh feature points."""
    world = compute_world_poses(model, _joint_positions_at_reference(model))
    sig_cache: dict[str, MeshSignature] = {}
    points: list[tuple[float, float, float]] = []
    for name, link in model.visual_links.items():
        if name not in world:
            continue
        features = _link_world_features(link, world[name], model.base_dir, sig_cache)
        if features:
            points.extend(features)
    if not points:
        return 0.0
    mins = [min(p[axis] for p in points) for axis in range(3)]
    maxs = [max(p[axis] for p in points) for axis in range(3)]
    return math.sqrt(sum((maxs[axis] - mins[axis]) ** 2 for axis in range(3)))


def _descriptor_feature_diagonal(
    descriptors: dict[str, PoseDescriptor],
) -> float:
    """Approximate q=0 assembly diagonal from transformed mesh moments."""
    if not descriptors:
        return 0.0
    points = [descriptor.centroid for descriptor in descriptors.values()]
    mins = [min(point[axis] for point in points) for axis in range(3)]
    maxs = [max(point[axis] for point in points) for axis in range(3)]
    center_diagonal = math.sqrt(sum(
        (maxs[axis] - mins[axis]) ** 2 for axis in range(3)
    ))
    # A single-link object has no centroid spread. Use its RMS radius as a
    # stable linear extent; for multi-link objects it also prevents tiny
    # centre separation from making the estimate numerically brittle.
    radius_margin = 2.0 * max(
        (descriptor.scale for descriptor in descriptors.values()), default=0.0,
    )
    return math.sqrt(center_diagonal ** 2 + radius_margin ** 2)


def _kinematic_tolerance(gt: URDFModel, th: Thresholds) -> float:
    """Scale-normalized pose tolerance, with an explicit legacy override."""
    if th.kinematic_tol is not None:
        return th.kinematic_tol
    return max(th.kinematic_abs_tol, th.kinematic_rel_tol * _model_feature_diagonal(gt))


def _joint_positions_at(model: URDFModel, t: float) -> dict[str, float]:
    """Position every movable joint at a fraction t ∈ [0, 1] of its range.

    Missing limits fall back to sensible defaults: CONTINUOUS → [0, 2π],
    REVOLUTE → [-π, π], PRISMATIC → [0, 0.1m]. These defaults match how
    the skill typically drives joints when limits aren't set, so the
    comparison stays meaningful.
    """
    out: dict[str, float] = {}
    for j in model.movable_joints.values():
        lo, hi = j.limit_lower, j.limit_upper
        if lo is None or hi is None:
            if j.type == "CONTINUOUS":
                lo, hi = 0.0, 2 * math.pi
            elif j.type == "REVOLUTE":
                lo, hi = -math.pi, math.pi
            elif j.type == "PRISMATIC":
                lo, hi = 0.0, 0.1
            else:
                lo, hi = 0.0, 0.0
        out[j.name] = lo + t * (hi - lo)
    return out


def _joint_positions_at_reference(model: URDFModel) -> dict[str, float]:
    """Canonical joint coordinates corresponding to source-URDF q=0."""
    return {
        joint.name: joint.reference_position
        for joint in model.movable_joints.values()
    }


def _motion_subtree_enabled() -> bool:
    """Per-joint motion verdicts use the joint's own subtree (default). Set
    EVO_MOTION_SUBTREE=0 for the previous behaviour (every link of the object)."""
    return os.environ.get("EVO_MOTION_SUBTREE", "1") != "0"


def _subtree_links(model: "URDFModel", link: str) -> set[str]:
    out, stack = set(), [link]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(j.child for j in model.children_of(cur))
    return out


def _configuration_feature_drifts(
    gt: URDFModel,
    pred: URDFModel,
    matching: dict[str, str],
    gt_positions: dict[str, float],
    pred_positions: dict[str, float],
    gt_sig_cache: dict[str, MeshSignature],
    pred_sig_cache: dict[str, MeshSignature],
) -> dict[str, float]:
    """Per-link feature drift for one pair of joint configurations."""
    gt_world = compute_world_poses(gt, gt_positions)
    pred_world = compute_world_poses(pred, pred_positions)
    gt_desc: dict[str, PoseDescriptor] = {}
    pred_desc: dict[str, PoseDescriptor] = {}
    for gt_link, pred_link in matching.items():
        if gt_link not in gt_world or pred_link not in pred_world:
            continue
        gli = gt.links.get(gt_link)
        pli = pred.links.get(pred_link)
        if gli is None or pli is None:
            continue
        gd = _gt_link_descriptor(gt, gt_link, gt_world[gt_link], gt_sig_cache)
        pd = _link_world_descriptor(pli, pred_world[pred_link], pred.base_dir, pred_sig_cache)
        if gd is not None and pd is not None:
            gt_desc[gt_link] = gd
            pred_desc[gt_link] = pd

    if not gt_desc or not pred_desc:
        return {}
    align = _descriptor_similarity_alignment(
        gt, pred, matching, gt_desc, pred_desc,
    )
    return {
        gt_link: _descriptor_distance(descriptor, pred_desc[gt_link], align)
        for gt_link, descriptor in gt_desc.items()
    }


def check_kinematic_equivalence(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str], th: Thresholds,
    reports: dict[str, JointReport] | None = None,
) -> CheckResult:
    """Compare each movable joint independently over its reachable trace.

    All other joints remain at canonical lower limits while one joint is
    sampled.  For every joint we evaluate both ``pred(t)`` and
    ``pred(1-t)`` and keep the better full-trace correspondence.  This is
    invariant to endpoint/parameter direction while preventing errors in one
    degree of freedom from being hidden by synchronized motion in another.
    """
    if not matching:
        return CheckResult(
            "kinematic_equivalence", False,
            "no matched link pairs — upstream link_matching must pass first",
        )

    samples = th.kinematic_samples
    if samples < 2:
        samples = 2
    ts = [i / (samples - 1) for i in range(samples)]

    gt_sig_cache: dict[str, MeshSignature] = {}
    pred_sig_cache: dict[str, MeshSignature] = {}

    tol = _kinematic_tolerance(gt, th)
    errors: list[str] = []
    joint_maxima: dict[str, tuple[float, str]] = {}

    # A rigid-only object still needs a meaningful full-stage geometry check.
    if not gt.movable_joints:
        drifts = _configuration_feature_drifts(
            gt, pred, matching,
            _joint_positions_at(gt, 0.0), _joint_positions_at(pred, 0.0),
            gt_sig_cache, pred_sig_cache,
        )
        if not drifts:
            return CheckResult(
                "kinematic_equivalence", False,
                "no matched links resolved in rigid FK pass",
            )
        max_drift = max(drifts.values())
        return CheckResult(
            "kinematic_equivalence", max_drift <= tol,
            f"rigid rest max feature drift {max_drift:.3f}m (tol={tol:.3f}m)",
        )

    for gj in gt.movable_joints.values():
        rep = (reports or {}).get(gj.name)
        if rep is not None:
            rep.motion_ok = False
        if gj.child not in matching:
            errors.append(f"{gj.child}: unmatched GT movable joint")
            continue
        pred_child = matching[gj.child]
        pj = pred.parent_of(pred_child)
        if pj is None or pj.type == "FIXED":
            errors.append(f"{gj.child}: missing predicted movable joint")
            continue

        # Links this joint moves: its child and everything below it. The joint
        # report scores the trace on these only, so one misplaced link elsewhere
        # is not charged to every joint of the object as a motion error (the
        # check verdict below still uses every link).
        moved = _subtree_links(gt, gj.child)
        direction_results: list[tuple[float, str]] = []
        own_results: list[float] = []
        for reverse in (False, True):
            trace_max = 0.0
            own_max = 0.0
            resolved = True
            for t in ts:
                # Every other joint sits at its source q=0 (the rest pose both
                # sides share), not at the canonical lower limit: parsing flips
                # a negative-dominant axis and moves that joint's reference to
                # `upper`, so "lower" would hold a pred's untraced joints fully
                # extended against a GT at rest and fail motion by the travel.
                gt_positions = _joint_positions_at_reference(gt)
                pred_positions = _joint_positions_at_reference(pred)
                gt_positions[gj.name] = _joint_positions_at(gt, t)[gj.name]
                pred_t = 1.0 - t if reverse else t
                pred_positions[pj.name] = _joint_positions_at(pred, pred_t)[pj.name]
                drifts = _configuration_feature_drifts(
                    gt, pred, matching, gt_positions, pred_positions,
                    gt_sig_cache, pred_sig_cache,
                )
                if not drifts:
                    resolved = False
                    break
                trace_max = max(trace_max, max(drifts.values()))
                own_max = max(own_max, max((v for k, v in drifts.items() if k in moved), default=0.0))
            if resolved:
                direction_results.append((trace_max, "reverse" if reverse else "forward"))
                own_results.append(own_max)

        if not direction_results:
            errors.append(f"{gj.child}: no comparable FK trace")
            continue
        best_drift, direction = min(direction_results, key=lambda item: item[0])
        joint_maxima[gj.child] = (best_drift, direction)
        if rep is not None:
            own_drift = min(own_results) if _motion_subtree_enabled() else best_drift
            rep.trace_drift_m = own_drift
            rep.motion_ok = own_drift <= tol
        if best_drift > tol:
            errors.append(
                f"{gj.child}: trace drift={best_drift:.3f}m "
                f"({direction}, tol={tol:.3f}m)"
            )

    if not joint_maxima and errors:
        return CheckResult(
            "kinematic_equivalence", False,
            f"0/{len(gt.movable_joints)} movable joint traces compared; {errors[:3]}",
        )

    max_drift = max((value[0] for value in joint_maxima.values()), default=float("inf"))
    passed = not errors and len(joint_maxima) == len(gt.movable_joints)
    if passed:
        reverse_count = sum(1 for _, direction in joint_maxima.values() if direction == "reverse")
        detail = (
            f"{len(joint_maxima)} independent joint trace(s) × {samples} samples; "
            f"max feature drift {max_drift:.3f}m (tol={tol:.3f}m, "
            f"reverse_matches={reverse_count})"
        )
    else:
        detail = (
            f"{len(errors)} invalid independent joint trace(s); "
            + "; ".join(errors[:3])
        )
    return CheckResult("kinematic_equivalence", passed, detail)


def check_valid_tree(model: URDFModel) -> CheckResult:
    """Check the predicted URDF forms a valid tree (single root, no cycles)."""
    root = model.root_link()
    if root is None:
        return CheckResult("valid_tree", False, "no unique root link found")

    # BFS to check connectivity and cycles
    visited: set[str] = set()
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node in visited:
            return CheckResult("valid_tree", False, f"cycle detected at '{node}'")
        visited.add(node)
        for j in model.children_of(node):
            queue.append(j.child)

    unvisited = set(model.links.keys()) - visited
    if unvisited:
        return CheckResult("valid_tree", False,
                           f"{len(unvisited)} disconnected links: {list(unvisited)[:5]}")

    return CheckResult("valid_tree", True, f"root='{root}', {len(visited)} links connected")


def check_model_integrity(model: URDFModel) -> CheckResult:
    """Fail malformed/non-finite models before geometric comparison."""
    errors: list[str] = []
    for link in model.visual_links.values():
        for mesh_ref in link.mesh_paths:
            path = resolve_mesh_path(mesh_ref, model.base_dir)
            try:
                if not path.is_file() or path.stat().st_size <= 0:
                    errors.append(f"{link.name}: missing/empty mesh {mesh_ref!r}")
            except OSError:
                errors.append(f"{link.name}: unreadable mesh {mesh_ref!r}")
        values = (*link.visual_xyz, *link.visual_rpy)
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{link.name}: non-finite visual transform")

    for joint in model.joints.values():
        if joint.parent not in model.links or joint.child not in model.links:
            errors.append(
                f"{joint.name}: unknown parent/child "
                f"({joint.parent!r}->{joint.child!r})"
            )
        numeric = [*joint.origin_xyz, *joint.origin_rpy, joint.reference_position]
        if joint.axis is not None:
            numeric.extend(joint.axis)
        if joint.limit_lower is not None:
            numeric.append(joint.limit_lower)
        if joint.limit_upper is not None:
            numeric.append(joint.limit_upper)
        if not all(math.isfinite(value) for value in numeric):
            errors.append(f"{joint.name}: non-finite joint parameter")
        if joint.type != "FIXED":
            if joint.axis is None or math.sqrt(sum(a * a for a in joint.axis)) < 1e-9:
                errors.append(f"{joint.name}: movable joint has no valid axis")
        if (
            joint.limit_lower is not None
            and joint.limit_upper is not None
            and joint.limit_lower > joint.limit_upper
        ):
            errors.append(f"{joint.name}: lower limit exceeds upper limit")

    return CheckResult(
        "model_integrity", not errors,
        "all mesh references and numeric joint fields valid"
        if not errors else f"{len(errors)} integrity error(s): {errors[:4]}",
    )


# ---------------------------------------------------------------------------
# Oracle: aggregate checks into 1-bit verdict
# ---------------------------------------------------------------------------

@dataclass
class OracleResult:
    passed: bool
    checks: list[CheckResult]
    matching: dict[str, str] | None = None
    # What the verdict cost, for oracles that spend money (the subagent
    # judge). None for the script oracle, which is free, and for a model
    # with no entry in evo.engine.pricing. Without this the round cost
    # tables count only executors and understate a subagent-oracle run
    # several-fold, since `votes` judges each read every rendered frame.
    cost_usd: float | None = None
    # Per-GT-joint structured outcomes (None for oracles that do not resolve
    # joints, e.g. the subagent judge). Keys are GT joint names.
    joints: dict[str, "JointReport"] | None = None
    # Number of independent executor sessions this result aggregates (1 = a
    # single session; >1 = majority checks + mean joint scores, see
    # evo.engine.loop._aggregate_oracle_results).
    reps: int = 1

    @property
    def score(self) -> float:
        """Fraction of scored checks that passed (for graded feedback)."""
        scored_checks = [check for check in self.checks if check.scored]
        if not scored_checks:
            return 0.0
        return sum(1 for c in scored_checks if c.passed) / len(scored_checks)

    def to_dict(self) -> dict:
        return {
            "oracle_version": ORACLE_VERSION,
            "passed": self.passed,
            "score": round(self.score, 3),
            "checks": {
                c.name: {
                    "passed": c.passed,
                    "scored": c.scored,
                    "detail": c.detail,
                }
                for c in self.checks
            },
            "matching": self.matching,
            "cost_usd": self.cost_usd,
            "joints": (
                {name: r.to_dict() for name, r in self.joints.items()}
                if self.joints is not None else None
            ),
            "reps": self.reps,
        }

    def report(self) -> str:
        lines = []
        lines.append(f"{'PASS' if self.passed else 'FAIL'} (score={self.score:.2f})")
        for c in self.checks:
            lines.append(str(c))
        return "\n".join(lines)


# Check-detail strings embed GT values ("GT=[lo, hi]", "GT axis=(...)",
# "GT={'REVOLUTE': 2}") for human review. Those values must NOT reach the
# generator: a past run read them and hardcoded GT joint limits as magic
# constants in a tool (reward hacking). The sanitized form keeps the pred
# side and error magnitudes; only the GT-side values are redacted.
_GT_VALUE_PATTERNS = [
    re.compile(r"GT(?:\s+\w+)?\s*=\s*\[[^\]]*\]"),
    re.compile(r"GT(?:\s+\w+)?\s*=\s*\([^)]*\)"),
    re.compile(r"GT(?:\s+\w+)?\s*=\s*\{[^}]*\}"),
    re.compile(r"GT(?:\s+\w+)?\s*=\s*[-+0-9.eE]+"),
]


def sanitize_detail(detail: str) -> str:
    """Redact GT-side values from a check-detail string."""
    for pat in _GT_VALUE_PATTERNS:
        detail = pat.sub("GT=<redacted>", detail)
    return detail


def sanitize_oracle_dict(d: dict) -> dict:
    """Return a copy of `OracleResult.to_dict()` safe to show the generator.

    Redacts GT values from every check detail. Pass/fail flags, score, and
    the link matching are kept — they carry the diagnostic signal without
    exposing copyable GT constants.
    """
    out = dict(d)
    out["checks"] = {
        name: {**c, "detail": sanitize_detail(c.get("detail", ""))}
        for name, c in (d.get("checks") or {}).items()
    }
    if d.get("joints"):
        # Error magnitudes and pred-side values only; the GT joint type and
        # child link name are copyable constants and stay private.
        out["joints"] = {
            name: {k: v for k, v in r.items() if k == "gt_joint" or not k.startswith("gt_")}
            for name, r in d["joints"].items()
        }
    return out


def run_oracle(
    gt_dir: Path,
    pred_dir: Path,
    thresholds: Thresholds | None = None,
    verbose: bool = False,
    stage: str = "full",
) -> OracleResult:
    """Run the oracle comparison.

    Args:
      stage: "full" (default) runs all objective checks and requires every
        scored check to pass. "static" skips kinematic_equivalence entirely
        (no joint sweep) but still requires every scored check it runs to pass.
        Non-scored diagnostics remain in both result modes. Use static for
        early-abort screening when the predicted URDF was produced
        before joint motion was validated.

    Returns OracleResult with 1-bit verdict and per-check details.
    """
    if stage not in ("full", "static"):
        raise ValueError(f"stage must be 'full' or 'static', got {stage!r}")
    if _sd.AVAILABLE:
        _sd.reset_caches()
    th = thresholds or Thresholds()
    if (Path(gt_dir) / "bake_transforms.json").is_file() and not th.score_pivot_alignment:
        th = dataclasses.replace(th, score_pivot_alignment=True)
    checks: list[CheckResult] = []

    # 1. Parse URDFs
    try:
        gt_urdf_path = find_urdf(gt_dir)
        gt = parse_urdf(
            gt_urdf_path,
            allow_duplicate_joint_names=True,
            repair_reversed_limits=True,
        )
    except Exception as e:
        return OracleResult(False, [CheckResult("parse_gt", False, str(e))])

    try:
        pred_urdf_path = find_urdf(pred_dir)
        prediction_is_reference = (
            pred_urdf_path.resolve() == gt_urdf_path.resolve()
        )
        pred = parse_urdf(
            pred_urdf_path,
            # Calibration invariant: a reference compared with the exact
            # same file must score perfectly, including the five legacy GT
            # files with duplicate joint identifiers.  A separately written
            # prediction never gets this exception.
            allow_duplicate_joint_names=prediction_is_reference,
            repair_reversed_limits=prediction_is_reference,
        )
    except Exception as e:
        return OracleResult(False, [CheckResult("parse_pred", False, str(e))])

    if verbose:
        print(f"GT: {gt_urdf_path} ({len(gt.links)} links, {len(gt.joints)} joints)")
        print(f"Pred: {pred_urdf_path} ({len(pred.links)} links, {len(pred.joints)} joints)")

    # 2. Structural check on predicted URDF
    checks.append(check_model_integrity(pred))
    checks.append(check_valid_tree(pred))

    # 3. Link count
    checks.append(check_link_count(gt, pred, th))

    # 4. Joint type distribution (coarse check, per-type tolerance)
    checks.append(check_joint_type_distribution(gt, pred, th))

    # 5. Match links by mesh geometry
    match_result = _match_links_detailed(
        gt, pred,
        max_geometry_cost=th.link_match_max_geometry_cost,
        verbose=verbose,
    )
    matching = match_result.matching
    if not matching:
        checks.append(CheckResult("link_matching", False, "could not match any links"))
        return OracleResult(False, _apply_scored_checks(checks))
    else:
        gt_vis_count = len(gt.visual_links)
        match_pct = len(matching) / max(gt_vis_count, 1) * 100
        max_cost = max(match_result.geometry_costs.values(), default=float("inf"))
        rejected_detail = ""
        if match_result.rejected:
            rejected_detail = (
                f", rejected={len(match_result.rejected)} "
                f"(max_geometry_cost={th.link_match_max_geometry_cost})"
            )
        checks.append(CheckResult(
            "link_matching",
            match_pct >= th.link_matching_min_pct * 100,
            f"globally matched {len(matching)}/{gt_vis_count} GT visual links "
            f"({match_pct:.0f}%, max accepted geometry cost={max_cost:.6f}"
            f"{rejected_detail})",
        ))

    # 6-10. Everything downstream depends on the GT<->Pred link assignment.
    # Interchangeable parts (two drawers, two doors, four legs) make several
    # assignments physically identical; the dataset names those classes in
    # gt/part_equivalence.json (derived from the GT geometry when absent),
    # and the closest assignment wins. The assignment the matcher chose is
    # scored first; alternatives are tried only when it does not pass.
    classes = load_part_equivalence(Path(gt_dir))
    transforms = load_part_equivalence_transforms(Path(gt_dir))
    if classes is None:
        classes = equivalence_classes(gt)
    candidates = candidate_matchings(matching, classes) if classes else [matching]
    gt_sigs: dict[str, MeshSignature] = {}
    pred_sigs: dict[str, MeshSignature] = {}
    if transforms:
        for name, link in gt.visual_links.items():
            if link.mesh_paths:
                gt_sigs[name] = link_signature(link, gt.base_dir)
        for name, link in pred.visual_links.items():
            if link.mesh_paths:
                pred_sigs[name] = link_signature(link, pred.base_dir)
    best: tuple[tuple, list[CheckResult], dict[str, JointReport], dict[str, str], dict] | None = None
    budget = 240
    for cand in candidates:
        for subs in _substitute_variants(gt, pred, cand, classes, transforms, gt_sigs, pred_sigs):
            gt.link_substitutes = subs
            tail, reports = _evaluate_matching(gt, pred, cand, th, stage)
            key = _matching_rank(tail, reports)
            if best is None or key > best[0]:
                best = (key, tail, reports, cand, subs)
            budget -= 1
            if budget <= 0 or (best[1] and all(c.passed for c in best[1] if c.scored)):
                break
        if budget <= 0 or (best is not None and best[1] and all(c.passed for c in best[1] if c.scored)):
            break
    assert best is not None
    _key, tail, reports, matching, subs = best
    gt.link_substitutes = subs
    if matching != candidates[0]:
        swapped = sorted(g for g in matching if matching[g] != candidates[0][g])
        checks[-1] = CheckResult(
            checks[-1].name, checks[-1].passed,
            checks[-1].detail + f"; interchangeable parts: scored the closest assignment "
            f"(swapped GT links {swapped})",
        )
    checks.extend(tail)
    _apply_scored_checks(checks)

    # Final verdict.
    #   "full"   — every scored structural, articulation, and FK check is a
    #              hard gate. Partial credit is the fraction of scored checks.
    #   "static" — gate on every scored check that ran. Non-scored diagnostics
    #              remain in the result for analysis but cannot change PASS.
    passed = all(c.passed for c in checks if c.scored)

    return OracleResult(passed=passed, checks=checks, matching=matching, joints=reports)


def _evaluate_matching(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str], th: Thresholds, stage: str,
) -> tuple[list[CheckResult], dict[str, JointReport]]:
    """Checks 6-10 (topology, joints, rest pose, pivots, motion) under one link assignment."""
    checks: list[CheckResult] = []
    # Structured per-joint record, filled by the joint-level checks below.
    reports: dict[str, JointReport] = {
        jname: JointReport(
            gt_joint=jname, gt_child=gj.child, gt_type=gj.type,
            matched=gj.child in matching,
        )
        for jname, gj in gt.movable_joints.items()
    }
    # 6. Topology check (parent-child under matching)
    checks.append(check_topology(gt, pred, matching))
    # 7. Joint type check (per matched pair)
    checks.append(check_joint_types(gt, pred, matching, reports=reports))
    # 8. Joint axis check
    checks.append(check_joint_axes(gt, pred, matching, th, reports=reports))
    # 9. Joint limit check
    checks.append(check_joint_limits(gt, pred, matching, th, reports=reports))
    # 9.5. Two distinct scored geometry checks, both using the shared
    # doubled pose tolerance (2 cm absolute / 4% of GT model scale):
    #   - final q=0 mesh placement (joint and visual transforms combined);
    #   - physical revolute axis-line location (direction is joint_axes).
    # Kinematic equivalence remains the final full-trace motion gate.
    checks.append(check_rest_pose_alignment(gt, pred, matching, th, reports=reports))
    checks.append(check_joint_pivot_alignment(gt, pred, matching, th, reports=reports))
    # 9.7. Per-link size (diagnostic unless the dataset scores it).
    checks.append(check_link_scale(gt, pred, matching, th, reports=reports))
    # 10. Kinematic equivalence under a joint-parameter sweep.
    # Skipped in "static" stage because no joint motion has been validated yet.
    if stage == "full":
        checks.append(check_kinematic_equivalence(gt, pred, matching, th, reports=reports))
    return checks, reports


def check_link_scale(
    gt: URDFModel, pred: URDFModel, matching: dict[str, str], th: Thresholds,
    reports: dict[str, JointReport] | None = None,
) -> CheckResult:
    """Per-link size: the matched prediction's mesh must have the GT part's
    rms radius about its centroid (within ``scale_rel_tol``). Invariant to how
    the part is placed or oriented, so it isolates the scale answer of views
    whose task parts are unit-cube normalized. Fills ``scale_ratio`` /
    ``scale_ok`` on the report of every joint driving a matched link."""
    errors: list[str] = []
    checked = 0
    ratios: dict[str, float] = {}
    for gt_link, pred_link in matching.items():
        gl = gt.links.get(gt_link); pl = pred.links.get(pred_link)
        if gl is None or pl is None or not gl.mesh_paths or not pl.mesh_paths:
            continue
        g = link_signature(gl, gt.base_dir)
        p = link_signature(pl, pred.base_dir)
        if g.rms_radius <= 1e-12 or p.rms_radius <= 1e-12:
            continue
        checked += 1
        ratio = p.rms_radius / g.rms_radius
        ratios[gt_link] = ratio
        if abs(ratio - 1.0) > th.scale_rel_tol:
            errors.append(f"{gt_link}: size ratio {ratio:.3f} (tol ±{th.scale_rel_tol:.0%})")
    for rep in (reports or {}).values():
        if rep.gt_child in ratios:
            rep.scale_ratio = ratios[rep.gt_child]
            rep.scale_ok = abs(ratios[rep.gt_child] - 1.0) <= th.scale_rel_tol
    passed = not errors
    detail = (f"checked {checked} matched link(s), all within ±{th.scale_rel_tol:.0%}" if passed
              else f"checked {checked} matched link(s), {len(errors)} off-size: " + "; ".join(errors[:8]))
    return CheckResult("link_scale", passed, detail, scored="link_scale" in SCORED_CHECKS)


def _matching_rank(checks: list[CheckResult], reports: dict[str, JointReport]) -> tuple:
    """Closest-wins ordering of a link assignment: all scored checks passed,
    then the number of scored checks passed, then the summed per-joint score."""
    scored = [c for c in checks if c.scored]
    n_pass = sum(1 for c in scored if c.passed)
    joint_total = sum(r.score() for r in reports.values()) if reports else 0.0
    return (n_pass == len(scored), n_pass, round(joint_total, 9))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="URDF comparison oracle for /assemble skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare predicted output against ground truth
  python check_urdf.py /path/to/gt_instance /path/to/pred_output

  # Verbose mode shows matching details
  python check_urdf.py /path/to/gt /path/to/pred --verbose

  # JSON output for programmatic use
  python check_urdf.py /path/to/gt /path/to/pred --json

  # Sanity check: compare GT against itself (should always pass)
  python check_urdf.py /path/to/gt /path/to/gt
        """,
    )
    parser.add_argument("gt_dir", type=Path, help="Ground-truth directory (contains scene.urdf)")
    parser.add_argument("pred_dir", type=Path, help="Predicted directory (contains *.urdf + meshes/)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show matching details")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--axis-tol", type=float, default=0.95,
                        help="Min |cos(angle)| for axis match (default: 0.95)")
    parser.add_argument("--limit-tol", type=float, default=0.30,
                        help="Relative tolerance for joint limits (default: 0.30)")
    args = parser.parse_args()

    th = Thresholds(axis_cos_min=args.axis_tol, limit_rel_tol=args.limit_tol)
    result = run_oracle(args.gt_dir, args.pred_dir, thresholds=th, verbose=args.verbose)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.report())

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
