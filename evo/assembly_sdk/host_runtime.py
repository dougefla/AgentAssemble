"""Host-process Assembly SDK — the pure-Python successor to the bpy runtime.

Same public API and error contract as the Blender-era ``runtime.py`` (see
ACTIONS.md), with the Blender data model replaced by:

- **State**: the assembly graph persists as JSON in a state file
  (``$OUTPUT_DIR/.assembly_state.json`` unless ``ASSEMBLY_STATE_PATH``
  overrides it), with the same revision compare-and-swap semantics the
  Blender scene key had.
- **Meshes**: parts are parsed straight from their source OBJ files into an
  in-process registry (vertices quantized through IEEE float32, exactly the
  representation Blender's mesh datablocks used, so vertex digests are
  byte-compatible with SDK 1.1.x state files). Parts reload lazily from
  ``source_path`` after a worker restart.
- **Rendering**: ``render_preview``/``render_joint_sweep`` flatten the
  current scene into (world matrix, mesh path) entries and delegate to the
  shared light-3js-phyx renderer (evo.oracle.render_lightphyx.render_scene).
- **Export**: meshes re-emit from the parsed source OBJ structure (vertex
  coordinates identical to import by construction), materials still flow
  through evo.assembly_sdk.materials.

Load via ``runtime.py`` (which injects the runtime globals) — this module is
importable for tests but expects ``get_assembly(globals_dict)`` to wire the
state path.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET

from evo.assembly_sdk.core import (
    AssemblyError,
    ErrorCode,
    JOINT_TYPES,
    MOVABLE_TYPES,
    OperationResult,
    SDK_VERSION,
    SCHEMA_VERSION,
    fail,
    normalize_axis,
    validate_identifier,
    validate_state,
    vec3,
)
from evo.assembly_sdk.materials import preserve_materials

STATE_FILENAME = ".assembly_state.json"
_UNSET = object()

_IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


# ---- rigid 4x4 math (row-major tuples) ---------------------------------

def _mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def _matrix(xyz, rpy):
    """Translation @ Rz(rpy[2]) @ Ry(rpy[1]) @ Rx(rpy[0]) — URDF convention,
    identical to mathutils' Matrix.Translation @ Rotation(Z) @ (Y) @ (X)."""
    x, y, z = (float(v) for v in xyz)
    r, p, w = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cw, sw = math.cos(w), math.sin(w)
    return (
        (cw * cp, cw * sp * sr - sw * cr, cw * sp * cr + sw * sr, x),
        (sw * cp, sw * sp * sr + cw * cr, sw * sp * cr - cw * sr, y),
        (-sp, cp * sr, cp * cr, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _axis_rotation(axis, angle):
    ux, uy, uz = axis
    mag = math.sqrt(ux * ux + uy * uy + uz * uz)
    if mag < 1e-12:
        return _IDENTITY
    ux, uy, uz = ux / mag, uy / mag, uz / mag
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return (
        (c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s, 0.0),
        (uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s, 0.0),
        (uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _translation(xyz):
    x, y, z = xyz
    return ((1.0, 0.0, 0.0, x), (0.0, 1.0, 0.0, y),
            (0.0, 0.0, 1.0, z), (0.0, 0.0, 0.0, 1.0))


def _uniform_scale(value: float):
    scale = float(value)
    return ((scale, 0.0, 0.0, 0.0), (0.0, scale, 0.0, 0.0),
            (0.0, 0.0, scale, 0.0), (0.0, 0.0, 0.0, 1.0))


def _mat_invert_rigid(M):
    R = [[M[i][j] for j in range(3)] for i in range(3)]
    t = [M[i][3] for i in range(3)]
    return (
        (R[0][0], R[1][0], R[2][0], -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2])),
        (R[0][1], R[1][1], R[2][1], -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2])),
        (R[0][2], R[1][2], R[2][2], -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def _transform_point(M, p):
    x, y, z = p
    return (
        M[0][0] * x + M[0][1] * y + M[0][2] * z + M[0][3],
        M[1][0] * x + M[1][1] * y + M[1][2] * z + M[1][3],
        M[2][0] * x + M[2][1] * y + M[2][2] * z + M[2][3],
    )


def _matrix_pose(M, operation: str):
    """Decompose a rigid 4x4 into (xyz, rpy). Fails on non-rigid scale."""
    for k in range(3):
        col = math.sqrt(sum(M[i][k] * M[i][k] for i in range(3)))
        if abs(col - 1.0) > 1e-5:
            fail(
                ErrorCode.INVARIANT_VIOLATION,
                operation,
                "derived transform contains non-rigid scale",
                scale=[
                    math.sqrt(sum(M[i][j] * M[i][j] for i in range(3)))
                    for j in range(3)
                ],
            )
    xyz = (float(M[0][3]), float(M[1][3]), float(M[2][3]))
    sp = max(-1.0, min(1.0, -M[2][0]))
    p = math.asin(sp)
    if abs(sp) < 1.0 - 1e-9:
        r = math.atan2(M[2][1], M[2][2])
        w = math.atan2(M[1][0], M[0][0])
    else:
        r = math.atan2(-M[1][2], M[1][1])
        w = 0.0
    return xyz, (float(r), float(p), float(w))


# ---- mesh registry -------------------------------------------------------

def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _vertex_digest(vertices) -> str:
    rows = sorted(
        struct.pack(
            "<fff",
            *(0.0 if abs(float(value)) < 1e-30 else float(value) for value in xyz),
        )
        for xyz in vertices
    )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row)
    return digest.hexdigest()


class MeshPart:
    """A parsed OBJ: float32-quantized vertices + raw structure for export."""

    __slots__ = ("source_path", "vertices", "local_min", "local_max",
                 "faces", "texcoords", "matrix_world")

    def __init__(self, source_path: Path):
        self.source_path = Path(source_path)
        vertices: list[tuple[float, float, float]] = []
        texcoords: list[tuple[float, float]] = []
        # faces: list of (source_usemtl or None, [(v_idx, vt_idx or None)])
        faces: list[tuple[str | None, list[tuple[int, int | None]]]] = []
        current_mtl: str | None = None
        with self.source_path.open(errors="replace") as stream:
            for line in stream:
                f = line.split()
                if not f:
                    continue
                key = f[0]
                if key == "v" and len(f) >= 4:
                    vertices.append((_f32(f[1]), _f32(f[2]), _f32(f[3])))
                elif key == "vt" and len(f) >= 3:
                    texcoords.append((float(f[1]), float(f[2])))
                elif key == "usemtl":
                    rest = line.split(None, 1)
                    current_mtl = rest[1].strip() if len(rest) > 1 else None
                elif key == "f":
                    refs = []
                    for token in f[1:]:
                        pieces = token.split("/")
                        v_idx = int(pieces[0])
                        # Relative (negative) indices refer to the vertices
                        # seen so far; rebase them now so export never has
                        # to guess the writer's ordering.
                        if v_idx < 0:
                            v_idx = len(vertices) + v_idx + 1
                        vt_idx = (
                            int(pieces[1])
                            if len(pieces) > 1 and pieces[1] else None
                        )
                        if vt_idx is not None and vt_idx < 0:
                            vt_idx = len(texcoords) + vt_idx + 1
                        refs.append((v_idx, vt_idx))
                    faces.append((current_mtl, refs))
        self.vertices = vertices
        self.texcoords = texcoords
        self.faces = faces
        if vertices:
            self.local_min = tuple(min(v[i] for v in vertices) for i in range(3))
            self.local_max = tuple(max(v[i] for v in vertices) for i in range(3))
        else:
            self.local_min = self.local_max = (0.0, 0.0, 0.0)
        self.matrix_world = _IDENTITY

    def bound_corners(self):
        (lx, ly, lz), (hx, hy, hz) = self.local_min, self.local_max
        return [(x, y, z) for x in (lx, hx) for y in (ly, hy) for z in (lz, hz)]

    def world_corners(self):
        return [_transform_point(self.matrix_world, c) for c in self.bound_corners()]


MESH_FRAME_LINK = "link"
MESH_FRAME_WORLD = "world"
MESH_FRAME_ENV = "ASSEMBLY_MESH_FRAME"
WORLD_BAKE_MARKER = "world_bake_q0"


def detect_mesh_frame(input_dir: Path) -> str:
    """Decide how imported OBJ vertices relate to the world.

    ``link`` (legacy): vertices are in each part's own link frame, whose origin
    is the joint frame. Nothing is placed until a joint origin says where the
    link goes, so placing parts and locating pivots are one decision.

    ``world``: vertices are already the rest-pose world coordinates (the
    ``artvip_beginner`` view bakes the GT chain at q=0 into every part). Parts
    load in position; joints only describe motion. The SDK keeps every mesh
    exactly where it was imported by re-solving each link's visual transform
    whenever a joint changes, and refuses the APIs that would move a mesh.

    Resolution: ``ASSEMBLY_MESH_FRAME`` env override, else the nearest
    ``dataset_variant.json`` above ``input_dir`` listing ``world_bake_q0`` in
    ``task_mesh_transforms``, else ``link``.
    """
    override = os.environ.get(MESH_FRAME_ENV, "").strip().lower()
    if override in (MESH_FRAME_LINK, MESH_FRAME_WORLD):
        return override
    if override:
        fail(ErrorCode.INVALID_ARGUMENT, "import_parts",
             f"{MESH_FRAME_ENV} must be 'link' or 'world'", value=override)
    probe = Path(input_dir).resolve()
    for _ in range(6):
        marker = probe / "dataset_variant.json"
        if marker.is_file():
            try:
                variant = json.loads(marker.read_text())
            except Exception:
                return MESH_FRAME_LINK
            transforms = variant.get("task_mesh_transforms") or []
            return MESH_FRAME_WORLD if WORLD_BAKE_MARKER in transforms else MESH_FRAME_LINK
        if probe.parent == probe:
            break
        probe = probe.parent
    return MESH_FRAME_LINK


class AssemblySDK:
    """Stable, explicit API for articulated assembly — host-process edition.

    Two mesh-frame modes, chosen at ``import_parts`` (see ``detect_mesh_frame``):
    ``link`` keeps the historical contract; ``world`` treats imported vertices
    as the rest-pose world coordinates and guarantees no mesh ever moves —
    joints describe motion only, and ``audit`` reports any drift.
    """

    version = SDK_VERSION

    def __init__(self, state_path: Path):
        self._state_path = Path(state_path)
        self._registry: dict[str, MeshPart] = {}

    def _output_path(self, value: str | os.PathLike, operation: str) -> Path:
        """Resolve an SDK artifact path inside this session's OUTPUT_DIR.

        Relative paths are intentionally useful for agents and resolve against
        the state-file directory. Absolute paths outside it are rejected before
        filesystem access, producing a recoverable contract error instead of a
        raw permission failure in locations such as ``/mnt/workspace``.
        """
        root = self._state_path.resolve().parent
        raw = Path(value).expanduser()
        resolved = (raw if raw.is_absolute() else root / raw).resolve()
        if not resolved.is_relative_to(root):
            fail(
                ErrorCode.INVALID_ARGUMENT,
                operation,
                "output path must stay inside OUTPUT_DIR",
                path=str(resolved),
                output_dir=str(root),
            )
        return resolved

    # ---- error/result boundary -----------------------------------------

    def safe_call(self, method: str, **kwargs) -> dict:
        """Call one SDK method and always return a structured dictionary."""
        if not isinstance(method, str):
            return AssemblyError(
                ErrorCode.INVALID_ARGUMENT, "safe_call",
                "method must be a string",
                details={"received": repr(method)},
            ).as_dict()
        target = getattr(self, method, None)
        if target is None or method.startswith("_") or not callable(target):
            return AssemblyError(
                ErrorCode.INVALID_ARGUMENT, "safe_call",
                "unknown public SDK method", details={"method": method},
            ).as_dict()
        try:
            signature = inspect.signature(target)
            signature.bind(**kwargs)
        except TypeError as exc:
            return AssemblyError(
                ErrorCode.INVALID_ARGUMENT, method, str(exc),
                details={"accepted": list(signature.parameters)},
            ).as_dict()
        try:
            value = target(**kwargs)
            if isinstance(value, OperationResult):
                return value.as_dict()
            warnings = (
                tuple(str(item) for item in value.get("warnings", []))
                if isinstance(value, dict) else ()
            )
            return OperationResult(ok=True, value=value, warnings=warnings).as_dict()
        except AssemblyError as exc:
            return exc.as_dict()
        except OSError as exc:
            return AssemblyError(
                ErrorCode.IO_FAILURE, method, "filesystem operation failed",
                details={"type": type(exc).__name__, "cause": str(exc)},
            ).as_dict()
        except Exception as exc:
            return AssemblyError(
                ErrorCode.SIM_RUNTIME, method,
                "unexpected simulation/Python failure",
                details={"type": type(exc).__name__, "cause": str(exc)},
            ).as_dict()

    # ---- state and identity -------------------------------------------

    @staticmethod
    def _identity_state() -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "sdk_version": SDK_VERSION,
            "revision": 0,
            "parts": {},
            "links": {},
            "joints": {},
            "joint_positions": {},
        }

    def _read_raw(self):
        try:
            return self._state_path.read_text()
        except FileNotFoundError:
            return None

    def _load(self) -> dict:
        raw = self._read_raw()
        if not raw:
            return self._identity_state()
        try:
            state = json.loads(raw)
        except Exception as exc:
            fail(ErrorCode.INVARIANT_VIOLATION, "load_state",
                 "persisted state is not valid JSON", cause=str(exc))
        if state.get("schema_version") != SCHEMA_VERSION:
            fail(
                ErrorCode.INVARIANT_VIOLATION, "load_state",
                "unsupported assembly state schema",
                expected=SCHEMA_VERSION, found=state.get("schema_version"),
            )
        return state

    # ---- mesh-frame mode ----------------------------------------------
    @staticmethod
    def _mesh_frame(state: dict) -> str:
        return str(state.get("mesh_frame") or MESH_FRAME_LINK)

    def _require_link_frame(self, state: dict, operation: str) -> None:
        """Refuse mesh-moving operations when parts are pre-placed."""
        if self._mesh_frame(state) == MESH_FRAME_WORLD:
            fail(
                ErrorCode.INVARIANT_VIOLATION, operation,
                "not available when mesh_frame='world': parts are imported in "
                "their rest-pose world coordinates and must never move; joints "
                "only describe motion (use create_joint / update_joint / "
                "set_joint_world_frame / reparent_joint)",
                mesh_frame=MESH_FRAME_WORLD,
            )

    def _sync_world_visuals(self, state: dict) -> None:
        """World mode invariant: T_world_link(q=0) @ T_link_visual == identity.

        Re-solved on every save, so whatever a joint operation did to a link's
        q=0 frame, its mesh stays at the imported world coordinates. Validation
        failures are left for the caller's own validate_state to report under
        the mutating operation's name.
        """
        if self._mesh_frame(state) != MESH_FRAME_WORLD or not state.get("links"):
            return
        zero_state = copy.deepcopy(state)
        self._zero_positions(zero_state)
        try:
            poses = self._link_world_transforms(zero_state, strict=False)
        except AssemblyError:
            return
        for link_name, pose in poses.items():
            xyz, rpy = _matrix_pose(_mat_invert_rigid(pose), "sync_world_visuals")
            state["links"][link_name]["visual_xyz"] = list(xyz)
            state["links"][link_name]["visual_rpy"] = list(rpy)

    def _world_frame_drift(self, state: dict) -> list[dict]:
        """Audit helper: links whose mesh is not at its imported world pose."""
        if self._mesh_frame(state) != MESH_FRAME_WORLD or not state.get("links"):
            return []
        zero_state = copy.deepcopy(state)
        self._zero_positions(zero_state)
        try:
            poses = self._link_world_transforms(zero_state, strict=False)
        except AssemblyError:
            return []
        issues = []
        for link_name, pose in poses.items():
            link = state["links"][link_name]
            world = _mat_mul(pose, _matrix(link["visual_xyz"], link["visual_rpy"]))
            drift = max(
                abs(world[i][j] - _IDENTITY[i][j]) for i in range(3) for j in range(4)
            )
            if drift > 1e-6:
                issues.append({
                    "code": "world_frame_drift",
                    "message": "mesh moved away from its imported world pose",
                    "details": {"link": link_name, "max_abs_drift": drift},
                })
        return issues

    def _save(self, candidate: dict, operation: str, *, complete: bool = False) -> None:
        self._sync_world_visuals(candidate)
        try:
            issues = validate_state(candidate, require_complete_tree=complete)
        except AssemblyError as exc:
            issues = list(exc.details.get("issues", []))
        if issues:
            fail(ErrorCode.INVARIANT_VIOLATION, operation,
                 "candidate state violates SDK invariants", issues=issues)
        expected_revision = int(candidate.get("revision", 0))
        raw = self._read_raw()
        if raw:
            try:
                actual_revision = int(json.loads(raw).get("revision", 0))
            except Exception as exc:
                fail(ErrorCode.INVARIANT_VIOLATION, operation,
                     "cannot compare revisions because persisted state is corrupt",
                     cause=str(exc))
        else:
            actual_revision = 0
        if actual_revision != expected_revision:
            fail(
                ErrorCode.INVARIANT_VIOLATION, operation,
                "stale state snapshot; reload before retrying",
                expected_revision=expected_revision,
                actual_revision=actual_revision,
            )
        candidate["revision"] = expected_revision + 1
        candidate["sdk_version"] = SDK_VERSION
        payload = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=self._state_path.parent, delete=False
        ) as stream:
            temp = Path(stream.name)
            stream.write(payload)
        temp.replace(self._state_path)

    def state(self) -> dict:
        """Return a detached, JSON-safe state snapshot."""
        return json.loads(json.dumps(self._load()))

    def _part(self, state: dict, handle: str, operation: str) -> MeshPart:
        if handle not in state["parts"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown part handle", handle=handle)
        entry = state["parts"][handle]
        if not isinstance(entry, dict):
            fail(
                ErrorCode.INVARIANT_VIOLATION, operation,
                "part registry entry is malformed",
                handle=handle, received=type(entry).__name__,
            )
        part = self._registry.get(handle)
        if part is None:
            # Lazy recovery (e.g. after a worker restart): reload from source.
            source = Path(entry.get("source_path", ""))
            if not source.is_file():
                fail(
                    ErrorCode.UNKNOWN_HANDLE, operation,
                    "part handle is stale; its source mesh is missing",
                    handle=handle, source_path=str(source),
                )
            part = MeshPart(source)
            self._registry[handle] = part
        scale = _finite_number(entry.get("scale", 1.0), operation, "part.scale")
        if scale <= 0.0:
            fail(
                ErrorCode.INVARIANT_VIOLATION,
                operation,
                "persisted part scale must be positive",
                handle=handle,
                scale=scale,
            )
        return part

    def set_part_scale(self, handle: str, scale: float) -> dict:
        """Set and persist one part's positive uniform semantic scale."""
        operation = "set_part_scale"
        state = self._load()
        self._require_link_frame(state, operation)
        self._part(state, handle, operation)
        value = _finite_number(scale, operation, "scale")
        if value <= 0.0:
            fail(
                ErrorCode.INVALID_ARGUMENT,
                operation,
                "scale must be greater than zero",
                scale=value,
            )
        state["parts"][handle]["scale"] = value
        self._save(state, operation)
        self._apply(state, strict=False)
        return {"handle": handle, "scale": value}

    def clear_scene(self) -> dict:
        """Drop all loaded meshes and reset SDK state."""
        self._registry.clear()
        self._state_path.unlink(missing_ok=True)
        return {"removed": True, "revision": 0}

    def import_parts(self, input_dir: str | os.PathLike, *, replace: bool = True) -> list[dict]:
        """Import OBJ files while preserving exact source vertex coordinates.

        Returns stable part handles; loader internals are never required by
        callers.
        """
        operation = "import_parts"
        root = Path(input_dir).resolve()
        if not root.is_dir():
            fail(ErrorCode.INVALID_ARGUMENT, operation, "input_dir is not a directory", input_dir=str(root))
        def _objs(candidates):
            return sorted(
                path for path in candidates
                if path.is_file() and path.suffix.lower() == ".obj"
            )
        paths = _objs(root.iterdir()) or _objs(root.glob("*/*"))
        if not paths:
            fail(ErrorCode.INVALID_ARGUMENT, operation, "no OBJ files found", input_dir=str(root))
        if replace:
            self.clear_scene()
        state = self._load()
        requested = {
            f"part:{path.relative_to(root).as_posix()}": path
            for path in paths
        }
        # A fresh sim call may need handles again after durable state has
        # persisted but the agent's Python locals have not. Repeating the
        # exact same replace=False import is therefore a read-only lookup.
        # Partial overlaps and different source directories still fail below.
        if not replace and requested and set(state["parts"]) == set(requested):
            same_sources = all(
                Path(state["parts"][handle].get("source_path", "")).resolve()
                == path.resolve()
                for handle, path in requested.items()
            )
            if same_sources:
                return [
                    {
                        "handle": handle,
                        "label": path.stem,
                        "vertices": int(
                            state["parts"][handle].get("source_vertex_count", 0)
                        ),
                    }
                    for handle, path in requested.items()
                ]
        if state["links"] or state["joints"]:
            fail(ErrorCode.INVARIANT_VIOLATION, operation,
                 "cannot import into a scene with an existing assembly; clear it first")

        imported = []
        added_handles: list[str] = []
        try:
            for path in paths:
                rel = path.relative_to(root).as_posix()
                handle = f"part:{rel}"
                if handle in state["parts"]:
                    fail(ErrorCode.AMBIGUOUS_GEOMETRY, operation,
                         "duplicate deterministic part handle", handle=handle)
                part = MeshPart(path)
                if not part.vertices:
                    fail(ErrorCode.INVALID_ARGUMENT, operation,
                         "OBJ contains no vertices", path=str(path))
                self._registry[handle] = part
                added_handles.append(handle)
                digest = _vertex_digest(part.vertices)
                state["parts"][handle] = {
                    "object_name": path.stem,
                    "source_path": str(path),
                    "source_vertex_count": len(part.vertices),
                    "source_vertex_digest": digest,
                    "scale": 1.0,
                }
                imported.append({"handle": handle, "label": path.stem,
                                 "vertices": len(part.vertices)})
        except AssemblyError:
            if replace:
                self.clear_scene()
            else:
                for handle in added_handles:
                    self._registry.pop(handle, None)
            raise
        except Exception as exc:
            if replace:
                self.clear_scene()
            else:
                for handle in added_handles:
                    self._registry.pop(handle, None)
            fail(ErrorCode.SIM_RUNTIME, operation, "OBJ import failed",
                 cause=str(exc), type=type(exc).__name__)
        state["mesh_frame"] = detect_mesh_frame(root)
        self._save(state, operation)
        return imported

    def inspect_parts(self) -> list[dict]:
        operation = "inspect_parts"
        state = self._load()
        if state["links"]:
            # Reconstruct pose and semantic scale before reporting world bounds,
            # including after a sim-worker restart emptied the mesh registry.
            self._apply(state, strict=False)
        result = []
        for handle in sorted(state["parts"]):
            part = self._part(state, handle, operation)
            world = part.world_corners()
            result.append({
                "handle": handle,
                "label": Path(state["parts"][handle]["source_path"]).stem,
                "vertex_count": len(part.vertices),
                "scale": float(state["parts"][handle].get("scale", 1.0)),
                "local_aabb": {"min": list(part.local_min), "max": list(part.local_max)},
                "world_aabb": {
                    "min": [min(p[i] for p in world) for i in range(3)],
                    "max": [max(p[i] for p in world) for i in range(3)],
                },
            })
        return result

    # ---- topology ------------------------------------------------------

    def create_link(
        self, name: str, parts: str | list[str], *,
        visual_xyz=(0.0, 0.0, 0.0), visual_rpy=(0.0, 0.0, 0.0),
    ) -> dict:
        operation = "create_link"
        validate_identifier(name, operation, "name")
        handles = _part_handle_list(parts, operation)
        state = self._load()
        if name in state["links"]:
            fail(ErrorCode.INVALID_ARGUMENT, operation, "link already exists", link=name)
        if self._mesh_frame(state) == MESH_FRAME_WORLD and (
            any(abs(float(v)) > 0.0 for v in vec3(visual_xyz, operation, "visual_xyz"))
            or any(abs(float(v)) > 0.0 for v in vec3(visual_rpy, operation, "visual_rpy"))
        ):
            fail(ErrorCode.INVALID_ARGUMENT, operation,
                 "visual_xyz/visual_rpy are not accepted when mesh_frame='world': "
                 "parts are pre-placed and the SDK owns the visual transform",
                 mesh_frame=MESH_FRAME_WORLD)
        owned = {h: link for link, info in state["links"].items() for h in info["part_handles"]}
        existing_owners = {owned[handle] for handle in handles if handle in owned}
        if existing_owners:
            # Exploratory agents commonly register anonymous parts under a
            # temporary name, then choose the semantic name after rendering.
            # Re-registering exactly one isolated group is an atomic rename.
            if len(existing_owners) == 1:
                old_name = next(iter(existing_owners))
                incident = [
                    joint_name
                    for joint_name, joint in state["joints"].items()
                    if joint["parent"] == old_name or joint["child"] == old_name
                ]
                old_handles = state["links"][old_name]["part_handles"]
                if not incident and set(old_handles) == set(handles):
                    # Semantic rename of a posed temporary link: keep its
                    # visual pose exactly like rename_link() does. Callers
                    # wanting a different pose use set_link_visual_pose().
                    old_link = state["links"].pop(old_name)
                    state["links"][name] = {
                        "part_handles": handles,
                        "visual_xyz": list(old_link["visual_xyz"]),
                        "visual_rpy": list(old_link["visual_rpy"]),
                    }
                    self._save(state, operation)
                    self._apply(state, strict=False)
                    return {
                        "link": name,
                        "parts": handles,
                        "renamed_from": old_name,
                    }
        for handle in handles:
            self._part(state, handle, operation)
            if handle in owned:
                fail(ErrorCode.INVARIANT_VIOLATION, operation,
                     "part already belongs to a link", part=handle, owner=owned[handle])
        state["links"][name] = {
            "part_handles": handles,
            "visual_xyz": list(vec3(visual_xyz, operation, "visual_xyz")),
            "visual_rpy": list(vec3(visual_rpy, operation, "visual_rpy")),
        }
        self._save(state, operation)
        self._apply(state, strict=False)
        return {"link": name, "parts": handles}

    def update_link_parts(self, name: str, parts: str | list[str]) -> dict:
        """Replace or atomically swap an isolated link's rigid part group."""
        operation = "update_link_parts"
        state = self._load()
        if name not in state["links"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown link", link=name)
        handles = _part_handle_list(parts, operation)
        owned = {
            handle: owner
            for owner, info in state["links"].items()
            if owner != name
            for handle in info["part_handles"]
        }
        conflicting_owners = {owned[h] for h in handles if h in owned}
        if len(conflicting_owners) == 1:
            other = next(iter(conflicting_owners))
            incident = [
                joint_name
                for joint_name, joint in state["joints"].items()
                if joint["parent"] == other or joint["child"] == other
            ]
            other_handles = list(state["links"][other]["part_handles"])
            current_handles = list(state["links"][name]["part_handles"])
            if (
                not incident
                and set(other_handles) == set(handles)
                and current_handles
            ):
                state["links"][name]["part_handles"] = handles
                state["links"][other]["part_handles"] = current_handles
                self._save(state, operation)
                self._apply(state, strict=False)
                return {
                    "link": name,
                    "parts": handles,
                    "swapped_with": other,
                    "released_parts": current_handles,
                }
        for handle in handles:
            self._part(state, handle, operation)
            if handle in owned:
                fail(ErrorCode.INVARIANT_VIOLATION, operation,
                     "part already belongs to another link",
                     part=handle, owner=owned[handle])
        state["links"][name]["part_handles"] = handles
        self._save(state, operation)
        self._apply(state, strict=False)
        return {"link": name, "parts": handles}

    def rename_link(self, name: str, new_name: str) -> dict:
        operation = "rename_link"
        validate_identifier(new_name, operation, "new_name")
        state = self._load()
        if name not in state["links"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown link", link=name)
        if new_name in state["links"]:
            fail(ErrorCode.INVALID_ARGUMENT, operation, "link already exists", link=new_name)
        state["links"][new_name] = state["links"].pop(name)
        for joint in state["joints"].values():
            if joint["parent"] == name:
                joint["parent"] = new_name
            if joint["child"] == name:
                joint["child"] = new_name
        self._save(state, operation)
        self._apply(state, strict=False)
        return {"old_name": name, "new_name": new_name}

    def delete_link(self, name: str) -> dict:
        """Delete an isolated link; remove incident joints explicitly first."""
        operation = "delete_link"
        state = self._load()
        if name not in state["links"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown link", link=name)
        incident = sorted(
            joint_name
            for joint_name, joint in state["joints"].items()
            if joint["parent"] == name or joint["child"] == name
        )
        if incident:
            fail(ErrorCode.INVARIANT_VIOLATION, operation,
                 "delete incident joints before deleting a link",
                 link=name, joints=incident)
        released = list(state["links"].pop(name)["part_handles"])
        self._save(state, operation)
        self._apply(state, strict=False)
        return {"link": name, "released_parts": released}

    def create_joint(
        self, name: str, joint_type: str, parent: str, child: str, origin_xyz,
        *, origin_rpy=(0.0, 0.0, 0.0), axis="Z", limits=None,
    ) -> dict:
        operation = "create_joint"
        state = self._load()
        validate_identifier(name, operation, "name")
        if name in state["joints"]:
            fail(ErrorCode.INVALID_ARGUMENT, operation, "joint already exists", joint=name)
        if parent not in state["links"] or child not in state["links"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation,
                 "parent and child must be registered links", parent=parent, child=child)
        joint_type = str(joint_type).upper()
        if joint_type not in JOINT_TYPES:
            fail(ErrorCode.INVALID_ARGUMENT, operation, "unsupported joint type", joint_type=joint_type)
        # Validate the axis for every type so a zero/malformed axis is
        # rejected consistently; FIXED joints then store the canonical Z.
        axis_value = list(normalize_axis(axis, operation))
        if joint_type == "FIXED":
            axis_value = [0.0, 0.0, 1.0]
        lower = upper = None
        if joint_type in ("REVOLUTE", "PRISMATIC"):
            if isinstance(limits, dict):
                limits = [limits.get("lower"), limits.get("upper")]
            if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                fail(ErrorCode.INVALID_ARGUMENT, operation, "bounded joint requires [lower, upper] limits")
            lower = _finite_number(limits[0], operation, "limits.lower")
            upper = _finite_number(limits[1], operation, "limits.upper")
            if lower > upper:
                fail(ErrorCode.INVALID_ARGUMENT, operation,
                     "limits must be finite and lower <= upper", limits=limits)
        elif limits is not None:
            fail(ErrorCode.INVALID_ARGUMENT, operation,
                 f"limits are not valid for a {joint_type} joint")
        state["joints"][name] = {
            "type": joint_type,
            "parent": parent,
            "child": child,
            "origin_xyz": list(vec3(origin_xyz, operation, "origin_xyz")),
            "origin_rpy": list(vec3(origin_rpy, operation, "origin_rpy")),
            "axis": axis_value,
            "limit_lower": lower,
            "limit_upper": upper,
        }
        state["joint_positions"][name] = 0.0
        self._save(state, operation)
        self._apply(state, strict=False)
        return {"joint": name, "parent": parent, "child": child}

    def update_joint(
        self, name: str, *, joint_type=None, origin_xyz=None, origin_rpy=None,
        axis=None, limits=_UNSET,
    ) -> dict:
        operation = "update_joint"
        state = self._load()
        if name not in state["joints"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown joint", joint=name)
        if (
            joint_type is None and origin_xyz is None and origin_rpy is None
            and axis is None and limits is _UNSET
        ):
            fail(ErrorCode.INVALID_ARGUMENT, operation, "no joint fields were provided")
        joint = copy.deepcopy(state["joints"][name])
        old_type = joint["type"]
        jt = str(joint_type or old_type).upper()
        if jt not in JOINT_TYPES:
            fail(ErrorCode.INVALID_ARGUMENT, operation, "unsupported joint type", joint_type=jt)
        if origin_xyz is not None:
            joint["origin_xyz"] = list(vec3(origin_xyz, operation, "origin_xyz"))
        if origin_rpy is not None:
            joint["origin_rpy"] = list(vec3(origin_rpy, operation, "origin_rpy"))
        if jt == "FIXED":
            if axis is not None:
                fail(ErrorCode.INVALID_ARGUMENT, operation, "axis is not valid for a FIXED joint")
            if limits is not _UNSET:
                fail(ErrorCode.INVALID_ARGUMENT, operation, "limits are not valid for a FIXED joint")
            joint["axis"] = [0.0, 0.0, 1.0]
            joint["limit_lower"] = joint["limit_upper"] = None
        else:
            joint["axis"] = list(normalize_axis(axis if axis is not None else joint["axis"], operation))
            if jt == "CONTINUOUS":
                if limits is not _UNSET:
                    fail(ErrorCode.INVALID_ARGUMENT, operation, "limits are not valid for a CONTINUOUS joint")
                joint["limit_lower"] = joint["limit_upper"] = None
            elif limits is not _UNSET:
                raw = [limits.get("lower"), limits.get("upper")] if isinstance(limits, dict) else limits
                if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                    fail(ErrorCode.INVALID_ARGUMENT, operation, "bounded joint requires [lower, upper] limits")
                lower = _finite_number(raw[0], operation, "limits.lower")
                upper = _finite_number(raw[1], operation, "limits.upper")
                if lower > upper:
                    fail(ErrorCode.INVALID_ARGUMENT, operation,
                         "limits must be finite and lower <= upper", limits=raw)
                joint["limit_lower"], joint["limit_upper"] = lower, upper
            elif joint.get("limit_lower") is None or joint.get("limit_upper") is None:
                fail(ErrorCode.INVALID_ARGUMENT, operation, "bounded joint requires [lower, upper] limits")
        joint["type"] = jt
        current_q = float(state["joint_positions"].get(name, 0.0))
        if jt == "FIXED" and abs(current_q) > 1e-12:
            fail(ErrorCode.INVALID_ARGUMENT, operation,
                 "reset the joint to q=0 before changing it to FIXED",
                 joint=name, current_position=current_q)
        if jt in ("REVOLUTE", "PRISMATIC") and not (
            joint["limit_lower"] - 1e-9 <= current_q <= joint["limit_upper"] + 1e-9
        ):
            fail(ErrorCode.INVALID_ARGUMENT, operation,
                 "current joint position would be outside the updated limits",
                 joint=name, current_position=current_q,
                 lower=joint["limit_lower"], upper=joint["limit_upper"])
        state["joints"][name] = joint
        self._save(state, operation)
        self._apply(state, strict=False)
        return copy.deepcopy(joint)

    def rename_joint(self, name: str, new_name: str) -> dict:
        operation = "rename_joint"
        validate_identifier(new_name, operation, "new_name")
        state = self._load()
        if name not in state["joints"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown joint", joint=name)
        if new_name in state["joints"]:
            fail(ErrorCode.INVALID_ARGUMENT, operation, "joint already exists", joint=new_name)
        state["joints"][new_name] = state["joints"].pop(name)
        state["joint_positions"][new_name] = state["joint_positions"].pop(name)
        self._save(state, operation)
        self._apply(state, strict=False)
        return {"old_name": name, "new_name": new_name}

    def delete_joint(self, name: str) -> dict:
        operation = "delete_joint"
        state = self._load()
        if name not in state["joints"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown joint", joint=name)
        removed = state["joints"].pop(name)
        state["joint_positions"].pop(name, None)
        self._save(state, operation)
        self._apply(state, strict=False)
        return {"joint": name, "parent": removed["parent"], "child": removed["child"]}

    def reparent_joint(self, name: str, new_parent: str) -> dict:
        """Reparent while preserving the child's complete q=0 world frame.

        Valid partial forests are supported: every forest root sits at the
        world identity, so the child's q=0 world frame is well defined even
        before the tree is connected.
        """
        operation = "reparent_joint"
        state = self._load()
        if name not in state["joints"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown joint", joint=name)
        if new_parent not in state["links"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown new parent link", parent=new_parent)
        zero_state = copy.deepcopy(state)
        self._zero_positions(zero_state)
        poses = self._link_world_transforms(zero_state, strict=False)
        joint = state["joints"][name]
        child = joint["child"]
        old_parent = joint["parent"]
        T_origin = _mat_mul(_mat_invert_rigid(poses[new_parent]), poses[child])
        xyz, rpy = _matrix_pose(T_origin, operation)
        joint["parent"] = new_parent
        joint["origin_xyz"] = list(xyz)
        joint["origin_rpy"] = list(rpy)
        self._save(state, operation, complete=False)
        self._apply(state, strict=False)
        return {"joint": name, "old_parent": old_parent, "new_parent": new_parent,
                "origin_xyz": xyz, "origin_rpy": rpy}

    def set_joint_world_frame(self, name: str, world_origin, world_axis=None) -> dict:
        """Set a joint's q=0 world frame, including in a valid partial forest."""
        operation = "set_joint_world_frame"
        state = self._load()
        if name not in state["joints"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown joint", joint=name)
        zero_state = copy.deepcopy(state)
        self._zero_positions(zero_state)
        # Agent assembly is incremental: other imported links may still be
        # separate roots when this joint is authored. Structural corruption
        # is still rejected by strict=False; only the one-root/complete-tree
        # requirement is relaxed until audit/export.
        poses = self._link_world_transforms(zero_state, strict=False)
        joint = state["joints"][name]
        parent_T = poses[joint["parent"]]
        point = vec3(world_origin, operation, "world_origin")
        local = _transform_point(_mat_invert_rigid(parent_T), point)
        joint["origin_xyz"] = [float(v) for v in local]
        if world_axis is not None and joint["type"] == "FIXED":
            fail(ErrorCode.INVALID_ARGUMENT, operation,
                 "world_axis is not valid for a FIXED joint", joint=name)
        if world_axis is not None:
            axis_world = normalize_axis(world_axis, operation)
            frame = _mat_mul(parent_T, _matrix((0, 0, 0), joint["origin_rpy"]))
            inv = _mat_invert_rigid(frame)
            axis_local = (
                inv[0][0] * axis_world[0] + inv[0][1] * axis_world[1] + inv[0][2] * axis_world[2],
                inv[1][0] * axis_world[0] + inv[1][1] * axis_world[1] + inv[1][2] * axis_world[2],
                inv[2][0] * axis_world[0] + inv[2][1] * axis_world[1] + inv[2][2] * axis_world[2],
            )
            joint["axis"] = list(normalize_axis(tuple(axis_local), operation))
        self._save(state, operation, complete=False)
        self._apply(state, strict=False)
        return {"joint": name, "origin_xyz": joint["origin_xyz"], "axis": joint["axis"]}

    # ---- poses/FK ------------------------------------------------------

    def _root(self, state: dict, operation: str) -> str:
        children = {joint["child"] for joint in state["joints"].values()}
        roots = sorted(set(state["links"]) - children)
        if len(roots) != 1:
            fail(ErrorCode.INVARIANT_VIOLATION, operation,
                 "assembly must have exactly one root", roots=roots)
        return roots[0]

    def _zero_positions(self, state: dict) -> None:
        state["joint_positions"] = {name: 0.0 for name in state["joints"]}

    @staticmethod
    def _motion(joint, q: float):
        if joint["type"] == "PRISMATIC":
            ax = joint["axis"]
            mag = math.sqrt(sum(a * a for a in ax)) or 1.0
            return _translation((ax[0] / mag * q, ax[1] / mag * q, ax[2] / mag * q))
        if joint["type"] in ("REVOLUTE", "CONTINUOUS"):
            return _axis_rotation(joint["axis"], q)
        return _IDENTITY

    def _link_world_transforms(self, state: dict, *, strict: bool) -> dict:
        if strict:
            validate_state(state, require_complete_tree=True)
        else:
            issues = validate_state(state, require_complete_tree=False)
            if issues:
                fail(ErrorCode.INVARIANT_VIOLATION, "forward_kinematics",
                     "assembly state failed validation", issues=issues)
        if not state["links"]:
            return {}
        children_set = {joint["child"] for joint in state["joints"].values()}
        roots = sorted(set(state["links"]) - children_set)
        if strict:
            roots = [self._root(state, "forward_kinematics")]
        poses = {root: _IDENTITY for root in roots}
        seen = set(roots)
        children = {}
        for name, joint in state["joints"].items():
            children.setdefault(joint["parent"], []).append((name, joint))
        queue = list(roots)
        while queue:
            parent = queue.pop(0)
            for joint_name, joint in sorted(
                children.get(parent, []), key=lambda item: item[0]
            ):
                child = joint["child"]
                if child in seen:
                    fail(ErrorCode.INVARIANT_VIOLATION, "forward_kinematics",
                         "kinematic graph revisits a link",
                         joint=joint_name, child=child)
                seen.add(child)
                q = float(state["joint_positions"].get(joint_name, 0.0))
                poses[child] = _mat_mul(
                    _mat_mul(poses[parent], _matrix(joint["origin_xyz"], joint["origin_rpy"])),
                    self._motion(joint, q),
                )
                queue.append(child)
        if strict and set(poses) != set(state["links"]):
            fail(ErrorCode.INVARIANT_VIOLATION, "forward_kinematics",
                 "not all links are reachable",
                 unreachable=sorted(set(state["links"]) - set(poses)))
        return poses

    def _apply(self, state: dict, *, strict: bool) -> dict:
        poses = self._link_world_transforms(state, strict=strict)
        for link_name, link in state["links"].items():
            if link_name not in poses:
                continue
            world = _mat_mul(poses[link_name], _matrix(link["visual_xyz"], link["visual_rpy"]))
            for handle in link["part_handles"]:
                part = self._part(state, handle, "apply_to_scene")
                scale = float(state["parts"][handle].get("scale", 1.0))
                part.matrix_world = _mat_mul(world, _uniform_scale(scale))
        return poses

    def set_link_visual_pose(self, name: str, world_xyz, world_rpy=(0.0, 0.0, 0.0)) -> dict:
        """Place only a link's visual mesh frame in q=0 WORLD coordinates.

        Child link frames are intentionally unaffected. To move a kinematic
        subtree, update the incoming joint with ``set_joint_world_frame``.
        Valid partial forests are supported during incremental authoring;
        ``audit`` and ``export_urdf`` still require one complete tree.
        """
        operation = "set_link_visual_pose"
        state = self._load()
        self._require_link_frame(state, operation)
        if name not in state["links"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown link", link=name)
        zero_state = copy.deepcopy(state)
        self._zero_positions(zero_state)
        poses = self._link_world_transforms(zero_state, strict=False)
        desired = _matrix(vec3(world_xyz, operation, "world_xyz"),
                          vec3(world_rpy, operation, "world_rpy"))
        visual = _mat_mul(_mat_invert_rigid(poses[name]), desired)
        xyz, rpy = _matrix_pose(visual, operation)
        state["links"][name]["visual_xyz"] = list(xyz)
        state["links"][name]["visual_rpy"] = list(rpy)
        self._save(state, operation, complete=False)
        self._apply(state, strict=False)
        return {"link": name, "visual_xyz": xyz, "visual_rpy": rpy}

    def set_link_pose(self, name: str, world_xyz, *, world_rpy=(0.0, 0.0, 0.0)) -> dict:
        """Deprecated compatibility alias for ``set_link_visual_pose``."""
        result = self.set_link_visual_pose(name, world_xyz, world_rpy=world_rpy)
        result["deprecated_alias"] = "set_link_pose"
        result["use_instead"] = "set_link_visual_pose"
        return result

    def bake_current_pose_as_rest(
        self, name: str, *, allow_nonzero_joint_positions: bool = False
    ) -> dict:
        """Bake the displayed visual pose into q=0.

        By default this refuses a nonzero configuration, preventing joint
        motion from being accidentally baked into the reference rest pose.
        """
        operation = "bake_current_pose_as_rest"
        state = self._load()
        self._require_link_frame(state, operation)
        if name not in state["links"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown link", link=name)
        nonzero = {
            joint: float(q)
            for joint, q in state["joint_positions"].items()
            if abs(float(q)) > 1e-9
        }
        if nonzero and not allow_nonzero_joint_positions:
            fail(ErrorCode.INVALID_ARGUMENT, operation,
                 "refusing to bake joint motion into q=0; reset joints first or opt in explicitly",
                 nonzero_joint_positions=nonzero)
        # Refresh every part's matrix_world from persisted state first: a
        # lazily reloaded MeshPart (worker restart, second process) would
        # otherwise still carry the identity matrix and bake a wrong pose.
        self._apply(state, strict=False)
        handles = state["links"][name]["part_handles"]
        matrices = []
        for handle in handles:
            part = self._part(state, handle, operation)
            scale = float(state["parts"][handle].get("scale", 1.0))
            matrices.append(
                _mat_mul(part.matrix_world, _uniform_scale(1.0 / scale))
            )
        if any(
            max(abs(float(matrices[0][r][c] - matrix[r][c])) for r in range(4) for c in range(4)) > 1e-6
            for matrix in matrices[1:]
        ):
            fail(ErrorCode.AMBIGUOUS_GEOMETRY, operation,
                 "parts in one link do not share a rigid transform", link=name)
        snapshot = matrices[0]
        self._zero_positions(state)
        poses = self._link_world_transforms(state, strict=True)
        xyz, rpy = _matrix_pose(_mat_mul(_mat_invert_rigid(poses[name]), snapshot), operation)
        state["links"][name]["visual_xyz"] = list(xyz)
        state["links"][name]["visual_rpy"] = list(rpy)
        self._save(state, operation, complete=True)
        self._apply(state, strict=True)
        return {"link": name, "visual_xyz": xyz, "visual_rpy": rpy}

    def set_joint_positions(self, positions: dict[str, float]) -> dict:
        operation = "set_joint_positions"
        if not isinstance(positions, dict):
            fail(ErrorCode.INVALID_ARGUMENT, operation,
                 "positions must be a joint->value dictionary")
        state = self._load()
        for name, value in positions.items():
            if name not in state["joints"]:
                fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown joint", joint=name)
            joint = state["joints"][name]
            q = _finite_number(value, operation, f"positions.{name}")
            if joint["type"] in ("REVOLUTE", "PRISMATIC") and not (
                joint["limit_lower"] - 1e-9 <= q <= joint["limit_upper"] + 1e-9
            ):
                fail(ErrorCode.INVALID_ARGUMENT, operation,
                     "joint value is outside limits", joint=name, value=q,
                     lower=joint["limit_lower"], upper=joint["limit_upper"])
            if joint["type"] == "FIXED" and abs(q) > 1e-12:
                fail(ErrorCode.INVALID_ARGUMENT, operation,
                     "fixed joint cannot move", joint=name, value=q)
            state["joint_positions"][name] = q
        self._save(state, operation)
        self._apply(state, strict=False)
        return copy.deepcopy(state["joint_positions"])

    def reset_joints(self) -> dict:
        state = self._load()
        self._zero_positions(state)
        self._save(state, "reset_joints")
        self._apply(state, strict=False)
        return copy.deepcopy(state["joint_positions"])

    def ground(self, z: float = 0.0) -> dict:
        operation = "ground"
        target = _finite_number(z, operation, "z")
        state = self._load()
        self._require_link_frame(state, operation)
        zero_state = copy.deepcopy(state)
        self._zero_positions(zero_state)
        self._apply(zero_state, strict=True)
        minimum = None
        for link in state["links"].values():
            for handle in link["part_handles"]:
                part = self._part(state, handle, operation)
                for corner in part.world_corners():
                    value = float(corner[2])
                    minimum = value if minimum is None else min(minimum, value)
        if minimum is None:
            fail(ErrorCode.AMBIGUOUS_GEOMETRY, operation, "assembly contains no mesh bounds")
        delta = target - minimum
        root = self._root(state, operation)
        state["links"][root]["visual_xyz"][2] += delta
        for joint in state["joints"].values():
            if joint["parent"] == root:
                joint["origin_xyz"][2] += delta
        self._save(state, operation, complete=True)
        self._apply(state, strict=True)
        return {
            "previous_min_z": minimum,
            "target_z": target,
            "delta_z": delta,
            "measured_at": "q=0",
        }

    # ---- objective checks and export ----------------------------------

    def audit(self, *, require_complete_tree: bool = True, verify_source_vertices: bool = True) -> dict:
        state = self._load()
        if require_complete_tree:
            try:
                validate_state(state, require_complete_tree=True)
            except AssemblyError as exc:
                issues = list(exc.details.get("issues", []))
            else:
                issues = []
        else:
            issues = validate_state(state, require_complete_tree=False)
        for handle in sorted(state["parts"]):
            try:
                part = self._part(state, handle, "audit")
            except AssemblyError as exc:
                issues.append({"code": exc.code.value, "message": exc.message, "details": exc.details})
                continue
            if verify_source_vertices:
                entry = state["parts"][handle]
                digest = _vertex_digest(part.vertices)
                if digest != entry.get("source_vertex_digest"):
                    issues.append({
                        "code": "source_vertex_drift",
                        "message": "loaded mesh vertices differ from imported OBJ",
                        "details": {"part": handle, "expected": entry.get("source_vertex_digest"), "actual": digest},
                    })
        issues.extend(self._world_frame_drift(state))
        return {
            "ok": not issues,
            "sdk_version": SDK_VERSION,
            "mesh_frame": self._mesh_frame(state),
            "revision": state["revision"],
            "counts": {"parts": len(state["parts"]), "links": len(state["links"]), "joints": len(state["joints"])},
            "issues": issues,
        }

    def collision_aabbs(self, *, epsilon: float = 1e-5) -> list[dict]:
        state = self._load()
        self._apply(state, strict=False)
        boxes = {}
        for link_name, link in state["links"].items():
            points = []
            for handle in link["part_handles"]:
                points.extend(self._part(state, handle, "collision_aabbs").world_corners())
            boxes[link_name] = (
                [min(p[i] for p in points) for i in range(3)],
                [max(p[i] for p in points) for i in range(3)],
            )
        collisions = []
        names = sorted(boxes)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                a, b = boxes[first], boxes[second]
                overlap = [min(a[1][k], b[1][k]) - max(a[0][k], b[0][k]) for k in range(3)]
                if all(v > epsilon for v in overlap):
                    collisions.append({"links": [first, second], "aabb_overlap": overlap})
        return collisions

    def _export_mesh(self, output: Path, state: dict, link_name: str, link: dict) -> list[str]:
        lines = [f"# Assembly SDK {SDK_VERSION}; link {link_name}"]
        sources = [
            (handle, Path(state["parts"][handle]["source_path"]))
            for handle in link["part_handles"]
        ]
        bundle = preserve_materials(link_name, sources, output.parent)
        warnings = list(bundle.warnings)
        if bundle.mtl_filename:
            lines.append(f"mtllib {bundle.mtl_filename}")
        vertex_offset = 0
        uv_offset = 0
        for handle in link["part_handles"]:
            part = self._part(state, handle, "export_urdf")
            scale = float(state["parts"][handle].get("scale", 1.0))
            source_map = bundle.material_names.get(handle, {})
            lines.append(f"o {Path(state['parts'][handle]['source_path']).stem}")
            for vertex in part.vertices:
                lines.append(
                    "v %.9g %.9g %.9g"
                    % tuple(float(value) * scale for value in vertex)
                )
            for uv in part.texcoords:
                lines.append("vt %.9g %.9g" % uv)
            current_material = None
            for source_mtl, refs in part.faces:
                material = source_map.get(source_mtl or "", "")
                if not material and source_mtl:
                    warnings.append(
                        f"cannot map source material {source_mtl!r} for {handle}"
                    )
                if material and material != current_material:
                    lines.append(f"usemtl {material}")
                    current_material = material
                tokens = []
                for v_idx, vt_idx in refs:
                    v_abs = v_idx + vertex_offset
                    if vt_idx is not None:
                        vt_abs = vt_idx + uv_offset
                        tokens.append(f"{v_abs}/{vt_abs}")
                    else:
                        tokens.append(str(v_abs))
                lines.append("f " + " ".join(tokens))
            vertex_offset += len(part.vertices)
            uv_offset += len(part.texcoords)
        output.write_text("\n".join(lines) + "\n")
        return warnings

    def export_urdf(self, output_dir: str | os.PathLike, *, history: bool = False) -> dict:
        operation = "export_urdf"
        audit = self.audit(require_complete_tree=True, verify_source_vertices=True)
        if not audit["ok"]:
            fail(ErrorCode.INVARIANT_VIOLATION, operation,
                 "refusing to export an invalid assembly", issues=audit["issues"])
        # Reload after audit: compare-and-swap requires the latest revision.
        state = self._load()
        for handle in sorted(state["parts"]):
            self._part(state, handle, operation)
        self._zero_positions(state)
        self._save(state, operation, complete=True)
        self._apply(state, strict=True)

        output = self._output_path(output_dir, operation)
        output.mkdir(parents=True, exist_ok=True)
        meshes = output / "meshes"
        # Stage every mesh/MTL/texture in a scratch directory, then swap it
        # in right before the URDF is replaced, so a failure mid-export
        # never leaves a URDF pointing at deleted or half-written meshes.
        staging = output / f".meshes.staging-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        warnings: list[str] = []
        try:
            for link_name in sorted(state["links"]):
                warnings.extend(self._export_mesh(
                    staging / f"{link_name}_visual.obj",
                    state, link_name, state["links"][link_name],
                ))
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        robot = ET.Element("robot", {"name": "furniture"})
        fmt = lambda values: " ".join(f"{float(v):.9g}" for v in values)
        for link_name in sorted(state["links"]):
            link = state["links"][link_name]
            element = ET.SubElement(robot, "link", {"name": link_name})
            visual = ET.SubElement(element, "visual")
            ET.SubElement(visual, "origin", {"xyz": fmt(link["visual_xyz"]), "rpy": fmt(link["visual_rpy"])})
            geometry = ET.SubElement(visual, "geometry")
            ET.SubElement(geometry, "mesh", {"filename": f"meshes/{link_name}_visual.obj"})
        for joint_name in sorted(state["joints"]):
            joint = state["joints"][joint_name]
            element = ET.SubElement(robot, "joint", {"name": joint_name, "type": joint["type"].lower()})
            ET.SubElement(element, "parent", {"link": joint["parent"]})
            ET.SubElement(element, "child", {"link": joint["child"]})
            ET.SubElement(element, "origin", {"xyz": fmt(joint["origin_xyz"]), "rpy": fmt(joint["origin_rpy"])})
            if joint["type"] in MOVABLE_TYPES:
                ET.SubElement(element, "axis", {"xyz": fmt(joint["axis"])})
            if joint["type"] in ("REVOLUTE", "PRISMATIC"):
                ET.SubElement(element, "limit", {
                    "lower": f"{joint['limit_lower']:.9g}",
                    "upper": f"{joint['limit_upper']:.9g}",
                    "effort": "10",
                    "velocity": "1",
                })
        ET.indent(robot, space="  ")
        urdf = output / "furniture.urdf"
        try:
            with tempfile.NamedTemporaryFile("wb", dir=output, delete=False) as stream:
                temp_path = Path(stream.name)
                ET.ElementTree(robot).write(stream, encoding="utf-8", xml_declaration=True)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        previous = output / f".meshes.previous-{os.getpid()}"
        if previous.exists():
            shutil.rmtree(previous)
        try:
            if meshes.exists():
                meshes.rename(previous)
            staging.rename(meshes)
        except BaseException:
            # Roll the directory swap back so the old URDF stays consistent.
            if not meshes.exists() and previous.exists():
                previous.rename(meshes)
            shutil.rmtree(staging, ignore_errors=True)
            temp_path.unlink(missing_ok=True)
            raise
        temp_path.replace(urdf)
        shutil.rmtree(previous, ignore_errors=True)
        if history:
            history_dir = output / "urdf_history"
            history_dir.mkdir(exist_ok=True)
            digest = hashlib.sha256(urdf.read_bytes()).hexdigest()[:12]
            shutil.copy2(urdf, history_dir / f"rev{state['revision']:04d}_{digest}.urdf")
        parts_map = {
            "version": 2,
            "links": {
                link_name: [
                    {
                        "path": state["parts"][handle]["source_path"],
                        "scale": float(state["parts"][handle].get("scale", 1.0)),
                    }
                    for handle in state["links"][link_name]["part_handles"]
                ]
                for link_name in sorted(state["links"])
            },
        }
        parts_map_path = output / "parts_map.json"
        with tempfile.NamedTemporaryFile("w", dir=output, delete=False) as stream:
            json.dump(parts_map, stream, indent=2, sort_keys=True)
            stream.write("\n")
            parts_map_temp = Path(stream.name)
        parts_map_temp.replace(parts_map_path)
        final_audit = self.audit(require_complete_tree=True, verify_source_vertices=True)
        return {
            "urdf": str(urdf),
            "parts_map": str(parts_map_path),
            "links": len(state["links"]),
            "joints": len(state["joints"]),
            "audit": final_audit,
            "warnings": sorted(set(warnings)),
        }

    # ---- safe rendering ------------------------------------------------

    @staticmethod
    def _camera_for(center, radius, azimuth_deg, elevation_deg):
        """Camera position + fov for one azimuth around a fixed frame.

        azimuth 0 looks at the front (+Y direction, camera on -Y), growing
        counter-clockwise seen from above, so az=45 is front-right. Same
        convention as the oracle renderer (evo.oracle.render_lightphyx).
        """
        azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
        direction = (
            math.cos(elevation) * math.sin(azimuth),
            -math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
        )
        cam_pos = tuple(center[i] + direction[i] * radius * 3.0 for i in range(3))
        # The Blender-era camera was orthographic with ortho_scale=radius*2.4;
        # at distance radius*3 the perspective equivalent frames the same
        # extent with fov = 2*atan((radius*1.2)/(radius*3)) = 2*atan(0.4).
        return cam_pos, math.degrees(2.0 * math.atan(0.4))

    def _render_one(self, entries, destination, center, radius, azimuth_deg,
                    elevation_deg, resolution, operation):
        """Render one camera into `destination` and return that path."""
        from evo.oracle.render_lightphyx import render_scene

        cam_pos, fov_deg = self._camera_for(
            center, radius, azimuth_deg, elevation_deg
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            render_scene(entries, destination, cam_pos=cam_pos, lookat=center,
                         fov_deg=fov_deg, resolution=resolution)
        except Exception as exc:
            fail(ErrorCode.SIM_RUNTIME, operation, "simulation render failed",
                 cause=str(exc))
        return destination

    @staticmethod
    def _extra_view_azimuths(azimuth_deg, views, operation):
        """Evenly spaced azimuths after the primary one (views=2 -> +180)."""
        count = int(views)
        if count < 1 or count > 8:
            fail(ErrorCode.INVALID_ARGUMENT, operation,
                 "views must be between 1 and 8", views=views)
        return [
            (float(azimuth_deg) + 360.0 * i / count) % 360.0
            for i in range(1, count)
        ]

    def render_preview(
        self, path: str | os.PathLike, *,
        azimuth_deg: float = 45.0, elevation_deg: float = 25.0,
        resolution: int = 512, views: int = 2,
    ) -> dict:
        """Render the current assembly with deterministic camera framing.

        `views` > 1 also renders evenly spaced extra azimuths (2 -> the
        opposite side) as `<stem>_az<NNN>.png`, reported under `views`. One
        camera is not enough: the dataset has no shared front convention, so a
        fixed azimuth shows the featureless back of some objects, and any
        judgement made from that view is guesswork. Pass `views=1` for the
        single-camera behaviour.
        """
        operation = "render_preview"
        extra = self._extra_view_azimuths(azimuth_deg, views, operation)
        from evo.oracle.render_lightphyx import render_scene

        state = self._load()
        poses = self._apply(state, strict=False)
        parts = [(handle, self._part(state, handle, operation)) for handle in sorted(state["parts"])]
        if not parts:
            fail(ErrorCode.AMBIGUOUS_GEOMETRY, operation, "nothing to render")
        fixed_frame = getattr(self, "_fixed_render_frame", None)
        if fixed_frame is None:
            points = [corner for _, part in parts for corner in part.world_corners()]
            minimum = tuple(min(p[i] for p in points) for i in range(3))
            maximum = tuple(max(p[i] for p in points) for i in range(3))
            center = tuple((minimum[i] + maximum[i]) * 0.5 for i in range(3))
            diagonal = math.sqrt(
                sum((maximum[i] - minimum[i]) ** 2 for i in range(3))
            )
            radius = max(diagonal * 0.75, 0.5)
        else:
            center = tuple(float(v) for v in fixed_frame["center"])
            radius = float(fixed_frame["radius"])
        destination = self._output_path(path, operation)
        if destination.suffix.lower() != ".png":
            destination = destination.with_suffix(destination.suffix + ".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        resolution = max(64, min(int(resolution), 2048))
        entries = [(part.matrix_world, str(part.source_path)) for _, part in parts]
        self._render_one(entries, destination, center, radius, azimuth_deg,
                         elevation_deg, resolution, operation)
        rendered_views = [{"azimuth_deg": float(azimuth_deg), "path": str(destination)}]
        for other in extra:
            side = destination.with_name(
                f"{destination.stem}_az{round(other) % 360:03d}.png"
            )
            rendered_views.append(
                {"azimuth_deg": float(other),
                 "path": str(self._render_one(
                     entries, side, center, radius, other, elevation_deg,
                     resolution, operation))}
            )
        return {
            "path": str(destination),
            "views": rendered_views,
            "bytes": destination.stat().st_size,
            "posed_links": sorted(poses),
            # Compatibility with SDK 1.1.0. Valid partial forests pose every
            # root at identity, so callers should use ``roots`` to diagnose
            # an assembly that is not connected yet.
            "unposed_links": sorted(set(state["links"]) - set(poses)),
            "roots": sorted(
                set(state["links"])
                - {joint["child"] for joint in state["joints"].values()}
            ),
            "camera": {
                "location": [float(v) for v in self._camera_for(
                    center, radius, azimuth_deg, elevation_deg)[0]],
                "lookat": [float(v) for v in center],
                "type": "PERSPECTIVE",
                "fov_deg": float(self._camera_for(
                    center, radius, azimuth_deg, elevation_deg)[1]),
                "fixed_across_sweep": fixed_frame is not None,
            },
        }

    def render_joint_sweep(
        self, joint: str, output_dir: str | os.PathLike, *,
        steps: int = 4, azimuth_deg: float = 45.0,
        elevation_deg: float = 25.0, resolution: int = 512, views: int = 2,
    ) -> list[dict]:
        """Render evenly spaced legal poses and always restore q=0.

        `views` cameras are spread evenly from `azimuth_deg` (2 -> the
        opposite side as well). One camera is not enough: the dataset has no
        shared front convention, so a fixed azimuth shows some objects' blank
        back, where the moving part is occluded and the frames are identical.
        """
        operation = "render_joint_sweep"
        state = self._load()
        if joint not in state["joints"]:
            fail(ErrorCode.UNKNOWN_HANDLE, operation, "unknown joint", joint=joint)
        definition = state["joints"][joint]
        if definition["type"] == "FIXED":
            fail(ErrorCode.INVALID_ARGUMENT, operation, "fixed joint has no motion sweep", joint=joint)
        count = int(steps)
        if count < 2 or count > 16:
            fail(ErrorCode.INVALID_ARGUMENT, operation, "steps must be between 2 and 16", steps=steps)
        if definition["type"] == "CONTINUOUS":
            lower, upper = 0.0, math.pi * 2.0
        else:
            lower, upper = float(definition["limit_lower"]), float(definition["limit_upper"])
        destination = self._output_path(output_dir, operation)
        destination.mkdir(parents=True, exist_ok=True)
        frames = []
        fixed_frame_installed = False
        try:
            # Fit one camera to the UNION envelope of all joint poses. Without
            # this, per-frame auto-framing creates fake camera motion/zoom.
            self.reset_joints()
            envelope_points = []
            for index in range(count):
                fraction = index / (count - 1)
                q = lower + (upper - lower) * fraction
                self.set_joint_positions({joint: q})
                pose_state = self._load()
                self._apply(pose_state, strict=False)
                for handle in sorted(pose_state["parts"]):
                    envelope_points.extend(
                        self._part(pose_state, handle, operation).world_corners()
                    )
            if not envelope_points:
                fail(
                    ErrorCode.AMBIGUOUS_GEOMETRY,
                    operation,
                    "nothing to render",
                )
            minimum = tuple(
                min(point[axis] for point in envelope_points)
                for axis in range(3)
            )
            maximum = tuple(
                max(point[axis] for point in envelope_points)
                for axis in range(3)
            )
            center = tuple((minimum[i] + maximum[i]) * 0.5 for i in range(3))
            diagonal = math.sqrt(
                sum((maximum[i] - minimum[i]) ** 2 for i in range(3))
            )
            self._fixed_render_frame = {
                "center": [float(v) for v in center],
                "radius": float(max(diagonal * 0.75, 0.5)),
            }
            fixed_frame_installed = True
            azimuths = [float(azimuth_deg)] + self._extra_view_azimuths(
                azimuth_deg, views, operation
            )
            # Group by camera, not by pose: a caller that forwards this list
            # verbatim to a validator then hands it one coherent rest->full
            # sequence per camera, instead of frames that alternate between
            # two viewpoints and read as motion that is not there.
            for order, az in enumerate(azimuths):
                for index in range(count):
                    fraction = index / (count - 1)
                    q = lower + (upper - lower) * fraction
                    self.set_joint_positions({joint: q})
                    stem = f"{joint}_{index:02d}_{round(fraction * 100):03d}pct"
                    frame = destination / (
                        f"{stem}.png" if order == 0
                        else f"{stem}_az{round(az) % 360:03d}.png"
                    )
                    rendered = self.render_preview(
                        frame, azimuth_deg=az, elevation_deg=elevation_deg,
                        resolution=resolution, views=1,
                    )
                    frames.append({
                        "q": q, "fraction": fraction,
                        "azimuth_deg": float(az), **rendered,
                    })
        finally:
            if fixed_frame_installed and hasattr(self, "_fixed_render_frame"):
                del self._fixed_render_frame
            self.reset_joints()
        return frames


def _contract_boundary(method):
    """Turn Python argument-type failures into structured SDK errors.

    An agent typo such as ``parent=["body"]`` or ``steps="four"`` raises a
    raw TypeError/ValueError deep inside the SDK. The sim worker would report
    that as an engine failure and the policy would abort the sample; as an
    ``invalid_argument`` rejection it is a normal, correctable result.
    """
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (AssemblyError, OSError):
            raise
        except (TypeError, ValueError) as exc:
            raise AssemblyError(
                ErrorCode.INVALID_ARGUMENT, method.__name__,
                f"{type(exc).__name__}: {exc}",
                details={"hint": "check argument types against ACTIONS.md"},
            ) from exc
    return wrapper


for _name, _member in list(vars(AssemblySDK).items()):
    if (
        not _name.startswith("_")
        and _name != "safe_call"
        and inspect.isfunction(_member)
    ):
        setattr(AssemblySDK, _name, _contract_boundary(_member))
del _name, _member


# ---- module-level helpers shared with the old runtime -------------------

def _finite_number(value, operation: str, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        fail(ErrorCode.INVALID_ARGUMENT, operation, f"{field_name} must be numeric",
             field=field_name, received=repr(value), cause=str(exc))
    if not math.isfinite(number):
        fail(ErrorCode.INVALID_ARGUMENT, operation, f"{field_name} must be finite",
             field=field_name, received=repr(value))
    return number


def _part_handle_list(value, operation: str) -> list[str]:
    if isinstance(value, str):
        handles = [value]
    elif isinstance(value, (list, tuple)):
        handles = list(value)
    else:
        fail(ErrorCode.INVALID_ARGUMENT, operation,
             "parts must be a handle string or a list/tuple of handles",
             received=repr(value))
    if not handles or not all(isinstance(handle, str) and handle for handle in handles):
        fail(ErrorCode.INVALID_ARGUMENT, operation, "part handles must be non-empty strings")
    if len(handles) != len(set(handles)):
        fail(ErrorCode.INVALID_ARGUMENT, operation, "part handles must be unique", parts=handles)
    return handles


_INSTANCES: dict[str, AssemblySDK] = {}


def resolve_state_path(globals_dict=None, environ=None) -> Path:
    """State file location: ASSEMBLY_STATE_PATH, else OUTPUT_DIR/.assembly_state.json."""
    g = globals_dict or {}
    env = environ if environ is not None else os.environ
    explicit = g.get("ASSEMBLY_STATE_PATH") or env.get("ASSEMBLY_STATE_PATH")
    if explicit:
        return Path(str(explicit)).resolve()
    output_dir = g.get("OUTPUT_DIR") or env.get("OUTPUT_DIR")
    if output_dir:
        return Path(str(output_dir)).resolve() / STATE_FILENAME
    raise RuntimeError(
        "cannot locate assembly state: set OUTPUT_DIR (injected runtime "
        "global) or ASSEMBLY_STATE_PATH"
    )


def get_assembly(globals_dict=None) -> AssemblySDK:
    """Process-wide AssemblySDK per state path (mesh cache survives re-exec)."""
    path = resolve_state_path(globals_dict)
    key = str(path)
    instance = _INSTANCES.get(key)
    if instance is None:
        instance = AssemblySDK(path)
        _INSTANCES[key] = instance
    return instance
