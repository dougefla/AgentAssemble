"""Interchangeable (symmetric) parts of a GT object.

Two drawers of one chest, the two doors of a cabinet, four identical legs: a
prediction that puts part A where the reference has part B and vice versa is
the same physical object, so the oracle must not punish the swap. The link
matcher already prefers the rest position among *exactly* equal meshes, but
real repeats are only nearly equal (vertex order, a handle a few millimetres
off), and its geometry term then forces the by-name assignment.

This module names the equivalence classes explicitly:

- :func:`equivalence_classes` derives them from the GT geometry: links whose
  mesh signatures are within ``max_distance`` (the rigid-invariant
  :meth:`MeshSignature.distance`), whose physical scale differs by at most
  ``max_scale_diff``, and which are driven by the same joint type (a drawer
  and a door of the same shape are not interchangeable);
- :func:`derive_part_equivalence` confirms each candidate pair by registering
  the two world-baked surfaces (a rotation from the cube group plus the
  bounding-box centres; residual below a few millimetres) and records, for
  every ordered pair (X -> Y), the transform that puts X's mesh exactly where
  Y sits. The oracle needs it: its pose metrics are vertex-distribution
  moments, so comparing a prediction that holds mesh X against the reference
  link that holds mesh Y must be done with X's own vertex cloud moved into
  Y's slot, never with Y's cloud;
- :func:`write_part_equivalence` records classes and transforms as
  ``gt/<task>/<inst>/part_equivalence.json`` so the datasets carry the truth
  and anyone can inspect or edit it;
- :func:`load_part_equivalence` / :func:`load_part_equivalence_transforms`
  read that file back (``None`` when absent; the oracle then derives the
  classes on the fly, without transforms, which only covers vertex-identical
  swaps);
- :func:`candidate_matchings` enumerates the alternative GT<->Pred link
  assignments that permute members within a class, which
  :func:`evo.oracle.check_urdf.run_oracle` scores and resolves by "closest
  wins".

Stdlib only.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from evo.oracle.check_urdf import URDFModel

EQUIVALENCE_NAME = "part_equivalence.json"
DEFAULT_MAX_DISTANCE = 0.02      # MeshSignature.distance: 4*count_error + radial-shape error
DEFAULT_MAX_SCALE_DIFF = 0.02    # relative rms-radius difference
MAX_CANDIDATE_MATCHINGS = 240    # product of class permutations the oracle is willing to score


def equivalence_classes(
    gt: "URDFModel",
    *,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    max_scale_diff: float = DEFAULT_MAX_SCALE_DIFF,
    same_joint_type: bool = True,
) -> list[list[str]]:
    """Classes (size >= 2) of GT link names whose parts are interchangeable."""
    from evo.oracle.check_urdf import link_signature

    child_type = {j.child: j.type for j in gt.joints.values()}
    # Limits travel with the joint name in the task templates, so two parts
    # are only interchangeable when their joints also share the travel range
    # (mirror-image doors with opposite ranges are not).
    child_limits = {j.child: (j.limit_lower, j.limit_upper) for j in gt.joints.values()}
    sigs = {}
    for name, link in gt.visual_links.items():
        if link.mesh_paths:
            sigs[name] = link_signature(link, gt.base_dir)
    names = sorted(sigs)
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(names, 2):
        sa, sb = sigs[a], sigs[b]
        if sa.vertex_count <= 0 or sb.vertex_count <= 0:
            continue
        if same_joint_type and child_type.get(a) != child_type.get(b):
            continue
        if same_joint_type and not _same_limits(child_limits.get(a), child_limits.get(b)):
            continue
        scale = abs(sa.rms_radius - sb.rms_radius) / max(sa.rms_radius, sb.rms_radius, 1e-9)
        if sa.distance(sb) > max_distance or scale > max_scale_diff:
            continue
        # The radial moments cannot tell a 15 cm drawer from a 24 cm one of the
        # same footprint; the principal variances (rotation-invariant) can.
        ea, eb = _principal_variances(sa.covariance), _principal_variances(sb.covariance)
        if any(abs(x - y) / max(x, y, 1e-12) > 2 * max_scale_diff for x, y in zip(ea, eb)):
            continue
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def _same_limits(la, lb, *, rel: float = 0.05, abs_tol: float = 0.01) -> bool:
    """Equal travel ranges (both missing counts as equal)."""
    if la is None and lb is None:
        return True
    if la is None or lb is None:
        return False
    for x, y in zip(la, lb):
        if x is None and y is None:
            continue
        if x is None or y is None:
            return False
        if abs(x - y) > max(abs_tol, rel * max(abs(x), abs(y))):
            return False
    return True


def _principal_variances(cov) -> tuple[float, float, float]:
    """Eigenvalues of a symmetric 3x3 covariance, descending (closed form)."""
    import math
    a, b, c = cov[0][0], cov[1][1], cov[2][2]
    d, e, f = cov[0][1], cov[0][2], cov[1][2]
    p1 = d * d + e * e + f * f
    if p1 < 1e-30:
        return tuple(sorted((a, b, c), reverse=True))
    q = (a + b + c) / 3.0
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1
    p = math.sqrt(p2 / 6.0)
    # B = (A - qI) / p ; r = det(B) / 2
    B = [[(a - q) / p, d / p, e / p], [d / p, (b - q) / p, f / p], [e / p, f / p, (c - q) / p]]
    r = (B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1])
         - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
         + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0])) / 2.0
    r = max(-1.0, min(1.0, r))
    phi = math.acos(r) / 3.0
    e1 = q + 2 * p * math.cos(phi)
    e3 = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    e2 = 3 * q - e1 - e3
    return (e1, e2, e3)


DEFAULT_RESIDUAL_ABS = 0.005     # m: registration residual accepted for "congruent"
DEFAULT_RESIDUAL_REL = 0.01      # fraction of the part's bounding-box diagonal


def _world_vertices(gt: "URDFModel", link: str, world_pose) -> "list[tuple[float, float, float]]":
    from evo.oracle.check_urdf import _mat4_mul, _mat4_rotate_rpy, _mat4_translate, resolve_mesh_path
    info = gt.links[link]
    T = _mat4_mul(world_pose, _mat4_mul(_mat4_translate(info.visual_xyz), _mat4_rotate_rpy(info.visual_rpy)))
    out = []
    with open(resolve_mesh_path(info.mesh_paths[0], gt.base_dir)) as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = (float(t) for t in line.split()[1:4])
                out.append((T[0][0] * x + T[0][1] * y + T[0][2] * z + T[0][3],
                            T[1][0] * x + T[1][1] * y + T[1][2] * z + T[1][3],
                            T[2][0] * x + T[2][1] * y + T[2][2] * z + T[2][3]))
    return out


def derive_part_equivalence(
    gt_dir: Path,
    *,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    max_scale_diff: float = DEFAULT_MAX_SCALE_DIFF,
    residual_abs: float = DEFAULT_RESIDUAL_ABS,
    residual_rel: float = DEFAULT_RESIDUAL_REL,
) -> tuple[list[list[str]], dict[tuple[str, str], dict]]:
    """Classes confirmed by surface registration, plus the ordered-pair transforms.

    Dataset-build time only (uses numpy). For every pair that passes the
    moment prefilter, the world-baked (q=0) vertex clouds are registered with
    a cube-group rotation ``Q`` and the bounding-box centres; the pair is kept
    when the nearest-neighbour RMS residual is within tolerance. The returned
    transform for ``(X, Y)`` is ``L = T_Y(0)^-1 @ G @ T_X(0)`` with
    ``G = [Q | c_Y - Q c_X]``: applied in Y's link frame to X's link-frame mesh
    it reproduces Y's world surface.
    """
    import itertools as _it
    import numpy as np
    from evo.oracle.check_urdf import (
        _joint_positions_at_reference, compute_world_poses, find_urdf, parse_urdf,
    )

    gt_dir = Path(gt_dir)
    gt = parse_urdf(find_urdf(gt_dir), allow_duplicate_joint_names=True, repair_reversed_limits=True)
    prelim = equivalence_classes(gt, max_distance=max_distance, max_scale_diff=max_scale_diff)
    if not prelim:
        return [], {}
    world = compute_world_poses(gt, _joint_positions_at_reference(gt))
    rots = []
    for perm in _it.permutations(range(3)):
        for signs in _it.product([1.0, -1.0], repeat=3):
            M = np.zeros((3, 3))
            for i, (pi, sg) in enumerate(zip(perm, signs)):
                M[i, pi] = sg
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                rots.append(M)
    clouds: dict[str, np.ndarray] = {}
    for cls in prelim:
        for link in cls:
            if link in world:
                clouds[link] = np.asarray(_world_vertices(gt, link, world[link]), dtype=float)
    rng = np.random.default_rng(0)

    def sample(V: np.ndarray, n: int) -> np.ndarray:
        return V if len(V) <= n else V[rng.choice(len(V), n, replace=False)]

    def residual(A: np.ndarray, B: np.ndarray) -> float:
        a = sample(A, 1500); b = sample(B, 4000)
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
        return float(np.sqrt((d ** 2).mean()))

    transforms: dict[tuple[str, str], dict] = {}
    verified_pairs: list[tuple[str, str]] = []
    for cls in prelim:
        for X, Y in _it.combinations(cls, 2):
            if X not in clouds or Y not in clouds:
                continue
            A, B = clouds[X], clouds[Y]
            cX, cY = (A.min(0) + A.max(0)) / 2, (B.min(0) + B.max(0)) / 2
            A0, B0 = A - cX, B - cY
            diag = float(np.linalg.norm(B.max(0) - B.min(0)))
            tol = max(residual_abs, residual_rel * diag)
            # every rotation that registers within tolerance is a valid way to
            # swap the parts (a thin symmetric panel admits several); the best
            # comes first, the rest are alternatives the oracle may also try
            scored = sorted(((residual(A0 @ Q.T, B0), k, Q) for k, Q in enumerate(rots)), key=lambda x: (x[0], x[1]))
            good = [(r, Q) for r, _k, Q in scored if r <= tol][:8]
            if not good:
                continue
            verified_pairs.append((X, Y))
            TX, TY = np.asarray(world[X], dtype=float), np.asarray(world[Y], dtype=float)
            fwd, back = [], []
            for r, Q in good:
                G = np.eye(4); G[:3, :3] = Q; G[:3, 3] = cY - Q @ cX
                fwd.append(([[float(v) for v in row] for row in (np.linalg.inv(TY) @ G @ TX)], round(float(r), 6)))
                back.append(([[float(v) for v in row] for row in (np.linalg.inv(TX) @ np.linalg.inv(G) @ TY)], round(float(r), 6)))
            transforms[(X, Y)] = {"matrix": fwd[0][0], "residual_m": fwd[0][1],
                                  "alternatives": [{"matrix": m, "residual_m": r} for m, r in fwd[1:]]}
            transforms[(Y, X)] = {"matrix": back[0][0], "residual_m": back[0][1],
                                  "alternatives": [{"matrix": m, "residual_m": r} for m, r in back[1:]]}
    # union-find over verified pairs only
    parent = {n: n for c in prelim for n in c}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for X, Y in verified_pairs:
        parent[find(X)] = find(Y)
    groups: dict[str, list[str]] = {}
    for n in parent:
        groups.setdefault(find(n), []).append(n)
    classes = sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])
    # a class member pair without a direct transform (A~B, B~C verified, A~C not) is composed
    for cls in classes:
        for X, Y in _it.permutations(cls, 2):
            if (X, Y) in transforms:
                continue
            for Z in cls:
                if (X, Z) in transforms and (Z, Y) in transforms:
                    L = np.asarray(transforms[(Z, Y)]["matrix"]) @ np.asarray(transforms[(X, Z)]["matrix"])
                    transforms[(X, Y)] = {"matrix": [[float(v) for v in row] for row in L],
                                          "residual_m": max(transforms[(X, Z)]["residual_m"], transforms[(Z, Y)]["residual_m"]),
                                          "alternatives": [], "composed_via": Z}
                    break
    return classes, transforms


def write_part_equivalence(
    gt_dir: Path,
    *,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    max_scale_diff: float = DEFAULT_MAX_SCALE_DIFF,
    with_transforms: bool = True,
) -> list[list[str]]:
    """Write ``part_equivalence.json`` for ``gt_dir/scene.urdf``.

    With ``with_transforms`` (default) the classes are confirmed by surface
    registration and every ordered pair gets its transform
    (:func:`derive_part_equivalence`, needs numpy); otherwise only the moment
    prefilter classes are written.
    """
    from evo.oracle.check_urdf import find_urdf, parse_urdf

    gt_dir = Path(gt_dir)
    gt = parse_urdf(find_urdf(gt_dir), allow_duplicate_joint_names=True, repair_reversed_limits=True)
    if with_transforms:
        classes, transforms = derive_part_equivalence(gt_dir, max_distance=max_distance, max_scale_diff=max_scale_diff)
    else:
        classes, transforms = equivalence_classes(gt, max_distance=max_distance, max_scale_diff=max_scale_diff), {}
    parts = {name: Path(link.mesh_paths[0]).name for name, link in gt.visual_links.items() if link.mesh_paths}
    child_type = {j.child: j.type for j in gt.joints.values()}
    payload = {
        "convention": "links inside one class own interchangeable (congruent) parts; a prediction may place "
                      "any member where the reference has another member of the same class, and the oracle "
                      "scores the closest such assignment. transforms[X->Y] is a 4x4 in Y's link frame that "
                      "moves X's link-frame mesh onto Y's surface (T_Y(0)^-1 @ G @ T_X(0)); the oracle "
                      "compares a prediction holding mesh X at Y's slot against X's own vertex cloud moved "
                      "by it, so vertex sampling never enters the verdict",
        "criteria": {"max_signature_distance": max_distance, "max_relative_scale_diff": max_scale_diff,
                     "same_joint_type": True, "same_limits": True, "principal_variances_within": 2 * max_scale_diff,
                     "surface_registration": {"rotations": "cube group (24)", "residual_abs_m": DEFAULT_RESIDUAL_ABS,
                                              "residual_rel_diag": DEFAULT_RESIDUAL_REL} if with_transforms else None},
        "classes": [
            {"links": c, "parts": [parts.get(n) for n in c], "joint_type": child_type.get(c[0], "ROOT")}
            for c in classes
        ],
        "transforms": [
            {"from": X, "to": Y, **info} for (X, Y), info in sorted(transforms.items())
        ],
    }
    (gt_dir / EQUIVALENCE_NAME).write_text(json.dumps(payload, indent=2) + "\n")
    return classes


def load_part_equivalence(gt_dir: Path) -> list[list[str]] | None:
    """Classes recorded in ``gt_dir/part_equivalence.json``, or ``None`` when absent/unreadable."""
    path = Path(gt_dir) / EQUIVALENCE_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    classes = []
    for entry in payload.get("classes") or []:
        links = entry.get("links") if isinstance(entry, dict) else entry
        if isinstance(links, list) and len(links) > 1:
            classes.append([str(x) for x in links])
    return classes


def load_part_equivalence_transforms(gt_dir: Path) -> dict[tuple[str, str], list[list[list[float]]]]:
    """``{(X, Y): [4x4, ...]}`` recorded in ``part_equivalence.json``, best registration
    first and any near-tied alternatives after it (empty when absent)."""
    path = Path(gt_dir) / EQUIVALENCE_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[tuple[str, str], list[list[list[float]]]] = {}
    for entry in payload.get("transforms") or []:
        try:
            mats = [[[float(v) for v in row] for row in entry["matrix"]]]
            for alt in entry.get("alternatives") or []:
                mats.append([[float(v) for v in row] for row in alt["matrix"]])
            out[(str(entry["from"]), str(entry["to"]))] = mats
        except (KeyError, TypeError, ValueError):
            continue
    return out


def candidate_matchings(
    matching: dict[str, str],
    classes: list[list[str]],
    *,
    limit: int = MAX_CANDIDATE_MATCHINGS,
) -> list[dict[str, str]]:
    """Every GT->Pred assignment reachable from ``matching`` by permuting the
    predicted links among the matched members of each class. The original
    assignment comes first; the list is capped at ``limit`` entries (classes
    are processed in order, so the first classes get their full permutation
    set before later ones are frozen).
    """
    groups = [[g for g in c if g in matching] for c in classes]
    groups = [g for g in groups if len(g) > 1]
    results = [dict(matching)]
    for group in groups:
        preds = [matching[g] for g in group]
        expanded: list[dict[str, str]] = []
        for base in results:
            for perm in itertools.permutations(preds):
                cand = dict(base)
                for g, p in zip(group, perm):
                    cand[g] = p
                expanded.append(cand)
                if len(expanded) >= limit:
                    break
            if len(expanded) >= limit:
                break
        results = expanded
        if len(results) >= limit:
            break
    # de-duplicate while keeping order (identity permutation repeats the base)
    seen: set[tuple[tuple[str, str], ...]] = set()
    unique: list[dict[str, str]] = []
    for cand in results:
        key = tuple(sorted(cand.items()))
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique[:limit]
