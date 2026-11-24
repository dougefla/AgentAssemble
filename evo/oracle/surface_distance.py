"""Surface distance between two posed meshes (oracle 3).

The pose checks of oracle 1.x/2.0 compared vertex moments (centroid,
covariance, third moment) and converted the covariance / third-moment
differences back to metres with a square / cube root. Two consequences:

* a part rotated about its own symmetry axis (a lazy Susan, a wheel, a knob)
  is the same solid but has different vertex moments wherever the tessellation
  is not perfectly regular, so physically identical answers failed;
* the root conversion is non-linear: a 0.3 % size residual became centimetres.

This module measures what the checks mean: how far the *surface* of the
predicted link is from the surface of the ground-truth link, in metres, as a
symmetric (two-sided) point-to-triangle distance. It is invariant to any
symmetry of the part, to tessellation, to vertex order and to how a transform
is split between the mesh vertices, the visual origin and the joint origin.

Needs numpy and scipy; ``AVAILABLE`` is False without them and the oracle then
falls back to the moment descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:  # optional: the legacy oracle path is stdlib-only
    import numpy as np
    from scipy.spatial import cKDTree
    AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without numpy/scipy
    np = None
    cKDTree = None
    AVAILABLE = False

# Field (target side): dense area-weighted samples that index their triangle.
FIELD_SAMPLES_MIN = 6000
FIELD_SAMPLES_MAX = 40000
FIELD_NEIGHBOURS = 8
# Probe (source side): every vertex up to a cap plus area-weighted samples.
PROBE_VERTICES_MAX = 1500
PROBE_SAMPLES = 1000
_SEED = 20261112


def _read_obj(path: Path):
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                if len(p) >= 4:
                    try:
                        verts.append((float(p[1]), float(p[2]), float(p[3])))
                    except ValueError:
                        continue
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    try:
                        i = int(tok.split("/")[0])
                    except ValueError:
                        idx = []
                        break
                    idx.append(i)
                for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                    faces.append((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    n = len(v)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f):
        f = np.where(f < 0, f + n, f - 1)  # OBJ is 1-based; negatives are relative
        f = f[((f >= 0) & (f < n)).all(axis=1)]
    return v, f


@dataclass
class MeshField:
    """One mesh in its own frame, ready for point-to-surface queries."""
    vertices: "np.ndarray"          # (n, 3)
    faces: "np.ndarray"             # (m, 3) or empty (point cloud)
    probes: "np.ndarray"            # (p, 3) points on the surface, mesh frame
    _samples_tree: object
    _sample_face: "np.ndarray"
    diagonal: float

    def distance(self, pts: "np.ndarray") -> "np.ndarray":
        """Unsigned distance of each point (mesh frame) to the surface."""
        if len(pts) == 0:
            return np.zeros(0)
        if len(self.faces) == 0:
            return self._samples_tree.query(pts)[0]
        k = min(FIELD_NEIGHBOURS, len(self._sample_face))
        _, idx = self._samples_tree.query(pts, k=k)
        if k == 1:
            idx = idx[:, None]
        tri = self.vertices[self.faces[self._sample_face[idx]]]      # (q, k, 3, 3)
        d = _point_triangle_distance(pts[:, None, :], tri[..., 0, :], tri[..., 1, :], tri[..., 2, :])
        return d.min(axis=1)


def _point_triangle_distance(p, a, b, c):
    """Distance from points p to triangles (a, b, c); broadcasts, Ericson 5.1.5."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    eps = 1e-30

    # Interior of the face (default), then override with edge / vertex regions.
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < eps, eps, denom)
    v = vb / denom
    w = vc / denom
    closest = a + ab * v[..., None] + ac * w[..., None]

    def put(mask, value):
        nonlocal closest
        closest = np.where(mask[..., None], value, closest)

    # Edge BC
    t_bc = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < eps, eps, (d4 - d3) + (d5 - d6))
    put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + (c - b) * t_bc[..., None])
    # Edge AC
    t_ac = d2 / np.where(np.abs(d2 - d6) < eps, eps, d2 - d6)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * t_ac[..., None])
    # Edge AB
    t_ab = d1 / np.where(np.abs(d1 - d3) < eps, eps, d1 - d3)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * t_ab[..., None])
    # Vertices
    put((d6 >= 0) & (d5 <= d6), c + np.zeros_like(closest))
    put((d3 >= 0) & (d4 <= d3), b + np.zeros_like(closest))
    put((d1 <= 0) & (d2 <= 0), a + np.zeros_like(closest))
    return np.sqrt(((p - closest) ** 2).sum(-1))


def _area_samples(v, f, n, rng):
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.sqrt((cross ** 2).sum(-1))
    total = float(area.sum())
    if total <= 0 or n <= 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    face = rng.choice(len(f), size=n, p=area / total)
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    t = tri[face]
    pts = (1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1] + (r1 * r2)[:, None] * t[:, 2]
    return pts, face


_FIELD_CACHE: dict[tuple, MeshField | None] = {}


def mesh_field(path: Path, scale: tuple[float, float, float] | None = None) -> MeshField | None:
    """Cached :class:`MeshField` for an OBJ with a URDF ``<mesh scale>`` applied."""
    path = Path(path)
    try:
        st = path.stat()
        key = (str(path.resolve()), st.st_mtime_ns, st.st_size, tuple(scale) if scale else None)
    except OSError:
        return None
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]
    try:
        v, f = _read_obj(path)
    except OSError:
        v, f = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    field: MeshField | None = None
    if len(v):
        if scale is not None:
            v = v * np.asarray(scale, dtype=np.float64)
        rng = np.random.default_rng(_SEED)
        ref = v[np.unique(f)] if len(f) else v
        diag = float(np.linalg.norm(ref.max(0) - ref.min(0)))
        if len(f):
            n_field = int(min(FIELD_SAMPLES_MAX, max(FIELD_SAMPLES_MIN, 4 * len(f))))
            s_pts, s_face = _area_samples(v, f, n_field, rng)
            cen = v[f].mean(axis=1)                       # every face is reachable
            s_pts = np.vstack([s_pts, cen])
            s_face = np.concatenate([s_face, np.arange(len(f))])
            p_extra, _ = _area_samples(v, f, PROBE_SAMPLES, rng)
        else:
            s_pts, s_face = v, np.zeros(len(v), dtype=np.int64)
            p_extra = np.zeros((0, 3))
        # Only vertices a face uses are surface points; OBJ exporters leave
        # stray unreferenced vertices that must not be probed.
        used = v[np.unique(f)] if len(f) else v
        if len(used) > PROBE_VERTICES_MAX:
            pick = np.linspace(0, len(used) - 1, PROBE_VERTICES_MAX).round().astype(np.int64)
            # A rigid or scale error is largest at the extremities, so the
            # support vertices along 26 directions are always probed.
            dirs = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                             if (x, y, z) != (0, 0, 0)], dtype=np.float64)
            extreme = np.unique((used @ dirs.T).argmax(axis=0))
            pv = used[np.unique(np.concatenate([pick, extreme]))]
        else:
            pv = used
        field = MeshField(
            vertices=v, faces=f if len(s_pts) else np.zeros((0, 3), dtype=np.int64),
            probes=np.vstack([pv, p_extra]), _samples_tree=cKDTree(s_pts if len(s_pts) else v),
            _sample_face=s_face, diagonal=diag,
        )
    if len(_FIELD_CACHE) >= 96:      # a field holds a KD-tree: keep the cache small
        _FIELD_CACHE.clear()
        _PAIR_CACHE.clear()
    _FIELD_CACHE[key] = field
    return field


def _as_rt(m) -> tuple["np.ndarray", "np.ndarray"]:
    a = np.asarray(m, dtype=np.float64)
    return a[:3, :3], a[:3, 3]


_PAIR_CACHE: dict[tuple, tuple[float, float]] = {}


def posed_surface_distance(
    gt_field: MeshField, gt_world, pred_field: MeshField, pred_world,
    scale: float = 1.0, translation=(0.0, 0.0, 0.0),
) -> tuple[float, float]:
    """(max, 95th percentile) two-sided surface distance in GT metres.

    ``gt_world`` / ``pred_world`` are the 4x4 mesh-frame-to-world transforms;
    prediction world coordinates are first mapped by ``p_gt = scale*p + translation``.
    """
    Rg, tg = _as_rt(gt_world)
    Rp, tp = _as_rt(pred_world)
    t = np.asarray(translation, dtype=np.float64)
    # pred mesh frame -> GT mesh frame:  x_g = A x_p + b   (A is scale * rotation
    # when both transforms are rigid; computed generally via a linear solve).
    Rg_inv = np.linalg.inv(Rg)
    A = Rg_inv @ (scale * Rp)
    b = Rg_inv @ (scale * tp + t - tg)
    key = (id(gt_field), id(pred_field), tuple(np.round(A, 7).ravel()), tuple(np.round(b, 7)))
    hit = _PAIR_CACHE.get(key)
    if hit is not None:
        return hit
    d_pg = gt_field.distance(pred_field.probes @ A.T + b)
    # GT probes into the pred mesh frame; distances come back in pred units.
    A_inv = np.linalg.inv(A)
    d_gp = pred_field.distance((gt_field.probes - b) @ A_inv.T)
    # Length scale of the pred frame relative to GT metres (uniform for rigid+scale).
    unit = float(np.cbrt(abs(np.linalg.det(A)))) or 1.0
    d_gp = d_gp * unit
    # A's rotation part may carry a non-uniform Rg scale in theory; distances
    # measured in the GT mesh frame are GT metres when Rg is a rotation.
    g_unit = float(np.cbrt(abs(np.linalg.det(Rg)))) or 1.0
    both = np.concatenate([d_pg * g_unit, d_gp * g_unit])
    out = (float(both.max()), float(np.percentile(both, 95)))
    if len(_PAIR_CACHE) >= 20000:
        _PAIR_CACHE.clear()
    _PAIR_CACHE[key] = out
    return out


def similarity_fit(src: "np.ndarray", dst: "np.ndarray") -> tuple[float, "np.ndarray", "np.ndarray"]:
    """Umeyama: scale s, rotation R, translation t with dst ~= s R src + t (index-paired)."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = float((xs ** 2).sum() / len(src))
    s = float((S * np.diag(D)).sum() / var) if var > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def reset_caches(*, fields: bool = False) -> None:
    """Drop the per-evaluation pair cache (its keys hold object ids, so it must
    not outlive the fields it refers to); optionally the mesh fields too."""
    _PAIR_CACHE.clear()
    if fields:
        _FIELD_CACHE.clear()
