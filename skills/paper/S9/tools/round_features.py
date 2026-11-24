"""Geometry-only evidence from closed, sharp circular mesh rims."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def find_round_features(vertices, faces, *, min_segments=12):
    """Return fitted circular sharp-edge loops, in the supplied mesh frame."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or not np.isfinite(v).all():
        raise ValueError('vertices must be finite N by 3 coordinates, N >= 3')
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError('faces must be an M by 3 integer array')
    if f.size and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError('face index outside vertices')
    if isinstance(min_segments, (bool, np.bool_)) or not isinstance(min_segments, (int, np.integer)) or min_segments < 8:
        raise ValueError('min_segments must be an integer >= 8')
    if not len(f):
        return []
    # OBJ seams often duplicate coordinates. Weld exact duplicates only.
    v, inverse = np.unique(v, axis=0, return_inverse=True)
    f = inverse[f]
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 0
    f, normals = f[good], normals[good] / lengths[good, None]
    if not len(f):
        return []
    edges = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    edge_normals = np.tile(normals, (3, 1))
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, edge_normals = edges[order], edge_normals[order]
    starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(edges)])
    sharp = counts == 1
    two = np.flatnonzero(counts == 2)
    dots = np.einsum('ij,ij->i', edge_normals[starts[two]], edge_normals[starts[two] + 1])
    sharp[two] = np.abs(dots) < np.cos(np.deg2rad(25))
    edges = edges[starts[sharp]]
    if not len(edges):
        return []
    ids, graph_edges = np.unique(edges, return_inverse=True)
    graph_edges = graph_edges.reshape(-1, 2)
    degree = np.bincount(graph_edges.ravel(), minlength=len(ids))
    rows = np.r_[graph_edges[:, 0], graph_edges[:, 1]]
    cols = np.r_[graph_edges[:, 1], graph_edges[:, 0]]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids)))
    _, labels = connected_components(graph, directed=False)
    grouped = np.argsort(labels, kind='stable')
    groups = np.split(grouped, np.flatnonzero(np.diff(labels[grouped])) + 1)
    features = []
    for group in groups:
        if len(group) < min_segments or np.any(degree[group] != 2):
            continue
        p = v[ids[group]]
        mean = p.mean(axis=0)
        _, singular, basis = np.linalg.svd(p - mean, full_matrices=False)
        if singular[1] <= singular[0] * 1e-8:
            continue
        xy = (p - mean) @ basis[:2].T
        solution = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], np.sum(xy * xy, axis=1), rcond=None)[0]
        radius_squared = solution[2] + solution[:2] @ solution[:2]
        if radius_squared <= 0:
            continue
        radius = float(np.sqrt(radius_squared))
        local = xy - solution[:2]
        plane = (p - mean) @ basis[2]
        error = float(np.sqrt(np.mean((np.linalg.norm(local, axis=1) - radius) ** 2 + plane ** 2)) / radius)
        angles = np.sort(np.arctan2(local[:, 1], local[:, 0]))
        gap = np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))
        if error > 0.025 or gap > np.pi / 4 + 1e-6:
            continue
        center = mean + solution[:2] @ basis[:2]
        axis = basis[2]
        axis *= 1 if axis[np.argmax(np.abs(axis))] >= 0 else -1
        features.append({'center': center.tolist(), 'axis': axis.tolist(), 'radius': radius,
                         'relative_fit_error': error, 'vertex_count': int(len(group)),
                         'origin_line_distance': float(np.linalg.norm(center - axis * (axis @ center)))})
    return sorted(features, key=lambda feature: -feature['radius'])
