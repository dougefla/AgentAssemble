"""Root-registered colour diagnostics; no simulator dependencies."""
from pathlib import Path
import numpy as np
from PIL import Image


def _directions(rgb):
    x = rgb[..., 0] - rgb[..., 1]
    y = (rgb[..., 0] + rgb[..., 1] - 2 * rgb[..., 2]) / np.sqrt(3.0)
    v = np.stack([x, y], axis=-1)
    length = np.linalg.norm(v, axis=-1)
    return v / np.maximum(length[..., None], 1e-12), length


def _regions(image, palette):
    rgb = np.asarray(image.convert('RGB'), dtype=float) / 255
    direction, chroma = _directions(rgb)
    pd, pc = _directions(palette)
    if np.any(pc < 0.03):
        raise ValueError('Palette colours must be chromatic (not grey).')
    similarities = direction @ pd.T
    label = np.argmax(similarities, axis=-1)
    valid = (chroma > 0.035) & (np.max(similarities, axis=-1) > np.cos(np.deg2rad(25)))
    regions = []
    for j in range(len(palette)):
        y, x = np.nonzero(valid & (label == j))
        if len(x) < 8:
            regions.append(None)
            continue
        lo = np.quantile(np.column_stack([x, y]), 0.01, axis=0)
        hi = np.quantile(np.column_stack([x, y]), 0.99, axis=0)
        regions.append({'bounds': np.array([lo, hi]), 'pixels': int(len(x))})
    return regions


def compare_views(reference_path, preview_path, colors, *, root, overlay_path=None):
    """Compare coloured regions after isotropic registration of the root."""
    if not isinstance(colors, dict) or not colors or root not in colors:
        raise ValueError('colors must be a nonempty name->RGB dict containing root.')
    names = list(colors)
    if any(not isinstance(n, str) for n in names):
        raise ValueError('Colour keys must be strings.')
    try:
        palette = np.asarray([colors[n] for n in names], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Each colour must be a numeric RGB triple in [0,1].') from exc
    if palette.shape != (len(names), 3) or not np.isfinite(palette).all() or np.any((palette < 0) | (palette > 1)):
        raise ValueError('Each colour must be a finite RGB triple in [0,1].')
    ref = Image.open(reference_path).convert('RGB')
    pred = Image.open(preview_path).convert('RGB')
    rr, pr = _regions(ref, palette), _regions(pred, palette)
    j = names.index(root)
    if rr[j] is None or pr[j] is None:
        raise ValueError('Root colour is absent or has fewer than eight pixels.')
    rb, pb = rr[j]['bounds'], pr[j]['bounds']
    rs, ps = rb[1] - rb[0], pb[1] - pb[0]
    if np.any(rs < 2) or np.any(ps < 2):
        raise ValueError('Root colour region is degenerate.')
    scale = float(np.dot(rs, ps) / np.dot(ps, ps))
    offset = rb.mean(0) - scale * pb.mean(0)
    error = float(np.linalg.norm(scale * ps - rs) / np.linalg.norm(rs))
    links = {}
    for name, r, p in zip(names, rr, pr):
        if r is None or p is None:
            links[name] = {'visible_in_reference': r is not None, 'visible_in_preview': p is not None}
            continue
        rbox, pbox = r['bounds'], scale * p['bounds'] + offset
        delta = rbox.mean(0) - pbox.mean(0)
        links[name] = {
            'reference_bounds_px': rbox.tolist(),
            'registered_preview_bounds_px': pbox.tolist(),
            'center_delta_px': delta.tolist(),
            'center_error_root_fraction': float(np.linalg.norm(delta) / np.linalg.norm(rs)),
            'reference_over_preview_span': ((rbox[1] - rbox[0]) / np.maximum(pbox[1] - pbox[0], 1e-9)).tolist(),
            'visible_in_reference': True, 'visible_in_preview': True,
        }
    output = None
    if overlay_path is not None:
        output = str(Path(overlay_path))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        aligned = pred.transform(ref.size, Image.Transform.AFFINE,
                                 (1 / scale, 0, -offset[0] / scale, 0, 1 / scale, -offset[1] / scale),
                                 resample=Image.Resampling.BILINEAR, fillcolor='white')
        panel = Image.new('RGB', (3 * ref.width, ref.height), 'white')
        panel.paste(ref, (0, 0))
        panel.paste(aligned, (ref.width, 0))
        panel.paste(Image.blend(ref, aligned, 0.5), (2 * ref.width, 0))
        panel.save(output)
    return {'preview_to_reference_scale': scale, 'preview_to_reference_offset_px': offset.tolist(),
            'root_span_residual': error, 'links': links, 'overlay_path': output}
