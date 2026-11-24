"""Deterministic local tests for skill helpers."""

def run_preflight(*, skill_dir):
    import tempfile
    from pathlib import Path
    from PIL import Image, ImageDraw
    import sys
    sys.path.insert(0, str(skill_dir))
    from tools.image_compare import compare_views
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)
        ref = Image.new('RGB', (160, 140), 'white')
        draw = ImageDraw.Draw(ref)
        draw.rectangle((20, 30, 80, 110), fill=(180, 55, 55))
        draw.rectangle((95, 45, 125, 65), fill=(55, 180, 55))
        ref.save(p / 'reference.png')
        pred = Image.new('RGB', (320, 280), 'white')
        draw = ImageDraw.Draw(pred)
        draw.rectangle((40, 60, 160, 220), fill=(240, 20, 20))
        draw.rectangle((210, 90, 270, 130), fill=(20, 240, 20))
        pred.save(p / 'preview.png')
        colors = {'base': [0.8, 0.2, 0.2], 'child': [0.2, 0.8, 0.2]}
        a = compare_views(p / 'reference.png', p / 'preview.png', colors, root='base')
        assert abs(a['preview_to_reference_scale'] - 0.5) < 0.015
        assert a['links']['child']['center_delta_px'][0] < -8
        b = compare_views(str(p / 'reference.png'), str(p / 'preview.png'), colors,
                          root='base', overlay_path=p / 'overlay.png')
        assert Path(b['overlay_path']).exists()
        assert b['root_span_residual'] < 0.015
    import numpy as np
    from tools.round_features import find_round_features
    from scipy.spatial.transform import Rotation
    # Capped cylinder: two circular end rims, under arbitrary rigid motion.
    count = 32
    theta = np.arange(count) * 2 * np.pi / count
    ring = np.c_[np.cos(theta), np.sin(theta), np.zeros(count)]
    v = np.vstack([ring, ring + [0, 0, 2], [[0, 0, 0], [0, 0, 2]]])
    f = []
    for i in range(count):
        j = (i + 1) % count
        f.extend([[i, j, i+count], [j, j+count, i+count],
                  [2*count, j, i], [2*count+1, i+count, j+count]])
    f = np.array(f, dtype=int)
    for angle in [[0, 0, 0], [.3, -.6, 1.2]]:
        rot = Rotation.from_euler('xyz', angle).as_matrix()
        shift = np.array([3., -2., .5])
        transformed = v @ rot.T + shift
        a = find_round_features(transformed, f)
        b = find_round_features(transformed.tolist(), f.tolist(), min_segments=8)
        assert len(a) == len(b) == 2
        for rim in a:
            assert abs(rim['radius'] - 1) < 1e-9
            assert abs(np.dot(rim['axis'], rot[:, 2])) > 1 - 1e-9
            assert rim['relative_fit_error'] < 1e-9
            c = np.array(rim['center'])
            assert min(np.linalg.norm(c - shift), np.linalg.norm(c - shift - 2*rot[:, 2])) < 1e-9
        # Independently duplicated vertices at every triangle still weld.
        assert len(find_round_features(transformed[f].reshape(-1, 3), np.arange(f.size).reshape(-1, 3))) == 2
    assert find_round_features(v, np.empty((0, 3), dtype=int)) == []
    assert find_round_features([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1]],
                               [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]) == []
    ellipse = v * [1, .5, 1]
    assert find_round_features(ellipse, f) == []
    for badv, badf, kw in [(v * np.nan, f, {}), (v, f.astype(float), {}),
                           (v, [[0, 1, len(v)]], {}), (v, f, {'min_segments': 7}),
                           (v, f, {'min_segments': True}), ([[0,0,0]], f, {})]:
        try:
            find_round_features(badv, badf, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid rim input accepted')
    return {'ok': True, 'checks': ['compare_views: paths, strings, default and explicit overlay, shading, signed displacement',
                                 'find_round_features: arrays/lists, defaults/explicit segments, rigid motion, seams, empty, ellipse rejection, invalid inputs']}
