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
    return {'ok': True, 'checks': ['compare_views: paths, strings, default and explicit overlay, shading, signed displacement']}
