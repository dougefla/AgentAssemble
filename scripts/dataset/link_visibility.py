#!/usr/bin/env python3
"""Per-link visible pixel counts in each task thumbnail -> gt/<task>/<inst>/link_visibility.json.

Every link is rendered in its own flat colour (golden-angle hue by link index),
so the hue of a foreground pixel names its link. A link with (almost) no
visible pixels cannot have its orientation or size read off the reference
image; tasks with such links are tagged, not removed (see build_tiers.py).

    .venv/bin/python scripts/dataset/link_visibility.py ~/Dataset/artic300_rotscale
"""
from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MIN_SATURATION = 0.12
HIDDEN_PX = 60            # of 512 x 512: fewer visible pixels than this = hidden
SMALL_FRAC = 0.004        # of the object's foreground: barely visible


def _hue(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.zeros_like(mx)
    nz = d > 1e-9
    rc = np.where(nz, (mx - r) / np.where(nz, d, 1), 0)
    gc = np.where(nz, (mx - g) / np.where(nz, d, 1), 0)
    bc = np.where(nz, (mx - b) / np.where(nz, d, 1), 0)
    h = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    sat = np.where(mx > 1e-9, d / np.where(mx > 1e-9, mx, 1), 0)
    return h, sat


def task_visibility(task_dir: Path, gt_dir: Path) -> dict:
    meta = json.loads((gt_dir / "metadata.json").read_text())
    links = meta["links"]
    hues = np.array([colorsys.rgb_to_hsv(*l["color_rgb"])[0] for l in links])
    im = np.asarray(Image.open(task_dir / "thumbnail.png").convert("RGB")).astype(np.float64) / 255.0
    h, sat = _hue(im.reshape(-1, 3))
    fg = sat > MIN_SATURATION
    hp = h[fg]
    d = np.abs(hp[:, None] - hues[None, :])
    d = np.minimum(d, 1.0 - d)
    lab = d.argmin(1)
    counts = np.bincount(lab, minlength=len(links))
    total = int(fg.sum())
    out = {"image_px": int(im.shape[0] * im.shape[1]), "foreground_px": total, "links": {}}
    for l, c in zip(links, counts):
        frac = float(c) / total if total else 0.0
        out["links"][l["link"]] = {"px": int(c), "frac": round(frac, 5),
                                   "hidden": bool(c < HIDDEN_PX), "small": bool(frac < SMALL_FRAC)}
    out["hidden_links"] = sorted(k for k, v in out["links"].items() if v["hidden"])
    out["small_links"] = sorted(k for k, v in out["links"].items() if v["small"] and not v["hidden"])
    return out


def main() -> int:
    for root in map(Path, sys.argv[1:]):
        root = root.expanduser().resolve()
        n = hid = 0
        for gt in sorted((root / "gt").glob("task_*/*/metadata.json")):
            task, inst = gt.parent.parent.name, gt.parent.name
            vis = task_visibility(root / "tasks" / task / inst, gt.parent)
            (gt.parent / "link_visibility.json").write_text(json.dumps(vis, indent=1))
            n += 1
            hid += bool(vis["hidden_links"])
        print(f"{root.name}: {n} tasks, {hid} with at least one hidden link")
    return 0


if __name__ == "__main__":
    sys.exit(main())
