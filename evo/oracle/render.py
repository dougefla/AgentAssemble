"""Render GT and Pred frames for one (task, instance) into
`<pred_dir>/oracle_render/`.

Both renders use the SAME mesh source — the original parts/*.obj from
the dataset (= what GT's URDF references, = what executor imported). The
only thing that differs between the GT render and the Pred render is the
URDF (different link tree, different joint origins, different visual
origins). This eliminates all material drift between sides of the
comparison.

Output filenames (in `<pred_dir>/oracle_render/`):
    gt_rest_az<NNN>.png                       (4 azs)
    gt_<joint>_az<NNN>_<PCT>pct.png           (4 az × 3 pct per GT joint)
    pred_rest_az<NNN>.png                     (4 azs)
    pred_<joint>_az<NNN>_<PCT>pct.png         (4 az × 3 pct per pred joint)

Implementation: reuses evo.oracle.render_motion.render_pose() with
its BLENDER_SCRIPT (sin/cos azimuth formula, sun-light + white-bg
render params identical to skills/v_minimal/tools/headless_render.py).
For GT, `parts_map` is None → mesh_path from URDF is resolved relative
to gt_dir (which points at `dataset/.../gt/<tid>/<inst>/assets/*.obj`).
For Pred, `parts_map` v2 is loaded from <pred_dir>/parts_map.json. Each source
entry carries the agent-authored uniform scale so rendering matches exported
geometry. Legacy link→path-list maps remain readable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from evo.oracle.render_motion import parse_urdf_raw, render_backend, render_pose


def _sweep_frame(links, joints, base_dir, parts_map, joint_qs):
    if render_backend() != "lightphyx":
        return None
    from evo.oracle.render_lightphyx import sweep_frame
    return sweep_frame(links, joints, base_dir, parts_map, joint_qs)


def _render_one_side(
    *,
    out_dir: Path,
    out_prefix: str,
    urdf_path: Path,
    base_dir: Path,
    parts_map: dict | None,
    azs: list[int],
    force: bool,
    verbose: bool,
) -> int:
    """Render rest + motion (0/50/100 per joint per az) for one side
    (GT or Pred). Returns count of newly-rendered files.
    """
    if not urdf_path.exists():
        if verbose:
            print(f"  [{out_prefix}] URDF missing: {urdf_path}")
        return 0

    links, joints = parse_urdf_raw(urdf_path)
    movable = {
        n: j for n, j in joints.items()
        if j["type"] in ("REVOLUTE", "PRISMATIC", "CONTINUOUS")
    }

    n = 0
    for az in azs:
        rest_out = out_dir / f"{out_prefix}_rest_az{az:03d}.png"
        if force or not rest_out.exists():
            if render_pose(out_dir, links, joints, {}, base_dir, rest_out,
                           float(az), parts_map=parts_map):
                n += 1
        for jname, j in movable.items():
            lo, hi = j["lower"], j["upper"]
            if lo == 0 and hi == 0 and j["type"] == "CONTINUOUS":
                lo, hi = 0.0, 2 * math.pi
            steps = []
            for pct, label in [(0.0, "000"), (0.5, "050"), (1.0, "100")]:
                joint_q = {jn: 0.0 for jn in movable}
                joint_q[jname] = lo + pct * (hi - lo)
                steps.append((label, joint_q))
            # The judge compares these three frames to read motion; the
            # camera must not move between them.
            frame = _sweep_frame(links, joints, base_dir, parts_map, [q for _l, q in steps])
            for label, joint_q in steps:
                out = out_dir / f"{out_prefix}_{jname}_az{az:03d}_{label}pct.png"
                if force or not out.exists():
                    if render_pose(out_dir, links, joints, joint_q, base_dir,
                                   out, float(az), parts_map=parts_map, frame=frame):
                        n += 1
    if verbose:
        print(f"  [{out_prefix}] {n} new renders → {out_dir}")
    return n


def render_for_oracle(
    gt_dir: Path,
    pred_dir: Path,
    azs: list[int],
    *,
    force: bool = False,
    verbose: bool = False,
) -> tuple[Path, int]:
    """Render GT and Pred frames into `<pred_dir>/oracle_render/`.

    Args:
        gt_dir:   `dataset/.../gt/<tid>/<inst>/` — contains scene.urdf +
                  assets/part_*.obj (the GT meshes).
        pred_dir: `logs/runs/.../outputs/<tid>/<inst>/` — contains
                  furniture.urdf + parts_map.json (executor-written
                  link→original-part mapping) + meshes/<link>_visual.obj
                  (fallback).
        azs:      4 azimuths to render at (e.g. [45, 135, 225, 315]).
        force:    If True, re-render even when files exist.
        verbose:  Print per-side progress.

    Returns: (oracle_render_dir, n_rendered)
    """
    oracle_dir = pred_dir / "oracle_render"
    oracle_dir.mkdir(parents=True, exist_ok=True)

    # --- GT side ---
    gt_urdf = gt_dir / "scene.urdf"
    n_gt = _render_one_side(
        out_dir=oracle_dir,
        out_prefix="gt",
        urdf_path=gt_urdf,
        base_dir=gt_dir,
        parts_map=None,  # GT URDF's mesh_path resolves to assets/part_*.obj
        azs=azs,
        force=force,
        verbose=verbose,
    )

    # --- Pred side ---
    pred_urdf = pred_dir / "furniture.urdf"
    parts_map: dict = {}
    parts_map_path = pred_dir / "parts_map.json"
    if parts_map_path.exists():
        try:
            parts_map = json.loads(parts_map_path.read_text())
        except Exception as e:
            if verbose:
                print(f"  [pred] parts_map.json read error: {e}")
    # No GT-asset reconstruction fallback: anonymous task meshes and metric GT
    # meshes are in different coordinate systems. When a map is absent, render
    # the prediction's exported per-link meshes instead.
    n_pred = _render_one_side(
        out_dir=oracle_dir,
        out_prefix="pred",
        urdf_path=pred_urdf,
        base_dir=pred_dir,
        parts_map=parts_map,
        azs=azs,
        force=force,
        verbose=verbose,
    )
    return oracle_dir, n_gt + n_pred
