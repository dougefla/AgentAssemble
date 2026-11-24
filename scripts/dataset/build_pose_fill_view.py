#!/usr/bin/env python3
"""Build the `artvip_posefill` view from `artvip_beginner`.

The root link's part (the base) is kept exactly as in `artvip_beginner`: world
coordinates, visual origin zero, given in the template. It anchors the world
frame, which the oracle needs because its geometry checks align translation and
scale only, never rotation. Every other task mesh is translated so that its
axis-aligned bounding-box center sits at the origin, then rotated by a random
rigid rotation about that origin (uniform over SO(3), seeded per part;
`--no-rotate` keeps the assembled orientation). Where those parts sit and how
they are oriented is therefore no longer in the mesh: it lives in the URDF, and
the task template blanks it out together with the joints. The solver fills

  * every non-root link's visual <origin xyz> and <origin rpy>   (part pose onto the base)
  * every joint's <origin xyz> and <axis xyz>

while link names, the link <-> part mapping, the kinematic tree, joint types and
limits are given. Compared with `artvip_jointfill` this puts placement back into
the task without leaking the pivot through the mesh origin (the ArtVIP link-frame
assets have their origin *on* the joint, which is why that view was rejected).

Layout (mirrors artvip_beginner):
  <out>/tasks/<task>/<inst>/parts/part_<i>.obj (+ .mtl, textures/)  <- centered
  <out>/tasks/<task>/<inst>/parts/template.urdf                      <- NEW
  <out>/tasks/<task>/<inst>/thumbnail.png                            <- copied
  <out>/gt/<task>/<inst>/part_transforms.json                        <- NEW, oracle-only
  <out>/gt/, gt_renders/, splits.json, mapping.json                  <- copied unchanged
  <out>/dataset_variant.json, README.md, verification.json

`task_mesh_transforms` deliberately does not contain `world_bake_q0`, so the
Assembly SDK imports these parts in `link` mode (nothing is placed until the
solver says so; `set_link_visual_pose` is allowed).

Verification (--verify, default on), per instance:
  1. leak scan: no GT origin/axis number, no subtracted center and no visual
     rpy of the answer appears in the template;
  2. GT refill: the template filled with the reference values
     (`evo.oracle.template_fill.posefill_values`), staged by `stage_filled_urdf`,
     must PASS the oracle;
  3. zero fill (visual xyz/rpy 0, joint origin 0, axis +z): the floor.
     Reported, not required.

Usage:
    .venv/bin/python scripts/dataset/build_pose_fill_view.py \
        --source ~/Dataset/artvip_beginner --output ~/Dataset/artvip_posefill
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evo.oracle.check_urdf import JointReport, Thresholds, run_oracle  # noqa: E402
from evo.oracle.template_fill import (  # noqa: E402
    TEMPLATE_NAME, build_template, fill_template, gt_fill_values, leaked_numbers,
    posefill_values, stage_filled_urdf, template_root_link,
)
import xml.etree.ElementTree as ET  # noqa: E402

SCORED_JOINT_DIMS = ["axis_ok", "pivot_ok", "motion_ok"]
# Object-level checks that count for score/pass: everything the template does
# not give (placement, pivots, axes, motion) plus limits and axes as filled.
SCORED_CHECKS = ["joint_axes", "joint_limits", "rest_pose_alignment", "joint_pivot_alignment",
                 "kinematic_equivalence"]
TRANSFORMS_NAME = "part_transforms.json"
TRANSFORM_TAG = "world_bake_q0_then_aabb_center_random_rotation"   # must NOT equal the SDK's world marker
TRANSFORM_TAG_NOROT = "world_bake_q0_then_aabb_center"
DEFAULT_SEED = 42
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------

def aabb_center(obj_path: Path) -> list[float]:
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                xyz = [float(t) for t in line.split()[1:4]]
                for i in range(3):
                    lo[i] = min(lo[i], xyz[i])
                    hi[i] = max(hi[i], xyz[i])
    if lo[0] == float("inf"):
        raise ValueError(f"{obj_path}: no vertices")
    return [(lo[i] + hi[i]) / 2.0 for i in range(3)]


def random_rotation(rng: random.Random) -> list[list[float]]:
    """Uniform random rotation matrix (Shoemake's quaternion method)."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q = (math.sqrt(1 - u1) * math.sin(2 * math.pi * u2), math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
         math.sqrt(u1) * math.sin(2 * math.pi * u3), math.sqrt(u1) * math.cos(2 * math.pi * u3))
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _apply(R: list[list[float]], v: list[float]) -> list[float]:
    return [sum(R[i][k] * v[k] for k in range(3)) for i in range(3)]


def center_obj(src: Path, dst: Path, center: list[float], rotation: list[list[float]] | None = None) -> int:
    """Rewrite ``v`` lines as ``R (v - center)`` and ``vn`` lines as ``R n``; everything else is kept."""
    R = rotation or IDENTITY
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.split()
                xyz = _apply(R, [float(t) - c for t, c in zip(parts[1:4], center)])
                rest = parts[4:]
                fout.write("v " + " ".join(f"{v:.6f}" for v in xyz) + ("" if not rest else " " + " ".join(rest)) + "\n")
                n += 1
            elif line.startswith("vn "):
                parts = line.split()
                nrm = _apply(R, [float(t) for t in parts[1:4]])
                fout.write("vn " + " ".join(f"{v:.6f}" for v in nrm) + "\n")
            else:
                fout.write(line)
    return n


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def plan_transforms(gt_urdf: Path, parts_dir: Path, *, rotate: bool, seed: int, tag: str) -> tuple[str, dict[str, dict]]:
    """``(root_part, {part: {center, rotation}})`` for one instance.

    The root link's part is anchored (center 0, identity); every other part is
    centered on its AABB center and, with ``rotate``, given a seeded uniform
    random rotation (``seed`` and ``tag`` make it deterministic per part).
    """
    root_link = template_root_link(gt_urdf)
    root_part = None
    for link_el in ET.parse(gt_urdf).getroot().findall("link"):
        mesh_el = link_el.find("visual/geometry/mesh")
        if link_el.attrib["name"] == root_link and mesh_el is not None:
            root_part = Path(mesh_el.attrib["filename"]).name
    transforms: dict[str, dict] = {}
    for obj in sorted(parts_dir.glob("part_*.obj")):
        if obj.name == root_part:
            transforms[obj.name] = {"center": [0.0, 0.0, 0.0], "rotation": IDENTITY, "anchored": True}
            continue
        c = [round(v, 6) for v in aabb_center(obj)]
        R = random_rotation(random.Random(f"{seed}:{tag}/{obj.name}")) if rotate else IDENTITY
        transforms[obj.name] = {"center": c, "rotation": [[round(v, 9) for v in row] for row in R], "anchored": False}
    if root_part is None or root_part not in transforms:
        raise ValueError(f"{gt_urdf}: root link {root_link!r} has no part in {parts_dir}")
    return root_part, transforms


def build_instance(src: Path, out: Path, task: str, inst: str, *, rotate: bool, seed: int) -> dict:
    gt_urdf = src / "gt" / task / inst / "scene.urdf"
    src_task = src / "tasks" / task / inst
    out_task = out / "tasks" / task / inst
    if out_task.exists():
        shutil.rmtree(out_task)
    shutil.copytree(src_task, out_task)
    root_part, transforms = plan_transforms(gt_urdf, src_task / "parts", rotate=rotate, seed=seed, tag=f"{task}/{inst}")
    for name, tf in transforms.items():
        if not tf["anchored"]:
            center_obj(src_task / "parts" / name, out_task / "parts" / name, tf["center"], tf["rotation"])
    gt_out = out / "gt" / task / inst
    gt_out.mkdir(parents=True, exist_ok=True)
    (gt_out / TRANSFORMS_NAME).write_text(json.dumps({
        "convention": "task mesh v' = rotation @ (v - center), v = world-baked vertex at source q=0; "
                      "answer: visual xyz = center - link world origin, visual rpy = rpy(rotation^T); "
                      "the anchored root part is untouched (world coordinates, visual origin 0)",
        "root_part": root_part,
        "seed": seed if rotate else None,
        "parts": transforms,
    }, indent=2) + "\n")
    template = build_template(gt_urdf, blank_visual_origins=True, header=False)   # bare URDF, no prompt text
    (out_task / "parts" / TEMPLATE_NAME).write_text(template)
    leaks = leaked_numbers(template, gt_urdf)
    present = set(_NUMBER_RE.findall(template))
    _joints, visuals = posefill_values(gt_urdf, transforms)
    for name, tf in transforms.items():
        for v in tf["center"]:
            tok = f"{v:.6f}"
            if abs(v) > 1e-9 and tok in present:
                leaks.append(f"{name}: center {tok}")
    for link, vis in visuals.items():
        for v in vis["rpy"]:
            tok = f"{v:.6f}"
            if abs(v) > 1e-9 and tok in present:
                leaks.append(f"{link}: rpy {tok}")
    return {"task": f"{task}/{inst}", "leaks": leaks, "n_parts": len(transforms)}


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def _oracle(gt_dir: Path, template: str, joint_values: dict, visual_xyz: dict, parts: Path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        filled = tmp / "filled.urdf"
        filled.write_text(fill_template(template, joint_values, visual_xyz))
        stage_filled_urdf(filled, tmp / "pred", mesh_dir=parts)
        result = run_oracle(gt_dir, tmp / "pred", Thresholds())
    checks = {c.name: c.passed for c in result.checks if c.scored}
    joints = {name: rep.flags() for name, rep in (result.joints or {}).items()}
    return result.passed, checks, joints


def verify_instance(out: Path, task: str, inst: str) -> dict:
    gt_dir = out / "gt" / task / inst
    parts = out / "tasks" / task / inst / "parts"
    template = (parts / TEMPLATE_NAME).read_text()
    transforms = json.loads((gt_dir / TRANSFORMS_NAME).read_text())["parts"]
    joints, visuals = posefill_values(gt_dir / "scene.urdf", transforms)
    ok, checks, _ = _oracle(gt_dir, template, joints, visuals, parts)
    zero_j = {j: {"origin_xyz": "0 0 0", "origin_rpy": "0 0 0", "axis": "0 0 1"} for j in gt_fill_values(gt_dir / "scene.urdf")}
    zero_v = {link: {"xyz": "0 0 0", "rpy": "0 0 0"} for link in visuals}
    zero_ok, zero_checks, zero_joints = _oracle(gt_dir, template, zero_j, zero_v, parts)
    return {
        "task": f"{task}/{inst}",
        "gt_refill_pass": ok,
        "gt_refill_failed_checks": sorted(k for k, v in checks.items() if not v),
        "zero_fill_pass": zero_ok,
        "zero_fill_failed_checks": sorted(k for k, v in zero_checks.items() if not v),
        "zero_fill_joint_dims": zero_joints,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("~/Dataset/artvip_beginner").expanduser())
    ap.add_argument("--output", type=Path, default=Path("~/Dataset/artvip_posefill").expanduser())
    ap.add_argument("--limit", type=int, default=None, help="only the first N instances (smoke)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--clean", action="store_true", help="delete --output first")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base seed of the per-part rotations")
    ap.add_argument("--no-rotate", action="store_true", help="translation only (keep the assembled orientation)")
    args = ap.parse_args(argv)
    rotate = not args.no_rotate

    src, out = args.source.resolve(), args.output.resolve()
    variant_path = src / "dataset_variant.json"
    if not (src / "gt").is_dir() or not (src / "tasks").is_dir() or not variant_path.is_file():
        ap.error(f"{src} does not look like a dataset root (needs gt/, tasks/, dataset_variant.json)")
    src_variant = json.loads(variant_path.read_text())
    if "world_bake_q0" not in src_variant.get("task_mesh_transforms", []):
        ap.error(f"{src} is not a world-baked view; build artvip_beginner first")
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    instances = sorted(
        (p.parent.name, p.name) for p in (src / "gt").glob("task_*/*") if (p / "scene.urdf").is_file())
    if args.limit:
        instances = instances[: args.limit]
    print(f"building {len(instances)} instances: {src} -> {out}")

    for name in ("gt", "gt_renders"):
        if (src / name).is_dir():
            shutil.copytree(src / name, out / name, dirs_exist_ok=True)
    # Interchangeable-part classes travel with the GT (oracle-only); derive
    # them here so a rebuilt view never ships without them.
    from evo.oracle.part_equivalence import write_part_equivalence
    for gt_dir in sorted(p.parent for p in (out / "gt").glob("task_*/*/scene.urdf")):
        write_part_equivalence(gt_dir)
    for name in ("splits.json", "mapping.json", "README.artvip_full.md"):
        if (src / name).is_file():
            shutil.copy2(src / name, out / name)

    leaks: list[str] = []
    for i, (task, inst) in enumerate(instances, 1):
        rec = build_instance(src, out, task, inst, rotate=rotate, seed=args.seed)
        leaks.extend(f"{rec['task']}: {x}" for x in rec["leaks"])
        if i % 20 == 0 or i == len(instances):
            print(f"  built {i}/{len(instances)}")
    for name in os.listdir(out / "tasks"):
        for inst_dir in (out / "tasks" / name).iterdir():
            assert not (inst_dir / "bake_transforms.json").exists(), inst_dir
            assert not (inst_dir / TRANSFORMS_NAME).exists(), inst_dir

    variant = {
        "dataset": "artvip_posefill",
        "source": str(src),
        "source_dataset": src.name,
        "dataset_version": 1,
        "task_mesh_policy": "root part kept as the world-baked artvip_beginner mesh (base anchors the world frame); "
                            "every other part translated so its AABB center is at the origin"
                            + (", then rotated by a per-part uniform random rotation" if rotate else
                               " (orientation unchanged)")
                            + f"; the applied transforms are oracle-only (gt/<task>/<inst>/{TRANSFORMS_NAME})",
        "task_mesh_transforms": [TRANSFORM_TAG if rotate else TRANSFORM_TAG_NOROT],
        "random_rotation": {"per_part": True, "seed": args.seed, "distribution": "uniform SO(3)"} if rotate else None,
        "mesh_frame": "part_centered",
        "task_template": f"parts/{TEMPLATE_NAME}",
        "given": ["link names and link <-> part mapping", "kinematic tree (parent/child)",
                  "joint types", "joint limits", "root part in place (visual origin 0, world-frame mesh)"],
        "hidden": ["visual origin xyz of every non-root link (part placement)",
                   "visual origin rpy of every non-root link (part orientation)",
                   "joint origin xyz", "joint axis xyz"],
        "joint_visibility": "template in tasks/ carries topology, types, limits and the root part's pose; other "
                            "visual origins (xyz, rpy), joint origins and axes blanked; gt/ is oracle-only",
        "executor_task": "place and orient every non-root part onto the fixed base (visual origin xyz + rpy per link) "
                         "and fill every joint's origin xyz and axis xyz in parts/template.urdf; nothing is derived "
                         "at finalize time",
        "scored_joint_dims": SCORED_JOINT_DIMS,
        "scored_checks": SCORED_CHECKS,
        "thumbnail_policy": "byte-for-byte reuse of source thumbnails",
        "split_policy": "byte-for-byte reuse of source splits.json",
        "n_instances": len(instances),
    }
    (out / "dataset_variant.json").write_text(json.dumps(variant, indent=2) + "\n")
    (out / "README.md").write_text(
        "# ArtVIP Assembly — pose-fill task view\n\n"
        "Derived from `artvip_beginner`. The root link's part (the base) is kept as the world-baked\n"
        "mesh with a zero visual origin given in the template: it anchors the world frame, which the\n"
        "oracle needs because its geometry checks align translation and scale only. Every other part\n"
        "mesh is the world-baked part translated so that its axis-aligned bounding-box center sits at the origin"
        + (", then rotated by a per-part\nuniform random rotation about that origin (seeded, see "
           "`dataset_variant.json`)" if rotate else " (orientation unchanged)")
        + ", so\nneither placement nor orientation is in the mesh. Each task carries\n"
        "`parts/template.urdf`: the GT URDF with every non-root link's visual `<origin xyz>` and\n"
        "`<origin rpy>` and every joint's `<origin xyz>` / `<axis xyz>` replaced by `? ? ?`. Link\n"
        "names, the link <-> part mapping, the kinematic tree, joint types, limits and the base's pose\n"
        "are given; the solver places and orients the other parts onto the base and fills the joints.\n"
        "Any part orientation that yields the assembled geometry is correct: the oracle scores\n"
        "geometry, not the rpy numbers.\n\n"
        "The template carries no explanatory comment. Conventions: joint `rpy` is zero, the root link frame is\n"
        "the world frame, a joint origin is the pivot in the parent link's frame, an axis is a\n"
        "world direction, a visual origin is the part center minus the link's world origin and\n"
        "its rpy is the part's orientation in the world (the inverse of the applied rotation).\n\n"
        "A filled template only needs its meshes staged next to it\n"
        "(`evo.oracle.template_fill.stage_filled_urdf`, done automatically by the loop); nothing\n"
        "is derived. The Assembly SDK imports these parts in `link` mode (parts are not placed\n"
        "until `set_link_visual_pose` says where).\n\n"
        f"`gt/<task>/<inst>/{TRANSFORMS_NAME}` holds the applied center/rotation per part and is oracle-only.\n"
        "Scored joint dimensions are `axis_ok`, `pivot_ok`, `motion_ok` (type and limits are\n"
        "given); placement is scored through the object-level rest-pose and motion checks.\n\n"
        "Built by `scripts/dataset/build_pose_fill_view.py`.\n")

    if leaks:
        print("FATAL: GT numbers leaked into templates:")
        for line in leaks[:20]:
            print("  ", line)
        return 2
    if args.no_verify:
        print("skipping verification")
        return 0

    JointReport.configure_dims(SCORED_JOINT_DIMS)
    print("verifying: GT refill must pass the oracle; zero fill reported as the floor")
    records: list[dict] = []
    fails: list[dict] = []
    floor = Counter()
    floor_dims = Counter()
    floor_dims_total = Counter()
    for i, (task, inst) in enumerate(instances, 1):
        rec = verify_instance(out, task, inst)
        records.append(rec)
        if not rec["gt_refill_pass"]:
            fails.append(rec)
        floor["pass"] += rec["zero_fill_pass"]
        for dims in rec["zero_fill_joint_dims"].values():
            for d, v in dims.items():
                floor_dims_total[d] += 1
                floor_dims[d] += bool(v)
        if i % 20 == 0 or i == len(instances):
            print(f"  verified {i}/{len(instances)}  GT-refill failures so far {len(fails)}")
    summary = {
        "n_instances": len(instances),
        "gt_refill_pass": len(instances) - len(fails),
        "zero_fill_pass": floor["pass"],
        "zero_fill_joint_dims": {d: f"{floor_dims[d]}/{floor_dims_total[d]}" for d in floor_dims_total},
        "instances": records,
    }
    (out / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"GT refill: {summary['gt_refill_pass']}/{len(instances)} pass; "
          f"zero fill: {floor['pass']}/{len(instances)} pass, joint dims {summary['zero_fill_joint_dims']}")
    for rec in fails[:10]:
        print("  GT-refill FAIL", rec["task"], rec["gt_refill_failed_checks"])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
