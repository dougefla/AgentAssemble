#!/usr/bin/env python3
"""Build the `artvip_jointfill` view from `artvip_beginner`.

The solver receives the world-baked parts of `artvip_beginner` plus the GT
URDF with every joint's `<origin xyz>` and `<axis xyz>` blanked out
(`parts/template.urdf`). Topology, joint types, limits and the link <-> part
mapping are given; the task is to fill in pivot and axis only.

Layout (mirrors artvip_beginner):
  <out>/tasks/<task>/<inst>/parts/part_<i>.obj (+ .mtl, textures/)  ← copied
  <out>/tasks/<task>/<inst>/parts/template.urdf                      ← NEW
  <out>/tasks/<task>/<inst>/thumbnail.png                            ← copied
  <out>/gt/, gt_renders/, splits.json, mapping.json                  ← copied unchanged
  <out>/dataset_variant.json, README.md, verification.json

Verification (--verify, default on), per instance:
  1. leak scan: no GT origin/axis number appears in the template;
  2. GT refill: the template filled with the reference values, finalized by
     `evo.oracle.template_fill.finalize_filled_urdf`, must PASS the oracle;
  3. zero fill (origin 0, axis +z): the floor. Reported, not required.

Usage:
    .venv/bin/python scripts/dataset/build_joint_fill_view.py \
        --source ~/Dataset/artvip_beginner --output ~/Dataset/artvip_jointfill
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evo.oracle.check_urdf import JointReport, Thresholds, run_oracle  # noqa: E402
from evo.oracle.template_fill import (  # noqa: E402
    TEMPLATE_NAME, build_template, fill_template, finalize_filled_urdf,
    gt_fill_values, leaked_numbers,
)

SCORED_JOINT_DIMS = ["axis_ok", "pivot_ok", "motion_ok"]


def build_instance(src: Path, out: Path, task: str, inst: str) -> dict:
    gt_urdf = src / "gt" / task / inst / "scene.urdf"
    src_task = src / "tasks" / task / inst
    out_task = out / "tasks" / task / inst
    if out_task.exists():
        shutil.rmtree(out_task)
    shutil.copytree(src_task, out_task)
    template = build_template(gt_urdf)
    (out_task / "parts" / TEMPLATE_NAME).write_text(template)
    return {"task": f"{task}/{inst}", "leaks": leaked_numbers(template, gt_urdf)}


def _oracle(gt_dir: Path, template: str, values: dict, parts: Path) -> tuple[bool, dict, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        filled = tmp / "filled.urdf"
        filled.write_text(fill_template(template, values))
        finalize_filled_urdf(filled, tmp / "pred", mesh_dir=parts)
        result = run_oracle(gt_dir, tmp / "pred", Thresholds())
    checks = {c.name: c.passed for c in result.checks if c.scored}
    joints = {name: rep.flags() for name, rep in (result.joints or {}).items()}
    return result.passed, checks, joints


def verify_instance(out: Path, task: str, inst: str) -> dict:
    gt_dir = out / "gt" / task / inst
    parts = out / "tasks" / task / inst / "parts"
    template = (parts / TEMPLATE_NAME).read_text()
    gt_values = gt_fill_values(gt_dir / "scene.urdf")
    ok, checks, _ = _oracle(gt_dir, template, gt_values, parts)
    zero = {j: {"origin_xyz": "0 0 0", "origin_rpy": "0 0 0", "axis": "0 0 1"} for j in gt_values}
    zero_ok, zero_checks, zero_joints = _oracle(gt_dir, template, zero, parts)
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
    ap.add_argument("--output", type=Path, default=Path("~/Dataset/artvip_jointfill").expanduser())
    ap.add_argument("--limit", type=int, default=None, help="only the first N instances (smoke)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--clean", action="store_true", help="delete --output first")
    args = ap.parse_args(argv)

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
        rec = build_instance(src, out, task, inst)
        leaks.extend(f"{rec['task']}: {x}" for x in rec["leaks"])
        if i % 20 == 0 or i == len(instances):
            print(f"  built {i}/{len(instances)}")
    for name in os.listdir(out / "tasks"):
        for inst_dir in (out / "tasks" / name).iterdir():
            assert not (inst_dir / "bake_transforms.json").exists(), inst_dir

    variant = {
        "dataset": "artvip_jointfill",
        "source": str(src),
        "source_dataset": src.name,
        "dataset_version": 1,
        "task_mesh_policy": src_variant.get("task_mesh_policy"),
        "task_mesh_transforms": list(src_variant.get("task_mesh_transforms", [])),
        "task_template": f"parts/{TEMPLATE_NAME}",
        "given": ["link names and link <-> part mapping", "kinematic tree (parent/child)",
                  "joint types", "joint limits"],
        "hidden": ["joint origin xyz", "joint axis xyz"],
        "joint_visibility": "template in tasks/ carries topology, types, limits; origin/axis blanked; gt/ is oracle-only",
        "executor_task": "fill every joint's origin xyz and axis xyz in parts/template.urdf; "
                         "visual origins are derived at finalize time",
        "scored_joint_dims": SCORED_JOINT_DIMS,
        "thumbnail_policy": "byte-for-byte reuse of source thumbnails",
        "split_policy": "byte-for-byte reuse of source splits.json",
        "n_instances": len(instances),
    }
    (out / "dataset_variant.json").write_text(json.dumps(variant, indent=2) + "\n")
    (out / "README.md").write_text(
        "# ArtVIP Assembly — joint-fill task view\n\n"
        "Derived from `artvip_beginner` (world-baked parts). Each task additionally carries\n"
        "`parts/template.urdf`: the GT URDF with every joint's `<origin xyz>` and `<axis xyz>`\n"
        "replaced by `? ? ?`. Link names, the link <-> part mapping, the kinematic tree, joint\n"
        "types and limits are given; the solver fills in pivot and axis only.\n\n"
        "The template has no visual origins on purpose: with world-baked meshes the visual\n"
        "transform is the inverse of the link's q=0 frame, which depends on the answer.\n"
        "`evo.oracle.template_fill.finalize_filled_urdf` derives them from the filled joints\n"
        "and writes a standard URDF the unchanged oracle can score.\n\n"
        "Scored joint dimensions are `axis_ok`, `pivot_ok`, `motion_ok` (type and limits are\n"
        "given, see `scored_joint_dims` in `dataset_variant.json`).\n\n"
        "`gt/` is byte-identical to the source and stays oracle-only.\n"
        "Built by `scripts/dataset/build_joint_fill_view.py`.\n")

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
        "zero_fill_joint_dim_pass_rate": {
            d: round(floor_dims[d] / floor_dims_total[d], 3) for d in sorted(floor_dims_total)
        },
    }
    (out / "verification.json").write_text(json.dumps({"summary": summary, "instances": records}, indent=1))
    for rec in fails[:10]:
        print(f"  FAIL {rec['task']}: {rec['gt_refill_failed_checks']}")
    print("summary:", json.dumps(summary))
    print("RESULT:", "OK" if not fails else f"{len(fails)} instance(s) failed GT refill")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
