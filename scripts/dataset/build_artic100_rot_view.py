#!/usr/bin/env python3
"""Build a "solved + randomly rotated links" view of artic100_simple.

Takes the tasks the starter skill solved (every task whose oracle verdict is
PASS in any of the given run dirs) and rebuilds them with a random rotation
applied to every link mesh about its own link origin (the parent-joint pivot):

  tasks/<t>/0/parts/part_<i>.obj   vertices v' = R_i v (R_i uniform on SO(3), seeded)
  tasks/<t>/0/parts/template.urdf  joint origin xyz / axis xyz blanked as before, plus every
                                   visual <origin rpy> blanked (collision elements dropped) ("? ? ?"); xyz stays 0 0 0
  gt/<t>/0/assets/part_<i>.obj     the same rotated meshes
  gt/<t>/0/scene.urdf              GT with visual/collision rpy = R_i^-1 so the assembled object is unchanged
  gt/<t>/0/link_rotations.json     per-link R_i (rpy + quaternion) — oracle-side record of the answer

With --unit-scale every non-root part is additionally scaled about its link origin so its longest AABB side is
1 (task parts only; gt assets stay full-size, and the template's <mesh scale="? ? ?"> is the blank the solver
fills with the factor that restores the true size — the root part is given at scale "1 1 1").

Because a rotation about the link origin keeps the pivot at the mesh origin, the solver still knows where
each part's hinge is; what it no longer gets for free is the part's orientation (and therefore the world
direction of any axis read off the mesh): it must recover R_i^-1 from the reference image and the mating
geometry. `--keep-root` leaves the root link unrotated (the base anchors the world frame).

Usage:
  .venv/bin/python scripts/dataset/build_artic100_rot_view.py \
      --source ~/Dataset/artic100_simple --out ~/Dataset/artic100_rot \
      --runs ~/workspace/evo-runs/runs/multi_1791986643 ~/workspace/evo-runs/runs/multi_1792005413 \
      [--seed 42] [--keep-root] [--max-angle-deg 180]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evo.oracle.check_urdf import ALL_CHECKS, _mat4_to_rpy  # noqa: E402

PLACEHOLDER = "? ? ?"


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------

def random_rotation(rng: random.Random, max_angle_deg: float):
    """Uniform random rotation (Shoemake) as a 3x3 matrix; with max_angle_deg < 180 a uniform
    random axis and an angle drawn uniformly from [0, max_angle_deg]."""
    if max_angle_deg >= 180.0:
        u1, u2, u3 = rng.random(), rng.random(), rng.random()
        q = (math.sqrt(1 - u1) * math.sin(2 * math.pi * u2), math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
             math.sqrt(u1) * math.sin(2 * math.pi * u3), math.sqrt(u1) * math.cos(2 * math.pi * u3))  # x, y, z, w
    else:
        z = rng.uniform(-1, 1); phi = rng.uniform(0, 2 * math.pi); s = math.sqrt(1 - z * z)
        axis = (s * math.cos(phi), s * math.sin(phi), z)
        ang = math.radians(rng.uniform(0, max_angle_deg))
        q = (axis[0] * math.sin(ang / 2), axis[1] * math.sin(ang / 2), axis[2] * math.sin(ang / 2), math.cos(ang / 2))
    x, y, z, w = q
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return R, q


def transpose(R): return [[R[j][i] for j in range(3)] for i in range(3)]
def apply(R, v): return [sum(R[i][k] * v[k] for k in range(3)) for i in range(3)]
def rpy_of(R):
    return _mat4_to_rpy(tuple(tuple(R[i]) + (0.0,) for i in range(3)) + ((0.0, 0.0, 0.0, 1.0),))
def fmt(v): return " ".join(f"{x:.9g}" for x in v)


def rotate_obj(src: Path, dst: Path, R, scale: float = 1.0) -> int:
    """Write ``dst`` = ``src`` with vertices R v * scale (normals rotated only)."""
    n = 0
    with open(src, errors="replace") as fi, open(dst, "w") as fo:
        for line in fi:
            if line.startswith("v ") or line.startswith("vn "):
                tag, *rest = line.split()
                xyz = apply(R, [float(c) for c in rest[:3]])
                if tag == "v" and scale != 1.0:
                    xyz = [c * scale for c in xyz]
                fo.write(f"{tag} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" + (" " + " ".join(rest[3:]) if len(rest) > 3 else "") + "\n")
                n += tag == "v"
            else:
                fo.write(line)
    return n


def obj_extent(path: Path) -> float:
    lo = [float("inf")] * 3; hi = [float("-inf")] * 3
    for line in open(path, errors="replace"):
        if line.startswith("v "):
            for i, c in enumerate(line.split()[1:4]):
                v = float(c); lo[i] = min(lo[i], v); hi[i] = max(hi[i], v)
    return max(hi[i] - lo[i] for i in range(3))


# ---------------------------------------------------------------------------
# URDF rewriting
# ---------------------------------------------------------------------------

HEADER_NOTE = (
    "\n  Rotated-link variant: every part_<i>.obj has additionally been rotated by an\n"
    "  unknown rotation about its own link origin (the pivot stays at the mesh origin).\n"
    "  Each visual <origin rpy> is therefore blank too: fill in the rotation that\n"
    "  brings the mesh back to its assembled orientation (xyz stays \"0 0 0\"). The\n"
    "  <collision> elements are omitted from this template. A visual whose rpy is\n"
    "  written as \"0 0 0\" (the root link's base part) was not rotated and is given in place.\n"
    "  thumbnail.png shows the assembled object; -Y is its front, Z is up.\n")
SCALE_NOTE = (
    "\n  Unit-scale variant: every part whose <mesh scale> is blank has also been\n"
    "  uniformly scaled about its link origin so that its longest bounding-box side\n"
    "  is 1. Fill scale with three equal numbers, the factor that restores the true\n"
    "  size (the root part, written as scale=\"1 1 1\", is given at its true size).\n")


def rewrite_template(src_text: str, given_links: set[str] = frozenset(), scaled_links: set[str] = frozenset()) -> str:
    """Blank every visual rpy except for ``given_links`` (an unrotated root keeps rpy 0 0 0);
    with ``scaled_links`` every visual mesh gets a scale attribute, blank for those links and "1 1 1" otherwise."""
    head, body = src_text.split("<robot", 1)
    head = head.replace('Visual/collision origins are "0 0 0" and\n  must stay so.',
                        'Visual origin xyz is "0 0 0" and must stay so;\n  visual rpy is blank (see below); there are no <collision> elements.')
    if "-->" in head:
        head = head.replace("-->", HEADER_NOTE + (SCALE_NOTE if scaled_links else "") + "-->", 1)
    root = ET.fromstring("<robot" + body)
    for link_el in root.findall("link"):
        # collision elements would need the same hidden rpy: drop them from the task template
        # (the oracle reads visuals only; gt/scene.urdf keeps its collisions).
        for col in link_el.findall("collision"):
            link_el.remove(col)
        for el in link_el.findall("visual"):
            origin = el.find("origin")
            if origin is None:
                origin = ET.Element("origin", {"xyz": "0 0 0"}); el.insert(0, origin)
            origin.set("xyz", "0 0 0"); origin.set("rpy", "0 0 0" if link_el.attrib["name"] in given_links else PLACEHOLDER)
            if scaled_links:
                for mesh_el in el.findall("geometry/mesh"):
                    mesh_el.set("scale", PLACEHOLDER if link_el.attrib["name"] in scaled_links else "1 1 1")
    ET.indent(root, space="  ")
    return head + ET.tostring(root, encoding="unicode") + "\n"


def rewrite_gt(src_text: str, inv_rpy: dict[str, tuple[float, float, float]]) -> str:
    root = ET.fromstring(src_text)
    for link_el in root.findall("link"):
        name = link_el.attrib["name"]
        if name not in inv_rpy:
            continue
        for tag in ("visual", "collision"):
            for el in link_el.findall(tag):
                origin = el.find("origin")
                if origin is None:
                    origin = ET.Element("origin", {"xyz": "0 0 0"}); el.insert(0, origin)
                origin.set("rpy", fmt(inv_rpy[name]))
    ET.indent(root, space="  ")
    return '<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def link_mesh_map(urdf_text: str) -> dict[str, str]:
    root = ET.fromstring(urdf_text.split("<robot", 1)[1].join(["<robot", ""]) if "<robot" in urdf_text else urdf_text)
    out = {}
    for link_el in root.findall("link"):
        m = link_el.find("visual/geometry/mesh")
        if m is not None:
            out[link_el.attrib["name"]] = Path(m.attrib["filename"]).name
    return out


def root_link(urdf_text: str) -> str:
    root = ET.fromstring(urdf_text)
    children = {j.find("child").attrib["link"] for j in root.findall("joint")}
    return next(l.attrib["name"] for l in root.findall("link") if l.attrib["name"] not in children)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def solved_tasks(runs: list[Path]) -> list[str]:
    ok: set[str] = set()
    for run in runs:
        for oj in (run / "round_0" / "outputs").glob("task_*/0/oracle.json"):
            try:
                if json.loads(oj.read_text()).get("passed"):
                    ok.add(oj.parent.parent.name)
            except (OSError, json.JSONDecodeError):
                pass
    return sorted(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("~/Dataset/artic100_simple").expanduser())
    ap.add_argument("--out", type=Path, default=Path("~/Dataset/artic100_rot").expanduser())
    ap.add_argument("--runs", type=Path, nargs="*", default=[], help="run dirs; tasks with a PASS oracle.json are selected")
    ap.add_argument("--tasks", type=Path, default=None, help="text file with one task_id per line (alternative to --runs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-root", action="store_true", help="leave the root link unrotated")
    ap.add_argument("--max-angle-deg", type=float, default=180.0, help="<180: random axis, angle uniform in [0, max]")
    ap.add_argument("--unit-scale", action="store_true", help="scale every non-root task part to a unit cube (gt stays full-size)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args()
    src, out = args.source.expanduser().resolve(), args.out.expanduser().resolve()
    if args.tasks:
        tasks = [l.strip() for l in args.tasks.read_text().splitlines() if l.strip()]
    elif args.runs:
        tasks = solved_tasks([r.expanduser().resolve() for r in args.runs])
    else:
        raise SystemExit("give --runs or --tasks")
    missing = [t for t in tasks if not (src / "tasks" / t).is_dir()]
    if missing:
        print(f"skipping {len(missing)} selected task(s) absent from {src}: {', '.join(missing)}")
        tasks = [t for t in tasks if t not in missing]
    if not tasks:
        raise SystemExit("no tasks selected")
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to rebuild")
        shutil.rmtree(out)
    (out / "tasks").mkdir(parents=True); (out / "gt").mkdir(); (out / "gt_renders").mkdir()

    records = {}
    for tid in tasks:
        inst = "0"
        s_task, s_gt, s_rend = src / "tasks" / tid / inst, src / "gt" / tid / inst, src / "gt_renders" / tid / inst
        d_task, d_gt = out / "tasks" / tid / inst, out / "gt" / tid / inst
        (d_task / "parts").mkdir(parents=True); (d_gt / "assets").mkdir(parents=True)
        gt_text = (s_gt / "scene.urdf").read_text()
        meshes = link_mesh_map(gt_text)            # link -> part_i.obj
        root = root_link(gt_text)
        rng = random.Random(f"{args.seed}:{tid}")
        rots = {}
        for link, mesh in meshes.items():
            if args.keep_root and link == root:
                R, q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0.0, 0.0, 0.0, 1.0)
            else:
                R, q = random_rotation(rng, args.max_angle_deg)
            Rinv = transpose(R)
            rots[link] = {"mesh": mesh, "R": R, "quat_xyzw": list(q), "rpy_applied": list(rpy_of(R)),
                          "visual_rpy_answer": list(rpy_of(Rinv)),
                          "angle_deg": math.degrees(2 * math.acos(max(-1.0, min(1.0, abs(q[3])))))}
            n = rotate_obj(s_gt / "assets" / mesh, d_gt / "assets" / mesh, R)          # gt: rotated, full size
            scale_applied = 1.0
            if args.unit_scale and link != root:
                scale_applied = 1.0 / obj_extent(d_gt / "assets" / mesh)
            rotate_obj(s_gt / "assets" / mesh, d_task / "parts" / mesh, R, scale_applied)  # task: rotated (+ unit-scaled)
            rots[link]["n_vertices"] = n
            rots[link]["scale_applied"] = scale_applied
            rots[link]["scale_answer"] = 1.0 / scale_applied         # <mesh scale> that restores the true size
            mtl = (s_gt / "assets" / mesh).with_suffix(".mtl")
            if mtl.is_file():
                shutil.copy2(mtl, d_task / "parts" / mtl.name); shutil.copy2(mtl, d_gt / "assets" / mtl.name)
        # template: the source task template (comment header + blanks) with visual/collision rpy blanked
        scaled = {l for l, r in rots.items() if r["scale_applied"] != 1.0}
        (d_task / "parts" / "template.urdf").write_text(rewrite_template(
            (s_task / "parts" / "template.urdf").read_text(), {root} if args.keep_root else frozenset(), scaled))
        shutil.copy2(s_task / "thumbnail.png", d_task / "thumbnail.png")
        # gt
        (d_gt / "scene.urdf").write_text(rewrite_gt(gt_text, {l: tuple(r["visual_rpy_answer"]) for l, r in rots.items()}))
        if (s_gt / "metadata.json").is_file():
            meta = json.loads((s_gt / "metadata.json").read_text())
            meta["link_rotations"] = {l: {k: v for k, v in r.items() if k != "R"} for l, r in rots.items()}
            meta["units"] = meta.get("units", "") + "; task/gt meshes are additionally rotated about the link origin by link_rotations[link].R (visual rpy in scene.urdf undoes it)"
            (d_gt / "metadata.json").write_text(json.dumps(meta, indent=1))
        (d_gt / "link_rotations.json").write_text(json.dumps(rots, indent=1))
        if s_rend.is_dir():
            shutil.copytree(s_rend, out / "gt_renders" / tid / inst)
        records[tid] = {l: round(r["angle_deg"], 1) for l, r in rots.items()}
        print(f"{tid}: {len(meshes)} links rotated, angles " + ", ".join(f"{a:.0f}°" for a in records[tid].values())
              + (", scale answers " + ", ".join(f"{r['scale_answer']:.3g}" for r in rots.values() if r["scale_applied"] != 1.0) if args.unit_scale else ""), flush=True)

    # manifests
    keep = set(tasks)
    rows = [r for r in json.loads((src / "splits.json").read_text()) if r["task_id"] in keep]
    (out / "splits.json").write_text(json.dumps(rows, indent=1) + "\n")
    idx = [r for r in json.loads((src / "index.json").read_text()) if r["task_id"] in keep]
    (out / "index.json").write_text(json.dumps(idx, indent=1) + "\n")
    mp = json.loads((src / "mapping.json").read_text())
    for key in ("task_to_source", "task_to_category", "task_to_name"):
        if key in mp:
            mp[key] = {k: v for k, v in mp[key].items() if k in keep}
    if "instances" in mp and isinstance(mp["instances"], dict):
        mp["instances"] = {k: v for k, v in mp["instances"].items() if k in keep}
    mp["note"] = (mp.get("note", "") + f" | {out.name}: {len(tasks)} tasks solved by the starter skill in "
                  + ", ".join(r.name for r in args.runs) + ", every link mesh rotated by a seeded random rotation (see dataset_variant.json)").strip(" |")
    (out / "mapping.json").write_text(json.dumps(mp, indent=1) + "\n")
    variant = json.loads((src / "dataset_variant.json").read_text())
    variant.update({
        "dataset": out.name, "source": str(src), "source_dataset": variant.get("dataset", src.name), "dataset_version": 1,
        "task_mesh_policy": (f"gt/<task>/<instance>/assets of {src.name} rotated about each link's origin by a seeded random rotation "
                             f"(seed {args.seed}, uniform on SO(3)" + (f", angle ≤ {args.max_angle_deg:g}°" if args.max_angle_deg < 180 else "")
                             + ("; root link kept upright" if args.keep_root else "; root link rotated too") + ")"
            + ("; task parts (not gt assets) additionally scaled about the link origin to a unit cube, root excluded" if args.unit_scale
               else "; task parts and gt assets are the same files")),
        "task_mesh_transforms": ["link_random_rotation"] + (["unit_cube_scale"] if args.unit_scale else []),
        "template_kind": "jointfill+rot" + ("+scale" if args.unit_scale else ""),
        "given": [g for g in variant.get("given", []) if "link-frame" not in g]
                 + ["link-frame meshes, each non-root one rotated by an unknown rotation about its link origin (pivot stays at the mesh origin)"
                    + ("; and scaled so its longest bounding-box side is 1 (the root part is given upright at its true size)" if args.unit_scale else "")
                    + ("; the root link's part is unrotated and given in place" if args.keep_root and not args.unit_scale else "")],
        "hidden": list(variant.get("hidden", [])) + ["per-link visual origin rpy (the inverse of the applied rotation)"]
                  + (["per-link mesh scale (the factor restoring each non-root part's true size; root given at 1)"] if args.unit_scale else []),
        "joint_visibility": variant.get("joint_visibility", "") + "; visual origin rpy blanked as '? ? ?' too; collision elements dropped from the template",
        "executor_task": ("fill every '? ? ?' in parts/template.urdf (joint origin xyz, joint axis xyz, each link's visual origin rpy"
                          + (", each non-root link's mesh scale" if args.unit_scale else "") + ") or export an equivalent complete URDF; visual xyz stays 0 0 0"),
        "selection": {"runs": [str(r) for r in args.runs], "tasks_file": str(args.tasks) if args.tasks else None, "n_tasks": len(tasks),
                      "criterion": "oracle PASS in any listed run (starter skill skills/v_io_sdk, single session)"},
        "rotation": {"seed": args.seed, "keep_root": args.keep_root, "max_angle_deg": args.max_angle_deg},
        "unit_scale": args.unit_scale,
        **({"scored_joint_dims": list(variant.get("scored_joint_dims", ["axis_ok", "pivot_ok", "motion_ok"])) + ["scale_ok"],
            "scored_checks": list(ALL_CHECKS)} if args.unit_scale else {}),
        "n_instances": len(tasks),
        "split_policy": f"inherited from {src.name} splits.json, filtered to the selected tasks",
    })
    (out / "dataset_variant.json").write_text(json.dumps(variant, indent=1) + "\n")
    from collections import Counter
    by_split = Counter(r["split"] for r in rows)
    readme = f"""# {out.name} — solved artic100 tasks with randomly rotated link meshes

Built {date.today().isoformat()} from `{src}` by `scripts/dataset/build_artic100_rot_view.py` (seed {args.seed}).
Selection: the {len(tasks)} tasks whose oracle verdict was PASS for the starter skill (`skills/v_io_sdk`, one executor
session each) in {', '.join(r.name for r in args.runs) or 'the given task list'}. Splits inherited: {dict(by_split)}.

Same layout and conventions as artic100_simple, with one change: **every link mesh is rotated by a random rotation
about its own link origin** (uniform on SO(3){', root link kept upright' if args.keep_root else ', root link included'}).
`tasks/<t>/0/parts/part_<i>.obj` and `gt/<t>/0/assets/part_<i>.obj` are the same rotated files; `gt/scene.urdf`
carries the compensating visual/collision `<origin rpy>` so the assembled GT object is unchanged (thumbnail and
gt_renders are reused from the source). The pivot still sits at each mesh's origin; what the solver must now
recover in addition to joint origin/axis is each link's orientation.

Template: the artic100_simple template with every visual `<origin rpy>` blanked as `? ? ?` (xyz stays `0 0 0`)
and the `<collision>` elements dropped (the oracle reads visuals only). Oracle-only answer per link: `gt/<t>/0/link_rotations.json` (`visual_rpy_answer`).
""" + (f"""
**Unit scale**: every non-root task part is also scaled about its link origin so its longest AABB side is 1
(`gt/assets` stay full-size). The template's visual `<mesh scale="? ? ?">` is the blank: three equal numbers, the
factor that restores the true size (`link_rotations.json` → `scale_answer`); the root part is given as `scale="1 1 1"`.
""" if args.unit_scale else "")
    (out / "README.md").write_text(readme)
    print(f"\n{out}: {len(tasks)} tasks, splits {dict(by_split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
