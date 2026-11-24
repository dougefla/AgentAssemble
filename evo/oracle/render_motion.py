#!/usr/bin/env python3
"""Render rest pose + per-joint motion frames for predicted URDFs.

For each task output dir at logs/runs/<run_id>/round_<N>/outputs/<tid>/<inst>/:
  - render_assembled.png   (rest pose, az=135, elev=30)
  - render_<joint>_az135_000pct.png (q at lower limit)
  - render_<joint>_az135_050pct.png (q midpoint)
  - render_<joint>_az135_100pct.png (q at upper limit)

Uses the *predicted* furniture.urdf in each output dir (not the GT).
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

from evo.engine.paths import blender_bin

ROOT = Path(__file__).resolve().parents[2]
BLENDER_BIN = str(blender_bin())


def _parse_xyz(s):
    if not s: return (0.0, 0.0, 0.0)
    a = s.strip().split()
    return (float(a[0]), float(a[1]), float(a[2]))


def parse_urdf_raw(urdf_path: Path):
    """Return (links, joints) as plain dicts (no canonicalization)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {}
    joints = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        visual = link.find("visual")
        if visual is None:
            links[name] = {"visual_xyz":(0,0,0),"visual_rpy":(0,0,0),"mesh_path":None}
            continue
        o = visual.find("origin")
        vx = _parse_xyz(o.attrib.get("xyz")) if o is not None else (0,0,0)
        vr = _parse_xyz(o.attrib.get("rpy")) if o is not None else (0,0,0)
        mesh = visual.find("geometry/mesh")
        mp = mesh.attrib.get("filename") if mesh is not None else None
        links[name] = {"visual_xyz": vx, "visual_rpy": vr, "mesh_path": mp}
    for j in root.findall("joint"):
        name = j.attrib["name"]
        if name in joints:
            # Some GT URDFs duplicate joint names by mistake (different
            # parent/child pair, same name). Rename so all rows survive.
            i = 2
            while f"{name}__dup{i}" in joints:
                i += 1
            name = f"{name}__dup{i}"
        jtype = j.attrib.get("type","fixed").upper()
        parent_el = j.find("parent")
        child_el = j.find("child")
        parent = parent_el.attrib["link"] if parent_el is not None else ""
        child = child_el.attrib["link"] if child_el is not None else ""
        oel = j.find("origin")
        oxyz = _parse_xyz(oel.attrib.get("xyz")) if oel is not None else (0,0,0)
        orpy = _parse_xyz(oel.attrib.get("rpy")) if oel is not None else (0,0,0)
        axis_el = j.find("axis")
        axis = _parse_xyz(axis_el.attrib.get("xyz")) if axis_el is not None else (0,0,0)
        limit_el = j.find("limit")
        lo = float(limit_el.attrib.get("lower", 0)) if limit_el is not None else 0.0
        hi = float(limit_el.attrib.get("upper", 0)) if limit_el is not None else 0.0
        joints[name] = {
            "parent": parent, "child": child, "type": jtype,
            "origin_xyz": oxyz, "origin_rpy": orpy,
            "axis": axis, "lower": lo, "upper": hi,
        }
    return links, joints


def find_root(joints, links):
    children = {j["child"] for j in joints.values()}
    parents = {j["parent"] for j in joints.values()}
    roots = parents - children
    if roots: return next(iter(roots))
    return next(iter(links))


def build_pose(joints, links, joint_q: dict[str, float]):
    """Compute world 4x4 per link given each joint's q value.

    For rest pose, joint_q = {} → all joints at q=0.
    """
    children_of = {}
    for jn, j in joints.items():
        children_of.setdefault(j["parent"], []).append((jn, j["child"]))
    root = find_root(joints, links)

    def rpy_to_mat(rpy):
        cr,sr = math.cos(rpy[0]),math.sin(rpy[0])
        cp,sp = math.cos(rpy[1]),math.sin(rpy[1])
        cy,sy = math.cos(rpy[2]),math.sin(rpy[2])
        return [
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, 0],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, 0],
            [-sp,   cp*sr,            cp*cr,            0],
            [0,0,0,1]
        ]
    def trans(xyz):
        return [[1,0,0,xyz[0]],[0,1,0,xyz[1]],[0,0,1,xyz[2]],[0,0,0,1]]
    def matmul(a,b):
        return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    def axis_angle_mat(axis, angle):
        mag = math.sqrt(sum(a*a for a in axis))
        if mag < 1e-9: return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        ux,uy,uz = axis[0]/mag, axis[1]/mag, axis[2]/mag
        c,s = math.cos(angle), math.sin(angle)
        C = 1-c
        return [
            [c+ux*ux*C, ux*uy*C-uz*s, ux*uz*C+uy*s, 0],
            [uy*ux*C+uz*s, c+uy*uy*C, uy*uz*C-ux*s, 0],
            [uz*ux*C-uy*s, uz*uy*C+ux*s, c+uz*uz*C, 0],
            [0,0,0,1]
        ]
    def axis_translate(axis, dist):
        mag = math.sqrt(sum(a*a for a in axis))
        if mag < 1e-9: return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        return trans((axis[0]/mag*dist, axis[1]/mag*dist, axis[2]/mag*dist))

    poses = {root: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
    stack = [root]
    while stack:
        cur = stack.pop()
        cm = poses[cur]
        for jn, child in children_of.get(cur, []):
            j = joints[jn]
            T = matmul(trans(j["origin_xyz"]), rpy_to_mat(j["origin_rpy"]))
            q = joint_q.get(jn, 0.0)
            if j["type"] == "REVOLUTE" or j["type"] == "CONTINUOUS":
                T = matmul(T, axis_angle_mat(j["axis"], q))
            elif j["type"] == "PRISMATIC":
                T = matmul(T, axis_translate(j["axis"], q))
            poses[child] = matmul(cm, T)
            stack.append(child)
    return poses


BLENDER_SCRIPT = r'''
# Render parameters mirror skills/v_minimal/tools/headless_render.py exactly
# (shared oblique-view camera framing) so
# pred renders are pixel-comparable to executor's renders and to GT
# 4-view renders at the same az label.
#
# Source mesh policy: prefer the ORIGINAL parts/*.obj (full PBR materials)
# via cfg["parts_map"] when available. Falls back to the link-keyed
# meshes/<link>_visual.obj from cfg["links"][...]["mesh_path"] only when
# the parts_map is missing (back-compat for older runs).
import bpy, json, math, mathutils, os
cfg = json.loads(open(os.environ["RENDER_CFG"]).read())

base_dir = cfg["base_dir"]
links = cfg["links"]
poses = cfg["poses"]
out_path = cfg["out_path"]
azimuth = cfg.get("azimuth", 135.0)
res = cfg.get("res", 512)
# parts_map v2:
# {"version": 2, "links": {link_name: [{"path": ..., "scale": ...}]}}
# Legacy {link_name: [path, ...]} remains readable with implicit scale=1.
parts_map = cfg.get("parts_map") or {}
parts_by_link = parts_map.get("links", {}) if parts_map.get("version") == 2 else parts_map

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

mn = mathutils.Vector((math.inf, math.inf, math.inf))
mx = mathutils.Vector((-math.inf, -math.inf, -math.inf))
for link_name, info in links.items():
    pose = poses.get(link_name)
    if pose is None:
        continue
    M_link = mathutils.Matrix(pose)
    vx,vy,vz = info["visual_xyz"]
    vr,vp,vy_ = info["visual_rpy"]
    M_vis = mathutils.Matrix.Translation((vx,vy,vz)) @ mathutils.Euler((vr,vp,vy_), 'XYZ').to_matrix().to_4x4()
    M = M_link @ M_vis

    # Source OBJ list for this link: original parts (preferred) or the
    # exported per-link mesh (fallback).
    src_entries = list(parts_by_link.get(link_name) or [])
    if not src_entries:
        mp = info.get("mesh_path")
        if mp:
            src_entries = [{"path": os.path.join(base_dir, mp), "scale": 1.0}]

    for entry in src_entries:
        if isinstance(entry, str):
            full, part_scale = entry, 1.0
        else:
            full = entry.get("path", "")
            part_scale = float(entry.get("scale", 1.0))
        if not os.path.exists(full):
            continue
        before = set(o.name for o in bpy.data.objects)
        try:
            bpy.ops.wm.obj_import(filepath=full, up_axis='Z', forward_axis='NEGATIVE_Y')
        except Exception:
            continue
        new = [o for o in bpy.data.objects if o.name not in before and o.type == 'MESH']
        for o in new:
            S = mathutils.Matrix.Diagonal((part_scale, part_scale, part_scale, 1.0))
            o.matrix_world = M @ S
            for c in o.bound_box:
                wp = o.matrix_world @ mathutils.Vector(c)
                for i in range(3):
                    if wp[i] < mn[i]: mn[i] = wp[i]
                    if wp[i] > mx[i]: mx[i] = wp[i]

if not all(v != math.inf for v in mn):
    raise RuntimeError("no meshes loaded")

center = (mn+mx)*0.5
extent = (mx-mn).length  # diagonal, matches headless_render.py
if extent <= 0: extent = 1.0

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 8
scene.cycles.use_denoising = True
scene.render.resolution_x = res
scene.render.resolution_y = res
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"

# White background via world nodes (matches headless_render)
scene.world = scene.world or bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg_node = scene.world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs[0].default_value = (1, 1, 1, 1)
    bg_node.inputs[1].default_value = 1.0

# Sun light
if not any(o.type == 'LIGHT' for o in bpy.context.scene.objects):
    light_data = bpy.data.lights.new('__render_light__', 'SUN')
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new('__render_light__', light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(50), math.radians(10), math.radians(30))

cam_data = bpy.data.cameras.new("__render_cam__"); cam_data.lens = 50
cam = bpy.data.objects.new("__render_cam__", cam_data); bpy.context.scene.collection.objects.link(cam)
scene.camera = cam

el = math.radians(25.0)
dist = extent * 1.8
az = math.radians(azimuth)
# Azimuth formula identical to headless_render.py.
cam.location = mathutils.Vector(center) + mathutils.Vector((
    dist*math.cos(el)*math.sin(az),
    -dist*math.cos(el)*math.cos(az),
    dist*math.sin(el),
))
direction = mathutils.Vector(center) - cam.location
cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
bpy.context.view_layer.update()
scene.render.filepath = out_path
bpy.ops.render.render(write_still=True)
print("RENDERED:", out_path)
'''


def render_backend() -> str:
    """Which renderer backs render_pose(): `lightphyx` (default) or `blender`.

    Set EVO_RENDER_BACKEND=blender to use the legacy one-process-per-frame
    Blender path (kept for A/B comparison during the backend migration).
    """
    backend = os.environ.get("EVO_RENDER_BACKEND", "lightphyx").strip().lower()
    if backend not in ("lightphyx", "blender"):
        raise ValueError(
            f"EVO_RENDER_BACKEND={backend!r} — expected 'lightphyx' or 'blender'"
        )
    return backend


def render_pose(out_dir: Path, links, joints, joint_q: dict, base_dir: Path,
                out_path: Path, azimuth: float = 135.0,
                parts_map: dict | None = None, frame=None) -> bool:
    """`frame` (center, extent) pins the camera across a sweep; see
    render_lightphyx.sweep_frame. The legacy Blender path ignores it."""
    if render_backend() == "lightphyx":
        from evo.oracle.render_lightphyx import render_pose_lightphyx
        return render_pose_lightphyx(out_dir, links, joints, joint_q, base_dir,
                                     out_path, azimuth, parts_map=parts_map,
                                     frame=frame)
    poses = build_pose(joints, links, joint_q)
    cfg = {
        "base_dir": str(base_dir),
        "links": links, "poses": poses,
        "out_path": str(out_path), "azimuth": azimuth, "res": 512,
        "parts_map": parts_map or {},
    }
    cfg_file = out_dir / "_cfg.json"; cfg_file.write_text(json.dumps(cfg))
    script_file = out_dir / "_render.py"; script_file.write_text(BLENDER_SCRIPT)
    env = os.environ.copy(); env["RENDER_CFG"] = str(cfg_file)
    res = subprocess.run([BLENDER_BIN,"-b","--python",str(script_file)],
                          env=env, capture_output=True, text=True, timeout=180)
    cfg_file.unlink(missing_ok=True); script_file.unlink(missing_ok=True)
    if res.returncode != 0:
        print(f"  [fail] {out_path.name}: {res.stderr[-200:]}")
        return False
    return True


def render_one_task(task_dir: Path, force: bool = False):
    """Render rest + motion for one task. Look for furniture.urdf in task_dir."""
    urdf = task_dir / "furniture.urdf"
    if not urdf.exists():
        print(f"  skip {task_dir} — no furniture.urdf")
        return False, 0
    links, joints = parse_urdf_raw(urdf)
    base = task_dir
    n_rendered = 0

    # Rest pose
    rest_out = task_dir / "render_assembled.png"
    if force or not rest_out.exists():
        if render_pose(task_dir, links, joints, {}, base, rest_out): n_rendered += 1

    # Motion: per movable joint, render at 0% / 50% / 100%
    movable = {n: j for n, j in joints.items()
               if j["type"] in ("REVOLUTE", "PRISMATIC", "CONTINUOUS")}
    for jname, j in movable.items():
        lo, hi = j["lower"], j["upper"]
        # If both 0 (e.g. continuous default), use [0, 2π]
        if lo == 0 and hi == 0 and j["type"] == "CONTINUOUS":
            lo, hi = 0.0, 2 * math.pi
        for pct, label in [(0.0, "000"), (0.5, "050"), (1.0, "100")]:
            q = lo + pct * (hi - lo)
            joint_q = {n: 0.0 for n in movable}
            joint_q[jname] = q
            out = task_dir / f"render_{jname}_az135_{label}pct.png"
            if force or not out.exists():
                if render_pose(task_dir, links, joints, joint_q, base, out, 135.0):
                    n_rendered += 1
    return True, n_rendered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/runs/multi_1778327182",
                    help="Run dir whose round_*/outputs to fill in")
    ap.add_argument("--rounds", default="1,2",
                    help="Comma-separated round indices")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = ROOT / args.run
    rounds = [int(r) for r in args.rounds.split(",")]

    tasks = []
    for r in rounds:
        out_root = run_dir / f"round_{r}" / "outputs"
        if not out_root.is_dir(): continue
        for tdir in sorted(out_root.iterdir()):
            for idir in sorted(tdir.iterdir()):
                if (idir / "furniture.urdf").exists():
                    tasks.append(idir)

    print(f"Will render {len(tasks)} task outputs")
    t0 = time.time()
    n_ok = 0
    for i, task_dir in enumerate(tasks):
        rel = task_dir.relative_to(run_dir)
        print(f"[{i+1}/{len(tasks)}] {rel}")
        ok, n = render_one_task(task_dir, force=args.force)
        if ok: n_ok += 1
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (len(tasks) - i - 1)
        print(f"   rendered {n} files, elapsed={elapsed:.0f}s ETA={eta:.0f}s")
    print(f"\nDone {n_ok}/{len(tasks)} ok, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
