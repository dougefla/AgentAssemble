#!/usr/bin/env python3
"""ArtiAssembly ground truth (URDF + per-link OBJ) -> GLB + joint JSON for the three.js viewer of the project page.

Every link mesh is baked to its world rest pose and every joint is written with its world pivot and world axis, so
the viewer only has to nest one pivot per joint along the kinematic tree (any depth). A mimic joint is written with
the joint it follows, its multiplier and its offset.

usage: python3 scripts/make_webgl_assets_artiassembly.py [--dataset ~/Dataset/artic300_simple]
"""
import argparse, json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "projectpage" / "assets" / "models"
ASSETS = [("task_203", "Cash register", "cash_register"), ("task_179", "Computer desk", "computer_desk"),
          ("task_112", "Tape measure", "tape_measure"), ("task_222", "Microscope", "microscope"),
          ("task_014", "Ceiling fan", "ceiling_fan"), ("task_263", "Slot machine", "slot_machine"),
          ("task_013", "Toaster", "toaster"), ("task_057", "Printer", "printer")]
MAX_FACES = 60000


def rpy_m(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], [-sp, cp*sr, cp*cr]])


def T_of(el):
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = rpy_m([float(v) for v in el.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def convert(ds: Path, task: str, label: str, slug: str) -> dict:
    base = ds / "gt" / task / "0"
    robot = ET.parse(base / "scene.urdf").getroot()
    links = [l.get("name") for l in robot.findall("link")]
    jel = {j.find("child").get("link"): j for j in robot.findall("joint")}
    root = next(l for l in links if l not in jel)

    def world(link):
        j = jel.get(link)
        return np.eye(4) if j is None else world(j.find("parent").get("link")) @ T_of(j.find("origin"))

    def depth(link):
        return 0 if link not in jel else 1 + depth(jel[link].find("parent").get("link"))

    scene, faces = trimesh.Scene(), 0
    per_link = max(2000, MAX_FACES // len(links))
    for l in robot.findall("link"):
        Tw, parts = world(l.get("name")), []
        for vis in l.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            mesh = trimesh.load(base / m.get("filename"), force="mesh", process=False)
            mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
            S = np.diag([float(v) for v in m.get("scale", "1 1 1").split()] + [1.0])
            mesh.apply_transform(Tw @ T_of(vis.find("origin")) @ S)
            parts.append(mesh)
        if not parts:
            continue
        mesh = trimesh.util.concatenate(parts)
        if len(mesh.faces) > per_link:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=per_link)
            except Exception as e:
                print(f"    decimation skipped for {l.get('name')}: {e}")
        _ = mesh.vertex_normals
        faces += len(mesh.faces)
        scene.add_geometry(mesh, node_name=l.get("name"), geom_name=l.get("name"))

    joints = []
    for child in sorted(jel, key=depth):
        j = jel[child]
        if j.get("type") == "fixed":
            typ, axis, lo, hi = "fixed", [0, 0, 1], 0.0, 0.0
        else:
            Tw = world(child)
            a = Tw[:3, :3] @ np.array([float(v) for v in j.find("axis").get("xyz").split()])
            axis, typ = (a / np.linalg.norm(a)).round(5).tolist(), j.get("type")
            lim = j.find("limit")
            lo, hi = (float(lim.get("lower", 0)), float(lim.get("upper", 0))) if lim is not None else (0.0, 0.0)
            if typ == "continuous" or hi <= lo:
                lo, hi = -np.pi, np.pi
        rec = dict(name=j.get("name"), type=typ, parent=j.find("parent").get("link"), child=child,
                   origin=world(child)[:3, 3].round(5).tolist(), axis=axis, lower=round(lo, 5), upper=round(hi, 5))
        mim = j.find("mimic")
        if mim is not None:
            rec["mimic"] = dict(joint=mim.get("joint"), multiplier=float(mim.get("multiplier", 1)), offset=float(mim.get("offset", 0)))
        joints.append(rec)
    scene.export(OUT / f"{slug}.glb")
    lo, hi = scene.bounds
    meta = dict(label=label, task=task, slug=slug, up="z", root=root, links=links, joints=joints,
                bbox=dict(min=lo.round(4).tolist(), max=hi.round(4).tolist()), faces=faces)
    (OUT / f"{slug}.json").write_text(json.dumps(meta, indent=1))
    mov = [j for j in joints if j["type"] != "fixed"]
    print(f"  {slug:14} links={len(links):2} joints={len(mov):2} depth={max(map(depth, links))} mimic={sum('mimic' in j for j in joints)} "
          f"faces={faces:6d} {(OUT / (slug + '.glb')).stat().st_size // 1024} KB")
    return dict(slug=slug, label=label, joints=len(mov), types=sorted({j["type"] for j in mov}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path.home() / "Dataset" / "artic300_simple")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = [convert(a.dataset, *x) for x in ASSETS]
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
