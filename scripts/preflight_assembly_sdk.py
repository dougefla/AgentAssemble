#!/usr/bin/env python3
"""Model-free Assembly SDK readiness gate on real ArtVIP OBJ tasks."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evo.engine.envfile import load_env_chain  # noqa: E402

load_env_chain()

from evo.engine.paths import PROJECT_ROOT  # noqa: E402
from evo.engine.sim_manager import SimWorkerManager, send_to_sim_worker  # noqa: E402

ASSEMBLY_SDK_DIR = PROJECT_ROOT / "evo" / "assembly_sdk"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vertices(path: Path) -> list[tuple[float, float, float]]:
    result = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            fields = line.split()
            result.append(tuple(float(value) for value in fields[1:4]))
    return result


def _xyz(element: ET.Element | None) -> tuple[float, float, float]:
    if element is None:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) for value in element.attrib.get("xyz", "0 0 0").split())


def _q0_fk_matches_grounded_source(
    parts_dir: Path, output: Path, *, scaled_part: str | None = None,
    scale: float = 1.0,
) -> bool:
    """Independently replay this preflight's fixed-joint URDF at q=0."""
    source_by_link = {
        path.stem: _vertices(path) for path in sorted(parts_dir.glob("*.obj"))
    }
    if not source_by_link or any(not values for values in source_by_link.values()):
        return False
    source_min_z = min(
        xyz[2] * (scale if link_name == scaled_part else 1.0)
        for link_name, vertices in source_by_link.items()
        for xyz in vertices
    )
    ground_delta = -source_min_z
    robot = ET.parse(output / "furniture.urdf").getroot()
    joints = robot.findall("joint")
    children = {joint.find("child").attrib["link"] for joint in joints}
    links = {link.attrib["name"]: link for link in robot.findall("link")}
    roots = sorted(set(links) - children)
    if len(roots) != 1:
        return False
    world = {roots[0]: (0.0, 0.0, 0.0)}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in list(pending):
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            if parent not in world:
                continue
            origin = joint.find("origin")
            if origin is not None and any(
                abs(float(value)) > 1e-9
                for value in origin.attrib.get("rpy", "0 0 0").split()
            ):
                return False
            offset = _xyz(origin)
            world[child] = tuple(world[parent][i] + offset[i] for i in range(3))
            pending.remove(joint)
            progressed = True
        if not progressed:
            return False
    for link_name, source_vertices in source_by_link.items():
        link = links.get(link_name)
        if link is None or link_name not in world:
            return False
        visual = link.find("visual/origin")
        if visual is not None and any(
            abs(float(value)) > 1e-9
            for value in visual.attrib.get("rpy", "0 0 0").split()
        ):
            return False
        visual_offset = _xyz(visual)
        exported_vertices = _vertices(output / "meshes" / f"{link_name}_visual.obj")
        if len(exported_vertices) != len(source_vertices):
            return False
        part_scale = scale if link_name == scaled_part else 1.0
        expected_min = [
            min(vertex[axis] * part_scale for vertex in source_vertices)
            + (ground_delta if axis == 2 else 0.0)
            for axis in range(3)
        ]
        actual_min = [
            min(vertex[axis] for vertex in exported_vertices)
            + world[link_name][axis]
            + visual_offset[axis]
            for axis in range(3)
        ]
        expected_max = [
            max(vertex[axis] * part_scale for vertex in source_vertices)
            + (ground_delta if axis == 2 else 0.0)
            for axis in range(3)
        ]
        actual_max = [
            max(vertex[axis] for vertex in exported_vertices)
            + world[link_name][axis]
            + visual_offset[axis]
            for axis in range(3)
        ]
        if max(
            abs(actual_min[i] - expected_min[i])
            for i in range(3)
        ) > 2e-6 or max(
            abs(actual_max[i] - expected_max[i])
            for i in range(3)
        ) > 2e-6:
            return False
    return True


def _scaled_extent_ok(result: dict) -> bool:
    """World extent of the lazily restored part must equal scale x source extent.

    Dataset-agnostic: artvip_full normalizes every part to a unit cube, but
    other views (e.g. artvip_simple) keep verbatim metric meshes.
    """
    world = result.get("restored_scaled_extent") or []
    local = result.get("restored_scaled_local_extent") or []
    if len(world) != 3 or len(local) != 3:
        return False
    return all(
        abs(float(w) - 0.375 * float(l)) < 2e-6 * max(1.0, float(l))
        for w, l in zip(world, local)
    )


def _select_tasks(dataset: Path, count: int) -> list[Path]:
    candidates = sorted(
        path for path in (dataset / "tasks").glob("task_*/*/parts")
        if 2 <= len(list(path.glob("*.obj"))) <= 12
    )
    if len(candidates) < count:
        raise RuntimeError(f"need {count} representative tasks, found {len(candidates)}")
    if count == 1:
        return [candidates[0]]
    # Spread samples across the sorted corpus rather than testing one factory.
    return [candidates[round(i * (len(candidates) - 1) / (count - 1))] for i in range(count)]


def _candidate_code(parts_dir: Path, output_a: Path, output_b: Path, render: Path) -> str:
    return f"""
ASSEMBLY_SDK_DIR = {str(ASSEMBLY_SDK_DIR)!r}
exec(open(f'{{ASSEMBLY_SDK_DIR}}/runtime.py').read())
import json
import_attempt = assembly.safe_call('import_parts', input_dir={str(parts_dir)!r}, replace=True)
if not import_attempt['ok']:
    raise RuntimeError(json.dumps(import_attempt, ensure_ascii=False))
parts = import_attempt['value']
ordered = sorted(parts, key=lambda item: item['label'])
scaled = ordered[-1]
assembly.set_part_scale(scaled['handle'], 0.375)
for item in ordered:
    assembly.create_link(item['label'], item['handle'])
root = ordered[0]['label']
for item in ordered[1:]:
    assembly.create_joint('fixed_' + item['label'], 'FIXED', root, item['label'], [0, 0, 0])
audit_before = assembly.audit()

# Stable handles must survive external interference: dropping the whole
# in-process mesh cache (a worker restart) forces lazy source recovery,
# and the label cache is non-semantic metadata.
first_handle = ordered[0]['handle']
assembly._registry.clear()
inspection = assembly.inspect_parts()
recovery_audit = assembly.audit(require_complete_tree=True)
repaired = bool(recovery_audit['ok']) and len(inspection) == len(ordered)

# Persisted non-unit scale must survive lazy mesh recovery from source.
scaled_handle = scaled['handle']
assembly._registry.pop(scaled_handle, None)
inspection_after_restore = assembly.inspect_parts()
restored_scaled = next(
    item for item in inspection_after_restore if item['handle'] == scaled_handle
)

first_ground = assembly.ground()
second_ground = assembly.ground()
render_result = assembly.render_preview({str(render)!r}, resolution=256)
export_a = assembly.export_urdf({str(output_a)!r})
export_b = assembly.export_urdf({str(output_b)!r})
result = {{
    'parts': ordered,
    'scaled_label': scaled['label'],
    'scaled_value': assembly.state()['parts'][scaled['handle']]['scale'],
    'restored_scaled_value': restored_scaled['scale'],
    'restored_scaled_extent': [
        restored_scaled['world_aabb']['max'][axis]
        - restored_scaled['world_aabb']['min'][axis]
        for axis in range(3)
    ],
    'restored_scaled_local_extent': [
        restored_scaled['local_aabb']['max'][axis]
        - restored_scaled['local_aabb']['min'][axis]
        for axis in range(3)
    ],
    'audit_before': audit_before,
    'audit_after': assembly.audit(),
    'rename_repaired': repaired,
    'inspection_count': len(inspection),
    'first_ground_delta': first_ground['delta_z'],
    'second_ground_delta': second_ground['delta_z'],
    'render_bytes': render_result['bytes'],
    'export_a': export_a,
    'export_b': export_b,
}}
"""


def _fresh_export_code(parts_dir: Path, output: Path) -> str:
    return f"""
ASSEMBLY_SDK_DIR = {str(ASSEMBLY_SDK_DIR)!r}
exec(open(f'{{ASSEMBLY_SDK_DIR}}/runtime.py').read())
parts = assembly.import_parts({str(parts_dir)!r}, replace=True)
ordered = sorted(parts, key=lambda item: item['label'])
assembly.set_part_scale(ordered[-1]['handle'], 0.375)
for item in ordered:
    assembly.create_link(item['label'], item['handle'])
root = ordered[0]['label']
for item in ordered[1:]:
    assembly.create_joint('fixed_' + item['label'], 'FIXED', root, item['label'], [0, 0, 0])
assembly.ground()
result = assembly.export_urdf({str(output)!r})
"""


async def run(args: argparse.Namespace) -> int:
    dataset = args.dataset.resolve()
    tasks = _select_tasks(dataset, args.samples)
    reports = []
    with tempfile.TemporaryDirectory(prefix="assembly-sdk-realdata-") as temporary:
        root = Path(temporary)
        samples = []
        runtime_globals = {
            "ASSEMBLY_SDK_DIR": str(ASSEMBLY_SDK_DIR),
            "OUTPUT_DIR": str(root),
        }
        with SimWorkerManager(port=args.port, runtime_globals=runtime_globals):
            for index, parts_dir in enumerate(tasks):
                task_root = root / f"sample_{index}"
                output_a, output_b, output_c = (
                    task_root / "a", task_root / "b", task_root / "c"
                )
                render = task_root / "preview.png"
                response = await asyncio.to_thread(
                    send_to_sim_worker,
                    _candidate_code(parts_dir, output_a, output_b, render),
                    args.port,
                )
                if response.get("status") != "ok":
                    reports.append({
                        "task": str(parts_dir),
                        "ok": False,
                        "error": response.get("message"),
                        "stdout": str(response.get("stdout", ""))[-2000:],
                    })
                    continue
                result = response.get("result") or {}
                files_a = sorted(path.relative_to(output_a) for path in output_a.rglob("*") if path.is_file())
                files_b = sorted(path.relative_to(output_b) for path in output_b.rglob("*") if path.is_file())
                deterministic = files_a == files_b and all(
                    _digest(output_a / relative) == _digest(output_b / relative)
                    for relative in files_a
                )
                in_process_ok = (
                    result.get("audit_before", {}).get("ok")
                    and result.get("audit_after", {}).get("ok")
                    and result.get("rename_repaired")
                    and result.get("inspection_count") == len(result.get("parts", []))
                    and abs(float(result.get("scaled_value", 0.0)) - 0.375) < 1e-12
                    and abs(float(result.get("restored_scaled_value", 0.0)) - 0.375) < 1e-12
                    and _scaled_extent_ok(result)
                    and abs(float(result.get("second_ground_delta", 1.0))) < 1e-6
                    # LightPhyx uses a compact deterministic PNG encoder;
                    # successful small renders can be below Blender's old
                    # ~800-byte heuristic. Rendering failures raise in the SDK.
                    and int(result.get("render_bytes", 0)) > 100
                    and deterministic
                )
                samples.append({
                    "parts_dir": parts_dir,
                    "output_a": output_a,
                    "output_c": output_c,
                    "result": result,
                    "in_process_ok": in_process_ok,
                    "deterministic": deterministic,
                })

        # Rebuild in a fresh worker process to catch importer/export ordering
        # dependencies hidden by a single long-lived session.
        with SimWorkerManager(port=args.port, runtime_globals=runtime_globals):
            for sample in samples:
                response = await asyncio.to_thread(
                    send_to_sim_worker,
                    _fresh_export_code(sample["parts_dir"], sample["output_c"]),
                    args.port,
                )
                output_a, output_c = sample["output_a"], sample["output_c"]
                files_a = sorted(
                    path.relative_to(output_a)
                    for path in output_a.rglob("*") if path.is_file()
                )
                files_c = sorted(
                    path.relative_to(output_c)
                    for path in output_c.rglob("*") if path.is_file()
                ) if response.get("status") == "ok" else []
                cross_process = files_a == files_c and all(
                    _digest(output_a / relative) == _digest(output_c / relative)
                    for relative in files_a
                )
                result = sample["result"]
                fk_replay = _q0_fk_matches_grounded_source(
                    sample["parts_dir"], output_a,
                    scaled_part=result.get("scaled_label"), scale=0.375,
                )
                ok = sample["in_process_ok"] and cross_process and fk_replay
                reports.append({
                    "task": str(sample["parts_dir"].relative_to(dataset)),
                    "ok": bool(ok),
                    "parts": len(result.get("parts", [])),
                    "source_vertex_preservation": result.get("audit_before", {}).get("ok"),
                    "stable_handle_rename_repair": result.get("rename_repaired"),
                    "non_unit_scale_lazy_restore": (
                        abs(float(result.get("restored_scaled_value", 0.0)) - 0.375)
                        < 1e-12
                        and _scaled_extent_ok(result)
                    ),
                    "ground_idempotent": abs(float(result.get("second_ground_delta", 1.0))) < 1e-6,
                    "deterministic_repeated_export": sample["deterministic"],
                    "deterministic_cross_process_export": cross_process,
                    "independent_q0_fk_replay": fk_replay,
                    "render_bytes": result.get("render_bytes"),
                })
    for report in reports:
        print(json.dumps(report, ensure_ascii=False))
    passed = sum(bool(report["ok"]) for report in reports)
    print(f"ASSEMBLY_SDK_PREFLIGHT {passed}/{len(reports)} passed")
    return 0 if reports and passed == len(reports) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "artvip_full",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--port", type=int, default=12001)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
