#!/usr/bin/env python3
"""Model-free sim-worker regression gate for Assembly SDK agent contracts."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evo.engine.envfile import load_env_chain  # noqa: E402

load_env_chain()

from evo.engine.paths import PROJECT_ROOT  # noqa: E402
from evo.engine.sim_manager import SimWorkerManager, send_to_sim_worker  # noqa: E402

ASSEMBLY_SDK_DIR = PROJECT_ROOT / "evo" / "assembly_sdk"


_CUBE_OBJ = """v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""


def _code(
    input_dir: Path,
    output_dir: Path,
    material_input: Path,
    material_output: Path,
) -> str:
    return f"""
import json
from pathlib import Path
ASSEMBLY_SDK_DIR = {str(ASSEMBLY_SDK_DIR)!r}
exec(open(f'{{ASSEMBLY_SDK_DIR}}/runtime.py').read())

bad_kwarg = assembly.safe_call('create_link', nmae='typo', parts='missing')
non_callable = assembly.safe_call('version')
parts = assembly.import_parts({str(input_dir)!r})
by_label = {{part['label']: part['handle'] for part in parts}}
exec(open(f'{{ASSEMBLY_SDK_DIR}}/runtime.py').read())
runtime_parts = parts
counts_before_retry = (len(assembly.state()['parts']), len(assembly._registry))
retry = assembly.safe_call('import_parts', input_dir={str(input_dir)!r}, replace=False)
counts_after_retry = (len(assembly.state()['parts']), len(assembly._registry))
bad_name = assembly.safe_call('create_link', name='bad name "quoted"', parts=by_label['body'])

assembly.create_link('temporary_body', by_label['body'])
semantic_takeover = assembly.create_link('body', by_label['body'])
assembly.create_link('door', by_label['door'])
swap_to_body = assembly.update_link_parts('body', by_label['door'])
swap_back = assembly.update_link_parts('body', by_label['body'])
assembly.set_part_scale(by_label['door'], 0.25)
partial_visual_pose = assembly.set_link_visual_pose(
    'door', [1.5, 0, 0], [0, 0, 0]
)
partial_render = assembly.render_preview({str(output_dir / 'partial_preview')!r}, resolution=128)
partial_collisions = assembly.collision_aabbs()
assembly.create_joint('hinge', 'REVOLUTE', 'body', 'door', [1, 0, 0], limits=[0, 1.5])
partial_joint_frame = assembly.set_joint_world_frame(
    'hinge', [1, 0, 0], [0, 0, 1]
)
orphan_audit = assembly.audit()

assembly.create_link('knob', by_label['knob'])
assembly.create_joint('knob_joint', 'FIXED', 'door', 'knob', [1, 0, 0])
assembly.set_link_visual_pose('door', [1.5, 0, 0])
assembly.set_link_visual_pose('knob', [2.0, 0, 0])
fixed_sweep = assembly.render_joint_sweep(
    'hinge', {str(output_dir / 'fixed_camera_sweep')!r}, steps=3, resolution=64
)
fixed_sweep_cameras = [frame['camera'] for frame in fixed_sweep]
fixed_sweep_azimuths = [frame['azimuth_deg'] for frame in fixed_sweep]
assembly.set_joint_positions({{'hinge': 0.4}})
assembly.update_joint('hinge', limits=[0, 1.4])
q_after_update = assembly.state()['joint_positions']['hinge']
assembly.set_link_visual_pose('door', [1.5, 0, 0])
q_after_visual_pose = assembly.state()['joint_positions']['hinge']
ground_result = assembly.ground()
q_after_ground = assembly.state()['joint_positions']['hinge']
ignored_limits = assembly.safe_call(
    'update_joint', name='hinge', joint_type='CONTINUOUS', limits='garbage'
)
bake_nonzero = assembly.safe_call('bake_current_pose_as_rest', name='door')
assembly.reset_joints()

assembly.rename_link('knob', 'knob_tmp')
assembly.rename_joint('knob_joint', 'knob_joint_tmp')
assembly.rename_joint('knob_joint_tmp', 'knob_joint')
assembly.rename_link('knob_tmp', 'knob')
assembly.delete_joint('knob_joint')
deleted_link = assembly.delete_link('knob')
assembly.create_link('knob', deleted_link['released_parts'])
assembly.create_joint('knob_joint', 'FIXED', 'door', 'knob', [1, 0, 0])
assembly.set_link_visual_pose('knob', [2.0, 0, 0])

# External interference: tamper the non-semantic label cache in the state
# file directly and drop the in-process mesh cache (worker-restart analog).
# Handle resolution must recover from source paths without a revision bump.
import json as _json
handle = by_label['body']
_tampered = assembly.state()
_tampered['parts'][handle]['object_name'] += '_external_rename'
assembly._state_path.write_text(_json.dumps(_tampered, sort_keys=True, separators=(',', ':')))
assembly._registry.clear()
revision_before_inspect = assembly.state()['revision']
inspection_after_tamper = assembly.inspect_parts()
state_after_inspect = assembly.state()
revision_after_inspect = state_after_inspect['revision']
name_repaired = (
    len(inspection_after_tamper) == len(state_after_inspect['parts'])
    and assembly.audit()['ok']
)

revision_before_export = state_after_inspect['revision']
export_result = assembly.export_urdf({str(output_dir / 'export')!r})
parts_map_v2 = json.loads(
    Path(export_result['parts_map']).read_text()
)
state_after_export = assembly.state()
revision_after_export = state_after_export['revision']
final_audit = assembly.audit()

assembly.clear_scene()
material_parts = assembly.import_parts({str(material_input)!r})
material_by_label = {{part['label']: part['handle'] for part in material_parts}}
for label in sorted(material_by_label):
    assembly.create_link(label, material_by_label[label])
for label in sorted(material_by_label):
    if label != 'left':
        assembly.create_joint('fixed_' + label, 'FIXED', 'left', label, [0, 0, 0])
stale_meshes = Path({str(material_output)!r}) / 'meshes'
stale_meshes.mkdir(parents=True)
(stale_meshes / 'stale_visual.mtl').write_text('stale')
(stale_meshes / 'textures').mkdir()
(stale_meshes / 'textures' / 'stale.png').write_bytes(b'stale')
material_export = assembly.export_urdf({str(material_output)!r})
assembly.clear_scene()
remaining_file_images = (
    [] if not assembly._registry and not assembly._state_path.exists()
    else ['stale_session_state']
)
result = {{
    'bad_kwarg': bad_kwarg,
    'runtime_parts_count': len(runtime_parts),
    'non_callable': non_callable,
    'retry': retry,
    'counts_before_retry': counts_before_retry,
    'counts_after_retry': counts_after_retry,
    'bad_name': bad_name,
    'semantic_takeover': semantic_takeover,
    'swap_to_body': swap_to_body,
    'swap_back': swap_back,
    'partial_visual_pose': partial_visual_pose,
    'partial_joint_frame': partial_joint_frame,
    'partial_render': partial_render,
    'partial_collisions_is_list': isinstance(partial_collisions, list),
    'orphan_codes': [issue['code'] for issue in orphan_audit['issues']],
    'fixed_sweep_cameras': fixed_sweep_cameras,
    'fixed_sweep_azimuths': fixed_sweep_azimuths,
    'q_after_update': q_after_update,
    'q_after_visual_pose': q_after_visual_pose,
    'q_after_ground': q_after_ground,
    'ground_result': ground_result,
    'ignored_limits': ignored_limits,
    'bake_nonzero': bake_nonzero,
    'revision_before_inspect': revision_before_inspect,
    'revision_after_inspect': revision_after_inspect,
    'name_repaired': name_repaired,
    'revision_before_export': revision_before_export,
    'revision_after_export': revision_after_export,
    'export_warnings': export_result['warnings'],
    'parts_map_v2': parts_map_v2,
    'final_audit': final_audit,
    'material_export_warnings': material_export['warnings'],
    'remaining_file_images': remaining_file_images,
}}
"""


def _cycle_setup_code(input_dir: Path) -> str:
    return f"""
import json
ASSEMBLY_SDK_DIR = {str(ASSEMBLY_SDK_DIR)!r}
exec(open(f'{{ASSEMBLY_SDK_DIR}}/runtime.py').read())
assembly.clear_scene()
parts = assembly.import_parts({str(input_dir)!r})
by_label = {{part['label']: part['handle'] for part in parts}}
for label in ('body', 'door', 'knob'):
    assembly.create_link(label, by_label[label])
state = assembly.state()
state['joints'] = {{
    'body_to_door': {{
        'type': 'FIXED', 'parent': 'body', 'child': 'door',
        'origin_xyz': [0, 0, 0], 'origin_rpy': [0, 0, 0],
        'axis': [0, 0, 1], 'limit_lower': None, 'limit_upper': None,
    }},
    'door_to_knob': {{
        'type': 'FIXED', 'parent': 'door', 'child': 'knob',
        'origin_xyz': [0, 0, 0], 'origin_rpy': [0, 0, 0],
        'axis': [0, 0, 1], 'limit_lower': None, 'limit_upper': None,
    }},
    'knob_to_door': {{
        'type': 'FIXED', 'parent': 'knob', 'child': 'door',
        'origin_xyz': [0, 0, 0], 'origin_rpy': [0, 0, 0],
        'axis': [0, 0, 1], 'limit_lower': None, 'limit_upper': None,
    }},
}}
state['joint_positions'] = {{name: 0.0 for name in state['joints']}}
assembly._state_path.write_text(json.dumps(
    state, sort_keys=True, separators=(',', ':')
))
result = {{'cycle_codes': [
    issue['code'] for issue in assembly.audit(require_complete_tree=False)['issues']
]}}
"""


def _timed_safe_call_code(method: str, **kwargs: object) -> str:
    return f"""
import time
ASSEMBLY_SDK_DIR = {str(ASSEMBLY_SDK_DIR)!r}
exec(open(f'{{ASSEMBLY_SDK_DIR}}/runtime.py').read())
started = time.monotonic()
call_result = assembly.safe_call({method!r}, **{kwargs!r})
result = {{
    'call': call_result,
    'elapsed_sec': time.monotonic() - started,
}}
"""


def _result(response: dict, probe: str) -> dict:
    if response.get("status") != "ok":
        raise RuntimeError(f"{probe} sim call failed: {response}")
    value = response.get("result")
    if not isinstance(value, dict):
        raise RuntimeError(f"{probe} returned a non-dictionary result: {value!r}")
    if value.get("ok") is False and isinstance(value.get("error"), dict):
        raise RuntimeError(
            f"{probe} stopped on an unexpected SDK contract error: "
            f"{value['error']}"
        )
    return value



def _sweep_cameras_fixed_per_view(cameras, azimuths, *, steps, views):
    """Frames are grouped by camera; each camera is identical across its group."""
    if len(cameras) != steps * views or len(azimuths) != steps * views:
        return False
    groups = [cameras[i * steps:(i + 1) * steps] for i in range(views)]
    az_groups = [azimuths[i * steps:(i + 1) * steps] for i in range(views)]
    if len({round(g[0]) for g in az_groups}) != views:
        return False
    return all(
        all(c == group[0] and c.get("fixed_across_sweep") is True for c in group)
        and len(set(az)) == 1
        for group, az in zip(groups, az_groups)
    )

def _cycle_failure_ok(probe: dict, deadline: float) -> bool:
    error = probe.get("call", {}).get("error", {})
    issues = error.get("details", {}).get("issues", [])
    return (
        error.get("code") == "invariant_violation"
        and any(issue.get("code") == "cycle" for issue in issues)
        and float(probe.get("elapsed_sec", deadline)) < deadline
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12003)
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=60.0,
        help="wall-clock socket deadline for the main contract probe",
    )
    parser.add_argument(
        "--hang-probe-timeout",
        type=float,
        default=5.0,
        help="per-call wall-clock deadline for operations tested against hangs",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="assembly-sdk-contracts-") as temporary:
        root = Path(temporary)
        input_dir = root / "input"
        output_dir = root / "output"
        material_input = root / "material_input"
        material_output = root / "material_output"
        input_dir.mkdir()
        output_dir.mkdir()
        for name in ("body", "door", "knob"):
            (input_dir / f"{name}.obj").write_text(_CUBE_OBJ)
        texture_payloads = {
            ("sub_a", "left", "mat_shared"): b"LEFT_TEXTURE",
            ("sub_b", "right", "mat_shared"): b"RIGHT_TEXTURE",
            ("sub_c", "literal", "wood.001"): b"LITERAL_TEXTURE",
        }
        for (directory, stem, material), payload in texture_payloads.items():
            source_dir = material_input / directory
            textures = source_dir / "textures"
            textures.mkdir(parents=True)
            obj_text = (
                f"mtllib {stem}.mtl\nusemtl {material}\n" + _CUBE_OBJ
            )
            (source_dir / f"{stem}.obj").write_text(obj_text)
            (source_dir / f"{stem}.mtl").write_text(
                f"newmtl {material}\nmap_Kd textures/tex.png\n"
            )
            (textures / "tex.png").write_bytes(payload)
        runtime_globals = {
            "ASSEMBLY_SDK_DIR": str(ASSEMBLY_SDK_DIR),
            "OUTPUT_DIR": str(root),
        }
        with SimWorkerManager(port=args.port, runtime_globals=runtime_globals):
            response = send_to_sim_worker(
                _code(input_dir, output_dir, material_input, material_output),
                args.port,
                timeout=args.call_timeout,
            )
            setup = send_to_sim_worker(
                _cycle_setup_code(input_dir),
                args.port,
                timeout=args.call_timeout,
            )
            cycle_setup = _result(setup, "cycle setup")
            collision_cycle = _result(
                send_to_sim_worker(
                    _timed_safe_call_code("collision_aabbs"),
                    args.port,
                    timeout=args.hang_probe_timeout,
                ),
                "cyclic collision_aabbs",
            )
            render_cycle = _result(
                send_to_sim_worker(
                    _timed_safe_call_code(
                        "render_preview",
                        path=str(output_dir / "cyclic_preview.png"),
                        resolution=64,
                    ),
                    args.port,
                    timeout=args.hang_probe_timeout,
                ),
                "cyclic render_preview",
            )
        result = _result(response, "main contracts")
        invalid_results = [
            result.get("bad_kwarg"), result.get("non_callable"),
            result.get("bad_name"), result.get("ignored_limits"),
            result.get("bake_nonzero"),
        ]
        exported_materials = set()
        used_materials = set()
        texture_contents = set()
        for mtl in (material_output / "meshes").glob("*_visual.mtl"):
            for line in mtl.read_text().splitlines():
                if line.startswith("newmtl "):
                    exported_materials.add(line.split(None, 1)[1])
                if line.startswith("map_Kd "):
                    texture = material_output / "meshes" / line.split()[-1]
                    if texture.is_file():
                        texture_contents.add(texture.read_bytes())
        for obj in (material_output / "meshes").glob("*_visual.obj"):
            for line in obj.read_text().splitlines():
                if line.startswith("usemtl "):
                    used_materials.add(line.split(None, 1)[1])
        material_identity_ok = (
            len(exported_materials) == 3
            and len(used_materials) == 3
            and used_materials == exported_materials
            and texture_contents == set(texture_payloads.values())
        )
        ok = (
            all(item.get("error", {}).get("code") == "invalid_argument" for item in invalid_results)
            and result.get("retry", {}).get("ok") is True
            and len(result.get("retry", {}).get("value", [])) == 3
            and result.get("runtime_parts_count") == 3
            and result.get("counts_before_retry") == result.get("counts_after_retry") == [3, 3]
            and result.get("semantic_takeover", {}).get("renamed_from") == "temporary_body"
            and result.get("swap_to_body", {}).get("swapped_with") == "door"
            and result.get("swap_back", {}).get("swapped_with") == "door"
            and abs(float(result.get("partial_visual_pose", {}).get("visual_xyz", [0])[0]) - 1.5) < 1e-6
            and result.get("partial_joint_frame", {}).get("origin_xyz") == [1.0, 0.0, 0.0]
            and result.get("partial_joint_frame", {}).get("axis") == [0.0, 0.0, 1.0]
            and result.get("partial_render", {}).get("path", "").endswith("partial_preview.png")
            and Path(result["partial_render"]["path"]).is_file()
            and set(result.get("partial_render", {}).get("posed_links", [])) == {"body", "door"}
            and set(result.get("partial_render", {}).get("roots", [])) == {"body", "door"}
            and result.get("partial_collisions_is_list")
            and "unassigned_part" in result.get("orphan_codes", [])
            # views=2 default: 3 poses x 2 cameras, grouped by camera. The
            # invariant is that a camera never drifts across ITS frames.
            and _sweep_cameras_fixed_per_view(
                result.get("fixed_sweep_cameras", []),
                result.get("fixed_sweep_azimuths", []),
                steps=3, views=2,
            )
            and abs(float(result.get("q_after_update", -1)) - 0.4) < 1e-9
            and abs(float(result.get("q_after_visual_pose", -1)) - 0.4) < 1e-9
            and abs(float(result.get("q_after_ground", -1)) - 0.4) < 1e-9
            and result.get("ground_result", {}).get("measured_at") == "q=0"
            and result.get("revision_before_inspect") == result.get("revision_after_inspect")
            and result.get("name_repaired")
            and result.get("revision_after_export") == result.get("revision_before_export") + 1
            and result.get("parts_map_v2", {}).get("version") == 2
            and abs(float(
                result.get("parts_map_v2", {})
                .get("links", {}).get("door", [{}])[0].get("scale", 0.0)
            ) - 0.25) < 1e-12
            and result.get("final_audit", {}).get("ok")
            and result.get("material_export_warnings") == []
            and material_identity_ok
            and not (material_output / "meshes" / "stale_visual.mtl").exists()
            and not (material_output / "meshes" / "textures" / "stale.png").exists()
            and result.get("remaining_file_images") == []
            and "cycle" in cycle_setup.get("cycle_codes", [])
            and _cycle_failure_ok(collision_cycle, args.hang_probe_timeout)
            and _cycle_failure_ok(render_cycle, args.hang_probe_timeout)
        )
        report = {
            "ok": bool(ok),
            "material_identity_ok": material_identity_ok,
            "cycle_setup": cycle_setup,
            "cyclic_collision_aabbs": collision_cycle,
            "cyclic_render_preview": render_cycle,
            **result,
        }
        print(json.dumps(report, ensure_ascii=False))
        print(f"ASSEMBLY_SDK_CONTRACTS {'1/1 passed' if ok else '0/1 passed'}")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
