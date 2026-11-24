#!/usr/bin/env python3
"""Bounded preflight for the evo-assemble formal training runtime.

This is intentionally not an assembly experiment. It makes two small model
runs — luna exercises Responses + the sim MCP tool + runtime-global
injection, then sol exercises the Generator's Read/Write tool loop — plus a
model-free coordinate-contract probe against the frozen Assembly SDK. The
model probes are skipped cleanly when no provider credentials are present;
any executed probe that fails exits nonzero.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evo.engine.envfile import load_env_chain
from evo.engine.model_defaults import DEFAULT_AGENT_EFFORT, DEFAULT_AGENT_MODEL

load_env_chain()

from evo.engine.agents import ASSEMBLY_SDK_DIR  # noqa: E402
from evo.engine.sim_manager import (  # noqa: E402
    SimWorkerManager,
    send_to_sim_worker,
)
from evo.engine.responses_backend import (  # noqa: E402
    SimResponsesMCPServer,
    _responses_api_key,
    _responses_base_url,
    run_responses_agent,
)


def _model_creds_present() -> bool:
    """True when the Responses provider has usable credentials."""
    try:
        return bool(_responses_api_key(_responses_base_url()))
    except Exception:  # noqa: BLE001 - missing/odd env is just "no creds"
        return False


def _tool_outputs(artifact: Path) -> str:
    payload = json.loads(artifact.read_text())
    chunks: list[str] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("function_call_output", "mcp_call"):
            chunks.append(json.dumps(item.get("output"), ensure_ascii=False))
    return "\n".join(chunks)


async def _probe_luna_sim(root: Path, port: int) -> dict:
    started = time.time()
    skill_dir = root / "runtime_skill"
    output_dir = root / "runtime_output"
    input_dir = root / "runtime_input"
    for path in (skill_dir, output_dir, input_dir):
        path.mkdir()
    artifact = root / "luna.responses.json"
    runtime_globals = {
        "ASSEMBLY_SDK_DIR": str(ASSEMBLY_SDK_DIR),
        "SKILL_DIR": str(skill_dir),
        "INPUT_DIR": str(input_dir),
        "OUTPUT_DIR": str(output_dir),
    }
    with SimWorkerManager(port=port, runtime_globals=runtime_globals):
        outcome = await run_responses_agent(
            name="training_preflight_luna",
            instructions=(
                "You are a bounded infrastructure probe. Call the sim MCP "
                "tool exactly once with code that prints ASSEMBLY_SDK_DIR, "
                "SKILL_DIR, INPUT_DIR, and OUTPUT_DIR, then reply exactly "
                "SIM_GLOBALS_OK."
            ),
            prompt=(
                "Call mcp__sim__execute_python_code with: "
                "print('GLOBALS', ASSEMBLY_SDK_DIR, SKILL_DIR, INPUT_DIR, OUTPUT_DIR). "
                "Then reply SIM_GLOBALS_OK."
            ),
            model=DEFAULT_AGENT_MODEL,
            effort=DEFAULT_AGENT_EFFORT,
            max_turns=3,
            cwd=root,
            read_roots=[root],
            write_root=root,
            sim_port=port,
            runtime_globals=runtime_globals,
            artifact_path=artifact,
        )
        # The formal executor may emit assignments to these names even though
        # its prompt says not to. Verify the adapter enforces runtime ownership
        # in a real sim call instead of merely prepending correct values.
        overwrite_probe = await SimResponsesMCPServer(
            port, runtime_globals
        ).call_tool(
            "mcp__sim__execute_python_code",
            {
                "code": (
                    "import os\n"
                    "ASSEMBLY_SDK_DIR = '/wrong-sdk'\n"
                    "SKILL_DIR = '/wrong-skill'\n"
                    "INPUT_DIR = '/wrong-input'\n"
                    "OUTPUT_DIR = '/wrong-output'\n"
                    "os.environ['SKILL_DIR'] = '/wrong-env-skill'\n"
                    "os.environ['INPUT_DIR'] = '/wrong-env-input'\n"
                    "os.environ['OUTPUT_DIR'] = '/wrong-env-output'\n"
                    "os.environ['ASSEMBLY_SDK_DIR'] = '/wrong-env-sdk'\n"
                    "print('PROTECTED', ASSEMBLY_SDK_DIR, SKILL_DIR, INPUT_DIR, OUTPUT_DIR)\n"
                    "print('ENV', os.environ['ASSEMBLY_SDK_DIR'], os.environ['SKILL_DIR'], "
                    "os.environ['INPUT_DIR'], os.environ['OUTPUT_DIR'])"
                )
            },
        )
    outputs = _tool_outputs(artifact)
    overwrite_outputs = "\n".join(
        getattr(item, "text", "") for item in overwrite_probe.content
    )
    expected = (str(ASSEMBLY_SDK_DIR), str(skill_dir), str(input_dir), str(output_dir))
    ok = (
        outcome.final_output.strip() == "SIM_GLOBALS_OK"
        and all(value in outputs for value in expected)
        and not overwrite_probe.isError
        and all(value in overwrite_outputs for value in expected)
        and "/wrong-" not in overwrite_outputs
    )
    return {
        "probe": "luna_responses_sim_globals",
        "ok": ok,
        "duration_sec": round(time.time() - started, 2),
        "response_id": outcome.response_id,
        "api_calls": outcome.usage_summary.get("n_api_calls", 0),
        "detail": (
            "runtime globals observed and overwrite-protected in the sim worker"
            if ok else (outputs + "\n" + overwrite_outputs)[-1000:]
        ),
    }


async def _probe_sol_tools(root: Path) -> dict:
    started = time.time()
    candidate = root / "candidate"
    candidate.mkdir()
    skill = candidate / "SKILL.md"
    skill.write_text("# PRECHECK_BEFORE\n")
    journal = root / "generator_journal.md"
    artifact = root / "sol.responses.json"
    outcome = await run_responses_agent(
        name="training_preflight_generator",
        instructions=(
            "You are a bounded Generator infrastructure probe. Use Read and "
            "Write only. Do exactly the requested edits, then reply exactly "
            "GENERATOR_WRITE_OK."
        ),
        prompt=(
            f"Read {skill}. Replace it by writing '# PRECHECK_AFTER\\n'. "
            f"Write {journal} with 'Verified: preflight\\n'. Then reply "
            "GENERATOR_WRITE_OK."
        ),
        model=DEFAULT_AGENT_MODEL,
        effort=DEFAULT_AGENT_EFFORT,
        max_turns=5,
        cwd=candidate,
        read_roots=[root],
        write_root=root,
        artifact_path=artifact,
    )
    ok = (
        outcome.final_output.strip() == "GENERATOR_WRITE_OK"
        and skill.read_text() == "# PRECHECK_AFTER\n"
        and journal.read_text() == "Verified: preflight\n"
    )
    return {
        "probe": "sol_responses_generator_tools",
        "ok": ok,
        "duration_sec": round(time.time() - started, 2),
        "response_id": outcome.response_id,
        "api_calls": outcome.usage_summary.get("n_api_calls", 0),
        "detail": "Read/Write/final output completed" if ok else outcome.final_output[-500:],
    }


async def _probe_vminimal_coordinate_contract(root: Path, port: int) -> dict:
    """Exercise frozen SDK import → FK → grounding → URDF without a model."""
    started = time.time()
    repo_root = Path(__file__).resolve().parents[1]
    sdk_dir = repo_root / "evo" / "assembly_sdk"
    input_dir = root / "coordinate_contract_input"
    output_dir = root / "coordinate_contract_output"
    input_dir.mkdir()
    output_dir.mkdir()
    cube_obj = """v -0.5 -0.5 -0.5
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
    (input_dir / "body_obj.obj").write_text(cube_obj)
    (input_dir / "door_obj.obj").write_text(cube_obj)
    (input_dir / "knob_obj.obj").write_text(cube_obj)
    code = f"""
import math
ASSEMBLY_SDK_DIR = {str(sdk_dir)!r}
exec(open(f'{{ASSEMBLY_SDK_DIR}}/runtime.py').read())
parts = assembly.import_parts({str(input_dir)!r})
by_label = {{p['label']: p['handle'] for p in parts}}
assembly.create_link('body', by_label['body_obj'])
assembly.create_link('door', by_label['door_obj'])
assembly.create_link('knob', by_label['knob_obj'])
assembly.create_joint('hinge', 'REVOLUTE', 'body', 'door',
                      [1.0, 0.0, 0.0], axis=[0.0, 0.0, 5.0],
                      limits=[0.0, math.pi / 2])
assembly.create_joint('knob_joint', 'REVOLUTE', 'body', 'knob',
                      [2.0, 0.0, 0.0], origin_rpy=[0.0, 0.0, math.pi / 2],
                      axis='X', limits=[-0.5, 0.5])
assembly.set_link_visual_pose('door', [1.5, 0.0, 0.0])
assembly.set_link_visual_pose('knob', [2.0, 0.0, 0.0])
def center(handle):
    item = next(p for p in assembly.inspect_parts() if p['handle'] == handle)
    return tuple(round((item['world_aabb']['min'][i] + item['world_aabb']['max'][i]) / 2, 6) for i in range(3))
rest = center(by_label['door_obj'])
knob_before_reparent = center(by_label['knob_obj'])
assembly.reparent_joint('knob_joint', 'door')
knob_after_reparent = center(by_label['knob_obj'])
assembly.set_joint_world_frame(
    'knob_joint', world_origin=[2.0, 0.0, 0.0], world_axis=[0.0, 1.0, 0.0]
)
knob_after_world_frame = center(by_label['knob_obj'])
assembly.set_joint_positions({{'hinge': math.pi / 2}})
moved = center(by_label['door_obj'])
assembly.reset_joints()
assembly.ground()
grounded = center(by_label['door_obj'])
state = assembly.state()
axis = tuple(round(v, 6) for v in state['joints']['hinge']['axis'])
knob_axis = tuple(round(v, 6) for v in state['joints']['knob_joint']['axis'])
knob_origin = tuple(round(v, 6) for v in state['joints']['knob_joint']['origin_xyz'])
knob_parent = state['joints']['knob_joint']['parent']
assembly.export_urdf({str(output_dir)!r})
result = {{'rest': rest, 'moved': moved, 'grounded': grounded, 'axis': axis,
          'knob_before_reparent': knob_before_reparent,
          'knob_after_reparent': knob_after_reparent,
          'knob_after_world_frame': knob_after_world_frame,
          'knob_axis': knob_axis, 'knob_origin': knob_origin,
          'knob_parent': knob_parent,
          'audit': assembly.audit()}}
"""
    probe_globals = {
        "ASSEMBLY_SDK_DIR": str(sdk_dir),
        "INPUT_DIR": str(input_dir),
        "OUTPUT_DIR": str(output_dir),
    }
    with SimWorkerManager(port=port, runtime_globals=probe_globals):
        response = await asyncio.to_thread(
            send_to_sim_worker, code, port
        )
    if response.get("status") != "ok":
        return {
            "probe": "assembly_sdk_coordinate_contract",
            "ok": False,
            "duration_sec": round(time.time() - started, 2),
            "detail": response.get("message") or str(response)[-1000:],
        }

    urdf = ET.parse(output_dir / "furniture.urdf").getroot()
    door = next(link for link in urdf.findall("link") if link.attrib["name"] == "door")
    visual_xyz = tuple(
        float(v) for v in door.find("visual/origin").attrib["xyz"].split()
    )
    hinge = next(j for j in urdf.findall("joint") if j.attrib["name"] == "hinge")
    joint_xyz = tuple(
        float(v) for v in hinge.find("origin").attrib["xyz"].split()
    )
    knob_joint = next(
        j for j in urdf.findall("joint") if j.attrib["name"] == "knob_joint"
    )
    knob_joint_xyz = tuple(
        float(v) for v in knob_joint.find("origin").attrib["xyz"].split()
    )
    knob_joint_axis = tuple(
        float(v) for v in knob_joint.find("axis").attrib["xyz"].split()
    )
    body = next(link for link in urdf.findall("link") if link.attrib["name"] == "body")
    body_visual_xyz = tuple(
        float(v) for v in body.find("visual/origin").attrib["xyz"].split()
    )
    result = response.get("result") or {}
    ok = (
        tuple(result.get("rest", ())) == (1.5, 0.0, 0.0)
        and tuple(result.get("moved", ())) == (1.0, 0.5, 0.0)
        and tuple(result.get("grounded", ())) == (1.5, 0.0, 0.5)
        and tuple(result.get("axis", ())) == (0.0, 0.0, 1.0)
        and tuple(result.get("knob_before_reparent", ())) == (2.0, 0.0, 0.0)
        and tuple(result.get("knob_after_reparent", ())) == (2.0, 0.0, 0.0)
        and tuple(result.get("knob_after_world_frame", ())) == (2.0, 0.0, 0.0)
        and tuple(result.get("knob_origin", ())) == (1.0, 0.0, 0.0)
        and tuple(result.get("knob_axis", ())) == (1.0, 0.0, 0.0)
        and result.get("knob_parent") == "door"
        and bool(result.get("audit", {}).get("ok"))
        and all(abs(a - b) < 1e-6 for a, b in zip(visual_xyz, (0.5, 0.0, 0.0)))
        and all(abs(a - b) < 1e-6 for a, b in zip(joint_xyz, (1.0, 0.0, 0.5)))
        and all(abs(a - b) < 1e-6 for a, b in zip(body_visual_xyz, (0.0, 0.0, 0.5)))
        and all(abs(a - b) < 1e-6 for a, b in zip(knob_joint_xyz, (1.0, 0.0, 0.0)))
        and all(abs(a - b) < 1e-6 for a, b in zip(knob_joint_axis, (1.0, 0.0, 0.0)))
    )
    return {
        "probe": "assembly_sdk_coordinate_contract",
        "ok": ok,
        "duration_sec": round(time.time() - started, 2),
        "detail": (
            "rest/FK/axis/URDF round-trip consistent"
            if ok else {
                "response": result,
                "visual_xyz": visual_xyz,
                "joint_xyz": joint_xyz,
                "body_visual_xyz": body_visual_xyz,
                "knob_joint_xyz": knob_joint_xyz,
                "knob_joint_axis": knob_joint_axis,
            }
        ),
    }


async def amain(args: argparse.Namespace) -> int:
    results: list[dict] = []
    expected = 3
    try:
        with tempfile.TemporaryDirectory(prefix="evo-training-preflight-") as td:
            root = Path(td)
            if _model_creds_present():
                results.append(await _probe_luna_sim(root, args.port))
                results.append(await _probe_sol_tools(root))
            else:
                expected = 1
                for probe in ("luna_responses_sim_globals",
                              "sol_responses_generator_tools"):
                    results.append({
                        "probe": probe,
                        "ok": None,
                        "skipped": True,
                        "detail": "no Responses provider credentials; "
                                  "model probe skipped",
                    })
            results.append(await _probe_vminimal_coordinate_contract(root, args.port))
    except Exception as exc:  # noqa: BLE001 - preflight must report and fail
        results.append({
            "probe": "uncaught_training_system_error",
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    executed = [r for r in results if not r.get("skipped")]
    passed = sum(bool(result.get("ok")) for result in executed)
    skipped = len(results) - len(executed)
    print(
        f"PREFLIGHT {passed}/{len(executed)} passed"
        + (f" ({skipped} skipped: no model creds)" if skipped else ""),
        flush=True,
    )
    return 0 if len(executed) == expected and passed == expected else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12000)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
