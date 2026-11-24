"""Contracts between the evolvable seed policy and frozen Assembly SDK."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILLS = [ROOT / "skills" / "v_minimal", ROOT / "skills" / "v_beginner"]
SDK = ROOT / "evo" / "assembly_sdk"


def test_sdk_python_sources_parse_and_have_docstrings() -> None:
    sources = sorted(SDK.glob("*.py"))
    assert {path.name for path in sources} >= {"__init__.py", "core.py", "runtime.py"}
    for path in sources:
        tree = ast.parse(path.read_text(), filename=str(path))
        assert ast.get_docstring(tree), path


@pytest.mark.parametrize("SKILL", SKILLS, ids=lambda p: p.name)
def test_seed_skill_uses_frozen_sdk_not_ambient_tool_globals(SKILL: Path) -> None:
    skill = (SKILL / "SKILL.md").read_text()
    actions = (SKILL / "ACTIONS.md").read_text()
    combined = skill + actions
    assert 'exec(open(f"{ASSEMBLY_SDK_DIR}/runtime.py").read())' in combined
    assert "assembly.create_link(" in actions
    assert "assembly.create_joint(" in actions
    assert "assembly.export_urdf(" in actions
    assert "Never mutate assembly state" in skill
    assert "LINK_NAME =" not in combined
    assert "MESH_NAME =" not in combined
    assert "WORLD_XYZ =" not in combined


@pytest.mark.parametrize("SKILL", SKILLS, ids=lambda p: p.name)
def test_seed_skill_preserves_all_keyword_only_sdk_signatures(SKILL: Path) -> None:
    actions = (SKILL / "ACTIONS.md").read_text()
    skill = (SKILL / "SKILL.md").read_text()
    # The AssemblySDK implementation lives in host_runtime.py; runtime.py is
    # the thin session loader the sim worker preloads.
    runtime_tree = ast.parse((SDK / "host_runtime.py").read_text())
    assembly_class = next(
        node for node in runtime_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AssemblySDK"
    )
    documented = {
        name: signature
        for name, signature in re.findall(
            r"\| `(\w+)\(([^`]*)\)`", actions
        )
    }
    runtime_kwonly = {
        method.name: [arg.arg for arg in method.args.kwonlyargs]
        for method in assembly_class.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not method.name.startswith("_")
        and method.args.kwonlyargs
        and method.name != "set_link_pose"  # deprecated alias, not documented
    }

    assert runtime_kwonly
    for method_name, keyword_only_names in runtime_kwonly.items():
        assert method_name in documented, method_name
        assert "*" in documented[method_name], method_name
        keyword_section = documented[method_name].split("*", 1)[1]
        for argument_name in keyword_only_names:
            assert re.search(
                rf"\b{re.escape(argument_name)}\b", keyword_section
            ), f"{method_name}.{argument_name}"

    assert "refreshes a read-only `parts` snapshot" in skill
    assert "never reuse any other Python local" in skill
    assert "must not edit\n`ACTIONS.md`" in actions


@pytest.mark.parametrize("SKILL", SKILLS, ids=lambda p: p.name)
def test_coordinate_contract_is_identical_in_seed_and_sdk(SKILL: Path) -> None:
    skill = (SKILL / "SKILL.md").read_text()
    runtime = (SDK / "host_runtime.py").read_text()
    equation = "T_world_mesh(q) = T_world_link(q) @ T_link_visual"
    assert equation in skill
    # FK composes the visual transform onto the link pose and joint motion
    # onto the joint origin — the two lines the SKILL.md equations describe.
    assert '_mat_mul(poses[link_name], _matrix(link["visual_xyz"]' in runtime
    assert "self._motion(joint, q)" in runtime


@pytest.mark.parametrize("SKILL", SKILLS, ids=lambda p: p.name)
def test_seed_forbids_reimplementation_of_deterministic_mechanics(SKILL: Path) -> None:
    skill = (SKILL / "SKILL.md").read_text()
    for contract in (
        "Never hand-write FK",
        "Never edit or copy code under `ASSEMBLY_SDK_DIR`",
        "Never access a part by loader-internal name",
        "Never invoke the simulation backend outside its MCP tool",
        "Never export unless",
    ):
        assert contract in skill


@pytest.mark.parametrize("SKILL", SKILLS, ids=lambda p: p.name)
def test_sdk_has_stable_identity_and_structured_failure_contracts(SKILL: Path) -> None:
    core = (SDK / "core.py").read_text()
    loader = (SDK / "runtime.py").read_text()
    runtime = (SDK / "host_runtime.py").read_text()
    for code in (
        "INVALID_ARGUMENT",
        "UNKNOWN_HANDLE",
        "AMBIGUOUS_GEOMETRY",
        "INVARIANT_VIOLATION",
        "SIM_RUNTIME",
    ):
        assert code in core
    # Stable part handles ("part:<file>") are the only part identity.
    assert 'handle = f"part:{rel}"' in runtime
    assert "safe_call" in runtime
    assert "source_vertex_digest" in runtime
    assert 'if "ASSEMBLY_SDK_DIR" not in globals()' in loader
    assert 'SDK_VERSION = "1.4.0"' in core
    assert "Valid partial forests are supported" in runtime
    assert "valid partial forest" in (SKILL / "SKILL.md").read_text()
    assert "parts = assembly.inspect_parts()" in loader
    assert "renamed_from" in runtime
    assert "swapped_with" in runtime


@pytest.mark.parametrize("SKILL", SKILLS, ids=lambda p: p.name)
def test_seed_skill_forbids_reusing_handles_after_clear(SKILL: Path) -> None:
    skill = (SKILL / "SKILL.md").read_text()
    actions = (SKILL / "ACTIONS.md").read_text()
    assert "clear_scene()` invalidates all of those handles" in skill
    assert "Never reuse a part handle after `clear_scene()`" in skill
    assert "invalidates every previously returned part handle" in actions


def test_sdk_joint_sweep_contract_locks_one_camera() -> None:
    runtime = (SDK / "host_runtime.py").read_text()
    assert 'self._fixed_render_frame = {' in runtime
    assert 'fixed_across_sweep' in runtime
    assert 'del self._fixed_render_frame' in runtime


@pytest.mark.parametrize("SKILL", SKILLS, ids=lambda p: p.name)
def test_sdk_is_outside_evolvable_skill_tree(SKILL: Path) -> None:
    assert SDK.is_dir()
    assert not (SKILL / "assembly_sdk").exists()


def test_sim_preflight_covers_pose_preserving_frame_mutators() -> None:
    preflight = (ROOT / "scripts" / "preflight_training_system.py").read_text()
    assert "assembly.reparent_joint('knob_joint', 'door')" in preflight
    assert "assembly.set_joint_world_frame(" in preflight
    assert "knob_before_reparent" in preflight
    assert "knob_after_reparent" in preflight
    assert "knob_joint_axis" in preflight
