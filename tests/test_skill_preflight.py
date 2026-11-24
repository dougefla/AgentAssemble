"""Model-free contract tests for Generator-written skill tools."""
from __future__ import annotations

from pathlib import Path

import pytest

from evo.engine.skill_preflight import preflight_skill


ROOT = Path(__file__).resolve().parents[1]


def _skill(tmp_path: Path) -> Path:
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("# skill\n")
    return root


def _write_tools_doc(root: Path, *signatures: str) -> None:
    sections = ["# Evolved helper interfaces\n"]
    for signature in signatures:
        sections.append(
            f"### `{signature}`\n\n"
            "- **Parameters:** Documented valid inputs and units.\n"
            "- **Returns:** Documented output shape.\n"
            "- **Raises:** `ValueError` for invalid input.\n"
            "- **Example:** `helper(...)`\n"
        )
    (root / "TOOLS.md").write_text("\n".join(sections))


def test_preflight_accepts_skill_without_python_tools(tmp_path):
    result = preflight_skill(_skill(tmp_path))
    assert result.ok
    assert not result.smoke_required


@pytest.mark.parametrize("skill", ["v_minimal", "v_beginner", "v_io_sdk"])
def test_seed_skill_passes_contract_preflight(skill):
    result = preflight_skill(ROOT / "skills" / skill)
    assert result.ok, result.errors


def test_generator_prompt_requires_machine_smoke_hook():
    prompt = (ROOT / "evo" / "agents" / "generator.md").read_text()
    assert "def run_preflight(*, skill_dir):" in prompt
    assert "rejects the candidate before adoption" in prompt
    assert "TOOLS.md" in prompt
    assert "Parameters/Returns/Raises/Example" in prompt
    assert "inspect.signature()" in prompt


def test_preflight_rejects_python_syntax_error(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("def broken(:\n")
    result = preflight_skill(root)
    assert not result.ok
    assert any("compile tools/geometry.py" in error for error in result.errors)


def test_preflight_requires_smoke_hook_for_tools(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("def axis(value):\n    return value\n")
    result = preflight_skill(root)
    assert not result.ok
    assert result.smoke_required
    assert any("tools/preflight.py is required" in error for error in result.errors)


def test_preflight_requires_interface_docs_for_public_helpers(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("def axis(value):\n    return value\n")
    (tools / "preflight.py").write_text(
        "from geometry import axis\n\n"
        "def run_preflight(*, skill_dir):\n"
        "    assert axis('x') == 'x'\n"
        "    return {'ok': True}\n"
    )
    result = preflight_skill(root)
    assert not result.ok
    assert any("TOOLS.md is required" in error for error in result.errors)


def test_preflight_imports_tools_and_executes_smoke(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text(
        "def axis(value):\n"
        "    value = str(value).lower().lstrip('+')\n"
        "    if value != 'x':\n"
        "        raise ValueError(value)\n"
        "    return (1, 0, 0)\n"
    )
    (tools / "preflight.py").write_text(
        "from geometry import axis\n\n"
        "def run_preflight(*, skill_dir):\n"
        "    assert axis('x') == axis('+x') == axis('X')\n"
        "    return {'ok': True, 'skill': str(skill_dir)}\n"
    )
    _write_tools_doc(
        root,
        "tools/geometry.py::axis(value)",
        "tools/preflight.py::run_preflight(*, skill_dir)",
    )
    result = preflight_skill(root)
    assert result.ok
    assert result.smoke_ran
    assert "tools/geometry.py" in result.files


def test_preflight_accepts_documented_framework_hook_without_recursive_coverage(
    tmp_path,
):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text(
        "def identity(value):\n    return value\n"
    )
    (tools / "preflight.py").write_text(
        "from geometry import identity\n\n"
        "def run_preflight(*, skill_dir):\n"
        "    assert identity(str(skill_dir))\n"
        "    return {'ok': True}\n"
    )
    _write_tools_doc(
        root,
        "tools/geometry.py::identity(value)",
        "tools/preflight.py::run_preflight(*, skill_dir)",
    )
    result = preflight_skill(root)
    assert result.ok, result.errors
    assert result.smoke_ran


def test_preflight_rejects_bad_smoke_signature(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("VALUE = 1\n")
    (tools / "preflight.py").write_text(
        "def run_preflight(required_extra):\n    return True\n"
    )
    result = preflight_skill(root)
    assert not result.ok
    assert any("missing a required argument" in error for error in result.errors)


def test_preflight_rejects_false_smoke_result(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("VALUE = 1\n")
    (tools / "preflight.py").write_text(
        "def run_preflight(*, skill_dir):\n    return {'ok': False}\n"
    )
    _write_tools_doc(
        root,
        "tools/preflight.py::run_preflight(*, skill_dir)",
    )
    result = preflight_skill(root)
    assert not result.ok
    assert result.smoke_ran


def test_preflight_rejects_unexercised_public_helper(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text(
        "def first():\n    return 1\n\n"
        "def forgotten():\n    return 2\n"
    )
    (tools / "preflight.py").write_text(
        "from geometry import first\n\n"
        "def run_preflight(*, skill_dir):\n"
        "    assert first() == 1\n"
        "    return {'ok': True}\n"
    )
    _write_tools_doc(
        root,
        "tools/geometry.py::first()",
        "tools/geometry.py::forgotten()",
        "tools/preflight.py::run_preflight(*, skill_dir)",
    )
    result = preflight_skill(root)
    assert not result.ok
    assert any("geometry.py:forgotten" in error for error in result.errors)


def test_preflight_rejects_stale_documented_signature(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("def axis(value):\n    return value\n")
    (tools / "preflight.py").write_text(
        "from geometry import axis\n\n"
        "def run_preflight(*, skill_dir):\n"
        "    assert axis('x') == 'x'\n"
        "    return {'ok': True}\n"
    )
    _write_tools_doc(root, "tools/geometry.py::axis(value, mode=None)")
    result = preflight_skill(root)
    assert not result.ok
    assert any("missing exact public helper signature" in e for e in result.errors)
    assert any("stale or non-public helper signature" in e for e in result.errors)


def test_preflight_requires_complete_interface_fields(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("def axis(value):\n    return value\n")
    (tools / "preflight.py").write_text(
        "from geometry import axis\n\n"
        "def run_preflight(*, skill_dir):\n"
        "    assert axis('x') == 'x'\n"
        "    return {'ok': True}\n"
    )
    (root / "TOOLS.md").write_text(
        "### `tools/geometry.py::axis(value)`\n\n"
        "- **Parameters:** value\n"
    )
    result = preflight_skill(root)
    assert not result.ok
    assert any("missing **Returns:**" in e for e in result.errors)
    assert any("missing **Raises:**" in e for e in result.errors)
    assert any("missing **Example:**" in e for e in result.errors)


def test_preflight_accepts_relative_imports_between_tools(tmp_path):
    """Helpers import each other the way the executor does (`tools` is a package),
    so `from .sibling import x` must load; run 18 round 3 was rejected for this."""
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "base.py").write_text("def _fk(x):\n    return x * 2\n")
    (tools / "review.py").write_text(
        "from .base import _fk\n\n\ndef render(value):\n    return _fk(value)\n"
    )
    (tools / "preflight.py").write_text(
        "def run_preflight(*, skill_dir):\n"
        "    import sys\n"
        "    sys.path.insert(0, str(skill_dir))\n"
        "    from tools.review import render\n"
        "    assert render(2) == 4\n"
        "    return {'ok': True, 'checks': ['render']}\n"
    )
    _write_tools_doc(root, "tools/review.py::render(value)", "tools/preflight.py::run_preflight(*, skill_dir)")
    result = preflight_skill(root)
    assert result.ok, result.errors


def test_preflight_import_failure_does_not_cascade_into_stale_signature(tmp_path):
    root = _skill(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "broken.py").write_text("import no_such_module_xyz\n\n\ndef f(a):\n    return a\n")
    (tools / "preflight.py").write_text(
        "def run_preflight(*, skill_dir):\n    return {'ok': True, 'checks': []}\n"
    )
    _write_tools_doc(root, "tools/broken.py::f(a)", "tools/preflight.py::run_preflight(*, skill_dir)")
    result = preflight_skill(root)
    assert not result.ok
    assert any(e.startswith("import tools/broken.py") for e in result.errors)
    assert not any("stale" in e for e in result.errors)


def test_check_skill_report_matches_controller_gate(tmp_path):
    from evo.engine.agents import check_skill_report, make_check_skill_tool
    root = _skill(tmp_path)
    (root / "ACTIONS.md").write_text("# frozen\n")
    tools = root / "tools"
    tools.mkdir()
    (tools / "broken.py").write_text("from .missing import x\n")
    report = check_skill_report(root, {"ACTIONS.md": "# frozen\n"})
    assert not report["ok"] and any("import tools/broken.py" in e for e in report["errors"])
    (root / "ACTIONS.md").write_text("# edited\n")
    report = check_skill_report(root, {"ACTIONS.md": "# frozen\n"})
    assert any("ACTIONS.md was modified" in e for e in report["errors"])
    tool = make_check_skill_tool(root, {"ACTIONS.md": "# frozen\n"})
    assert tool.name == "check_skill"
