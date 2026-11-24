"""Model-free contract gate for an evolvable skill candidate.

The frozen Assembly SDK has its own repository preflight.  This module guards
the *evolvable* Python surface written by Generator: every Python source must
compile, every module under ``tools/`` must import in a fresh interpreter,
every public helper must have an exact interface section in ``TOOLS.md``, and
skills that expose tools must provide an executable ``tools/preflight.py``
smoke hook. The hook is deliberately model/API/simulator-free so it can run
before spending candidate A/B calls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class SkillPreflightResult:
    ok: bool
    files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    smoke_required: bool = False
    smoke_ran: bool = False
    duration_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


_RUNNER = r'''
import importlib.machinery
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

root = Path(sys.argv[1]).resolve()
tool_dir = root / "tools"
python_files = sorted(
    path for path in root.rglob("*.py")
    if "__pycache__" not in path.parts
)
tool_files = sorted(
    path for path in tool_dir.rglob("*.py")
    if path.name not in {"__init__.py", "preflight.py"}
) if tool_dir.is_dir() else []
smoke_path = tool_dir / "preflight.py"
interface_path = root / "TOOLS.md"
errors = []
loaded = []
public_functions = []
smoke_module = None
smoke_hook = None

sys.path.insert(0, str(root))
if tool_dir.is_dir():
    sys.path.insert(0, str(tool_dir))
os.environ["SKILL_DIR"] = str(root)

for path in python_files:
    relative = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    except Exception as exc:
        errors.append(f"compile {relative}: {type(exc).__name__}: {exc}")

# Tools are loaded the way the executor loads them — as submodules of the
# ``tools`` package (``sys.path.insert(0, SKILL_DIR); from tools.x import f``) —
# so relative imports between helpers (``from .geometry import fit``) resolve
# here exactly as they do at runtime. Loading each file as a free-standing
# module used to reject such imports that the executor would have run fine
# (run 18, round 3: tools/review.py).
PKG = "_evo_skill_tools"
if tool_dir.is_dir():
    pkg_spec = importlib.machinery.ModuleSpec(PKG, None, is_package=True)
    pkg_spec.submodule_search_locations = [str(tool_dir)]
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules[PKG] = pkg
failed_files = set()
for index, path in enumerate(tool_files):
    relative = path.relative_to(root).as_posix()
    try:
        rel_mod = ".".join(path.relative_to(tool_dir).with_suffix("").parts)
        module = importlib.import_module(f"{PKG}.{rel_mod}")
        loaded.append(relative)
        public_functions.extend(
            (relative, function.__name__, str(inspect.signature(function)))
            for name, function in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_") and function.__module__ == module.__name__
        )
    except Exception as exc:
        failed_files.add(relative)
        errors.append(
            f"import {relative}: {type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(exc))[-2000:]
        )

# ``run_preflight`` is itself Generator-authored and public.  It belongs in
# the exact interface contract even though the coverage tracer must not ask
# the hook to recursively exercise itself.
interface_functions = list(public_functions)
if smoke_path.is_file():
    try:
        spec = importlib.util.spec_from_file_location(
            "_evo_skill_smoke_preflight", smoke_path
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not create the smoke-test import spec")
        smoke_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke_module)
        smoke_hook = getattr(smoke_module, "run_preflight", None)
        if not callable(smoke_hook):
            raise TypeError(
                "tools/preflight.py must define callable "
                "run_preflight(*, skill_dir)"
            )
        signature = inspect.signature(smoke_hook)
        signature.bind(skill_dir=root)
        interface_functions.append(
            ("tools/preflight.py", "run_preflight", str(signature))
        )
    except Exception as exc:
        errors.append(
            f"import tools/preflight.py: {type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(exc))[-2000:]
        )

if interface_functions:
    if not interface_path.is_file():
        errors.append(
            "TOOLS.md is required whenever the skill exposes Python tools; "
            "document every public helper interface"
        )
    else:
        try:
            interface_text = interface_path.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"read TOOLS.md: {type(exc).__name__}: {exc}")
            interface_text = ""

        expected_markers = {
            f"{relative}::{name}{signature}"
            for relative, name, signature in interface_functions
        }
        documented_markers = set()
        for line in interface_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("### `tools/") and stripped.endswith("`"):
                documented_markers.add(stripped[len("### `"):-1])

        for marker in sorted(expected_markers):
            heading = f"### `{marker}`"
            if marker not in documented_markers:
                errors.append(
                    "TOOLS.md missing exact public helper signature: " + heading
                )
                continue
            start = interface_text.find(heading) + len(heading)
            next_heading = interface_text.find("\n### `tools/", start)
            section = interface_text[
                start:next_heading if next_heading >= 0 else len(interface_text)
            ].lower()
            for label in ("parameters", "returns", "raises", "example"):
                if f"**{label}:**" not in section:
                    errors.append(
                        f"TOOLS.md section {heading} missing **{label.title()}:**"
                    )

        for marker in sorted(documented_markers - expected_markers):
            # a file that failed to import has no readable signatures; its
            # headings are not "stale", the import error already says it all
            if any(marker.startswith(f"{failed}::") for failed in failed_files):
                continue
            errors.append(
                "TOOLS.md contains stale or non-public helper signature: "
                f"### `{marker}`"
            )

smoke_required = bool(tool_files)
smoke_ran = False
if smoke_required and not smoke_path.is_file():
    errors.append(
        "tools/preflight.py is required whenever the skill exposes Python "
        "tools; define run_preflight(*, skill_dir)"
    )
elif smoke_path.is_file() and not errors:
    try:
        if smoke_hook is None:
            raise TypeError(
                "tools/preflight.py must define callable "
                "run_preflight(*, skill_dir)"
            )
        smoke_ran = True
        called_functions = set()
        def trace_calls(frame, event, _arg):
            if event == "call":
                try:
                    source = Path(frame.f_code.co_filename).resolve()
                    relative = source.relative_to(root).as_posix()
                except (OSError, ValueError):
                    return trace_calls
                if relative.startswith("tools/") and relative != "tools/preflight.py":
                    called_functions.add((relative, frame.f_code.co_name))
            return trace_calls
        sys.settrace(trace_calls)
        try:
            value = smoke_hook(skill_dir=root)
        finally:
            sys.settrace(None)
        if inspect.isawaitable(value):
            raise TypeError("run_preflight must be synchronous")
        passed = value is True or (
            isinstance(value, dict) and value.get("ok") is True
        )
        if not passed:
            raise AssertionError(f"run_preflight returned {value!r}")
        callable_functions = {
            (relative, name) for relative, name, _signature in public_functions
        }
        uncovered = sorted(callable_functions - called_functions)
        if uncovered:
            names = ", ".join(f"{path}:{name}" for path, name in uncovered)
            raise AssertionError(
                "run_preflight did not exercise every public helper: " + names
            )
    except Exception as exc:
        errors.append(
            f"smoke tools/preflight.py: {type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(exc))[-2000:]
        )

print(json.dumps({
    "ok": not errors,
    "files": [path.relative_to(root).as_posix() for path in python_files],
    "loaded": loaded,
    "errors": errors,
    "smoke_required": smoke_required,
    "smoke_ran": smoke_ran,
}, ensure_ascii=False))
'''


PREFLIGHT_TIMEOUT_ENV = "EVO_SKILL_PREFLIGHT_TIMEOUT_S"
# Run 12 (gpt-6-astra, 2026-09-29): the round-8 candidate's own preflight
# tests took 62 s (image-sheet composition on real renders) and the loop
# rejected it on the old 20 s cap before any A/B; the retained skill's
# preflight already took 13 s under two concurrent runs. The preflight runs
# once per candidate, so a generous cap costs nothing.
DEFAULT_PREFLIGHT_TIMEOUT_S = 120.0


def _default_preflight_timeout() -> float:
    import os
    return float(os.environ.get(PREFLIGHT_TIMEOUT_ENV, DEFAULT_PREFLIGHT_TIMEOUT_S))


def preflight_skill(
    skill_dir: str | Path,
    *,
    timeout_sec: float | None = None,
) -> SkillPreflightResult:
    """Run the candidate contract gate in a disposable Python interpreter."""
    if timeout_sec is None:
        timeout_sec = _default_preflight_timeout()
    started = time.monotonic()
    root = Path(skill_dir).resolve()
    if not root.is_dir():
        return SkillPreflightResult(
            ok=False,
            errors=(f"skill directory does not exist: {root}",),
            duration_sec=time.monotonic() - started,
        )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SKILL_DIR"] = str(root)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(root)],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=max(0.1, float(timeout_sec)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SkillPreflightResult(
            ok=False,
            errors=(f"skill preflight exceeded {timeout_sec:g}s",),
            duration_sec=time.monotonic() - started,
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        payload = {}
    errors = [str(item) for item in payload.get("errors", [])]
    if completed.returncode != 0:
        errors.append(
            f"preflight interpreter exited {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    if not payload:
        errors.append(
            "preflight produced no machine-readable result: "
            + (completed.stderr or completed.stdout)[-2000:]
        )
    return SkillPreflightResult(
        ok=bool(payload.get("ok")) and not errors,
        files=tuple(str(item) for item in payload.get("files", [])),
        errors=tuple(errors),
        smoke_required=bool(payload.get("smoke_required")),
        smoke_ran=bool(payload.get("smoke_ran")),
        duration_sec=time.monotonic() - started,
    )
