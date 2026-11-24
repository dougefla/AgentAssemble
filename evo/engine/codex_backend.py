#!/usr/bin/env python3
"""Codex (OpenAI) agent backend — `codex exec` as an alternative to Claude.

Selected per role via ``EvoConfig.executor_backend`` / ``generator_backend ==
"codex"``; dispatched from ``evo.engine.agents.run_executor`` /
``run_generator``. Returns the same ``ExecutorResult`` / ``GeneratorResult`` as
the Claude backend so the loop is backend-agnostic.

Requirements:
  - the ``codex`` CLI on PATH (auth via your local ``codex login``).

The executor drives the persistent sim worker through the in-repo stdio MCP
bridge (``evo.engine.sim_mcp_bridge``), which forwards ``execute_python_code``
calls to the per-worker TCP port. The loop passes the COMPLETE server
definition (command/args/env) via ``-c`` overrides — no
``~/.codex/config.toml`` prerequisite. Codex has no Claude "Agent" subagents,
so the skill's validate-* specialist steps (e.g. validate-joints) fold into the
executor's inline check.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from evo.engine.sim_manager import SimWorkerManager

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY_SDK_DIR = PROJECT_ROOT / "evo" / "assembly_sdk"
CODEX_BIN = os.environ.get("CODEX_BIN", "codex")


def _model_args(model: str | None) -> list[str]:
    """Pass -m only for a genuine Codex/OpenAI model. A claude-* name (the
    loop's default) means "let Codex pick its configured default model"."""
    if model and not model.lower().startswith("claude"):
        return ["-m", model]
    return []


def _common_exec_args(cwd: Path, last_message_file: Path, model: str | None,
                      *, full_access: bool = False) -> list[str]:
    # full_access: the executor drives the sim worker + writes $2 autonomously (parity
    # with the Claude executor's bypassPermissions); it also bypasses any
    # per-tool approval_mode="approve" in the user's ~/.codex config.
    sandbox = (["--dangerously-bypass-approvals-and-sandbox"] if full_access
               else ["-s", "workspace-write", "-c", 'approval_policy="never"'])
    return [
        CODEX_BIN, "exec",
        *_model_args(model),
        "-C", str(cwd),
        "--add-dir", str(cwd),
        "--skip-git-repo-check",
        *sandbox,
        "-o", str(last_message_file),
        "--color", "never",
    ]


def _sim_mcp_config_args(port: int) -> list[str]:
    """Fully define the ``[mcp_servers.sim]`` entry via ``-c`` overrides.

    The stdio bridge ships in-repo (``evo.engine.sim_mcp_bridge``), so unlike
    the legacy blender-mcp setup there is no ``~/.codex/config.toml``
    prerequisite: command, args, and env are all provided here. The bridge
    reads the per-worker sim port from ``EVO_SIM_PORT`` and needs the repo
    root on ``PYTHONPATH`` to import ``evo``."""
    return [
        "-c", f'mcp_servers.sim.command="{sys.executable}"',
        "-c", 'mcp_servers.sim.args=["-m","evo.engine.sim_mcp_bridge"]',
        "-c", f'mcp_servers.sim.env.EVO_SIM_PORT="{port}"',
        "-c", f'mcp_servers.sim.env.PYTHONPATH="{PROJECT_ROOT}"',
    ]


async def _run_codex(
    args: list[str],
    *,
    prompt: str,
    env: dict[str, str],
    timeout: float,
    log_handle,
) -> tuple[int, str]:
    """Run `codex exec`, feeding the prompt on stdin; return (returncode, tail)."""
    from evo.engine.agents import _write_log  # lazy: avoid import cycle

    _shown = ["mcp_servers.*" if a.startswith("mcp_servers") else a for a in args[1:]]
    _write_log(log_handle, "codex", "exec " + " ".join(_shown))
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        _write_log(log_handle, "codex", f"TIMEOUT after {timeout}s")
        return 124, f"codex exec timed out after {timeout}s"
    text = (out or b"").decode("utf-8", errors="replace")
    tail = text[-4000:]
    _write_log(log_handle, "codex", f"returncode={proc.returncode}")
    _write_log(log_handle, "codex-output", tail)
    return proc.returncode or 0, tail


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

async def run_executor_codex(
    *,
    factory: str,
    instance: str,
    input_dir: Path,
    output_dir: Path,
    skill_dir: Path,
    thumbnail_path: Path | None = None,
    model: str | None = None,
    max_turns: int = 60,           # accepted for signature parity (unused)
    log_file: Path | None = None,
    skill_sha: str | None = None,
    extra_result_fields: dict[str, Any] | None = None,
    sim_port: int = 9876,
    sim_headless: bool = True,
    stage: str = "full",
    effort: str | None = None,
    subagent_model: str | None = None,
    subagent_effort: str | None = None,
    auth: str | None = "local",
    timeout: float = 1800.0,
):
    """Run the /assemble skill on one task via `codex exec` driving the sim worker."""
    from evo.engine.agents import (  # lazy: avoid import cycle
        ExecutorResult,
        _load_skill_as_executor_context,
        _write_log,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    skill_append, _subagents = _load_skill_as_executor_context(
        skill_dir,
        thumbnail_path=thumbnail_path,
        subagent_model=subagent_model,
        subagent_effort=subagent_effort,
        runtime_globals_injected=True,
    )
    thumb_line = (
        f"  Reference thumbnail: {thumbnail_path}\n" if thumbnail_path else ""
    )
    prompt = (
        f"{skill_append}\n\n"
        "----------------------------------------------------------------\n"
        "Run the /assemble skill above on these arguments:\n"
        f"  $1 (input OBJ directory, READ-ONLY) = {input_dir}\n"
        f"  $2 (output directory) = {output_dir}\n"
        f"{thumb_line}"
        "Drive the simulation backend ONLY through the "
        "`mcp__sim__execute_python_code` tool (a persistent Python session "
        "with the frozen `assembly` SDK preloaded; variables persist across "
        "calls). The Claude `Agent` tool is NOT available here — wherever "
        "the skill says to call a `validate-*` specialist such as `validate-joints` "
        "subagents, instead render the images and do that visual check "
        "yourself inline. Write `$2/furniture.urdf` + per-link meshes. "
        "Respond in English. Begin."
    )

    env = dict(os.environ)
    if not env.get("OPENAI_API_KEY"):
        log.info("codex: no OPENAI_API_KEY in env; using local `codex login`")

    f = open(log_file, "w") if log_file else None
    t0 = time.time()
    last_msg = output_dir / ".codex_last_message.txt"
    rc, tail = 1, ""
    try:
        _write_log(f, "config", f"backend=codex model={model or '(codex default)'} port={sim_port} stage={stage} effort={effort or '-'}")
        _write_log(f, "prompt", prompt[:2000])
        with SimWorkerManager(
            port=sim_port, headless=sim_headless,
            runtime_globals={
                "ASSEMBLY_SDK_DIR": str(ASSEMBLY_SDK_DIR),
                "SKILL_DIR": str(skill_dir),
                "INPUT_DIR": str(input_dir),
                "OUTPUT_DIR": str(output_dir),
            },
        ):
            _write_log(f, "sim", "Sim worker ready")
            args = (
                _common_exec_args(output_dir, last_msg, model, full_access=True)
                + _sim_mcp_config_args(sim_port)
            )
            rc, tail = await _run_codex(args, prompt=prompt, env=env, timeout=timeout, log_handle=f)
    except Exception as e:  # noqa: BLE001
        # A crash/timeout is a failure even if a (possibly half-written)
        # furniture.urdf exists — keep urdf_path for debugging, but never
        # send an unconfirmed artifact to the oracle as a success.
        urdf = output_dir / "furniture.urdf"
        result = ExecutorResult(
            factory=factory, instance=instance, output_dir=output_dir,
            urdf_path=urdf if urdf.exists() else None,
            success=False, error=str(e), duration=time.time() - t0,
        )
        if extra_result_fields:
            for k, v in extra_result_fields.items():
                setattr(result, k, v)
        if f:
            f.close()
        return result

    urdf = output_dir / "furniture.urdf"
    # Success requires BOTH a clean codex exit AND the artifact: a non-zero
    # rc (timeout=124, CLI crash, ...) with a leftover urdf is still a
    # failure — don't let half-finished runs pollute the oracle scores.
    success = rc == 0 and urdf.exists()
    result = ExecutorResult(
        factory=factory, instance=instance, output_dir=output_dir,
        urdf_path=urdf if urdf.exists() else None, success=success,
        error=None if success else (
            f"codex exec rc={rc}; furniture.urdf "
            f"{'present (untrusted)' if urdf.exists() else 'missing'}"
        ),
        duration=time.time() - t0, transcript=[tail],
    )
    if extra_result_fields:
        for k, v in extra_result_fields.items():
            setattr(result, k, v)
    _write_log(f, "done", f"success={success} duration={result.duration:.1f}s")
    if f:
        f.close()
    return result


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

async def run_generator_codex(
    *,
    current_skill_dir: Path,
    current_skill_content: dict[str, str],
    feedback_history: list[dict[str, Any]],
    output_skill_dir: Path,
    model: str | None = None,
    max_turns: int = 200,          # accepted for parity (unused)
    log_file: Path | None = None,
    effort: str | None = None,
    sim_port: int | None = None,          # accepted for parity (unused)
    sim_headless: bool = True,            # accepted for parity (unused)
    auth: str | None = "local",
    timeout: float = 1800.0,
):
    """Improve the skill in `output_skill_dir` via `codex exec` (file edits only)."""
    import json as _json

    from evo.engine.agents import GeneratorResult, _write_log  # lazy

    # Pre-populate the output skill dir with the current skill (caller may also
    # do this; copytree is idempotent with dirs_exist_ok).
    if output_skill_dir.exists():
        shutil.rmtree(output_skill_dir)
    if current_skill_dir.exists():
        shutil.copytree(current_skill_dir, output_skill_dir)

    role = ""
    gen_md = PROJECT_ROOT / "evo" / "agents" / "generator.md"
    if gen_md.exists():
        role = gen_md.read_text()

    feedback_json = _json.dumps(feedback_history[-12:], indent=2, default=str)[:24000]
    prompt = (
        f"{role}\n\n"
        "----------------------------------------------------------------\n"
        "You are improving the /assemble skill in your current working "
        "directory (SKILL.md, tools/*.py, agents/*.md). ACTIONS.md is the "
        "frozen Assembly SDK contract: read it, but never edit it or restate "
        "its signatures differently. Preserve every keyword-only marker. Edit the "
        "files in place to fix the failures shown in the feedback below. Keep "
        "changes minimal and targeted; do not rewrite working tools wholesale. "
        "Do NOT touch any path outside this directory.\n\n"
        f"Per-task feedback from the latest round:\n{feedback_json}\n"
    )

    env = dict(os.environ)
    if not env.get("OPENAI_API_KEY"):
        log.info("codex: no OPENAI_API_KEY in env; using local `codex login`")

    f = open(log_file, "w") if log_file else None
    last_msg = output_skill_dir / ".codex_last_message.txt"
    try:
        _write_log(f, "config", f"backend=codex(gen) model={model or '(codex default)'} effort={effort or '-'}")
        args = _common_exec_args(output_skill_dir, last_msg, model)
        rc, tail = await _run_codex(args, prompt=prompt, env=env, timeout=timeout, log_handle=f)
    finally:
        # last-message file is an artifact, not part of the skill
        try:
            last_msg.unlink()
        except OSError:
            pass

    # Compute modified files by diffing against the pre-edit content.
    modified: list[str] = []
    for fp in output_skill_dir.rglob("*"):
        if fp.is_file() and fp.suffix in (".md", ".py"):
            rel = str(fp.relative_to(output_skill_dir))
            try:
                current = fp.read_text()
            except OSError:
                continue
            if current_skill_content.get(rel) != current:
                modified.append(rel)

    result = GeneratorResult(
        skill_dir=output_skill_dir,
        changes_summary=tail[-2000:],
        files_modified=modified,
        raw_response=tail,
    )
    _write_log(f, "result", f"modified={modified}")
    if f:
        f.close()
    return result
