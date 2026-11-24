#!/usr/bin/env python3
"""Agent definitions for the OpenAI Responses, Claude, and Codex backends.

Three agents drive the co-evolution loop:
  - Executor: runs /assemble skill on a task (produces URDF)
  - Verifier: checks predicted URDF without seeing GT (produces diagnostics)
  - Generator: improves SKILL.md based on feedback (produces new skill version)

Information isolation is enforced by WHAT THE PROMPT CONTAINS, not by tool
permissions. Each agent receives only the information it should see, embedded
directly in its prompt. No agent is told about sensitive file paths (oracle.json,
mapping.json, gt/, evolution.jsonl, etc.).

  Executor:  sees task input dir + output dir. Prompt contains NO paths to
             GT, logs, or evolution metadata.
  Verifier:  sees predicted URDF content + part filenames. ZERO tools.
             All context embedded in prompt. Cannot access filesystem.
  Generator: sees current SKILL.md/tools content + feedback history.
             Has Write/Edit to modify skill files. Prompt contains NO paths
             to oracle results, GT, or evolution logs. cwd = isolated skill dir.
"""
from __future__ import annotations

import json
import os
import re
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import yaml

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from evo.engine.model_defaults import DEFAULT_AGENT_EFFORT, DEFAULT_AGENT_MODEL
from evo.oracle.check_urdf import JointReport
from evo.engine.sim_manager import (
    SimStartupError,
    SimWorkerManager,
    send_to_sim_worker,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY_SDK_DIR = PROJECT_ROOT / "evo" / "assembly_sdk"


def _snapshot_frozen_sdk() -> dict[str, bytes]:
    """Capture immutable SDK sources before a Generator gets sim access."""
    return {
        str(path.relative_to(ASSEMBLY_SDK_DIR)): path.read_bytes()
        for path in sorted(ASSEMBLY_SDK_DIR.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".md"}
    }


def _restore_frozen_sdk(snapshot: dict[str, bytes]) -> list[str]:
    """Restore SDK sources and report any attempted Generator mutations."""
    current = {
        str(path.relative_to(ASSEMBLY_SDK_DIR)): path
        for path in sorted(ASSEMBLY_SDK_DIR.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".md"}
    }
    mutated = sorted(
        relative
        for relative in set(snapshot) | set(current)
        if relative not in snapshot
        or relative not in current
        or current[relative].read_bytes() != snapshot[relative]
    )
    if not mutated:
        return []
    for relative in set(current) - set(snapshot):
        current[relative].unlink()
    for relative, content in snapshot.items():
        destination = ASSEMBLY_SDK_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.read_bytes() != content:
            destination.write_bytes(content)
    return mutated

# Hard-disable Claude Code's fast lane for every SDK subprocess.  `effort`
# controls reasoning depth and is intentionally independent of this setting.
# Keep both the process-level gate and the explicit session settings so a
# user/global Claude configuration cannot turn fast mode back on.
CLAUDE_FAST_MODE_DISABLED_ENV = {"CLAUDE_CODE_DISABLE_FAST_MODE": "1"}
# Largest single JSON message the claude-agent-sdk transport accepts (default 1 MiB).
CLAUDE_SDK_MAX_BUFFER_BYTES = 16 * 1024 * 1024
CLAUDE_FAST_MODE_DISABLED_SETTINGS = json.dumps({
    "fastMode": False,
    "fastModePerSessionOptIn": True,
})


# Sim MCP server config. We use an SDK in-process MCP server (NOT a stdio
# subprocess) so that proxy-routed runs (--auth proxy) also see the tool:
# the bundled `claude` CLI gates stdio MCP off when ANTHROPIC_BASE_URL is
# set, but lets sdk-type MCP through. The exposed tool name is
# `mcp__sim__execute_python_code` — the same name the Responses adapter and
# the Codex stdio bridge register, so SKILL.md is backend-agnostic.
#
# `make_sim_mcp_server(port)` is parallel-safe: each call closes over its
# own `port`, so workers in the same process can each have their own sim
# worker socket.

def make_sim_mcp_server(port: int = 9876):
    """Build an in-process MCP server bound to a specific sim-worker port."""

    @tool(
        "execute_python_code",
        "Execute Python code in a persistent Python session with the frozen "
        "`assembly` SDK preloaded and the immutable runtime globals "
        "(ASSEMBLY_SDK_DIR, SKILL_DIR, INPUT_DIR, OUTPUT_DIR) injected. "
        "Variables, imports, and helper functions persist across calls. "
        "Returns captured stdout/stderr plus the value of a trailing "
        "expression or a variable named `result`. Plain interpreter, not a "
        "Jupyter kernel: `display()`, IPython helpers, and `%` magics do not "
        "exist; use print() and write images to files (view PNGs with the "
        "Read tool). Each session has a resident-memory cap (default 6 GB); a "
        "snippet that crosses it is interrupted with a MemoryError.",
        {"code": str},
    )
    async def execute_python_code(args: dict) -> dict:
        code = args.get("code", "")
        try:
            resp = send_to_sim_worker(code, port)
        except Exception as e:
            from evo.engine.sim_manager import memory_kill_notice
            notice = memory_kill_notice(port)
            return {"content": [{"type": "text", "text": notice or f"socket error: {e}"}]}
        # Worker response: status "ok"/"error", result (ok), message (error),
        # optional stdout/stderr. Surface them as text.
        parts: list[str] = []
        if resp.get("stdout"):
            parts.append(str(resp["stdout"]).rstrip())
        if resp.get("stderr"):
            parts.append("STDERR:\n" + str(resp["stderr"]).rstrip())
        if resp.get("status") == "ok":
            result = resp.get("result", {})
            if result not in ({}, None, ""):
                parts.append("RESULT: " + json.dumps(result, default=str))
        else:
            parts.append(f"SIM ERROR: {resp.get('message', '?')}")
        text = "\n".join(parts) if parts else "(no output)"
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(
        name="sim",
        version="0.1.0",
        tools=[execute_python_code],
    )


def _parse_agent_md(
    text: str, stem: str
) -> tuple[str, str, list[str] | None, str | None, str | None, str]:
    """Parse a subagent .md file (YAML frontmatter + body).

    Returns (name, description, tools, model, effort, body). The frontmatter
    is parsed with yaml.safe_load (so quoted strings, lists, and `:` inside
    values all work); `tools:` accepts either a YAML list or a comma-separated
    string. Malformed frontmatter is warned about and ignored (the block is
    still stripped from the body).
    """
    name, description = stem, ""
    tools: list[str] | None = None
    model: str | None = None
    effort: str | None = None
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            body = text[end + 4:].lstrip("\n")
            try:
                fm = yaml.safe_load(text[3:end])
            except yaml.YAMLError as e:
                log.warning(f"agents/{stem}.md: malformed YAML frontmatter ({e}); ignoring it")
                fm = None
            if isinstance(fm, dict):
                if fm.get("name"):
                    name = str(fm["name"])
                description = str(fm.get("description") or "")
                raw_tools = fm.get("tools")
                if isinstance(raw_tools, str):
                    tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
                elif isinstance(raw_tools, list):
                    tools = [str(t).strip() for t in raw_tools if str(t).strip()]
                if fm.get("model"):
                    model = str(fm["model"])
                if fm.get("effort"):
                    effort = str(fm["effort"])
    return name, description, tools, model, effort, body


def _format_joint_report(rep: dict[str, Any]) -> str:
    """One line per GT joint: pred joint/type, then ✓/✗ with the error size."""
    if not rep.get("matched"):
        return "unmatched (GT child link matched no predicted link)"
    if not rep.get("pred_joint"):
        return "no predicted joint drives the matched link"
    def mark(ok):
        return "✓" if ok else "✗"
    # Only the dimensions this dataset scores (JointReport.DIMS): a view that
    # hands the executor the joint types or limits must not show them as
    # something to optimize.
    scored = set(JointReport.DIMS)
    parts = [f"pred={rep.get('pred_joint')} ({rep.get('pred_type')})"]
    if "type_ok" in scored:
        parts.append(f"type{mark(rep.get('type_ok'))}")
    if rep.get("axis_ok") is not None:
        deg = rep.get("axis_angle_deg")
        parts.append(f"axis{mark(rep['axis_ok'])}" + (f" {deg:.0f}°" if deg is not None else ""))
    if rep.get("pivot_ok") is not None:
        dist = rep.get("pivot_distance_m")
        parts.append(f"pivot{mark(rep['pivot_ok'])}" + (f" {dist*100:.1f}cm" if dist is not None else ""))
    elif rep.get("pred_type") == "REVOLUTE" and rep.get("axis_ok") is False:
        parts.append("pivot? (not comparable until the axis is right)")
    if rep.get("limits_ok") is not None and "limits_ok" in scored:
        ratio = rep.get("limit_ratio")
        parts.append(f"limits{mark(rep['limits_ok'])}" + (f" ×{ratio:.2f}" if ratio is not None else ""))
    if rep.get("motion_ok") is not None:
        drift = rep.get("trace_drift_m")
        parts.append(f"motion{mark(rep['motion_ok'])}" + (f" drift {drift*100:.1f}cm" if drift is not None else ""))
    if rep.get("scale_ok") is not None and "scale_ok" in scored:
        ratio = rep.get("scale_ratio")
        parts.append(f"scale{mark(rep['scale_ok'])}" + (f" ×{ratio:.2f}" if ratio is not None else ""))
    # Oracle 3: the child link's rest pose against its tolerance and, when it
    # is out, which single correction of the link would bring it in.
    rd, tol = rep.get("rest_drift_m"), rep.get("pose_tol_m")
    if rd is not None and tol:
        rest = f"rest {rd*100:.1f}cm/tol {tol*100:.1f}cm"
        cause = rep.get("pose_cause")
        if cause:
            extra = []
            if rep.get("rot_err_deg") is not None and "orientation" in cause:
                extra.append(f"{rep['rot_err_deg']:.0f}° off")
            if rep.get("pos_err_m") is not None and "position" in cause:
                extra.append(f"pivot {rep['pos_err_m']*100:.1f}cm off")
            rest += f" → fix {cause}" + (f" ({', '.join(extra)})" if extra else "")
        parts.append(rest)
    tail = f"  score {rep.get('score', 0):.2f}"
    if rep.get("reps", 1) > 1:
        spread = rep.get("score_spread")
        tail += f" ({rep['reps']} reps" + (f", spread {spread:.2f}" if spread is not None else "") + ")"
        if spread is not None and spread >= 0.5:
            tail += " UNSTABLE"
    return "  ".join(parts) + tail



def _dataset_task_block(tasks_root: Path | None) -> str:
    """The task definition the Generator must optimize for, from the dataset's
    ``dataset_variant.json`` (sibling of ``tasks/``).

    The views differ in what the executor has to produce: ``artvip_beginner``
    pre-places every part and asks for the whole joint annotation;
    ``artvip_jointfill`` gives the tree/types/limits and asks for pivots and
    axes; ``artvip_posefill`` scrambles the non-root parts and asks for their
    poses as well. A Generator prompt written for one view misdiagnoses the
    others, so the definition is read from the data, never hard-coded.
    """
    variant: dict = {}
    marker = Path(tasks_root).parent / "dataset_variant.json" if tasks_root else None
    if marker is not None and marker.is_file():
        try:
            variant = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            variant = {}
    transforms = variant.get("task_mesh_transforms") or []
    preplaced = "world_bake_q0" in transforms
    lines = [f"### Task definition (dataset `{variant.get('dataset', 'unknown')}`)"]
    if not variant:
        # A missing marker used to fall through silently to the beginner-view
        # defaults below and mis-describe every other dataset; say so instead.
        lines.append(f"- WARNING: no readable dataset_variant.json at {marker}; the lines "
                     "below are generic defaults, not this dataset's definition.")
    task = variant.get("executor_task")
    if task:
        lines.append(f"- Executor task: {task}")
    else:
        lines.append("- Executor task: annotate the joints of pre-placed parts (tree, type, pivot, axis, limits) and export a URDF.")
    if variant.get("given"):
        lines.append("- Given to the executor: " + "; ".join(variant["given"]) + ".")
    if variant.get("hidden"):
        lines.append("- Hidden, to be produced: " + "; ".join(variant["hidden"]) + ".")
    policy = variant.get("task_mesh_policy")
    if policy:
        lines.append(f"- Task meshes: {policy}.")
    if preplaced:
        lines.append("- Parts are pre-placed at the GT rest pose (SDK `mesh_frame='world'`): the SDK never "
                     "moves a mesh, a rest render must match the reference, and `audit()` reports any drift.")
    else:
        lines.append("- Parts are NOT pre-placed (SDK `mesh_frame='link'`): placing parts is the executor's job, "
                     "the SDK offers no no-move guarantee, and a wrong rest pose is a placement error, not an SDK violation.")
    dims = variant.get("scored_joint_dims")
    if dims:
        lines.append("- Scored joint dimensions: " + ", ".join(d[:-3] for d in dims)
                     + " (the other dimensions are given and never counted).")
    from evo.oracle.check_urdf import ALL_CHECKS, DEFAULT_SCORED_CHECKS
    scored_checks = variant.get("scored_checks") or list(DEFAULT_SCORED_CHECKS)
    diag = [c for c in ALL_CHECKS if c not in set(scored_checks)]
    lines.append("- Scored oracle checks (score = fraction passed; all must pass): "
                 + ", ".join(c for c in ALL_CHECKS if c in set(scored_checks)) + ".")
    if diag:
        lines.append("- Diagnostic-only checks (reported, never counted): " + ", ".join(diag) + ".")
    return "\n".join(lines)

def _format_joint_summary(runs: list[dict[str, Any]]) -> str:
    """Batch-level per-joint pass counts grouped by the predicted joint type.

    Only the dimensions this dataset scores (``JointReport.DIMS``) are listed.
    """
    dims = tuple(JointReport.DIMS)
    by_type: dict[str, dict[str, list[int]]] = {}
    total = 0
    for run in runs:
        for rep in (run.get("joint_reports") or {}).values():
            total += 1
            key = rep.get("pred_type") or ("unmatched" if not rep.get("matched") else "missing")
            slot = by_type.setdefault(key, {d: [0, 0] for d in dims})
            for d in dims:
                v = rep.get(d)
                if v is None:
                    continue
                slot[d][1] += 1
                slot[d][0] += int(bool(v))
    if not total:
        return ""
    lines = [f"  Per-joint outcomes across the batch ({total} GT joints, grouped by PREDICTED type):"]
    for key, slot in sorted(by_type.items()):
        cells = "  ".join(
            f"{d[:-3]} {ok}/{n}" for d, (ok, n) in slot.items() if n
        )
        lines.append(f"    {key:<10} {cells}")
    reps = max((rep.get("reps", 1) for run in runs for rep in (run.get("joint_reports") or {}).values()), default=1)
    if reps > 1:
        lines.append(f"  Each task ran {reps} independent executor sessions; ✓/✗ are majority votes and scores are means. "
                     "A joint marked UNSTABLE passed in some sessions and failed in others — that is executor noise, "
                     "not a skill rule to fix; stable failures are the ones worth a tool.")
    lines.append("  (per task below: each GT joint → predicted joint, ✓/✗ per dimension with the error size)")
    return "\n".join(lines)


def _format_previous_candidate_outcome(prev: dict[str, Any], *, same_tasks: bool = False) -> str:
    """Tell the Generator what became of the skill it wrote last round.

    Without this it optimizes blind. The per-check trajectory it is shown
    compares a DIFFERENT sample of tasks each round, so a discarded edit can
    look like it worked: an unchanged skill scored 1/10 then 3/10 on two
    disjoint tranches, and the Generator — never told its candidate had been
    rejected — read that as confirmation and doubled down, producing a
    candidate that dropped the same-asset replay from 3/10 to 0/10.

    `prev` is the feedback entry for the round that produced the candidate,
    which is `feedback_history[-2]`: the A/B verdict is written onto that
    entry after its generator turn, while `feedback_history[-1]` is the
    current round's entry whose A/B has not run yet.
    """
    if not prev:
        return ""
    ab = prev.get("candidate_ab_test") or {}
    validation_reason = prev.get("generator_output_rejected")
    updates = prev.get("updates") or []
    if not ab and not validation_reason and not updates:
        if prev.get("generator_changed"):
            return (
                "\n### Your previous candidate was ADOPTED\n"
                "The skill in your working directory includes last round's edit, and this "
                "round's evaluation below ran with it: compare it with the previous round's "
                "numbers"
                + (" (same tasks both rounds; executor sampling noise remains).\n" if same_tasks
                   else " knowing the task sample differs.\n")
            )
        return ""
    if len(updates) > 1:
        lines = ["\n### Last round ran several independent updates, each A/B-tested on the same assets"]
        for u in updates:
            t = u.get("paired_t")
            lines.append(
                f"  update {u.get('update')}: {u.get('action', '?')} ({u.get('reason')}); "
                f"paired n={u.get('paired_n')} mean_diff={u.get('paired_mean_diff', 0) or 0:+.3f}"
                + (f" t={t:.2f}" if isinstance(t, (int, float)) else "")
                + (f" — {u.get('summary')}" if u.get("summary") else "")
            )
        lines.append("  The skill in your working directory contains every ACCEPTED update and none of the rejected ones.")
        lines.append(
            "\nTreat these as the only controlled evidence about your last changes. The per-check "
            "trajectory below is NOT a controlled comparison. Do not re-apply a rejected change "
            "without new evidence.\n"
        )
        return "\n".join(lines) + "\n"

    lines: list[str] = []
    if validation_reason and not ab:
        # No gate ran: the acceptance check (frozen contract, TOOLS.md,
        # compile/import, preflight hook) refused the turn's last attempt.
        return (
            "\n### NOTE — your previous candidate was REJECTED by the acceptance check and DISCARDED\n"
            "The skill in your working directory does NOT contain last round's edits; "
            "the loop kept the earlier skill.\n"
            f"  reason: {validation_reason}\n"
        )
    accepted = bool(ab.get("accepted"))
    if accepted:
        lines.append(
            "\n### Your previous candidate was ACCEPTED and is in effect\n"
            "The skill in your working directory includes last round's edits."
        )
    else:
        lines.append(
            "\n### NOTE — your previous candidate was REJECTED and was DISCARDED\n"
            "The skill in your working directory does NOT contain last round's "
            "edits. They were replayed on the exact same Train assets and did "
            "not survive the gate, so the loop kept the earlier skill."
        )
    if ab:
        lines.append(
            f"  verdict: {ab.get('action', '?')}"
            + (f" ({ab.get('reason')})" if ab.get("reason") else "")
        )
        b_pass, c_pass = ab.get("baseline_pass_rate"), ab.get("candidate_pass_rate")
        b_score, c_score = ab.get("baseline_mean_score"), ab.get("candidate_mean_score")
        n = ab.get("sample_count")
        if b_pass is not None and c_pass is not None:
            lines.append(
                f"  pass rate:  {b_pass:.0%} -> {c_pass:.0%}"
                + (f"   (same {n} assets, replayed)" if n else "")
            )
        if b_score is not None and c_score is not None:
            lines.append(f"  mean score: {b_score:.3f} -> {c_score:.3f}")
        for key, label in (
            ("critical_regressions", "checks that PASSED before and FAILED under your candidate"),
            ("candidate_tool_contract_errors", "helper contract errors your candidate introduced"),
            ("executor_code_errors", "executor code errors under your candidate"),
        ):
            items = ab.get(key) or []
            if not items:
                continue
            lines.append(f"  {label} ({len(items)}):")
            for item in list(items)[:12]:
                lines.append(f"    - {item}")
            if len(items) > 12:
                lines.append(f"    - ... and {len(items) - 12} more")
    lines.append(
        "\nTreat this as the only controlled evidence about your last change. "
        "The per-check trajectory below is NOT a controlled comparison: each "
        "round samples different tasks, so its movement reflects task "
        "difficulty as much as anything you edited. Do not re-apply a "
        "rejected change without new evidence, and do not infer that a "
        "rejected change helped because the trajectory rose.\n"
    )
    return "\n".join(lines) + "\n"


def _load_skill_as_executor_context(
    skill_dir: Path,
    thumbnail_path: Path | None = None,
    subagent_model: str | None = None,
    subagent_effort: str | None = None,
    runtime_globals_injected: bool = False,
) -> tuple[str, dict[str, AgentDefinition]]:
    """Read a skill package into an inline system-prompt append + subagents dict.

    Lets us run the /assemble skill WITHOUT loading user-level settings from
    `~/.claude/` — so the executor doesn't acquire the auto-memory system
    (which would leak context across evaluation runs).

    Skill layout:
      SKILL.md                   → main instructions
      ACTIONS.md                 → frozen Assembly SDK reference (appended)
      TOOLS.md                   → evolved helper interface reference (appended)
      agents/<name>.md           → subagent definitions (YAML frontmatter + body)

    Pre-resolution of placeholders:
      - Runs with injected runtime globals leave ASSEMBLY_SDK_DIR /
        SKILL_DIR / INPUT_DIR / OUTPUT_DIR as variables because the sim
        worker injects them into every call. Legacy call sites without
        injection still resolve SKILL_DIR to the absolute skill path.
      - $1/render_thumbnail.png (and the legacy $1/thumbnail.png) → concrete
        `thumbnail_path` (if it exists), or a "(no thumbnail available)"
        marker. On-disk SKILL.md assumes the thumbnail is a sibling of the
        OBJ files inside $1, but callers pass `parts/` as $1 with the file
        at `$1.parent/render_thumbnail.png`. Resolving the full path here
        decouples the executor from that contract mismatch.

    A header banner is prepended explaining what was pre-resolved so the
    agent uses concrete paths directly and does not re-discover them.
    """
    skill_dir_abs = str(skill_dir.resolve())
    thumbnail_abs = str(thumbnail_path.resolve()) if thumbnail_path and thumbnail_path.exists() else None
    thumbnail_marker = thumbnail_abs or "(no thumbnail available for this task)"

    def _inline(text: str) -> str:
        # Drop the interactive-mode discovery hint; it's wrong in this context.
        # (The isolation notice is already in the header banner prepended below.)
        text = text.replace(
            "Python scripts in `tools/` run inside the sim worker via MCP. "
            "`SKILL_DIR` is the absolute path of this skill directory — "
            "when invoked interactively via `/assemble`, find it via "
            "`readlink ~/.claude/skills/assemble`.",
            "Python scripts in `tools/` run inside the sim worker via MCP. "
            "(Paths below are pre-resolved — see header notes at the top.)",
        )
        # Substitute every $1/render_thumbnail.png (and legacy $1/thumbnail.png)
        # reference with the concrete path (or absence marker). Must run BEFORE
        # the SKILL_DIR substitution so the marker text — which may contain
        # plain words — is not reshaped.
        text = text.replace("$1/render_thumbnail.png", thumbnail_marker)
        text = text.replace("$1/thumbnail.png", thumbnail_marker)
        if runtime_globals_injected:
            return text
        # Legacy MCP backends do not inject runtime globals.
        return text.replace("SKILL_DIR", skill_dir_abs)

    skill_md = _inline((skill_dir / "SKILL.md").read_text())
    actions_md_path = skill_dir / "ACTIONS.md"
    actions_md = _inline(actions_md_path.read_text()) if actions_md_path.exists() else ""
    tools_md_path = skill_dir / "TOOLS.md"
    tools_md = _inline(tools_md_path.read_text()) if tools_md_path.exists() else ""

    # Header describing the isolation context. NOT put through _inline, so the
    # phrase "SKILL_DIR" can appear here as an explanatory token.
    thumbnail_line = (
        f"- Reference thumbnail: `{thumbnail_abs}` (already substituted for every `$1/render_thumbnail.png`\n"
        "  reference below). Read it once before making geometry decisions.\n"
        if thumbnail_abs
        else "- Reference thumbnail: NOT PROVIDED. Every `$1/render_thumbnail.png` reference has been\n"
             "  replaced with a '(no thumbnail available)' marker. Skip thumbnail-based analysis;\n"
             "  use geometry + per-part renders alone.\n"
    )
    runtime_line = (
        "- `ASSEMBLY_SDK_DIR`, `SKILL_DIR`, `INPUT_DIR`, and `OUTPUT_DIR` are injected into every `mcp__sim__execute_python_code` call.\n"
        "  They are Python globals and are mirrored in `os.environ`. Use the globals directly\n"
        "  and unchanged; do not assign, discover, or repair these runtime-owned values.\n"
        "- The sim session is persistent: your other Python variables, imports, and helpers\n"
        "  DO survive across calls, and `assembly` is already preloaded (re-executing\n"
        "  runtime.py is allowed and idempotent).\n"
        if runtime_globals_injected
        else f"- `SKILL_DIR` has been resolved to `{skill_dir_abs}` throughout the text below.\n"
    )
    header = (
        "# /assemble Skill (inlined by the isolated executor — NOT from ~/.claude/)\n\n"
        "Execution environment notes (read before starting):\n"
        + runtime_line
        + thumbnail_line
        + (
            "- Tool source is intentionally not inlined. "
            + "; ".join(filter(None, [
                "ACTIONS.md is the frozen SDK contract" if actions_md else "",
                "TOOLS.md is the evolved-helper contract" if tools_md else "",
            ]))
            + ". Follow the exact signatures and examples. Read a specific source only if\n"
            "  those contracts are insufficient.\n"
            if (actions_md or tools_md) else ""
        )
        + "- Do NOT run `readlink ~/.claude/skills/...` or `ls ~/.claude/skills/...`.\n"
        "  Those paths may be missing or stale in this isolated run.\n"
        "- The `/assemble` slash command is NOT registered here; all skill instructions\n"
        "  are in this system prompt already.\n"
    )
    append_parts = [header, skill_md]
    if actions_md:
        append_parts.append("---\n\n# ACTIONS.md\n\n" + actions_md)
    if tools_md:
        append_parts.append("---\n\n# TOOLS.md\n\n" + tools_md)

    append = "\n\n".join(append_parts)

    # Parse subagent md files (YAML frontmatter + body)
    agents: dict[str, AgentDefinition] = {}
    agents_dir = skill_dir / "agents"
    if agents_dir.is_dir():
        for md in sorted(agents_dir.glob("*.md")):
            name, description, tools, sub_model, sub_effort, body = _parse_agent_md(
                md.read_text(), md.stem
            )
            # Explicit experiment-level overrides take precedence over
            # evolvable skill frontmatter. This makes model-comparison arms
            # auditable: a Generator cannot silently route a validator to a
            # different model or reasoning effort.
            chosen_model = subagent_model or sub_model
            chosen_effort = (
                subagent_effort
                if subagent_effort is not None
                else sub_effort
            )
            agent_kwargs: dict[str, Any] = dict(
                description=description or f"Subagent {name}",
                prompt=body.strip(),
                tools=tools,
                model=chosen_model,
                memory=None,  # explicitly no memory for subagents
            )
            if chosen_effort is not None:
                agent_kwargs["effort"] = chosen_effort
            agents[name] = AgentDefinition(**agent_kwargs)
    return append, agents


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ExecutorResult:
    factory: str
    instance: str
    output_dir: Path
    urdf_path: Path | None = None
    success: bool = False
    error: str | None = None
    duration: float = 0.0
    transcript: list[str] = field(default_factory=list)


@dataclass
class GeneratorResult:
    skill_dir: Path | None = None
    changes_summary: str = ""
    files_modified: list[str] = field(default_factory=list)
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Streaming log helper
# ---------------------------------------------------------------------------

def _write_log(f: IO | None, tag: str, text: str) -> None:
    """Write a tagged line to a log file, flushing immediately."""
    if f is None:
        return
    timestamp = time.strftime("%H:%M:%S")
    for line in text.splitlines():
        f.write(f"[{timestamp}] [{tag}] {line}\n")
    f.flush()


def _fmt_usage(u: dict[str, Any] | None) -> str:
    """Compact one-line usage summary. Anthropic usage shape:
    input_tokens / output_tokens / cache_read_input_tokens / cache_creation_input_tokens.
    """
    if not u:
        return "(no usage)"
    return (
        f"in={u.get('input_tokens', 0)} "
        f"out={u.get('output_tokens', 0)} "
        f"cache_r={u.get('cache_read_input_tokens', 0)} "
        f"cache_w={u.get('cache_creation_input_tokens', 0)}"
    )


# Pricing for cost estimation (USD / 1M tokens). Matches the third-party
# proxy invoice we observed; change here if moving to a different gateway.
# Kept loosely co-located with _summarize_session below — the two are always
# edited together.
_PRICE_INPUT = 3.57
_PRICE_CACHE_READ = 0.36
_PRICE_CACHE_WRITE_5M = 4.46
_PRICE_CACHE_WRITE_1H = 5.57
_PRICE_OUTPUT = 17.86


def setup_auth_env(mode: str) -> dict[str, str | None]:
    """Configure the env the spawned `claude` CLI subprocess inherits.

    mode == "local":
        Use the Claude Code desktop/CLI's own credentials (OAuth token
        or API key stored in ~/.claude/ by `claude setup-token` /
        `claude login`). ANTHROPIC_BASE_URL / AUTH_TOKEN / MODEL are
        stripped from the environment so the CLI falls back to its
        native auth path.

    mode == "proxy":
        Use whatever the shell already exported (typically via
        a `.env` file at the repo root — see evo/engine/envfile.py).
        No mutation here.

    Must be called BEFORE the first run_executor /
    run_generator invocation — the SDK captures env
    when it spawns the claude subprocess.

    Returns a small dict describing the resolved configuration, safe
    to log (token is redacted).
    """
    import os
    info: dict[str, str | None] = {"mode": mode}
    if mode == "local":
        for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL"):
            info[key] = "unset" if os.environ.pop(key, None) is not None else "not-present"
    else:
        info["ANTHROPIC_BASE_URL"] = os.environ.get("ANTHROPIC_BASE_URL", "(unset)")
        info["ANTHROPIC_AUTH_TOKEN"] = (
            "(redacted)" if os.environ.get("ANTHROPIC_AUTH_TOKEN") else "(unset)"
        )
        info["ANTHROPIC_MODEL"] = os.environ.get("ANTHROPIC_MODEL", "(unset)")
    return info


def _git_sha(repo_dir: Path) -> str | None:
    """Short git SHA of repo_dir. Returns None if not a git repo or git is
    not available. Used to stamp which code version produced each sample."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _summarize_session(jsonl_path: Path | None) -> dict:
    """Parse a session JSONL into an aggregate usage/cost dict.

    Dedupes by message.id — a single API response can span many JSONL
    records when parallel_tool_calls produces multiple content blocks
    (the same high-signal stdout-dedup logic used across the loop). Returns
    zeros if path is missing.
    """
    s = {
        "n_api_calls": 0,
        "fresh_input": 0,
        "cache_read": 0,
        "cache_write_5m": 0,
        "cache_write_1h": 0,
        "output": 0,
        "cost_usd": 0.0,
        "cache_misses": 0,
    }
    if not jsonl_path or not jsonl_path.exists():
        return s
    seen: set[str] = set()
    try:
        with jsonl_path.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") != "assistant":
                    continue
                msg = rec.get("message") or {}
                mid = msg.get("id")
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                u = msg.get("usage") or {}
                cc = u.get("cache_creation") or {}
                fi = u.get("input_tokens", 0)
                cr = u.get("cache_read_input_tokens", 0)
                cw5 = cc.get("ephemeral_5m_input_tokens", 0)
                cw1 = cc.get("ephemeral_1h_input_tokens", 0)
                ot = u.get("output_tokens", 0)
                s["fresh_input"] += fi
                s["cache_read"] += cr
                s["cache_write_5m"] += cw5
                s["cache_write_1h"] += cw1
                s["output"] += ot
                s["n_api_calls"] += 1
                s["cost_usd"] += (
                    fi * _PRICE_INPUT
                    + cr * _PRICE_CACHE_READ
                    + cw5 * _PRICE_CACHE_WRITE_5M
                    + cw1 * _PRICE_CACHE_WRITE_1H
                    + ot * _PRICE_OUTPUT
                ) / 1e6
                # "Whole-prefix rewrite" heuristic — same threshold as analyzer
                if cr == 0 and cw5 > 5000:
                    s["cache_misses"] += 1
    except Exception as e:
        log.warning(f"_summarize_session failed on {jsonl_path}: {e}")
    return s


def _copy_session_jsonl(
    session_id: str | None,
    cwd: Path,
    log_file: Path | None,
) -> Path | None:
    """Copy the Claude Code CLI's native session JSONL alongside `log_file`.

    The SDK spawns the `claude` CLI, which writes a complete structured
    trajectory to `~/.claude/projects/{escaped_cwd}/{session_id}.jsonl`.
    That file captures every message block (tool_use, tool_result, images,
    thinking, per-turn usage) with zero loss — strictly richer than the
    text log produced by `_log_message`.

    Path convention matches `log_file`: `path/to/executor.log` becomes
    `path/to/executor.session.jsonl`. If `log_file` is None, the JSONL
    lands in the cwd as `session.jsonl`.
    """
    if not session_id:
        return None
    # Claude Code's project-dir escape: both '/' and '_' collapse to '-'.
    # Example: <repo>/logs/runs/interactive_run_6_160600
    # → -Users-me-workspace-assemble-skill-output-interactive-run-6-160600
    escaped = str(cwd.resolve()).replace("/", "-").replace("_", "-")
    src_dir = Path.home() / ".claude" / "projects" / escaped
    src = src_dir / f"{session_id}.jsonl"
    if not src.exists():
        return None
    if log_file is not None:
        dest = log_file.parent / f"{log_file.stem}.session.jsonl"
    else:
        dest = cwd / "session.jsonl"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        # Subagent trajectories (verify-assembly, verify-joints, Task-spawned)
        # live in `{session_id}/subagents/agent-*.jsonl` with matching .meta.json.
        # Each subagent call is a separate fresh claude session; without this
        # we'd miss their token usage entirely. Copy the whole subdir verbatim.
        sub_src = src_dir / session_id / "subagents"
        if sub_src.is_dir():
            sub_dest = dest.parent / f"{dest.stem}.subagents"
            if sub_dest.exists():
                shutil.rmtree(sub_dest)
            shutil.copytree(sub_src, sub_dest)
        return dest
    except Exception as e:
        log.warning(f"Failed to copy session JSONL from {src} to {dest}: {e}")
        return None


class APIError(RuntimeError):
    """The API/proxy returned an error response and no retry was attempted.

    Repo policy: API errors are immediate run-level failures — we never
    silently retry, fall back to a different model, or wait out a 5xx.
    The configured (model, base_url, token) is what runs. If the proxy
    is busy or the model is unavailable, mark the run failed cleanly
    so batch evaluation can move on.
    """


def _raise_on_api_error(f: IO | None, msg: Any) -> None:
    """Fail-fast on the first API retry signal.

    The bundled Claude Code CLI retries up to 10× internally on 5xx /
    server-busy responses. That's fine for interactive use but bad for
    batch evaluation: a transient proxy hiccup burns ~5 min of wall
    time and reports nothing useful when retries finally exhaust. We
    abort on the first retry so the failure is fast and the proxy
    response code is preserved in the log.
    """
    if not isinstance(msg, SystemMessage):
        return
    if getattr(msg, "subtype", None) != "api_retry":
        return
    d = msg.data or {}
    err_msg = (
        f"API error (fail-fast policy — retries disabled): "
        f"status={d.get('error_status')!r} error={d.get('error')!r} "
        f"attempt={d.get('attempt')}/{d.get('max_retries')}"
    )
    _write_log(f, "api_error", err_msg)
    raise APIError(err_msg)


def _unwrap_tool_output(raw: Any) -> str:
    """Tool outputs are stored as the repr of a content list
    (``[{'type': 'input_text', 'text': ...}, ...]``); return the plain text,
    with inline images replaced by ``<image>``."""
    text = str(raw)
    try:
        import ast
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        parsed = None
    if isinstance(parsed, list):
        parts = []
        for part in parsed:
            if not isinstance(part, dict):
                parts.append(str(part))
            elif part.get("type") in ("input_text", "text", "output_text"):
                parts.append(str(part.get("text", "")))
            elif part.get("type") == "input_image":
                parts.append("<image>")
        text = "\n".join(parts)
    return re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "<image>", text)


def _log_responses_trajectory(f: IO | None, artifact_path: Path | None,
                              *, call_chars: int = 400, result_chars: int = 600,
                              error_chars: int = 2000) -> None:
    """Append the Responses session's tool calls and results to the log.

    Results that carry an error (``SIM ERROR``, ``socket error``, ``Error``)
    keep more text so the traceback is readable in place.
    """
    if f is None or artifact_path is None or not artifact_path.exists():
        return
    try:
        items = json.loads(artifact_path.read_text(errors="replace")).get("items") or []
    except (OSError, ValueError, AttributeError):
        return
    names: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "function_call":
            name = str(item.get("name", "?"))
            names[str(item.get("call_id"))] = name
            args = str(item.get("arguments", ""))
            if name == "mcp__sim__execute_python_code":
                try:
                    args = json.loads(args).get("code", args)
                except (ValueError, AttributeError):
                    pass
            _write_log(f, "tool_call", f"{name}: {args[:call_chars]}"
                       + (" …" if len(args) > call_chars else ""))
        elif kind == "function_call_output":
            out = _unwrap_tool_output(item.get("output", ""))
            limit = error_chars if re.search(r"SIM ERROR|socket error|Error|Traceback", out) else result_chars
            _write_log(f, "tool_result", f"[{names.get(str(item.get('call_id')), '?')}] "
                       + out[:limit] + (" …" if len(out) > limit else ""))
        elif kind == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    _write_log(f, "assistant", str(part.get("text", ""))[:result_chars])


def _log_message(f: IO | None, msg: Any) -> str:
    """Extract text from a message and write to log. Returns extracted text."""
    collected = ""
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                _write_log(f, "text", block.text)
                collected += block.text
            elif isinstance(block, ToolUseBlock):
                _write_log(f, "tool_call", f"{block.name}({json.dumps(block.input)[:200]})")
            elif isinstance(block, ToolResultBlock):
                content_str = str(block.content)[:300] if block.content else ""
                _write_log(f, "tool_result", content_str)
        if msg.usage is not None:
            _write_log(f, "usage", _fmt_usage(msg.usage))
    elif isinstance(msg, ResultMessage):
        cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd is not None else "n/a"
        _write_log(
            f, "result",
            f"finished: {msg.subtype} turns={msg.num_turns} cost={cost} "
            f"api_ms={msg.duration_api_ms}",
        )
        if msg.usage:
            _write_log(f, "usage_total", _fmt_usage(msg.usage))
        if msg.model_usage:
            # Per-model breakdown (most runs use a single model)
            for model_name, mu in msg.model_usage.items():
                _write_log(f, "usage_model", f"{model_name} → {_fmt_usage(mu)}")
    return collected


# ---------------------------------------------------------------------------
# Executor agent
# ---------------------------------------------------------------------------

async def run_executor(
    factory: str,
    instance: str,
    input_dir: Path,
    output_dir: Path,
    skill_dir: Path,
    thumbnail_path: Path | None = None,
    model: str = DEFAULT_AGENT_MODEL,
    max_turns: int = 60,
    log_file: Path | None = None,
    skill_sha: str | None = None,
    extra_result_fields: dict[str, Any] | None = None,
    sim_port: int = 9876,
    sim_headless: bool = False,
    stage: str = "full",
    effort: str | None = None,
    subagent_model: str | None = None,
    subagent_effort: str | None = None,
    backend: str = "responses",
    auth: str | None = "local",
) -> ExecutorResult:
    """Execute the /assemble skill on a single task.

    Prompt contains ONLY the input and output paths. No references to
    GT, oracle, evolution logs, or mapping files.

    `skill_dir` is read inline into the system prompt + subagent registry,
    so we can run WITHOUT `setting_sources=["user"]` — which also pulls the
    auto-memory system into the context and pollutes evaluation runs.

    `thumbnail_path` (optional) is pre-resolved by the caller and threaded
    into the inlined skill text so the agent reads the right file even when
    `$1` does not match SKILL.md's "flat directory with thumbnail + OBJs"
    assumption (e.g., when `input_dir = <instance>/parts/`).

    `skill_sha` (optional): short git SHA of the code / skill repo at run
    time. Written into result.json for batch reproducibility. Pass
    `_git_sha(skill_dir)` or a project-level SHA from the caller.

    `extra_result_fields` (optional): caller-supplied fields that get
    merged into result.json — e.g. batch_id, sample_idx. Used by the
    batch driver to stamp each sample's provenance without needing a
    separate manifest pass.

    `stage` (default "full"): "full" runs the complete SKILL.md workflow.
    "static" keeps input inspection, structural inference, link/joint
    registration, rest rendering, assembly validation, and URDF export,
    while skipping joint driving/validation and animation. This is
    used by the generator-driven loop for early-abort screening — when
    the static structure is already wrong, paying for joint motion
    validation and animation is wasted turns.
    """
    if stage not in ("full", "static"):
        raise ValueError(f"stage must be 'full' or 'static', got {stage!r}")
    result = ExecutorResult(
        factory=factory,
        instance=instance,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze a private copy of the skill into the run's output_dir and
    # execute against that copy. Why:
    #   (1) An editor save / git pull / concurrent run to the original
    #       skill_dir cannot retroactively change this run's behavior.
    #   (2) The executor has Write + bypassPermissions — if it misbehaves,
    #       it can only corrupt its own snapshot, not the source of truth.
    #   (3) The snapshot is the ground-truth reproducer; skill_sha is a
    #       shortcut ("roughly this commit") that can't capture
    #       working-tree edits. The snapshot wins if they diverge.
    # Disk cost: ~70 KB per sample. Negligible next to the API cost.
    if backend == "codex":
        from evo.engine.codex_backend import run_executor_codex
        return await run_executor_codex(
            factory=factory, instance=instance, input_dir=input_dir,
            output_dir=output_dir, skill_dir=skill_dir, thumbnail_path=thumbnail_path,
            model=model, max_turns=max_turns, log_file=log_file, skill_sha=skill_sha,
            extra_result_fields=extra_result_fields, sim_port=sim_port,
            sim_headless=sim_headless, stage=stage, effort=effort,
            subagent_model=subagent_model,
            subagent_effort=subagent_effort, auth=auth,
        )
    if backend not in ("responses", "claude"):
        raise ValueError(f"unknown executor backend: {backend!r}")
    original_skill_dir = skill_dir
    snapshot_dir = output_dir / "skill_snapshot"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    shutil.copytree(
        skill_dir, snapshot_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    skill_dir = snapshot_dir  # downstream uses the frozen copy

    # The sim worker injects the runtime globals for BOTH backends now
    # (SimWorkerManager hands them to the worker via EVO_SIM_GLOBALS and the
    # worker re-injects before every call), so the old Claude-branch
    # asymmetry — no injected globals, SKILL_DIR textually resolved — is
    # fixed: both branches get the injected-globals skill text.
    skill_append, subagents = _load_skill_as_executor_context(
        skill_dir, thumbnail_path=thumbnail_path,
        subagent_model=subagent_model,
        subagent_effort=subagent_effort,
        runtime_globals_injected=True,
    )

    # Prompt contains only input/output paths — no sensitive paths leaked
    thumbnail_line = (
        f"  Reference thumbnail = {thumbnail_path}\n"
        if thumbnail_path and thumbnail_path.exists()
        else "  Reference thumbnail = (not provided; proceed without one)\n"
    )
    # "Respond in English." overrides an Anthropic account-level Chinese
    # default that bleeds into reasoning text blocks. Without it, the
    # agent's narration is in Chinese while everything functional
    # (paths, code, tool args) stays English — harmless but ugly in
    # log review and a small risk for downstream string-matching
    # generators / verifiers.
    if stage == "static":
        stage_line = (
            "STAGE OVERRIDE: stage=static. Complete input inspection, "
            "structural inference, link/joint registration, one assembled "
            "rest render, the skill's own visual checks, sanity checks, and URDF export. "
            "Do NOT drive joints, call validate_joints, or render motion/"
            "animation. A later full-stage run handles motion validation.\n"
        )
    else:
        stage_line = ""
    prompt = (
        f"You have the /assemble skill loaded in your system prompt. "
        f"Execute it on the following arguments:\n"
        f"  $1 (input OBJ directory, READ-ONLY) = {input_dir}\n"
        f"  $2 (output directory) = {output_dir}\n"
        f"{thumbnail_line}"
        f"{stage_line}"
        f"Respond in English. "
        f"Begin the assembly workflow."
    )
    log.info(f"Executor: {factory}/{instance}")

    f = open(log_file, "w") if log_file else None
    # Predeclared so the finally block's result.json write works even if we
    # exception out before the query loop runs.
    t0 = time.time()
    session_id: str | None = None
    jsonl_dest: Path | None = None
    responses_artifact: Path | None = None
    response_summary: dict[str, Any] | None = None
    try:
        effective_effort = effort or (
            DEFAULT_AGENT_EFFORT if backend == "responses" else "high"
        )
        effective_subagent_effort = subagent_effort or (
            DEFAULT_AGENT_EFFORT if backend == "responses" else "high"
        )
        _write_log(f, "config", f"backend={backend} model={model} max_turns={max_turns} stage={stage} effort={effective_effort} subagent_model_override={subagent_model or '(skill/default)'} subagent_effort_override={subagent_effort or '(skill/default)'} effective_subagent_default_effort={effective_subagent_effort}")
        _write_log(f, "config", f"skill_dir={original_skill_dir} (source)")
        _write_log(f, "config", f"skill_snapshot={skill_dir} (frozen; this is what actually runs)")
        _write_log(f, "config", f"subagents={list(subagents.keys())}")
        _write_log(f, "config", f"thumbnail_path={thumbnail_path}")
        _write_log(f, "config", f"skill_sha={skill_sha}")
        _write_log(f, "isolation", "setting_sources=None → no user memory, no user skills, no user CLAUDE.md")
        _write_log(f, "prompt", prompt)

        _write_log(
            f, "sim",
            f"Starting fresh sim worker for executor "
            f"(port={sim_port}, headless={sim_headless})",
        )
        # These paths are runtime-owned invariants, not values the assembly
        # model should have to remember. The manager passes them to the
        # worker (EVO_SIM_GLOBALS) which re-injects them before every call —
        # identically for the Responses and Claude branches, fixing the old
        # asymmetry where only the Blender-MCP Responses adapter injected.
        runtime_globals = {
            "ASSEMBLY_SDK_DIR": str(ASSEMBLY_SDK_DIR),
            "SKILL_DIR": str(skill_dir),
            "INPUT_DIR": str(input_dir),
            "OUTPUT_DIR": str(output_dir),
        }
        with SimWorkerManager(
            port=sim_port, headless=sim_headless, runtime_globals=runtime_globals,
        ):
            _write_log(f, "sim", "Sim worker ready")
            if backend == "responses":
                from evo.engine.responses_backend import run_responses_agent

                responses_artifact = output_dir / "executor.responses.json"
                outcome = await run_responses_agent(
                    name="assembly_executor",
                    instructions=skill_append,
                    prompt=prompt,
                    model=model,
                    effort=effort or DEFAULT_AGENT_EFFORT,
                    max_turns=max_turns,
                    cwd=output_dir,
                    read_roots=[
                        input_dir,
                        output_dir,
                        skill_dir,
                        *([thumbnail_path.parent] if thumbnail_path else []),
                    ],
                    write_root=output_dir,
                    subagent_dir=skill_dir / "agents",
                    subagent_model=subagent_model,
                    subagent_effort=subagent_effort,
                    subagent_default_effort=subagent_effort or DEFAULT_AGENT_EFFORT,
                    sim_port=sim_port,
                    # Defense in depth: the adapter also injects/protects the
                    # runtime globals per call (the worker re-injects too).
                    runtime_globals=runtime_globals,
                    artifact_path=responses_artifact,
                )
                session_id = outcome.response_id
                response_summary = outcome.usage_summary
                # The compact trajectory the Generator reads to diagnose a
                # failure itself: every tool call and (truncated) result, in
                # order, so executor.log carries what the raw Responses
                # artifact (blocked from wholesale Read) carries.
                _log_responses_trajectory(f, responses_artifact)
                if outcome.final_output:
                    result.transcript.append(outcome.final_output)
                    _write_log(f, "text", outcome.final_output)
            else:
                opts_kwargs: dict[str, Any] = dict(
                    model=model,
                    env=CLAUDE_FAST_MODE_DISABLED_ENV,
                    settings=CLAUDE_FAST_MODE_DISABLED_SETTINGS,
                    setting_sources=None,
                    # A sim tool result above the SDK's 1 MB default killed an
                    # Opus session mid-task (test task_079, 2026-11-03); the
                    # Responses backend bounds tool output itself.
                    max_buffer_size=CLAUDE_SDK_MAX_BUFFER_BYTES,
                    mcp_servers={"sim": make_sim_mcp_server(sim_port)},
                    system_prompt={
                        "type": "preset",
                        "preset": "claude_code",
                        "append": skill_append,
                    },
                    agents=subagents,
                    permission_mode="bypassPermissions",
                    max_turns=max_turns,
                    cwd=str(output_dir),
                )
                if effort is not None:
                    opts_kwargs["effort"] = effort
                async for msg in query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(**opts_kwargs),
                ):
                    _raise_on_api_error(f, msg)
                    text = _log_message(f, msg)
                    if isinstance(msg, ResultMessage):
                        session_id = msg.session_id
                    elif isinstance(msg, AssistantMessage) and session_id is None:
                        session_id = msg.session_id
                    if text:
                        result.transcript.append(text)
        _write_log(f, "sim", "Sim worker stopped")

    except Exception as e:
        result.error = str(e)
        # The SDK sometimes surfaces only a generic "Command failed with
        # exit code N" without the upstream API status / message body. The
        # real cause (e.g. linkapi `up_bad_request`, model-routing 400)
        # is, however, written to the [text] stream and ends up in the log
        # file. Re-scan the log tail and extract the most recent API
        # error so transient-error detection upstream actually has a
        # chance to match it.
        if log_file is not None:
            try:
                tail = log_file.read_text(errors="replace").splitlines()[-200:]
                api_lines = [
                    ln for ln in tail
                    if "API Error" in ln or "api_error" in ln
                       or "up_bad_request" in ln or "up_error" in ln
                       or "上游错误" in ln or "上游" in ln
                ]
                if api_lines:
                    result.error = (
                        f"{e} | upstream: {api_lines[-1].strip()[:500]}"
                    )
            except Exception:
                pass
        _write_log(f, "error", f"Exception: {e}")
        log.error(f"Executor error: {result.error}")
    finally:
        result.duration = time.time() - t0
        # Capture the session JSONL even on exception. `error_max_turns`
        # emits a ResultMessage (so session_id is set) but the claude
        # subprocess exits non-zero, which the SDK surfaces as an
        # exception AFTER the async loop. Without this branch we'd lose
        # the trajectory + cost data for every max_turns hit — a ~$10
        # data-loss event each time, since the API work was actually
        # done. Re-copying if already copied is harmless (shutil.copy2
        # is idempotent).
        if backend == "claude" and session_id and jsonl_dest is None:
            jsonl_dest = _copy_session_jsonl(session_id, output_dir, log_file)
        if responses_artifact:
            _write_log(f, "responses_artifact", str(responses_artifact))
        elif jsonl_dest:
            _write_log(f, "session_jsonl", str(jsonl_dest))
        else:
            _write_log(f, "session_jsonl", f"(not found; session_id={session_id})")

        # URDF detection: do this AFTER the copy so it reflects final
        # state even on exception. Only mark success if we actually
        # produced a URDF AND there was no exception setting result.error.
        urdfs = list(output_dir.glob("*.urdf"))
        if urdfs and not result.error:
            result.urdf_path = urdfs[0]
            result.success = True
            _write_log(f, "success", f"URDF: {result.urdf_path.name}")
        elif not urdfs:
            failed = output_dir / "FAILED.txt"
            if failed.exists() and not result.error:
                result.error = failed.read_text()[:500]
            elif not result.error:
                result.error = "No URDF produced"
            _write_log(f, "error", result.error or "")
        else:
            # URDF present but an exception happened — log both for clarity
            result.urdf_path = urdfs[0]  # record the path for forensics
            _write_log(f, "partial",
                       f"URDF present ({urdfs[0].name}) but execution raised: {result.error}")
        # Write machine-readable result.json — the file analyses read to
        # classify pass/fail + cost without re-parsing the full JSONL.
        # Deliberately tolerant: missing session/subagent files just emit
        # zeros. If something throws, swallow it so the executor return
        # value is still usable.
        try:
            parent_summary = response_summary or _summarize_session(jsonl_dest)
            sub_dir = None
            if jsonl_dest is not None:
                sub_dir = jsonl_dest.parent / f"{jsonl_dest.stem}.subagents"
            subagent_summaries: list[dict[str, Any]] = []
            if sub_dir and sub_dir.is_dir():
                for sub_jsonl in sorted(sub_dir.glob("agent-*.jsonl")):
                    summary = _summarize_session(sub_jsonl)
                    summary["jsonl"] = sub_jsonl.name
                    meta = sub_jsonl.with_suffix(".meta.json")
                    if meta.exists():
                        try:
                            m = json.loads(meta.read_text())
                            summary["agent_type"] = m.get("agentType")
                            summary["description"] = m.get("description")
                        except Exception:
                            pass
                    subagent_summaries.append(summary)
            total_cost = parent_summary["cost_usd"] + sum(
                s["cost_usd"] for s in subagent_summaries
            )
            result_doc: dict[str, Any] = {
                "task_id": factory,
                "instance": instance,
                "success": result.success,
                "urdf_path": str(result.urdf_path) if result.urdf_path else None,
                "error": result.error,
                "duration_sec": round(result.duration, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": model,
                "backend": backend,
                "effort": effort or (
                    DEFAULT_AGENT_EFFORT if backend == "responses" else None
                ),
                "stage": stage,
                "skill_dir_source": str(original_skill_dir),
                "skill_dir": str(skill_dir),  # the frozen snapshot that actually ran
                "skill_sha": skill_sha,
                "session_id": session_id,
                "session_jsonl": str(jsonl_dest) if jsonl_dest else None,
                "responses_artifact": (
                    str(responses_artifact) if responses_artifact else None
                ),
                "parent_session": parent_summary,
                "subagents": subagent_summaries,
                "total_cost_usd": round(total_cost, 4),
                "n_turns_total": parent_summary["n_api_calls"]
                + sum(s["n_api_calls"] for s in subagent_summaries),
            }
            if extra_result_fields:
                result_doc.update(extra_result_fields)
            (output_dir / "result.json").write_text(
                json.dumps(result_doc, indent=2)
            )
            _write_log(f, "result_json", str(output_dir / "result.json"))
            _write_log(f, "cost", f"total=${total_cost:.4f}")
        except Exception as e:
            _write_log(f, "result_json", f"(failed to write: {e})")
            log.warning(f"Failed to write result.json: {e}")
        _write_log(f, "done", f"duration={result.duration:.1f}s success={result.success}")
        if f:
            f.close()

    return result


# ---------------------------------------------------------------------------
# Generator agent — Write/Edit only, cwd isolated, no sensitive paths in prompt
# ---------------------------------------------------------------------------

# EVO_FEEDBACK_FORMAT: "digest" (default; evo/engine/feedback_digest.py) or
# "flat" (the Run 32 prompt: one line per task, no Evaluator digest, and the
# pre-digest generator.md wording). The flat variant is the control for the
# digest / clustering experiments.
# "cluster" (evo/engine/failure_clusters.py): every failure of the round grouped
# into clusters, try_skill(cluster=...) scores a cluster's tasks with the oracle
# (the per-edit gate), and generator_cluster.md asks for one bounded edit per
# cluster instead of one feature per round.
# "score" (ablation: no structured feedback): the flat prompt and workflow, but the
# Evolver is told only each task's pass/fail and continuous score. Failed checks,
# their details, per-joint reports and the per-check rates are withheld, and the
# loop moves the outputs' oracle.json files out of the Evolver's reach
# (loop._hide_oracle_files).
FEEDBACK_FORMAT = os.environ.get("EVO_FEEDBACK_FORMAT", "digest").strip().lower()
if FEEDBACK_FORMAT not in ("digest", "flat", "cluster", "score"):
    raise ValueError(f"EVO_FEEDBACK_FORMAT={FEEDBACK_FORMAT!r}: expected 'digest', 'flat', 'cluster' or 'score'")
GENERATOR_SYSTEM_PROMPT = (
    PROJECT_ROOT / "evo" / "agents"
    / {"flat": "generator_flat.md", "score": "generator_flat.md", "cluster": "generator_cluster.md"}.get(FEEDBACK_FORMAT, "generator.md")
).read_text()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def check_skill_report(skill_dir: Path, current_skill_content: dict[str, str] | None = None) -> dict:
    """The controller's acceptance checks on a candidate skill directory, as data.

    Exactly what the loop runs after the Generator turn (frozen ACTIONS.md
    unchanged; compile/import of every tool; TOOLS.md signatures and fields;
    the run_preflight hook in a fresh interpreter), so the Generator can run it
    itself before finishing and fix what fails.
    """
    from evo.engine.skill_preflight import preflight_skill

    errors: list[str] = []
    if current_skill_content is not None:
        expected = current_skill_content.get("ACTIONS.md")
        if expected is not None:
            actual = skill_dir / "ACTIONS.md"
            if not actual.exists():
                errors.append("frozen SDK contract ACTIONS.md was removed; restore it")
            elif actual.read_text() != expected:
                errors.append("frozen SDK contract ACTIONS.md was modified; restore it exactly")
    result = preflight_skill(skill_dir)
    errors.extend(result.errors)
    return {
        "ok": result.ok and not errors,
        "errors": errors,
        "files_checked": list(result.files),
        "smoke_hook_ran": result.smoke_ran,
        "duration_sec": round(result.duration_sec, 2),
    }


def make_check_skill_tool(skill_dir: Path, current_skill_content: dict[str, str] | None = None):
    """`check_skill` tool for the Generator's Responses agent."""
    from agents import function_tool

    @function_tool(name_override="check_skill")
    def check_skill() -> str:
        """Run the controller's candidate-skill check on your working directory
        (ACTIONS.md unchanged, every tool compiles and imports as tools.<name>,
        TOOLS.md signatures/fields exact, run_preflight passes in a fresh
        interpreter). Returns JSON {ok, errors, ...}. Call it before you finish
        and fix every error; a turn that ends with errors is rejected."""
        return json.dumps(check_skill_report(skill_dir, current_skill_content), indent=2)

    return check_skill


def _render_feedback_block(
    feedback_history: list[dict[str, Any]],
    round_dir: Path,
    per_task_block: str,
) -> str:
    """Digest for the prompt; the digest and the flat list go to files too.

    Falls back to the flat list if the digest cannot be built, so a
    formatting bug in the Evaluator never blocks a Generator turn.
    """
    from evo.engine.feedback_digest import build_digest

    run_dir = round_dir.parent
    full_list = round_dir / "feedback_tasks.md"
    try:
        round_dir.mkdir(parents=True, exist_ok=True)
        full_list.write_text(
            "# Per-task results of this round (one line per task; the digest "
            "in the prompt ranks them)\n\n" + per_task_block + "\n"
        )
        digest = build_digest(
            feedback_history, run_dir=run_dir, full_list_path=full_list,
        )
        (round_dir / "feedback_digest.md").write_text(digest + "\n")
        return digest
    except Exception as e:  # pragma: no cover - defensive
        log.warning(f"feedback digest unavailable, using the flat list: {e}")
        return per_task_block


async def run_generator(
    current_skill_dir: Path,
    current_skill_content: dict[str, str],
    feedback_history: list[dict[str, Any]],
    output_skill_dir: Path,
    model: str = DEFAULT_AGENT_MODEL,
    max_turns: int = 200,
    log_file: Path | None = None,
    effort: str | None = None,
    sim_port: int | None = None,
    sim_headless: bool = True,
    backend: str = "responses",
    auth: str | None = "local",
    add_dirs: list[str] | None = None,
    dataset_tasks_root: Path | None = None,
    repair_note: str | None = None,
    update_note: str | None = None,
    extra_tools: list[Any] | None = None,
    same_tasks_each_round: bool = False,
    trial_task: str | None = None,
    cluster_trial_tasks: dict[str, list[str]] | None = None,
) -> GeneratorResult:
    """Generate an improved skill based on accumulated feedback.

    The output_skill_dir is pre-populated with a copy of the current skill
    so the generator only needs to Edit what's wrong, not recreate everything.

    Args:
        current_skill_dir: directory holding the active skill (will be copied)
        current_skill_content: dict of filename → content (also embedded in prompt)
        feedback_history: list of feedback dicts from loop.py
        output_skill_dir: where generator writes improved skill files
        add_dirs: file-tool access scope. The loop passes a narrowed list
            (run dir + GT-free dataset inputs + oracle source) so the
            generator's Read/Glob/Grep cannot reach gt/ answer files —
            a past run read GT URDFs and hardcoded their values into a
            tool. None falls back to the legacy broad scope (tests only).
        repair_note: set when this is a same-round retry after the previous
            candidate was rejected before A/B (validation or preflight);
            the reason is shown so the Generator fixes it instead of
            starting over.
        dataset_tasks_root: real tasks/ root, used in the bootstrap prompt
            so example paths don't assume the dataset lives in-repo.
    """
    if backend == "codex":
        from evo.engine.codex_backend import run_generator_codex
        return await run_generator_codex(
            current_skill_dir=current_skill_dir,
            current_skill_content=current_skill_content,
            feedback_history=feedback_history, output_skill_dir=output_skill_dir,
            model=model, max_turns=max_turns, log_file=log_file, effort=effort,
            sim_port=sim_port, sim_headless=sim_headless, auth=auth,
        )
    if backend not in ("responses", "claude"):
        raise ValueError(f"unknown generator backend: {backend!r}")

    result = GeneratorResult(skill_dir=output_skill_dir)
    frozen_sdk_snapshot = _snapshot_frozen_sdk()

    # Pre-populate output dir with a copy of the current skill
    # so the generator only edits what's wrong, not recreates everything
    if output_skill_dir.exists():
        shutil.rmtree(output_skill_dir)
    shutil.copytree(
        current_skill_dir, output_skill_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )

    # Per-check trajectory across recent rounds (compact trend view).
    # The generator can also `find` past round dirs itself; this is a
    # quick at-a-glance signal for "did my last edit help or hurt".
    trajectory_lines: list[str] = []
    if feedback_history:
        # Aggregate all check names that appear in any round.
        all_checks: list[str] = []
        seen: set[str] = set()
        for fb in feedback_history:
            for k in (fb.get("per_check_pass_rate") or {}):
                if k not in seen:
                    all_checks.append(k); seen.add(k)
        # Header
        recent = feedback_history[-6:]
        header = "  check_name           " + " ".join(
            f"R{fb.get('round', '?')}".rjust(6) for fb in recent
        )
        trajectory_lines.append(header)
        for name in sorted(all_checks):
            row = f"  {name:<22}"
            for fb in recent:
                pcp = fb.get("per_check_pass_rate") or {}
                row += f" {pcp.get(name, '-'):>5}"
            trajectory_lines.append(row)
    trajectory_block = "\n".join(trajectory_lines) if trajectory_lines else "(no prior rounds)"
    trajectory_note = (
        "Every round evaluates the SAME tasks, so movement here reflects the\n"
        "skill (plus executor sampling noise: one session per task), not the sample."
        if same_tasks_each_round else
        "Each round evaluates a DIFFERENT sample of tasks, so movement here mixes your\n"
        "edits with task difficulty. It is a hint about which check is the bottleneck,\n"
        "never evidence that a change worked."
    )

    _trial_task_label = (
        f"`{trial_task}`, the first task of this round's Train list"
        if trial_task else "the first task of this round's Train list"
    )
    edit_rule_line = (
        "Up to THREE cluster edits this round, largest clusters first, each gated with "
        "`try_skill(cluster=...)` before you move on (see the system prompt); list each under **Changed**."
        if FEEDBACK_FORMAT == "cluster" else
        "Exactly ONE feature this round (see the system prompt); name it under\n**Changed**."
    )
    if FEEDBACK_FORMAT == "cluster" and cluster_trial_tasks:
        _cluster_list = ", ".join(
            f"{cid}: {', '.join(tasks)}" for cid, tasks in cluster_trial_tasks.items() if tasks
        )
        trial_block = f"""### Edit, gate, repeat; then check
Work cluster by cluster, largest first, at most THREE clusters this round:
1. read every member of the cluster (their output_dirs are in the report), form one hypothesis that
   explains the whole cluster, and make ONE bounded edit for it (a rule or a helper; keep it small —
   a few dozen SKILL.md lines or one helper function);
2. call `try_skill(cluster="<id>")`: the controller runs the cluster's try_skill tasks ({_cluster_list})
   with the skill in your working directory and scores them with the batch oracle. It returns, per
   task, the verdict and failed checks before (this round's batch) and after (your edit) and whether
   the cluster improved (more passes, or the same passes with fewer failed checks);
3. if it did not improve, revise the edit once and gate again, or revert it (Edit the files back);
   never keep an edit that failed its gate. An executor crash in the trial is read from its
   executor.log and fixed like any failure.
`try_skill()` with no argument is the old smoke test on {_trial_task_label}. Every trial costs a few
cents and 2–4 minutes; the session cap is 10 calls. Then call `check_skill` (ACTIONS.md unchanged,
tools compile/import, TOOLS.md exact, preflight hook) and fix every error it reports; a turn that
ends with errors is rejected and you get another turn with the reason."""
        journal_extra = (
            "\n**Changed** must list one line per cluster you edited for, in the form "
            "`C3 — <edit in ≤ 12 words> — kept` or `C3 — <edit> — reverted`; the controller reads these lines."
        )
    else:
        journal_extra = ""
        trial_block = f"""### Smoke-test, then check
`try_skill()` runs ONE real executor session with the skill in your working
directory on a fixed Train task ({_trial_task_label}; always the same one, no
argument) and reports whether the executor finished (URDF written) or failed,
with its raw error / FAILED.txt text and the path of its executor.log. The
controller does not classify causes: when it failed, you read the log (every
tool call and result is in it) and work out why. It does not score the output
and says nothing about effectiveness — the round's evaluation measures that.
After your change, call it once; if it failed, fix the cause and call it
again, until the executor finishes. Your own
helper calls passing is not evidence that the executor can use them. Then call
`check_skill` (the controller's acceptance check: ACTIONS.md unchanged, tools
compile/import, TOOLS.md exact, preflight hook) and fix every error it
reports; a turn that ends with errors is rejected and you get another turn
with the reason."""

    # Current round = last entry of feedback_history.
    cur = feedback_history[-1] if feedback_history else {}
    cur_round = cur.get("round", 0)
    cur_pass = cur.get("pass_rate", 0.0)
    cur_per_check = cur.get("per_check_pass_rate", {}) or {}
    per_check_str = "\n".join(
        f"  {k}: {v}" for k, v in sorted(cur_per_check.items())
    ) or "(none)"

    bcs = cur.get("batch_cost_summary") or {}
    cost_header = (
        f"  total: ${bcs.get('total_usd', 0):.2f}  "
        f"mean: ${bcs.get('mean_usd_per_task', 0):.3f}/task  "
        f"output_tokens: {bcs.get('total_output_tokens', 0):,}  "
        f"cache_misses: {bcs.get('total_cache_misses', 0)}"
    ) if bcs else "(no cost data)"
    prev_gen = cur.get("prev_generator_cost") or {}
    prev_gen_str = (
        f"  prev generator turn: ${prev_gen.get('cost_usd', 0):.2f}  "
        f"output_tokens: {prev_gen.get('output_tokens', 0):,}"
    ) if prev_gen else ""

    # Per-task path block — every task in the batch (PASS + FAIL),
    # so generator can compare what worked vs what didn't and decide
    # which to read deeply. Each entry is a one-line summary + the
    # output_dir; the generator opens what it wants.
    runs = cur.get("runs", [])
    per_task_block_lines: list[str] = []
    for run in runs:
        tag = run.get("tag", "?")
        verdict = run.get("oracle_verdict", "?")
        failed = run.get("failed_oracle_checks", []) or []
        cs = run.get("cost_summary") or {}
        out_dir = run.get("output_dir", "")
        per_task_block_lines.append(
            f"  - {tag}  [{verdict}]  "
            + (f"failed: {failed}  " if failed else "")
            + (f"score: {run['fitness']:.2f}  "
               if FEEDBACK_FORMAT == "score" and run.get("fitness") is not None else "")
            + f"cost: ${cs.get('cost_usd', 0):.2f}  "
            f"out_tokens: {cs.get('output_tokens', 0):,}"
        )
        per_task_block_lines.append(f"      output_dir: {out_dir}")
        if run.get("error"):
            per_task_block_lines.append(f"      executor error: {str(run['error'])[:200]}")
        # The judge's own diagnosis. Without it the Generator sees only that
        # a check named `motion_direction` failed and has to guess the cause.
        for name, detail in (run.get("failed_check_details") or {}).items():
            per_task_block_lines.append(f"      {name}: {detail}")
        # Per-joint outcomes from the script oracle: which predicted joint
        # resolved to each GT joint and how far off every dimension was.
        for jname, rep in sorted((run.get("joint_reports") or {}).items()):
            per_task_block_lines.append(f"      joint {jname}: {_format_joint_report(rep)}")
    joint_summary = _format_joint_summary(runs)
    per_task_block = "\n".join(
        ([joint_summary] if joint_summary else []) + per_task_block_lines
    ) or "(no runs)"
    # The Evaluator's digest replaces the flat list in the prompt: failure
    # modes ranked, representative tasks, flips since the last edit and the
    # feature ledger (evo/engine/feedback_digest.py). The flat list is still
    # written next to the journal so the Evolver can Read any task it wants.
    feedback_block = per_task_block
    feedback_heading = "Tasks in this round (read what you need with Read)"
    if feedback_history and FEEDBACK_FORMAT == "digest":
        feedback_block = _render_feedback_block(
            feedback_history, output_skill_dir.parent, per_task_block,
        )
        feedback_heading = (
            "Evaluator digest of this round (read the representative tasks' "
            "output_dirs with Read)"
        )
    elif feedback_history and FEEDBACK_FORMAT == "cluster":
        from evo.engine.failure_clusters import build_cluster_report
        round_dir = output_skill_dir.parent
        round_dir.mkdir(parents=True, exist_ok=True)
        full_list = round_dir / "feedback_tasks.md"
        full_list.write_text(
            "# Per-task results of this round (one line per task; the cluster "
            "report in the prompt groups them)\n\n" + per_task_block + "\n"
        )
        try:
            feedback_block, _clusters = build_cluster_report(
                feedback_history, run_dir=round_dir.parent, full_list_path=full_list,
                trial_tasks=cluster_trial_tasks,
            )
            (round_dir / "feedback_clusters.md").write_text(feedback_block + "\n")
            feedback_heading = (
                "Failure clusters of this round (read every member you edit for; "
                "output_dirs are listed per task)"
            )
        except Exception as e:  # pragma: no cover - defensive
            log.warning(f"cluster report unavailable, using the flat list: {e}")
    elif feedback_history:
        # Flat (Run 32) format: the list is the prompt; keep the file for records.
        round_dir = output_skill_dir.parent
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "feedback_tasks.md").write_text(
            "# Per-task results of this round (flat format, as in the prompt)\n\n"
            + per_task_block + "\n"
        )

    run_dir_hint = ""
    if runs and runs[0].get("output_dir"):
        out_dir = Path(runs[0]["output_dir"])
        for parent in out_dir.parents:
            if parent.name.startswith("round_"):
                run_dir_hint = str(parent.parent)
                break

    # Journal paths: read predecessor's, write your own. Lives in
    # round_<N>/generator_journal.md (sibling of generator.log, not
    # inside the skill dir — keeps executors' prompt clean). For
    # round 0, the predecessor is the bootstrap phase (if any),
    # whose journal lives in <run_dir>/bootstrap/generator_journal.md.
    this_round_dir = output_skill_dir.parent  # = run_dir/round_<N>
    this_journal = (this_round_dir / "generator_journal.md").resolve()

    def _prev_journal_locator(prev_round_dir: Path) -> str:
        prev_j = prev_round_dir / "generator_journal.md"
        if prev_j.exists():
            return str(prev_j.resolve())
        return (
            f"(predecessor wrote no journal; expected at {prev_j.resolve()}. "
            f"Inspect {prev_round_dir / 'generator.log'} and the matching "
            "generator.responses.json or generator.session.jsonl trace.)"
        )

    prev_journal_str = "(this is the first round — no predecessor yet)"
    if run_dir_hint:
        if cur_round >= 1:
            prev_journal_str = _prev_journal_locator(
                Path(run_dir_hint) / f"round_{cur_round - 1}"
            )
        else:  # cur_round == 0 — look for bootstrap phase
            boot_dir = Path(run_dir_hint) / "bootstrap"
            if boot_dir.exists():
                prev_journal_str = _prev_journal_locator(boot_dir)

    is_bootstrap = not feedback_history
    _tasks_root = dataset_tasks_root or (
        PROJECT_ROOT / "dataset" / "artvip_sanitized" / "tasks"
    )
    if is_bootstrap:
        # No executor data yet. Generator's job is to populate tools/
        # from a blank skill, working from ENVIRONMENT.md, the dataset,
        # and the oracle source. Test what you write before saving.
        this_journal = (output_skill_dir.parent / "generator_journal.md").resolve()
        prompt = f"""## Bootstrap phase (round 0 has not run yet)

Your working directory (cwd) — a copy of a **blank starter skill**. The `tools/`
directory is empty. Your job this turn is to populate it before any executor
runs:
  `{output_skill_dir.resolve()}`

Use relative paths to edit skill files (`Edit SKILL.md`, `Write tools/foo.py`).
Use absolute paths to read external files.

{_dataset_task_block(_tasks_root)}

### Read FIRST
1. `Read SKILL.md` and `Read ACTIONS.md` — the runtime contract (injected globals, persistent sim session) and the frozen SDK API you will build on.
2. Browse a real input task: `Glob` below `{_tasks_root}` and inspect one `parts/` directory.
3. `Read {PROJECT_ROOT}/evo/oracle/check_urdf.py` — exactly what your output URDF will be measured against.
4. The thumbnail at `{_tasks_root}/<tid>/<inst>/thumbnail.png` is what the executor sees as the assembly target.

### Then design tools

You have `mcp__sim__execute_python_code` (a persistent Python session with the frozen `assembly` SDK preloaded) and, only if the skill ships `agents/*.md`, the registered validation specialist tools. **Use them**:
- Prototype an OBJ import inline; check the result is what you expect.
- Decide on a representation (link/joint/state/FK convention) and write minimal Python files in `tools/` to support it.
- Use the `Read` tool on rendered PNGs (and any registered specialist) to inspect assembly or joint images when useful.
- Iterate until at least the import → assemble → URDF export pipeline runs end-to-end on one task.

### Update SKILL.md and TOOLS.md

Document the pipeline you've designed: what tools exist, how they fit together, what conventions you adopted. The executor will follow this verbatim.

For every public function under `tools/`, create an exact `TOOLS.md` interface
section as required by generator.md. Include the `inspect.signature()` heading
and Parameters/Returns/Raises/Example fields. Any later tool-source edit must
update `TOOLS.md` in the same turn; the controller rejects stale documentation.

### Before you finish
Run `try_skill()` — a smoke test: one real executor session with your skill on a
fixed Train task ({_trial_task_label}). It reports whether the executor finished
or failed, with the raw error and the executor.log path; it does not classify
causes or measure effectiveness. If it failed, read the log, fix the cause and
run it again, until the executor finishes. Then call
`check_skill` and fix every error it reports (it is the controller's own
acceptance check).

### Your journal (write LAST)
Path: `{this_journal}`
Required sections (≤ 300 words): **Observed** (what you found in the data), **Investigated** (what you prototyped), **Changed** (files written), **Verified** (what you self-tested + result), **Unresolved** (open questions for round-1 generator).

Begin. Respond in English."""
    else:
        # The A/B verdict for the previous turn's candidate lives on the entry
        # that produced it — feedback_history[-2]. [-1] is this round's own
        # entry, whose A/B has not run yet.
        candidate_outcome_note = _format_previous_candidate_outcome(
            feedback_history[-2] if len(feedback_history) >= 2 else {},
            same_tasks=same_tasks_each_round,
        )
        rollback_note = ""
        if cur.get("skill_rolled_back"):
            rollback_note = (
                "\n### NOTE — previous edits were ROLLED BACK\n"
                "The skill you are editing is the best-scoring checkpoint, "
                "NOT the latest edit: the last round's fitness regressed "
                "past tolerance, so the loop reverted. Read the predecessor "
                "journals to see what was tried, and do not re-apply the "
                "same change without new evidence.\n"
            )
        update_block = f"\n{update_note}\n" if update_note else ""
        repair_block = ""
        if repair_note:
            repair_block = (
                "\n### NOTE — your previous candidate THIS round was REJECTED by the acceptance check\n"
                f"Reason: {repair_note}\n"
                "The evaluation data below is unchanged and your working directory is a fresh "
                "copy of the current skill (the rejected edits are gone). Fix the cause above "
                "first — `check_skill` tells you whether it is fixed; a candidate that fails "
                "again is discarded and the round is re-evaluated from scratch.\n"
            )
        oracle_file_note = "`oracle.json` (the judge's full per-check verdict), "
        if FEEDBACK_FORMAT == "score":
            oracle_file_note = ""
            per_check_str = ("  (not provided: in this run the evaluator reports only each task's "
                             "pass/fail and score)")
            trajectory_block = "(not provided)"
        cur_mean_score = cur.get("mean_oracle_score")
        mean_score_line = (
            f"\n- Mean oracle score (fraction of checks passed): {cur_mean_score}. "
            f"Checkpoint selection ranks the binary pass rate first and uses this "
            f"score only as a tie-break."
            if cur_mean_score is not None else ""
        )
        prompt = f"""## Round {cur_round}

Your working directory (cwd) — a fresh copy of the current skill, this is what
you will edit and what next round's executors will run:
  `{output_skill_dir.resolve()}`

Use relative paths to edit skill files (`Edit SKILL.md`, `Edit tools/foo.py`).
Use absolute paths to read external files (logs, images, past round dirs).

{_dataset_task_block(_tasks_root)}

{edit_rule_line}
{rollback_note}{candidate_outcome_note}{update_block}{repair_block}
### Predecessor's journal (read FIRST, before forming hypotheses)
{prev_journal_str}

{trial_block}

### Your journal (write LAST, before your final summary)
Path: `{this_journal}`
Required sections (≤ 300 words total): **Observed**, **Investigated**, **Changed**, **Verified**, **Unresolved**. See generator.md "Journal" for definitions.{journal_extra}

### Aggregate
- Pass rate: {cur_pass}{mean_score_line}
- Per-check pass-rate (across this batch):
{per_check_str}

### Cost (this round)
{cost_header}
{prev_gen_str}

### Per-check trajectory (recent rounds)
{trajectory_note}
```
{trajectory_block}
```

### {feedback_heading}
In each `output_dir`: `render_thumbnail.png` (the target), `furniture.urdf`,
{oracle_file_note}`executor.log` (trajectory;
prefer it over `executor.responses.json`, which is blocked from wholesale Read —
Grep it), plus whatever renders the skill's workflow produced (names and
layout depend on the skill: `Glob` the directory; SDK sweep frames are
`<joint>_<index>_<pct>pct.png` with `_az<NNN>` for the second camera).

{feedback_block}

Run dir for navigating prior rounds: `{run_dir_hint or '(unavailable)'}` — use `Glob` to inspect `round_*/` siblings if you want history.

Begin. Respond in English."""

    # Load subagents from output_skill_dir/agents/ — generator can call
    # these (validate-assembly, validate-joints, plus any new ones it
    # writes) via the Agent tool to test newly designed tools or get a
    # context-isolated visual judgement.
    _gen_subagents: dict[str, AgentDefinition] = {}
    _gen_agents_dir = output_skill_dir / "agents"
    if _gen_agents_dir.is_dir():
        for md in sorted(_gen_agents_dir.glob("*.md")):
            name, description, tools_list, sub_model, sub_effort, body = _parse_agent_md(
                md.read_text(), md.stem
            )
            kw: dict[str, Any] = dict(
                description=description or f"Subagent {name}",
                prompt=body.strip(),
                tools=tools_list,
                model=sub_model,
                memory=None,
            )
            if sub_effort is not None:
                kw["effort"] = sub_effort
            _gen_subagents[name] = AgentDefinition(**kw)

    f = open(log_file, "w") if log_file else None
    responses_artifact: Path | None = None
    response_summary: dict[str, Any] | None = None
    jsonl_dest: Path | None = None
    try:
        _write_log(f, "config", f"model={model} max_turns={max_turns}")
        _write_log(f, "info", f"output_dir={output_skill_dir}")
        _write_log(f, "info", f"skill_files={list(current_skill_content.keys())}")
        _write_log(f, "info", f"feedback_rounds={len(feedback_history)}")
        effective_effort = effort or (
            DEFAULT_AGENT_EFFORT if backend == "responses" else "high"
        )
        _write_log(f, "info", f"backend={backend} model={model} effort={effective_effort}")
        _write_log(f, "info", f"subagents={list(_gen_subagents.keys())}")
        _write_log(f, "info", f"sim_port={sim_port} headless={sim_headless}")
        _write_log(f, "isolation", "Prompt contains paths + per-check summary only. Generator Reads files itself. No GT/oracle paths.")
        _write_log(f, "prompt", prompt)

        full_response = ""
        session_id: str | None = None
        # Generator now has the sim MCP tool + Agent so it can write tools
        # and spawn subagents to test them. add_dirs lets it Read prior round
        # output dirs (logs, images, URDFs).
        gen_opts_kwargs: dict[str, Any] = dict(
            model=model,
            env=CLAUDE_FAST_MODE_DISABLED_ENV,
            settings=CLAUDE_FAST_MODE_DISABLED_SETTINGS,
            max_buffer_size=CLAUDE_SDK_MAX_BUFFER_BYTES,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            disallowed_tools=["WebFetch", "WebSearch"],
            permission_mode="acceptEdits",
            max_turns=max_turns,
            cwd=str(output_skill_dir),
            add_dirs=add_dirs if add_dirs is not None else [str(PROJECT_ROOT), "/tmp"],
            agents=_gen_subagents or None,
        )
        if sim_port is not None:
            gen_opts_kwargs["mcp_servers"] = {
                "sim": make_sim_mcp_server(sim_port),
            }
        if effort is not None:
            gen_opts_kwargs["effort"] = effort

        # Runtime-owned globals for the generator's sim session. Injected by
        # the manager for BOTH backends (the worker re-injects per call), so
        # the old Claude-branch asymmetry (no injected globals) is fixed.
        gen_runtime_globals = {
            "ASSEMBLY_SDK_DIR": str(ASSEMBLY_SDK_DIR),
            "SKILL_DIR": str(output_skill_dir),
            "OUTPUT_DIR": str(
                output_skill_dir.parent / "generator_sim_output"
            ),
        }

        # Per-round sim worker for the generator. One process; subagents
        # share the persistent session (cleaning state is each subagent's
        # responsibility). No-op if sim_port is None (tests without a worker).
        if sim_port is not None:
            sim_ctx = SimWorkerManager(
                port=sim_port, headless=sim_headless,
                runtime_globals=gen_runtime_globals,
            )
        else:
            import contextlib
            sim_ctx = contextlib.nullcontext()

        with sim_ctx:
            if sim_port is not None:
                _write_log(f, "sim", f"Sim worker ready on port {sim_port}")
            if backend == "responses":
                from evo.engine.responses_backend import run_responses_agent

                responses_artifact = (
                    log_file.parent / "generator.responses.json"
                    if log_file else output_skill_dir / "generator.responses.json"
                )
                outcome = await run_responses_agent(
                    name="skill_generator",
                    instructions=GENERATOR_SYSTEM_PROMPT,
                    prompt=prompt,
                    model=model,
                    effort=effort or DEFAULT_AGENT_EFFORT,
                    max_turns=max_turns,
                    cwd=output_skill_dir,
                    read_roots=[
                        output_skill_dir,
                        *(Path(p) for p in (
                            add_dirs if add_dirs is not None
                            else [str(PROJECT_ROOT), "/tmp"]
                        )),
                    ],
                    # Includes the skill plus its sibling journal/artifacts.
                    write_root=output_skill_dir.parent,
                    subagent_dir=output_skill_dir / "agents",
                    subagent_default_effort=DEFAULT_AGENT_EFFORT,
                    sim_port=sim_port,
                    runtime_globals=gen_runtime_globals,
                    artifact_path=responses_artifact,
                    extra_tools=[make_check_skill_tool(output_skill_dir, current_skill_content), *(extra_tools or [])],
                )
                session_id = outcome.response_id
                response_summary = outcome.usage_summary
                full_response = outcome.final_output
                _write_log(f, "text", full_response)
            else:
                async for msg in query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(**gen_opts_kwargs),
                ):
                    # Generator is a single serial heavy call. Don't fail-fast on
                    # 529 / 5xx — let the CLI's built-in 10× retry ride through
                    # transient Anthropic overload. (Executors keep fail-fast so
                    # the batch evaluator gets quick verdicts.)
                    text = _log_message(f, msg)
                    if isinstance(msg, ResultMessage):
                        session_id = msg.session_id
                    elif isinstance(msg, AssistantMessage) and session_id is None:
                        session_id = msg.session_id
                    full_response += text

        result.raw_response = full_response
        result.changes_summary = full_response[-2000:]
        if backend == "claude":
            jsonl_dest = _copy_session_jsonl(session_id, output_skill_dir, log_file)
            if jsonl_dest:
                _write_log(f, "session_jsonl", str(jsonl_dest))
        elif responses_artifact:
            _write_log(f, "responses_artifact", str(responses_artifact))

    except Exception as e:
        result.changes_summary = f"Generator error: {e}"
        _write_log(f, "error", f"Exception: {e}")
        log.error(f"Generator error: {e}")
    finally:
        sdk_mutations = _restore_frozen_sdk(frozen_sdk_snapshot)
        if sdk_mutations:
            result.changes_summary = (
                "Generator error: immutable Assembly SDK mutation was blocked "
                f"and restored: {sdk_mutations}"
            )
            _write_log(f, "isolation", result.changes_summary)

        # Detect which files were modified — by comparing current content
        # with what was passed in. This runs even on exception so we capture
        # partial work from a truncated generator run.
        for fp in output_skill_dir.rglob("*"):
            if fp.is_file() and fp.suffix in (".md", ".py"):
                rel = str(fp.relative_to(output_skill_dir))
                try:
                    current = fp.read_text()
                except Exception:
                    continue
                original = current_skill_content.get(rel)
                if original is None or current != original:
                    result.files_modified.append(rel)

        _write_log(f, "result", f"modified={result.files_modified}")
        try:
            result_path = (
                log_file.parent / "generator.result.json"
                if log_file else output_skill_dir / "generator.result.json"
            )
            result_path.write_text(json.dumps({
                "backend": backend,
                "model": model,
                "effort": effort or (
                    DEFAULT_AGENT_EFFORT if backend == "responses" else None
                ),
                "response_id": session_id,
                "responses_artifact": (
                    str(responses_artifact) if responses_artifact else None
                ),
                "usage": response_summary or _summarize_session(jsonl_dest),
                "files_modified": result.files_modified,
                "error": (
                    result.changes_summary
                    if result.changes_summary.startswith("Generator error:")
                    else None
                ),
            }, indent=2))
        except Exception as e:
            _write_log(f, "result_json", f"(failed to write: {e})")
        if f:
            f.close()

    return result
