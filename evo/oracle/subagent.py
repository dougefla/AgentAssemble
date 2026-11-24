"""Subagent oracle: LLM visual judge for articulated URDF assemblies.

Drop-in replacement for `oracle.check_urdf.run_oracle` when the loop is run
with `--oracle-mode subagent`. Returns an `OracleResult` so downstream
artifact construction, logging, and feedback aggregation work unchanged.

What the subagent sees per call
-------------------------------
The oracle renders BOTH sides itself into `<pred_dir>/oracle_render/`,
using the ORIGINAL parts/*.obj for both GT and Pred so material drift
is zero. Files passed to the subagent:

    <pred_dir>/oracle_render/gt_rest_az<NNN>.png       (4 azs)
    <pred_dir>/oracle_render/gt_<joint>_az<NNN>_<PCT>pct.png
    <pred_dir>/oracle_render/pred_rest_az<NNN>.png     (4 azs)
    <pred_dir>/oracle_render/pred_<joint>_az<NNN>_<PCT>pct.png

PCT ∈ {000, 050, 100}. Subagent treats the comparison as PASS up to a
global Z-axis rotation φ ∈ {0°,90°,180°,270°}; see _build_prompt for
details.

Judges run on the same Responses provider as the executor and generator
(`OPENAI_API_BASE` + `OPENROUTER_EVO_KEY`, resolved through
`evo.engine.envfile.load_env_chain`); set `EVO_ORACLE_BACKEND=claude` to fall
back to the Claude Agent SDK. Either way the judge reads the PNGs through a
scoped `Read` tool, so the prompt is backend-agnostic.

Each judge emits one JSON line with three sub-judgements
(`rest_pose_match`, `motion_type_match`, `motion_direction_match`) plus an
overall `verdict` and `summary`. By default 3 independent judges run
concurrently and are combined by strict majority — rerun experiments showed
single-call verdicts flip on ~50% of cases, which is far too noisy to drive
skill evolution. The result carries `subagent_verdict` plus one CheckResult
per sub-judgement, so per-check feedback aggregation has real dimensions to
work with instead of a single opaque bit.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from evo.engine.model_defaults import DEFAULT_AGENT_EFFORT, DEFAULT_AGENT_MODEL
from evo.oracle.check_urdf import CheckResult, OracleResult

ROOT = Path(__file__).resolve().parents[2]

# Judge backend. `responses` shares the executor/generator provider; `claude`
# keeps the original Claude Agent SDK path for A/B comparison.
DEFAULT_ORACLE_BACKEND = "responses"
# The judge shares the executor/generator model so a round is not scored by a
# different model than it was produced by. Override per run with
# --oracle-model or EVO_ORACLE_MODEL (`gpt-5.6-luna` is ~10x cheaper, and the
# oracle is the highest-volume vision call at `votes` judges per task).
DEFAULT_ORACLE_MODEL = {
    "responses": DEFAULT_AGENT_MODEL,
    "claude": "claude-opus-4-7",
}
ORACLE_MAX_TURNS = 40


SYSTEM = (
    "You are a visual oracle for articulated furniture URDFs. You compare "
    "predicted articulation to ground truth from rendered frames, judging the "
    "rest pose and, above all, the joint motion: type, direction relative to "
    "the body, and where the part pivots. You read multiple PNGs per case via "
    "the Read tool. Be lenient on small offsets and color differences; strict "
    "on wrong joint type, reversed motion, and hinges on the wrong edge. "
    "Output a single JSON line."
)


_OK_PCTS = {"000", "050", "100"}


def _collect_oracle_frames(
    oracle_dir: Path,
    azs: list[int],
    side: str,
) -> tuple[dict[int, Path], dict[str, dict]]:
    """Return (rest_by_az, motion[joint][(az, pct)] = Path) for one side.

    `side` is "gt" or "pred"; files match `<side>_rest_az<NNN>.png` and
    `<side>_<joint>_az<NNN>_<PCT>pct.png` in <pred_dir>/oracle_render/.
    Only PCT ∈ {000, 050, 100} are included (filters out stray files).
    """
    rest: dict[int, Path] = {}
    for az in azs:
        p = oracle_dir / f"{side}_rest_az{az:03d}.png"
        if p.exists():
            rest[az] = p
    motion: dict[str, dict] = defaultdict(dict)
    pat = re.compile(rf"{side}_(.+)_az(\d+)_(\d+)pct\.png")
    for p in oracle_dir.glob(f"{side}_*_az*_*pct.png"):
        m = pat.match(p.name)
        if not m:
            continue
        jname, az_s, pct = m.group(1), m.group(2), m.group(3)
        if pct not in _OK_PCTS:
            continue
        if int(az_s) in azs:
            motion[jname][(az_s, pct)] = p
    return rest, dict(motion)



def _dataset_preplaced(gt_dir: Path) -> bool:
    """Whether the task parts arrive at their GT rest pose (``world_bake_q0``).

    Read from the nearest ``dataset_variant.json`` above ``gt_dir``. Views
    derived from the world-baked one (``artvip_posefill``) copy ``gt/`` wholesale,
    ``bake_transforms.json`` included, so that file's presence is no longer a
    valid signal; it remains the fallback only when no variant file exists.
    """
    probe = gt_dir.resolve()
    for _ in range(6):
        marker = probe / "dataset_variant.json"
        if marker.is_file():
            try:
                variant = json.loads(marker.read_text())
            except (OSError, json.JSONDecodeError):
                return False
            return "world_bake_q0" in (variant.get("task_mesh_transforms") or [])
        if probe.parent == probe:
            break
        probe = probe.parent
    return (gt_dir / "bake_transforms.json").is_file()

def _build_prompt(
    *,
    tag: str,
    gt_rest: dict[int, Path],
    gt_motion: dict[str, dict],
    pred_rest: dict[int, Path],
    pred_motion: dict[str, dict],
    azs: list[int],
    preplaced: bool = False,
) -> str:
    """Construct the multi-image prompt for one (task, instance).

    `preplaced` (the artvip_beginner view) tells the judge that parts were
    imported at their GT rest pose, so the rest views must match at φ=0 and
    the verdict rides on the joints alone.

    All file paths point at `<pred_dir>/oracle_render/` files rendered
    by `oracle.render_for_oracle`. GT and Pred sides use the same mesh
    source (original parts/*.obj) so material is identical between
    sides — only URDF differs.
    """
    parts: list[str] = [
        ("Compare predicted articulation to ground truth (GT). The parts were "
         "imported already in their GT rest pose; only the JOINTS were "
         "authored. Confirm the rest views match, then judge the joints.")
        if preplaced else
        "Compare predicted articulated assembly to ground truth (GT). "
        "Judge BOTH the rest pose AND the joint motion.",
        "",
        f"GROUND TRUTH — same object rendered at {len(azs)} azimuths {azs}:",
    ]
    for az, p in sorted(gt_rest.items()):
        parts.append(f"- GT rest at az{az:03d}: {p}")
    for jname, frames in gt_motion.items():
        for (az, pct), p in sorted(frames.items()):
            parts.append(f"- GT joint `{jname}` at az{az} q={pct}%: {p}")
    parts.append("")
    parts.append(
        f"PREDICTED — one object rendered at {len(azs)} azimuths {azs} "
        "(rotations around the vertical axis). Same mesh source as GT — "
        "any visual differences come from URDF differences only."
    )
    for az, p in sorted(pred_rest.items()):
        parts.append(f"- pred rest at az{az:03d}: {p}")
    for jname, frames in pred_motion.items():
        for (az, pct), p in sorted(frames.items()):
            parts.append(f"- pred joint `{jname}` at az{az} q={pct}%: {p}")
    if preplaced:
        parts.extend([
            "",
            "REST POSE — a guard, not the question:",
            "  Parts are pre-placed at the GT rest pose, so every pred rest view",
            "  should be pixel-for-pixel the same object as the GT view at the",
            "  SAME azimuth (φ=0; no rotation search is needed). If any part sits",
            "  somewhere else, is missing, or is duplicated, rest_pose_match FAILS",
            "  — that means the prediction moved a mesh, which is never allowed.",
            "",
            "HOW TO JUDGE THE JOINTS:",
            f"  1. Glance at the {len(azs)} GT and pred rest views to confirm they match.",
            "  2. For each joint, read the GT motion frames and the pred motion",
            "     frames at the azimuths where the motion is visible, and decide:",
            "     a. TYPE — does the part rotate (revolute) or slide (prismatic)",
            "        exactly as in GT?",
            "     b. PIVOT — does it rotate about the same edge / axis line as GT?",
            "        A door hinged on the wrong side, a lid pivoting at its middle,",
            "        or a drawer sliding along the wrong axis is a direction failure",
            "        even if the part ends up roughly open.",
            "     c. DIRECTION — does it open the same way relative to the body",
            "        (outward not inward, pull out not push in)? Endpoint order in",
            "        the sweep (open→closed vs closed→open) is NOT a failure by",
            "        itself; the physical path is what counts.",
            "     d. RANGE — same order of magnitude (don't penalize 90° vs 130°).",
            "",
            "PASS criteria (ALL must hold):",
            "  - Every pred rest view matches its GT view at the same azimuth.",
            "  - Every joint has the GT motion type.",
            "  - Every joint pivots where GT pivots and opens the way GT opens.",
            "",
            "FAIL criteria (ANY triggers FAIL):",
            "  - A part is not where GT has it at rest, or is missing / duplicated.",
            "  - Wrong joint type, or a part that should move does not (or vice versa).",
            "  - Hinge on the wrong edge, slide along the wrong axis, or reversed",
            "    opening direction relative to the body.",
            "",
            "Small sub-cm offsets and color/material differences are OK.",
            "",
            "Output ONE JSON line with three sub-judgements plus the overall verdict:",
            '  {"rest_pose_match":"PASS"|"FAIL",',
            '   "motion_type_match":"PASS"|"FAIL",',
            '   "motion_direction_match":"PASS"|"FAIL",',
            '   "verdict":"PASS"|"FAIL",',
            '   "summary":"<short reasoning naming the joint(s) and what was wrong>"}',
            "Where:",
            "  - rest_pose_match: all pred rest views match GT at the same azimuth.",
            "  - motion_type_match: every joint's motion type (revolute vs prismatic)",
            "    matches GT.",
            "  - motion_direction_match: every joint pivots at the GT location and",
            "    opens in the GT direction relative to the body.",
            "  - verdict: PASS iff all three sub-judgements PASS.",
        ])
        return "\n".join(parts)
    parts.extend([
        "",
        "GLOBAL ROTATION IS OK — judge equivalence up to a Z-axis rotation:",
        "  We only care that the parts are correctly placed RELATIVE to each",
        "  other (e.g. doors on the body's open face, hinges on the correct",
        "  edges, doors swing outward). The whole assembly may be rotated by",
        "  0°, 90°, 180°, or 270° around the vertical axis relative to GT and",
        "  that is still PASS — the pred and GT are the same physical object,",
        "  just placed at different yaw.",
        "",
        f"  Concretely: find a rotation offset φ ∈ {{0, 90, 180, 270}} such",
        f"  that pred at az(X+φ mod 360) matches GT at az(X) for ALL X in",
        f"  {azs}. If any φ makes every pair match, PASS. Only if NO φ works",
        "  (parts are misplaced relative to each other, not just globally",
        "  rotated) is it FAIL.",
        "",
        "HOW TO JUDGE:",
        f"  1. Read every GT rest azimuth ({len(azs)} views). Identify GT's distinguishing",
        "     features and which azimuth shows which (e.g. \"GT shows two glass",
        "     doors at az045/az315 and solid brown back at az135/az225\").",
        f"  2. Read every pred rest azimuth ({len(azs)} views). Identify pred's features",
        "     analogously (e.g. \"pred shows two glass doors at az135/az315\").",
        "  3. Try to match pred's pattern to GT's pattern under each of the 4",
        "     rotation offsets:",
        "       φ=0:   pred-az045 vs GT-az045, pred-az135 vs GT-az135, ...",
        "       φ=90:  pred-az045 vs GT-az315, pred-az135 vs GT-az045, ...",
        "       φ=180: pred-az045 vs GT-az225, pred-az135 vs GT-az315, ...",
        "       φ=270: pred-az045 vs GT-az135, pred-az135 vs GT-az225, ...",
        "     The pattern of \"which feature appears at which screen position\"",
        "     should align under exactly one φ if the assembly is correct.",
        "  4. For each joint: read GT motion frames + 2-3 pred motion frames",
        "     at the azimuth where motion is visible (under the matched φ).",
        "     Motion direction should match (door swings outward, drawer pulls",
        "     out, etc.) — direction is relative to the body, not absolute world.",
        "",
        "PASS criteria (ALL must hold):",
        "  - Some φ ∈ {0,90,180,270} makes every pred rest view match its GT view.",
        "  - For each joint: motion type matches (revolute vs prismatic) and",
        "    motion direction relative to body matches GT (swing outward, not",
        "    inward; pull out, not push in).",
        "  - Motion magnitude in same order of magnitude (don't penalize 90° vs 130°).",
        "",
        "FAIL criteria (ANY triggers FAIL):",
        "  - NO rotation φ makes pred match GT — i.e. parts are placed on the",
        "    wrong relative side, or one part is mirrored, or geometry is broken.",
        "  - Wrong link count / part swapped to wrong slot.",
        "  - Joint motion reversed (door rotates INTO body instead of OUT).",
        "  - Joint type wrong (revolute when should be prismatic).",
        "",
        "Small sub-cm offsets, color/material differences, and global yaw are OK.",
        "Mirror flips (true reflection, not rotation), parts on wrong relative face,",
        "and reversed joint direction are NOT OK.",
        "",
        "Output ONE JSON line with three sub-judgements plus the overall verdict:",
        '  {"rest_pose_match":"PASS"|"FAIL",',
        '   "motion_type_match":"PASS"|"FAIL",',
        '   "motion_direction_match":"PASS"|"FAIL",',
        '   "verdict":"PASS"|"FAIL",',
        '   "summary":"<short reasoning; cite the chosen φ if PASS, or which',
        '     mismatches you saw at every φ if FAIL>"}',
        "Where:",
        "  - rest_pose_match: some φ aligns ALL 4 pred rest views with GT.",
        "  - motion_type_match: every joint's motion type (revolute vs prismatic)",
        "    matches GT.",
        "  - motion_direction_match: every joint's motion direction relative to",
        "    the body matches GT.",
        "  - verdict: PASS iff all three sub-judgements PASS.",
    ])
    return "\n".join(parts)


_SUB_CHECKS = {
    # vote-JSON key            -> CheckResult name
    "rest_pose_match": "subagent_rest_pose",
    "motion_type_match": "subagent_motion_type",
    "motion_direction_match": "subagent_motion_direction",
}


def _parse_vote(text: str) -> dict:
    """Extract one judge vote from raw response text.

    Returns {"verdict": "PASS"|"FAIL"|"?", "summary": str, <sub-keys...>}.
    Unparseable output keeps verdict "?" — counted as FAIL downstream
    (conservative).
    """
    vote: dict = {"verdict": "?", "summary": text[:300]}
    m = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", text)
    if not m:
        return vote
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return vote
    vote["verdict"] = (d.get("verdict") or "?").upper()
    vote["summary"] = d.get("summary") or vote["summary"]
    for k in _SUB_CHECKS:
        if d.get(k):
            vote[k] = str(d[k]).upper()
    return vote


def _aggregate_votes(votes: list[dict]) -> tuple[bool, list[CheckResult]]:
    """Majority-vote aggregation over independent judge votes.

    Rerun experiments showed single-call verdicts flip on ~50% of cases;
    a strict-majority vote over n independent judges is the cheapest
    variance reduction available. Non-PASS (including "?" parse failures)
    counts against, so a missing/garbled vote can never flip a FAIL to PASS.

    Returns (passed, checks): `subagent_verdict` plus one CheckResult per
    sub-judgement that at least one vote reported.
    """
    n = len(votes)
    checks: list[CheckResult] = []

    def _majority(key: str) -> tuple[bool, str]:
        tallied = [v for v in votes if key in v]
        n_pass = sum(1 for v in tallied if v[key] == "PASS")
        ok = bool(tallied) and n_pass * 2 > len(tallied)
        return ok, f"{n_pass}/{len(tallied)} votes PASS"

    passed, tally = _majority("verdict")
    # Cite a summary from the majority side so the detail explains the verdict.
    side = [v for v in votes if (v.get("verdict") == "PASS") == passed]
    summary = (side or votes or [{}])[0].get("summary", "no votes returned")
    checks.append(CheckResult(
        name="subagent_verdict",
        passed=passed,
        detail=f"[{tally}] {summary}",
    ))
    for key, check_name in _SUB_CHECKS.items():
        if not any(key in v for v in votes):
            continue
        ok, sub_tally = _majority(key)
        checks.append(CheckResult(name=check_name, passed=ok, detail=sub_tally))
    if n == 0:
        passed = False
    return passed, checks


def oracle_backend(backend: str | None = None) -> str:
    """Resolve the judge backend: explicit argument, then env, then default."""
    name = (backend or os.environ.get("EVO_ORACLE_BACKEND") or "").strip().lower()
    name = name or DEFAULT_ORACLE_BACKEND
    if name not in DEFAULT_ORACLE_MODEL:
        raise ValueError(
            f"EVO_ORACLE_BACKEND={name!r} — expected one of "
            f"{sorted(DEFAULT_ORACLE_MODEL)}"
        )
    return name


def oracle_model(backend: str, model: str | None = None) -> str:
    """Resolve the judge model for `backend`: argument, then env, then default."""
    return (
        model
        or os.environ.get("EVO_ORACLE_MODEL")
        or DEFAULT_ORACLE_MODEL[backend]
    )


async def _query_subagent_responses(
    prompt: str,
    *,
    model: str,
    effort: str,
    max_turns: int,
    cwd: Path,
) -> tuple[dict, float | None]:
    """Judge once through the shared Responses provider.

    Runs on the same client as the executor and generator, so one set of
    credentials (`OPENAI_API_BASE` + `OPENROUTER_EVO_KEY`) covers every agent
    in the loop. The scoped file tools are rooted at `cwd` (the prediction
    directory holding `oracle_render/`) and are read-only, so the judge can
    reach the frames it was given and nothing else.
    """
    from evo.engine.responses_backend import run_responses_agent

    outcome = await run_responses_agent(
        name="oracle-judge",
        instructions=SYSTEM,
        prompt=prompt,
        model=model,
        effort=effort,
        max_turns=max_turns,
        cwd=cwd,
        read_roots=[cwd],
        write_root=None,
    )
    # `votes` judges each read every rendered frame, which makes the oracle
    # the most expensive stage per task. Report its cost rather than leaving
    # the loop's only vision spend invisible; None means the model has no
    # entry in evo.engine.pricing.
    summary = outcome.usage_summary or {}
    cost = summary.get("cost_usd") if summary.get("cost_priced") else None
    return _parse_vote(outcome.final_output), cost


async def _query_subagent_claude(
    prompt: str,
    *,
    model: str,
    effort: str,
    max_turns: int,
    cwd: Path,
) -> tuple[dict, float | None]:
    """Judge once through the Claude Agent SDK (EVO_ORACLE_BACKEND=claude).

    Imports `claude_agent_sdk` lazily so the default Responses path never pays
    for it, and so test runs that mock this stay light.
    """
    from claude_agent_sdk import (
        AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query,
    )

    opts = ClaudeAgentOptions(
        model=model,
        env={"CLAUDE_CODE_DISABLE_FAST_MODE": "1"},
        settings=json.dumps({
            "fastMode": False,
            "fastModePerSessionOptIn": True,
        }),
        system_prompt=SYSTEM,
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        cwd=str(cwd),
        add_dirs=[str(cwd)],
        disallowed_tools=["WebFetch", "WebSearch"],
        effort=effort,
    )
    text = ""
    cost = None
    async for msg in query(prompt=prompt, options=opts):
        if isinstance(msg, AssistantMessage):
            for c in msg.content:
                if isinstance(c, TextBlock):
                    text += c.text
        if isinstance(msg, ResultMessage):
            cost = getattr(msg, "total_cost_usd", None)
            break

    return _parse_vote(text), cost


async def _query_subagent(
    prompt: str,
    *,
    backend: str | None = None,
    model: str | None = None,
    effort: str = DEFAULT_AGENT_EFFORT,
    max_turns: int = ORACLE_MAX_TURNS,
    cwd: Path | None = None,
) -> tuple[dict, float | None]:
    """Call one judge and return (vote_dict, cost_usd)."""
    name = oracle_backend(backend)
    runner = (
        _query_subagent_responses if name == "responses"
        else _query_subagent_claude
    )
    return await runner(
        prompt,
        model=oracle_model(name, model),
        effort=effort,
        max_turns=max_turns,
        cwd=(cwd or ROOT).resolve(),
    )


async def run_subagent_oracle_async(
    gt_dir: Path,
    pred_dir: Path,
    *,
    tag: str,
    backend: str | None = None,
    model: str | None = None,
    effort: str = DEFAULT_AGENT_EFFORT,
    azs: list[int] | None = None,
    votes: int = 3,
    verbose: bool = False,
) -> OracleResult:
    """Async impl of the subagent oracle. Call from an event loop.

    Use `run_subagent_oracle` (sync wrapper) when no loop is running.

    `backend` and `model` default to `EVO_ORACLE_BACKEND` / `EVO_ORACLE_MODEL`
    and then to `DEFAULT_ORACLE_BACKEND` / `DEFAULT_ORACLE_MODEL` — i.e. the
    same Responses provider the executor and generator use.

    `azs` defaults to [45, 135, 225, 315] — pred is rendered at all 4
    canonical viewpoints so the subagent can pick the orientation that
    aligns with GT, since the assembled pred isn't guaranteed to share
    GT's coordinate frame.

    `votes` independent judge calls run concurrently and are combined by
    strict majority (see `_aggregate_votes`). votes=1 restores the old
    single-call behavior.
    """
    if azs is None:
        azs = [45, 135, 225, 315]
    votes = max(1, int(votes))
    backend = oracle_backend(backend)
    model = oracle_model(backend, model)

    # Render BOTH sides into <pred_dir>/oracle_render/. Same mesh source
    # (original parts/*.obj) → zero material drift. Idempotent: skips
    # files that already exist unless we force from outside.
    from evo.oracle.render import render_for_oracle  # local import to avoid cycle
    oracle_dir = await asyncio.to_thread(
        lambda: render_for_oracle(gt_dir, pred_dir, azs, verbose=verbose)[0],
    )

    gt_rest, gt_motion = _collect_oracle_frames(oracle_dir, azs, "gt")
    pred_rest, pred_motion = _collect_oracle_frames(oracle_dir, azs, "pred")

    if not pred_rest and not pred_motion:
        return OracleResult(
            passed=False,
            checks=[CheckResult(
                name="subagent_verdict",
                passed=False,
                detail=f"no pred renders found in {oracle_dir}; cannot judge visually",
            )],
            matching=None,
        )

    prompt = _build_prompt(
        tag=tag,
        gt_rest=gt_rest, gt_motion=gt_motion,
        pred_rest=pred_rest, pred_motion=pred_motion,
        azs=azs,
        preplaced=_dataset_preplaced(Path(gt_dir)),
    )

    t0 = time.time()
    raw = await asyncio.gather(
        *(
            _query_subagent(
                prompt,
                backend=backend,
                model=model,
                effort=effort,
                cwd=pred_dir.resolve(),
            )
            for _ in range(votes)
        ),
        return_exceptions=True,
    )
    vote_dicts: list[dict] = []
    total_cost = 0.0
    have_cost = False
    for r in raw:
        if isinstance(r, BaseException):
            vote_dicts.append({
                "verdict": "?",
                "summary": f"subagent call failed: {type(r).__name__}: {str(r)[:200]}",
            })
            continue
        vote, cost = r
        vote_dicts.append(vote)
        if cost is not None:
            total_cost += cost
            have_cost = True
    elapsed = time.time() - t0

    passed, checks = _aggregate_votes(vote_dicts)
    # Models with no entry in evo.engine.pricing leave cost None; report the
    # run shape regardless so the detail always says which judge produced it.
    spend = f"${total_cost:.4f} / " if have_cost else ""
    checks[0].detail = (
        f"{checks[0].detail} | {spend}{elapsed:.0f}s / {votes} votes "
        f"/ {backend}:{model}"
    )
    if verbose:
        print(f"  [subagent {tag}] {'PASS' if passed else 'FAIL'}: "
              f"{checks[0].detail[:140]}")

    return OracleResult(
        passed=passed,
        checks=checks,
        matching=None,
        cost_usd=round(total_cost, 6) if have_cost else None,
    )


def run_subagent_oracle(
    gt_dir: Path,
    pred_dir: Path,
    **kwargs,
) -> OracleResult:
    """Sync wrapper around `run_subagent_oracle_async` for use OUTSIDE an
    event loop (CLI / standalone scripts). Raises RuntimeError if called
    from inside a running asyncio loop — use the `_async` variant there.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_subagent_oracle() called from inside a running asyncio "
            "loop — use run_subagent_oracle_async(...) and await it."
        )
    return asyncio.run(run_subagent_oracle_async(gt_dir, pred_dir, **kwargs))
