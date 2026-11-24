"""Failure clusters: every failure of a round, grouped, for the Skill Evolver.

The Evaluator digest (feedback_digest.py) ranks exclusive failure *modes* and
shows three representative tasks per mode; the Evolver then reads those three
and writes one feature.  On artic92 E1 that meant 3 of 24–32 failures were
read in depth each round and "scale" was picked 10 rounds out of 10.

This module is the alternative feedback block (``EVO_FEEDBACK_FORMAT=cluster``):

1. every failed scored dimension of every GT joint of every evaluated task is a
   *failure item* (plus one item per task for execution / structure failures);
2. items are grouped into clusters by ``(mode, joint type, error bucket)`` —
   e.g. ``axis · PRISMATIC · flipped (≥150°)`` or ``scale · REVOLUTE · small
   (0.5–0.95)`` — and clusters are ranked by the number of distinct tasks;
3. the report lists EVERY cluster with EVERY member task (one line each with
   the numbers the oracle reported and the output_dir), how many members were
   in the same cluster last round (persistence), and which tasks the loop will
   use when the Evolver calls ``try_skill(cluster=<id>)``;
4. the edit ledger of previous rounds: for each cluster edit the Evolver made,
   the try_skill verdict (before → after on the cluster's trial tasks) and the
   batch result, so rejected edits are visible and not re-proposed.

Nothing here reads ground truth: all numbers come from the sanitised run
artifacts (``JointReport.to_dict(redact_gt=True)`` and ``sanitize_detail``).
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from evo.engine.feedback_digest import (
    STRUCTURE_CHECKS, _drift_ratio, _rest_ratio, summarize_executor_log,
)

# Dimension order per joint: the first failing one names the item (later ones
# are usually consequences: a wrong axis also fails pivot and motion).
JOINT_DIMS = ("type_ok", "axis_ok", "scale_ok", "pivot_ok", "motion_ok", "limits_ok")
DIM_MODE = {
    "type_ok": "joint_type", "axis_ok": "axis", "scale_ok": "scale",
    "pivot_ok": "pivot", "motion_ok": "motion", "limits_ok": "limits",
}
MAX_CLUSTERS_FOR_TRIALS = 6
TRIAL_TASKS_PER_CLUSTER = 3

_RATIO_RE = re.compile(r"size ratio ([0-9.]+)")


def _axis_bucket(deg: float | None) -> str:
    if deg is None:
        return "angle unknown"
    if deg >= 150:
        return "flipped sign (≥150°)"
    if deg >= 60:
        return "wrong axis (60–150°)"
    if deg >= 20:
        return "tilted 20–60°"
    return "tilted <20°"


def _scale_bucket(ratio: float | None) -> str:
    if not ratio or ratio <= 0:
        return "ratio unknown"
    if ratio <= 0.5:
        return "far too small (≤0.5×)"
    if ratio < 0.95:
        return "too small (0.5–0.95×)"
    if ratio <= 1.05:
        return "within 5% (another link off)"
    if ratio <= 2.0:
        return "too large (1.05–2×)"
    return "far too large (>2×)"


def _pivot_bucket(m: float | None) -> str:
    if m is None:
        return "distance unknown"
    cm = m * 100
    if cm < 2:
        return "off <2 cm"
    if cm < 10:
        return "off 2–10 cm"
    return "off ≥10 cm"


def _motion_bucket(m: float | None) -> str:
    if m is None:
        return "drift unknown"
    cm = m * 100
    if cm < 5:
        return "trace drift <5 cm"
    if cm < 20:
        return "trace drift 5–20 cm"
    return "trace drift ≥20 cm"


def _limits_bucket(ratio: float | None) -> str:
    if ratio is None:
        return "ratio unknown"
    return "range too small" if ratio < 1 else "range too large"


def _ratio_bucket(r: float | None, what: str) -> str:
    if r is None:
        return f"{what} (size unknown)"
    if r < 1.5:
        return f"{what} within 1.5× tol"
    if r < 4:
        return f"{what} 1.5–4× tol"
    return f"{what} ≥4× tol"


def _rest_within_tol(rep: dict[str, Any]) -> bool:
    d, tol = rep.get("rest_drift_m"), rep.get("pose_tol_m")
    return d is not None and bool(tol) and d <= tol


def _rot_bucket(deg: float | None) -> str:
    if deg is None:
        return "angle unknown"
    if deg >= 150:
        return "flipped (≥150°)"
    if deg >= 60:
        return "quarter turn (60–150°)"
    if deg >= 20:
        return "tilted 20–60°"
    return "tilted <20°"


def _pose_cause_item(rep: dict[str, Any]) -> tuple[str, str, str]:
    """(mode, bucket, numbers) of a rest-pose failure from the oracle's pose_cause."""
    cause = rep.get("pose_cause") or "unknown"
    rot, pos, ratio = rep.get("rot_err_deg"), rep.get("pos_err_m"), rep.get("scale_ratio")
    drift, tol = rep.get("rest_drift_m"), rep.get("pose_tol_m")
    nums = []
    if drift is not None and tol:
        nums.append(f"rest drift {drift * 100:.1f} cm = {drift / tol:.1f}× tol")
    if rot is not None:
        nums.append(f"orientation {rot:.0f}° off")
    if ratio is not None:
        nums.append(f"size ratio {ratio}")
    if pos is not None:
        nums.append(f"pivot {pos * 100:.1f} cm off")
    if cause == "orientation":
        bucket = _rot_bucket(rot)
    elif cause == "scale":
        bucket = _scale_bucket(ratio)
    elif cause == "position":
        bucket = _pivot_bucket(pos)
    elif cause == "unknown":
        bucket = "cause unknown"
    else:
        bucket = "several corrections needed" + (f", {_rot_bucket(rot)}" if "orientation" in cause else "")
    return f"pose·{cause}", bucket, ", ".join(nums)


# Joint dimensions a dataset view declares diagnostic (dataset_variant.json
# lists scored_joint_dims without them); set by the loop. They name no failure.
DIAGNOSTIC_DIMS: set[str] = set()


def failure_items(run: dict[str, Any]) -> list[dict[str, Any]]:
    """All failure items of one evaluated task (empty for a PASS)."""
    tag = run.get("tag", "?")
    verdict = run.get("oracle_verdict")
    out: list[dict[str, Any]] = []
    if verdict in ("execution_failed", "no_oracle_result"):
        err = (run.get("error") or "no oracle result")[:120]
        out.append({"tag": tag, "mode": "execution_failed", "jtype": "task", "bucket": "no URDF",
                    "joint": None, "numbers": err})
        return out
    if verdict == "PASS":
        return out
    failed = set(run.get("failed_oracle_checks") or [])
    details = run.get("failed_check_details") or {}
    if failed & STRUCTURE_CHECKS:
        names = sorted(failed & STRUCTURE_CHECKS)
        out.append({"tag": tag, "mode": "structure", "jtype": "task", "bucket": "+".join(names),
                    "joint": None, "numbers": (details.get(names[0]) or "")[:120]})
    joints = run.get("joint_reports") or {}
    joint_level_fail = False
    for jname, rep in sorted(joints.items()):
        if not rep.get("matched") or not rep.get("pred_joint"):
            out.append({"tag": tag, "mode": "structure", "jtype": "task", "bucket": "unmatched joint",
                        "joint": jname, "numbers": "GT child link matched no predicted joint"})
            joint_level_fail = True
            continue
        jtype = rep.get("pred_type") or "?"
        for dim in JOINT_DIMS:
            if dim in DIAGNOSTIC_DIMS:
                continue          # a diagnostic dimension of this dataset view names no failure
            if rep.get(dim) is False:
                mode = DIM_MODE[dim]
                if dim == "axis_ok":
                    bucket, num = _axis_bucket(rep.get("axis_angle_deg")), f"axis {rep.get('axis_angle_deg')}°"
                elif dim == "scale_ok":
                    bucket, num = _scale_bucket(rep.get("scale_ratio")), f"size ratio {rep.get('scale_ratio')}"
                elif dim in ("pivot_ok", "motion_ok") and rep.get("pose_cause"):
                    # Oracle 3 names which correction of the child link (size,
                    # orientation about the pivot, pivot position) would bring
                    # its rest pose within tolerance: cluster by that cause, not
                    # by the downstream symptom (trace drift).
                    mode, bucket, num = _pose_cause_item(rep)
                elif dim == "motion_ok" and _rest_within_tol(rep):
                    d = rep.get("trace_drift_m")
                    bucket = "rest pose correct, sweep wrong (direction / axis line)"
                    num = f"trace drift {d * 100:.1f} cm, rest drift {rep.get('rest_drift_m', 0) * 100:.1f} cm" if d is not None else "drift ?"
                elif dim == "pivot_ok":
                    d = rep.get("pivot_distance_m")
                    bucket, num = _pivot_bucket(d), f"pivot {d * 100:.1f} cm" if d is not None else "pivot ?"
                elif dim == "motion_ok":
                    d = rep.get("trace_drift_m")
                    bucket, num = _motion_bucket(d), f"trace drift {d * 100:.1f} cm" if d is not None else "drift ?"
                elif dim == "limits_ok":
                    bucket, num = _limits_bucket(rep.get("limit_ratio")), f"limit ratio {rep.get('limit_ratio')}"
                else:
                    bucket, num = f"pred {jtype}", f"type {jtype}"
                out.append({"tag": tag, "mode": mode, "jtype": jtype, "bucket": bucket,
                            "joint": jname, "pred_joint": rep.get("pred_joint"), "numbers": num})
                joint_level_fail = True
                break  # first failing dimension names the item
    # Task-level placement failures that no joint dimension explains.
    if "rest_pose_alignment" in failed and not joint_level_fail:
        r = _rest_ratio(details.get("rest_pose_alignment"))
        out.append({"tag": tag, "mode": "pose", "jtype": "task", "bucket": _ratio_bucket(r, "rest drift"),
                    "joint": None, "numbers": (details.get("rest_pose_alignment") or "")[:140]})
    elif "kinematic_equivalence" in failed and not joint_level_fail:
        r = _drift_ratio(details.get("kinematic_equivalence"))
        out.append({"tag": tag, "mode": "motion", "jtype": "task", "bucket": _ratio_bucket(r, "trace drift"),
                    "joint": None, "numbers": (details.get("kinematic_equivalence") or "")[:140]})
    elif "link_scale" in failed and not joint_level_fail:
        rs = [float(x) for x in _RATIO_RE.findall(details.get("link_scale") or "")]
        worst = max(rs, key=lambda x: abs(math.log(x)) if x > 0 else 0, default=None)
        out.append({"tag": tag, "mode": "scale", "jtype": "task", "bucket": _scale_bucket(worst),
                    "joint": None, "numbers": (details.get("link_scale") or "")[:140]})
    elif not out and failed:
        name = sorted(failed)[0]
        out.append({"tag": tag, "mode": name, "jtype": "task", "bucket": "other",
                    "joint": None, "numbers": (details.get(name) or "")[:140]})
    return out


def cluster_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group the failure items of a round; clusters sorted by distinct tasks."""
    by_key: dict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
    by_tag = {r.get("tag"): r for r in runs}
    for run in runs:
        for item in failure_items(run):
            key = (item["mode"], item["jtype"], item["bucket"])
            c = by_key.setdefault(key, {"key": key, "tasks": OrderedDict(), "n_items": 0})
            c["tasks"].setdefault(item["tag"], []).append(item)
            c["n_items"] += 1
    clusters = sorted(by_key.values(), key=lambda c: (-len(c["tasks"]), -c["n_items"], c["key"]))
    for i, c in enumerate(clusters, 1):
        c["id"] = f"C{i}"
        mode, jtype, bucket = c["key"]
        c["label"] = f"{mode} · {jtype} · {bucket}"
        c["n_tasks"] = len(c["tasks"])
        c["runs"] = {tag: by_tag.get(tag) for tag in c["tasks"]}
    return clusters


def cluster_key_of(c: dict[str, Any]) -> str:
    return "|".join(c["key"])


def persistence_counts(clusters: list[dict[str, Any]], prev_runs: list[dict[str, Any]] | None) -> None:
    """Annotate each cluster with how many of its tasks sat in the same cluster last round."""
    prev_members: dict[tuple[str, str, str], set[str]] = {}
    if prev_runs:
        for c in cluster_runs(prev_runs):
            prev_members[c["key"]] = set(c["tasks"])
    for c in clusters:
        prev = prev_members.get(c["key"], set())
        c["persistent"] = sorted(t for t in c["tasks"] if t in prev)


def select_trial_tasks(clusters: list[dict[str, Any]], *, per_cluster: int = TRIAL_TASKS_PER_CLUSTER,
                       max_clusters: int = MAX_CLUSTERS_FOR_TRIALS) -> dict[str, list[str]]:
    """Trial tasks per cluster for try_skill: persistent members first."""
    out: dict[str, list[str]] = {}
    for c in clusters[:max_clusters]:
        persistent = list(c.get("persistent") or [])
        rest = [t for t in c["tasks"] if t not in persistent]
        out[c["id"]] = (persistent + rest)[:per_cluster]
    return out


def _item_line(item: dict[str, Any]) -> str:
    j = f" {item['joint']}→{item.get('pred_joint')}" if item.get("joint") else ""
    return f"{item['tag']}{j}: {item['numbers']}"


def build_cluster_report(
    feedback_history: list[dict[str, Any]],
    *,
    run_dir: Path | None = None,
    full_list_path: Path | None = None,
    trial_tasks: dict[str, list[str]] | None = None,
    read_logs: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Render the cluster report for the LAST round; returns (text, clusters)."""
    if not feedback_history:
        return "(no evaluation yet)", []
    cur = feedback_history[-1]
    prev = feedback_history[-2] if len(feedback_history) >= 2 else None
    runs = cur.get("runs") or []
    clusters = cluster_runs(runs)
    persistence_counts(clusters, (prev or {}).get("runs"))
    if trial_tasks is None:
        trial_tasks = select_trial_tasks(clusters)
    n = len(runs)
    n_fail = sum(1 for r in runs if r.get("oracle_verdict") != "PASS")
    out: list[str] = []
    out.append(f"#### 1. Failure clusters  ({n} tasks, {n_fail} failed; every failed joint dimension of every "
               "task is one item; a task can be in several clusters; persistent = the same task was in the same "
               "cluster in the previous round)")
    out.append("| id | cluster (mode · joint type · error bucket) | tasks | items | persistent | try_skill tasks |")
    out.append("|---|---|---|---|---|---|")
    for c in clusters:
        trial = ", ".join(trial_tasks.get(c["id"], [])) or "—"
        out.append(f"| {c['id']} | {c['label']} | {c['n_tasks']} | {c['n_items']} | "
                   f"{len(c.get('persistent') or []) if prev else '—'} | {trial} |")
    out.append("Clusters are ranked by how many tasks they touch. The first failing dimension of a joint names its "
               "item (a wrong axis also fails pivot and motion; those are not counted again). Read ALL members of a "
               "cluster before editing for it: one rule that fits only the first member will not move the cluster.")

    out.append("")
    out.append("#### 2. Cluster members  (one line per failure item; output_dir holds the trajectory, URDF and renders)")
    for c in clusters:
        out.append(f"- **{c['id']}  {c['label']}**  ({c['n_tasks']} tasks)")
        for tag, items in c["tasks"].items():
            run = c["runs"].get(tag) or {}
            mark = " [persistent]" if tag in (c.get("persistent") or []) else ""
            lines = "; ".join(_item_line(it) for it in items)
            out.append(f"  - {lines}{mark}")
            out.append(f"      output_dir: {run.get('output_dir', '')}")
        # executor summaries for the trial tasks only (log reads are not free)
        if read_logs:
            for tag in trial_tasks.get(c["id"], []):
                run = c["runs"].get(tag) or {}
                if run.get("output_dir"):
                    out.append(f"      executor on {tag}: "
                               f"{summarize_executor_log(Path(run['output_dir']) / 'executor.log')}")
    out.append("")
    out.append("#### 3. Edit gate")
    out.append("`try_skill(cluster=\"C3\")` runs the cluster's try_skill tasks (listed above) with the skill in "
               "your working directory and scores them with the same oracle as the batch. It reports, per task, "
               "the verdict and failed checks before (this round's batch) and after (your edit), and whether the "
               "cluster improved. Make one bounded edit per cluster, then gate it; an edit whose cluster did not "
               "improve is reverted or revised before you move on.")
    ledger = edit_ledger(feedback_history)
    if ledger:
        out.append("")
        out.append("#### 4. Edit ledger  (previous rounds: each cluster edit, its try_skill verdict and the batch result)")
        out.append("| round | cluster | edit | try_skill before → after | kept in skill | batch pass after | gate |")
        out.append("|---|---|---|---|---|---|---|")
        for row in ledger:
            out.append("| " + " | ".join(str(x) for x in row) + " |")
        out.append("A rejected edit (not improved) must not be re-proposed in the same form; a kept edit whose batch "
                   "result did not move is a candidate for removal.")
    if full_list_path is not None:
        out.append("")
        out.append(f"#### 5. Full per-task list: `{full_list_path}`  (Read it for a task's full oracle detail)")
    return "\n".join(out), clusters


def edit_ledger(feedback_history: list[dict[str, Any]]) -> list[list[str]]:
    """Rows from each round's recorded cluster trials and the next round's batch."""
    rows: list[list[str]] = []
    for i, fb in enumerate(feedback_history[:-1]):
        trials = fb.get("cluster_trials") or []
        nxt = feedback_history[i + 1]
        batch_after = f"{round((nxt.get('pass_rate') or 0) * len(nxt.get('runs') or []))}/{len(nxt.get('runs') or [])}"
        gate = "rolled back" if nxt.get("skill_rolled_back") else ("new best" if nxt.get("new_best") else "kept")
        by_cluster: dict[str, dict[str, Any]] = OrderedDict()
        for t in trials:
            by_cluster[t.get("cluster") or "smoke"] = t  # last call per cluster wins
        if not by_cluster:
            continue
        for cid, t in by_cluster.items():
            before, after = t.get("pass_before"), t.get("pass_after")
            n = t.get("n_tasks")
            verdict = f"{before}/{n} → {after}/{n}" + (" improved" if t.get("improved") else " not improved")
            rows.append([f"R{fb.get('round')}", cid, (t.get("edit") or "?")[:60], verdict,
                         "yes" if t.get("kept", True) else "reverted", batch_after, gate])
    return rows
