"""Evaluator feedback digest: the structured, ranked report the Skill Evolver
reads instead of a flat per-task list.

Run 32 showed the problem with the flat list: every round the Evolver was
handed 43 task lines of equal weight (about 56k characters), opened only the
first four of them in depth, summarised the rest as "rest pose failed 27/43",
and 10 of its 15 proposals were rolled back.  Nothing told it which failure
mode dominated, which tasks were the clearest examples of it, which tasks its
last edit had flipped, or which features had already been tried and rejected.

This module derives all of that deterministically from data the Evaluator
already produces (the sanitised oracle verdicts, the per-joint reports, the
executor logs and the loop's own gate records).  It never reads ground truth:
every number here comes from `oracle.json` details that `sanitize_detail`
has already stripped of GT constants, or from `JointReport.to_dict(redact_gt=True)`.

Sections of the digest (see `build_digest`):

1. failure modes — every evaluated task is assigned ONE primary mode by the
   rules in `classify_run`, ordered from "the URDF never existed" down to
   "missed the tolerance by a hair"; the table counts tasks per mode, how many
   of those failures are persistent (the same task failed in the same mode in
   the previous round too: with one session per task that is the available
   evidence against sampling noise), and the change since the previous round;
2. representative tasks — per mode, the few persistent tasks with the largest
   error, each with a one-line summary of what the executor itself did and
   claimed (helpers used, images inspected, its final statement) so the
   Evolver can see where the Skill's own checks were blind;
3. flips — tasks whose verdict changed since the previous round, by check;
4. feature ledger — one row per past round: the feature the Evolver added,
   the train result it produced and what the regression gate decided;
5. the path of the full per-task list, which the Evolver can still Read.
"""
from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

# Primary failure modes, in priority order: a task is labelled with the FIRST
# mode whose rule matches, because the later ones are usually consequences of
# the earlier ones (a wrongly scaled link is also mis-posed at rest, a
# mis-posed link also fails its motion trace).
MODES: tuple[tuple[str, str], ...] = (
    ("execution_failed", "no URDF: executor crashed, hit a cap or wrote FAILED.txt"),
    ("structure", "wrong tree / link count / link matching / topology"),
    ("joint_type", "joint type differs from the reference"),
    ("axis", "joint axis direction off"),
    ("scale", "one or more links at the wrong size"),
    ("pose", "link placed or oriented wrongly at rest (scale ok)"),
    ("pivot", "revolute pivot off while axis and rest pose are right"),
    ("motion", "rest pose right but the joint trace moves the wrong way"),
    ("limits", "joint travel limits off"),
    ("near_miss", "every check within 1.5x its tolerance, one just outside"),
    ("pass", "all scored checks passed"),
)
MODE_ORDER = [m for m, _ in MODES]
MODE_HELP = dict(MODES)

STRUCTURE_CHECKS = {
    "model_integrity", "valid_tree", "link_count", "joint_type_distribution",
    "link_matching", "topology",
}
NEAR_MISS_RATIO = 1.5
DETAIL_CAP = 260      # per check line in the digest; the full list is uncapped


def _cap(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n].rstrip() + " …"

_REST_RE = re.compile(r"by > ([0-9.]+)m \(max=([0-9.]+)m\)")
_VOTES_RE = re.compile(r"^\[(\d+)/(\d+) votes PASS\]\s*")
_SCALE_RE = re.compile(r"size ratio ([0-9.]+)")
_DRIFT_RE = re.compile(r"trace drift=([0-9.]+)m \(\w+, tol=([0-9.]+)m\)")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _rest_ratio(detail: str | None) -> float | None:
    """max rest drift / tolerance from a rest_pose_alignment detail, if parsable."""
    if not detail:
        return None
    m = _REST_RE.search(detail)
    if not m:
        return None
    tol, mx = float(m.group(1)), float(m.group(2))
    return mx / tol if tol > 0 else None


def _drift_ratio(detail: str | None) -> float | None:
    """Largest trace drift / tolerance from a kinematic_equivalence detail."""
    if not detail:
        return None
    ratios = [float(a) / float(b) for a, b in _DRIFT_RE.findall(detail) if float(b) > 0]
    return max(ratios) if ratios else None


def _scale_error(run: dict[str, Any]) -> float:
    """Largest |log size ratio| over joints and the link_scale detail (0 if none)."""
    worst = 0.0
    for rep in (run.get("joint_reports") or {}).values():
        r = rep.get("scale_ratio")
        if rep.get("scale_ok") is False and r:
            worst = max(worst, abs(math.log(r)))
    detail = (run.get("failed_check_details") or {}).get("link_scale") or ""
    for r in _SCALE_RE.findall(detail):
        if float(r) > 0:
            worst = max(worst, abs(math.log(float(r))))
    return worst


def classify_run(run: dict[str, Any]) -> tuple[str, float]:
    """Primary failure mode of one evaluated task and a severity for ranking.

    Severity is unit-free: a ratio of the observed error to the check's own
    tolerance where the oracle reports one (rest drift, trace drift), the
    |log| size ratio for scale, degrees for axis, centimetres for pivot.
    Bigger means a clearer example of the mode.
    """
    verdict = run.get("oracle_verdict")
    if verdict in ("execution_failed", "no_oracle_result"):
        return "execution_failed", 0.0
    if verdict == "PASS":
        return "pass", 0.0
    failed = set(run.get("failed_oracle_checks") or [])
    details = run.get("failed_check_details") or {}
    joints = (run.get("joint_reports") or {}).values()

    if failed & STRUCTURE_CHECKS:
        return "structure", float(len(failed & STRUCTURE_CHECKS))
    if "joint_types" in failed or any(j.get("type_ok") is False for j in joints):
        return "joint_type", float(sum(j.get("type_ok") is False for j in joints))
    bad_axis = [j.get("axis_angle_deg") or 0.0 for j in joints if j.get("axis_ok") is False]
    if "joint_axes" in failed or bad_axis:
        return "axis", max(bad_axis, default=0.0)
    scale_err = _scale_error(run)
    if "link_scale" in failed or any(j.get("scale_ok") is False for j in joints):
        return "scale", scale_err
    rest_ratio = _rest_ratio(details.get("rest_pose_alignment"))
    drift_ratio = _drift_ratio(details.get("kinematic_equivalence"))
    bad_pivot = [(j.get("pivot_distance_m") or 0.0) * 100 for j in joints if j.get("pivot_ok") is False]
    if "rest_pose_alignment" in failed:
        r = rest_ratio if rest_ratio is not None else (drift_ratio or 0.0)
        others = {"rest_pose_alignment", "kinematic_equivalence", "joint_pivot_alignment"}
        if r < NEAR_MISS_RATIO and failed <= others and not bad_pivot:
            return "near_miss", r
        return "pose", r
    if "joint_pivot_alignment" in failed or bad_pivot:
        return "pivot", max(bad_pivot, default=0.0)
    if "kinematic_equivalence" in failed:
        r = drift_ratio or 0.0
        if r < NEAR_MISS_RATIO and failed == {"kinematic_equivalence"}:
            return "near_miss", r
        return "motion", r
    if "joint_limits" in failed:
        return "limits", float(sum(j.get("limits_ok") is False for j in joints))
    # A diagnostic-only failure with every scored check passing is still a FAIL
    # verdict only when the loop says so; fall back to the first failed name.
    return (sorted(failed)[0] if failed else "pose"), 0.0


def stability(run: dict[str, Any]) -> str | None:
    """'stable' when every failed check failed in every session, 'unstable'
    when some session passed one of them, None when the task ran once."""
    reps = run.get("executor_reps") or 1
    if reps <= 1:
        return None
    for detail in (run.get("failed_check_details") or {}).values():
        m = _VOTES_RE.match(detail or "")
        if m and int(m.group(1)) > 0:
            return "unstable"
    for rep in (run.get("joint_reports") or {}).values():
        if (rep.get("score_spread") or 0) >= 0.5:
            return "unstable"
    return "stable"


def persistence(tag: str, mode: str, feedback_history: list[dict[str, Any]]) -> int:
    """How many consecutive earlier rounds the same task failed in the same
    mode (0 = new this round).  Every round evaluates the same train pool, so
    a failure that survives several Skill versions is not sampling noise."""
    n = 0
    for fb in reversed(feedback_history[:-1]):
        prev = next((r for r in (fb.get("runs") or []) if r.get("tag") == tag), None)
        if prev is None or classify_run(prev)[0] != mode:
            break
        n += 1
    return n


# ---------------------------------------------------------------------------
# Executor trajectory summary (rule-based, from executor.log)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"^\[[0-9:]+\] \[tool_call\] (\w+)(?::|$)")
_HELPER_IMPORT_RE = re.compile(r"from tools\.(\w+) import ([\w, ]+)")
_HELPER_ATTR_RE = re.compile(r"\btools\.(\w+)\.(\w+)\s*\(")
_TEXT_RE = re.compile(r"^\[[0-9:]+\] \[text\] (.*)$")
_IMAGE_READ_RE = re.compile(r"\[tool_call\] Read: .*\.(?:png|jpg|jpeg)", re.I)


def summarize_executor_log(log_path: Path, *, max_claim: int = 160) -> str:
    """One line: which helpers the executor used, how much it inspected and
    what it finally claimed.  Nothing here is scored; it is for the Evolver to
    set against the oracle verdict of the same task."""
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return "(executor.log unavailable)"
    calls: Counter[str] = Counter()
    helpers: "OrderedDict[str, None]" = OrderedDict()
    images = 0
    claims: list[str] = []
    for line in text.splitlines():
        m = _TOOL_CALL_RE.match(line)
        if m:
            calls[m.group(1)] += 1
        if _IMAGE_READ_RE.search(line):
            images += 1
        for mod, names in _HELPER_IMPORT_RE.findall(line):
            for n in names.split(","):
                n = n.strip()
                if n:
                    helpers.setdefault(f"{mod}.{n}", None)
        for mod, fn in _HELPER_ATTR_RE.findall(line):
            helpers.setdefault(f"{mod}.{fn}", None)
        t = _TEXT_RE.match(line)
        if t and t.group(1).strip():
            claims.append(t.group(1).strip())
    sim = calls.get("mcp__sim__execute_python_code", 0)
    claim = claims[0] if claims else "(no final statement)"
    if len(claim) > max_claim:
        claim = claim[:max_claim].rstrip() + " …"
    names = [h for h in helpers if not any(o != h and o.startswith(h) for o in helpers)]
    helper_str = ", ".join(names[:8]) if names else "none"
    failed_txt = " wrote FAILED.txt;" if "FAILED.txt" in text and "FAILED.txt" in claim else ""
    return (f"helpers: {helper_str}; sim calls {sim}; images inspected {images};"
            f"{failed_txt} final claim: \"{claim}\"")


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

_CHANGED_RE = re.compile(r"^##\s*Changed\s*$", re.M)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def feature_from_journal(journal: Path, *, max_len: int = 70) -> str | None:
    """The feature named under '## Changed' in a generator journal."""
    try:
        text = journal.read_text(errors="ignore")
    except OSError:
        return None
    m = _CHANGED_RE.search(text)
    if not m:
        return None
    body = text[m.end():]
    nxt = re.search(r"^##\s", body, re.M)
    body = body[: nxt.start()] if nxt else body
    body = body.strip()
    if not body:
        return None
    b = _BOLD_RE.search(body)
    name = b.group(1) if b else body.splitlines()[0]
    # journals often open with "ONE feature:" / "Added ONE feature," boilerplate
    name = re.sub(r"^(?:added|removed|modified)?\s*one feature[:,]?\s*(?:the\s+)?", "", name, flags=re.I)
    name = re.sub(r"\s+", " ", name).strip(" .:-")
    if len(name) > max_len:
        name = name[:max_len].rstrip() + " …"
    return name


def _pass_count(fb: dict[str, Any]) -> tuple[int, int]:
    runs = fb.get("runs") or []
    return sum(r.get("oracle_verdict") == "PASS" for r in runs), len(runs)


def feature_ledger(feedback_history: list[dict[str, Any]], run_dir: Path | None) -> list[dict[str, Any]]:
    """One row per completed round: the feature written in that round and what
    the NEXT round's evaluation and gate made of it."""
    rows: list[dict[str, Any]] = []
    best = -1.0
    best_pass: int | None = None     # train passes of the incumbent checkpoint
    for i, fb in enumerate(feedback_history):
        n_pass, n = _pass_count(fb)
        fitness = fb.get("selection_fitness")
        outcome: str
        if i == 0:
            outcome = "seed"
        else:
            prev = feedback_history[i - 1]
            if prev.get("generator_output_rejected") and not prev.get("generator_changed"):
                outcome = "rejected by acceptance check"
            elif fb.get("skill_rolled_back"):
                outcome = "rolled back"
            elif fitness is not None and fitness > best + 1e-9:
                outcome = "new best"
            else:
                outcome = "kept"
        # Δ is measured against the checkpoint the proposal was forked from,
        # which is what the regression gate compares against.
        delta = None if best_pass is None else n_pass - best_pass
        if fitness is not None and fitness > best:
            best, best_pass = fitness, n_pass
        feature = None
        if run_dir is not None and i > 0:
            feature = feature_from_journal(Path(run_dir) / f"round_{i - 1}" / "generator_journal.md")
        gate = fb.get("gate") or {}
        rows.append({
            "round": fb.get("round", i),
            "feature": feature or ("hand-written seed" if i == 0 else "(journal missing)"),
            "written_in_round": None if i == 0 else fb.get("round", i) - 1,
            "train_pass": n_pass, "n": n,
            "delta": delta,
            # Gate split (val): scalar summary only, never per-task results.
            "val_pass": gate.get("pass"), "val_n": gate.get("n"),
            "val_delta": (None if gate.get("pass") is None or gate.get("checkpoint_pass") is None
                          else gate["pass"] - gate["checkpoint_pass"]),
            "val_split": gate.get("split"),
            "outcome": outcome,
        })
    return rows


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------

def _mode_counts(
    runs: list[dict[str, Any]], history: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in MODE_ORDER}
    for r in runs:
        mode, sev = classify_run(r)
        by_mode.setdefault(mode, []).append({
            **r, "_mode": mode, "_severity": sev, "_stability": stability(r),
            "_persist": persistence(r.get("tag", ""), mode, history) if history else 0,
        })
    return by_mode


def _fmt_delta(d: int | None) -> str:
    if d is None:
        return "—"
    return f"{d:+d}" if d else "0"


def _severity_str(mode: str, sev: float) -> str:
    if mode in ("pose", "motion", "near_miss"):
        return f"error {sev:.1f}× tol"
    if mode == "scale":
        return f"size off ×{math.exp(sev):.2f}"
    if mode == "axis":
        return f"axis off {sev:.0f}°"
    if mode == "pivot":
        return f"pivot off {sev:.1f} cm"
    return ""


def _flips(cur: dict[str, Any], prev: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not prev:
        return [], []
    prev_runs = {r.get("tag"): r for r in prev.get("runs") or []}
    gained, lost = [], []
    for r in cur.get("runs") or []:
        p = prev_runs.get(r.get("tag"))
        if not p:
            continue
        a, b = p.get("oracle_verdict"), r.get("oracle_verdict")
        if a != "PASS" and b == "PASS":
            fixed = sorted(set(p.get("failed_oracle_checks") or []))
            gained.append(f"{r['tag']} ({', '.join(fixed) or a})")
        elif a == "PASS" and b != "PASS":
            broke = sorted(set(r.get("failed_oracle_checks") or []))
            lost.append(f"{r['tag']} ({', '.join(broke) or b})")
    return gained, lost


def build_digest(
    feedback_history: list[dict[str, Any]],
    *,
    run_dir: Path | None = None,
    full_list_path: Path | None = None,
    n_representatives: int = 3,
    read_logs: bool = True,
) -> str:
    """Render the digest for the LAST entry of `feedback_history`."""
    if not feedback_history:
        return "(no evaluation yet)"
    cur = feedback_history[-1]
    prev = feedback_history[-2] if len(feedback_history) >= 2 else None
    runs = cur.get("runs") or []
    reps = max((r.get("executor_reps") or 1 for r in runs), default=1)
    by_mode = _mode_counts(runs, feedback_history)
    prev_by_mode = _mode_counts(prev.get("runs") or [], feedback_history[:-1]) if prev else None
    n = len(runs)
    out: list[str] = []

    # 1. modes
    out.append(f"#### 1. Failure modes  ({n} tasks, {reps} session{'s' if reps > 1 else ''} per task; "
               "persistent = the same task failed in the same mode in the previous round too)")
    out.append("| mode | tasks | persistent | Δ vs prev | meaning |")
    out.append("|---|---|---|---|---|")
    for mode in MODE_ORDER:
        items = by_mode.get(mode, [])
        if not items and mode != "pass":
            continue
        persistent = sum(1 for r in items if r["_persist"] > 0)
        persistent_str = "—" if mode == "pass" or prev is None else str(persistent)
        delta = None
        if prev_by_mode is not None:
            delta = len(items) - len(prev_by_mode.get(mode, []))
        out.append(f"| {mode} | {len(items)} | {persistent_str} | {_fmt_delta(delta)} | {MODE_HELP[mode]} |")
    out.append("Modes are exclusive and ordered: a task is counted under the first mode that explains it "
               "(a wrongly scaled link is not also counted as mis-posed). Fix the largest persistent mode "
               "that the Skill's workflow can address; near misses move by tolerance luck, not by rules.")

    # 2. representatives
    out.append("")
    out.append(f"#### 2. Representative tasks  (per mode: the {n_representatives} clearest cases, "
               "most persistent first, then largest error)")
    for mode in MODE_ORDER:
        if mode == "pass":
            continue
        items = by_mode.get(mode, [])
        if not items:
            continue
        ranked = sorted(items, key=lambda r: (r["_stability"] == "unstable", -r["_persist"], -r["_severity"]))
        out.append(f"- **{mode}** ({len(items)} tasks)")
        for r in ranked[:n_representatives]:
            sev = _severity_str(mode, r["_severity"])
            if r["_persist"]:
                stab = f", same mode for {r['_persist'] + 1} rounds"
            else:
                stab = ", new this round" if prev is not None else ""
            if r["_stability"] == "unstable":
                stab += ", unstable across sessions"
            head = f"  - {r.get('tag')}" + (f"  {sev}" if sev else "") + stab
            if r.get("error"):
                head += f"  error: {str(r['error'])[:160]}"
            out.append(head)
            for name, detail in sorted((r.get("failed_check_details") or {}).items()):
                out.append(f"      {name}: {_cap(detail, DETAIL_CAP)}")
            if read_logs and r.get("output_dir"):
                log = Path(r["output_dir"]) / "executor.log"
                out.append(f"      executor: {summarize_executor_log(log)}")
            out.append(f"      output_dir: {r.get('output_dir', '')}")
    out.append("When the executor's final claim says a check passed and the oracle says the same thing failed, "
               "the Skill's own check is blind to that error: that is the most actionable evidence here.")

    # 3. flips
    out.append("")
    gained, lost = _flips(cur, prev)
    if prev is None:
        out.append("#### 3. Flips since the previous round\n(first evaluated round)")
    else:
        feat = None
        if run_dir is not None:
            feat = feature_from_journal(Path(run_dir) / f"round_{prev.get('round', len(feedback_history) - 2)}" / "generator_journal.md")
        label = f" (edit under test: {feat})" if feat else ""
        out.append(f"#### 3. Flips since the previous round{label}")
        out.append(f"FAIL→PASS ({len(gained)}): " + (", ".join(gained) if gained else "none"))
        out.append(f"PASS→FAIL ({len(lost)}): " + (", ".join(lost) if lost else "none"))
        if reps <= 1:
            out.append("One session per task: a single flip is within sampling noise; a pattern across several tasks is not.")

    # 4. ledger
    out.append("")
    ledger = feature_ledger(feedback_history, run_dir)
    has_gate = any(r["val_pass"] is not None for r in ledger)
    gate_name = next((r["val_split"] for r in ledger if r["val_split"]), "val")
    gate_mode = (cur.get("gate") or {}).get("mode")
    out.append("#### 4. Feature ledger  (each row: the feature written in a round, evaluated by the next round's batch)")
    if has_gate:
        out.append(f"| evaluated in | feature | train pass | Δ vs checkpoint | {gate_name} pass | {gate_name} Δ vs checkpoint | gate |")
        out.append("|---|---|---|---|---|---|---|")
    else:
        out.append("| evaluated in | feature | train pass | Δ vs checkpoint | gate |")
        out.append("|---|---|---|---|---|")
    for row in ledger:
        cells = [f"R{row['round']}", row["feature"], f"{row['train_pass']}/{row['n']}", _fmt_delta(row["delta"])]
        if has_gate:
            vp = "—" if row["val_pass"] is None else f"{row['val_pass']}/{row['val_n']}"
            cells += [vp, _fmt_delta(row["val_delta"])]
        cells.append(row["outcome"])
        out.append("| " + " | ".join(cells) + " |")
    if has_gate and gate_mode == "select":
        out.append(f"Checkpoints and rollbacks are decided on the {gate_name} split, which you never learn from: "
                   f"a feature that raises train but lowers {gate_name} fits the training objects, not the task. "
                   f"You see only these counts of the {gate_name} split; its objects and their results are not available to you.")
    elif has_gate:
        out.append(f"The {gate_name} split is evaluated every round for the record; selection uses the train fitness. "
                   f"You see only these counts of the {gate_name} split; its objects and their results are not available to you.")
    out.append("Do not re-propose a rolled-back feature without new evidence; a feature that was kept "
               "but gained nothing is a candidate for removal.")

    # 5. full list
    out.append("")
    if full_list_path is not None:
        out.append(f"#### 5. Full per-task list: `{full_list_path}`  (Read it for a task not shown above)")
    return "\n".join(out)
