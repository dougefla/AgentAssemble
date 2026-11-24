"""Sub-evolvers (EVO_SUB_EVOLVERS=K): one proposal per failure cluster, then one integrator.

A single Evolver turn reads a handful of failures in depth and edits for those.
With K > 0 the round instead runs, in parallel and with isolated contexts, one
sub-evolver per failure cluster (largest clusters first). Each one starts from
the same skill, may read every member of its cluster, makes ONE bounded edit
and gates it on the cluster's trial tasks. The loop then *verifies* every
proposal on cluster members the sub-evolver was not given as trial tasks, and
the main Evolver receives only the proposals (diff, journal, development and
verification verdicts) and writes the round's patch from them.

This module holds the pure parts: cluster selection, the notes injected into
the two kinds of turn, and the proposal summary. The loop owns the agents.
"""
from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any

MIN_CLUSTER_TASKS = 4          # dev trial tasks + at least one unseen member to verify on
VERIFY_TASKS = 6               # unseen cluster members a proposal is re-tested on
SUB_TRIAL_REPS = 2             # sessions per task on each side (edited / unedited skill) in dev and verify gates
MIN_VERIFY_TASKS = 3          # fewer unseen cluster members than this cannot verify an edit
VERIFY_MIN_GAIN = 0.05         # mean fitness gain on the verification tasks for "verified" (a screen; the
                               # paired gate on the gate split is the test)
MAX_DIFF_LINES = 220
SKILL_LINE_BUDGET = 400


def sub_evolver_count() -> int:
    try:
        return max(0, int(os.environ.get("EVO_SUB_EVOLVERS", "0")))
    except ValueError:
        return 0


def select_clusters(clusters: list[dict[str, Any]], trial_keys: dict[str, list[str]], k: int,
                    *, cooled: set[str] | None = None) -> list[dict[str, Any]]:
    """Up to ``k`` clusters to hand to sub-evolvers: those with trial tasks,
    at least MIN_CLUSTER_TASKS members and not cooling down, largest first.
    Returns dicts with id, label, key, dev (trial task tags) and verify tags."""
    out = []
    for c in clusters:
        cid = c["id"]
        dev = list(trial_keys.get(cid) or [])
        if not dev or c["n_tasks"] < MIN_CLUSTER_TASKS:
            continue
        key = "|".join(c["key"])
        if cooled and key in cooled:
            continue
        rest = [t for t in c["tasks"] if t not in dev]
        out.append({"id": cid, "label": c["label"], "key": key, "n_tasks": c["n_tasks"],
                    "dev": dev, "verify": rest[:VERIFY_TASKS]})
        if len(out) >= k:
            break
    return out


def is_verified(rep: dict[str, Any] | None) -> bool:
    """A verification trial report counts as verified when it ran, improved and the
    mean fitness gain reaches ``VERIFY_MIN_GAIN``. The gain is recomputed from the
    before/after means when the report carries no ``fitness_diff``."""
    if not rep or not rep.get("ok") or not rep.get("improved"):
        return False
    if int(rep.get("n_tasks") or 0) < MIN_VERIFY_TASKS:
        return False
    diff = rep.get("fitness_diff")
    if diff is None:
        try:
            diff = float(rep["fitness_after"]) - float(rep["fitness_before"])
        except (KeyError, TypeError, ValueError):
            return False
    return float(diff) >= VERIFY_MIN_GAIN - 1e-9


def cooled_clusters(feedback_history: list[dict[str, Any]], *, misses: int = 2, cool: int = 2) -> set[str]:
    """Cluster keys that got a sub-evolver in each of the last ``misses`` rounds
    without a verified proposal: skipped for ``cool`` rounds."""
    recent = [fb.get("sub_evolvers") or [] for fb in feedback_history]
    cooled: set[str] = set()
    for i in range(len(recent)):
        window = recent[max(0, i - misses + 1): i + 1]
        if len(window) < misses:
            continue
        keys = None
        for rnd in window:
            failed = {p["key"] for p in rnd if not p.get("verified")}
            keys = failed if keys is None else keys & failed
        if keys and len(recent) - 1 - i < cool:
            cooled |= keys
    return cooled


def sub_note(c: dict[str, Any]) -> str:
    return (
        f"### You are a SUB-EVOLVER for one failure cluster: {c['id']} — {c['label']} ({c['n_tasks']} tasks)\n"
        "Other sub-evolvers are working on the other clusters in parallel from the same skill, and a separate "
        "integrator turn will merge the proposals. Your job this turn:\n"
        f"1. Read the cluster report for {c['id']}: every member task's numbers, and the trajectories "
        "(executor.log) and outputs of several members, not only the trial tasks. Work out what the executors "
        "did that produced this specific error and why the current skill let them.\n"
        f"2. Make exactly ONE bounded edit that addresses {c['id']} (one rule, one workflow step, or one helper "
        "with its TOOLS.md section and preflight coverage). Prefer a deterministic helper or a measurable check "
        "over more prose; prefer replacing or deleting text over adding it. Do not touch anything unrelated to "
        "this cluster and do not make edits for other clusters.\n"
        f"3. Gate it with try_skill(cluster=\"{c['id']}\"). If the fitness did not improve, read the "
        "after-trajectories, revise once and gate again. If it still does not improve, REVERT the edit: a "
        "turn that changes nothing is a valid result and tells the integrator this cluster has no fix yet.\n"
        "4. If you conclude the failures of this cluster are not the skill's fault (the scoring looks wrong, "
        "or the reference image and the parts cannot determine the answer), change nothing and say so in the "
        "journal under Unresolved, with the task ids and the evidence.\n"
        "After your turn the controller re-tests your edit on other members of the cluster that were NOT your "
        "trial tasks; an edit written for the trial tasks only will fail there and be dropped.\n"
        f"Journal: one Changed line `{c['id']} — <edit in ≤ 12 words> — kept|reverted`."
    )


def skill_diff(base: Path, new: Path, *, max_lines: int = MAX_DIFF_LINES) -> tuple[str, int, int]:
    """(unified diff of the .md/.py files, lines added, lines removed)."""
    def files(root: Path) -> dict[str, list[str]]:
        out = {}
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in {".md", ".py"} and "__pycache__" not in p.parts:
                try:
                    out[str(p.relative_to(root))] = p.read_text().splitlines()
                except (OSError, UnicodeDecodeError):
                    pass
        return out
    a, b = files(base), files(new)
    lines: list[str] = []
    added = removed = 0
    for name in sorted(set(a) | set(b)):
        if a.get(name) == b.get(name):
            continue
        d = list(difflib.unified_diff(a.get(name, []), b.get(name, []), f"a/{name}", f"b/{name}", lineterm="", n=2))
        added += sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
        removed += sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
        lines += d
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more diff lines; read the files under the proposal's skill dir)"]
    return "\n".join(lines), added, removed


def journal_section(journal: Path, name: str) -> str:
    try:
        text = journal.read_text()
    except OSError:
        return ""
    out, on = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            on = line[3:].strip().lower().startswith(name.lower())
            continue
        if on:
            out.append(line)
    return "\n".join(out).strip()


def _gate_line(tag: str, g: dict[str, Any] | None) -> str:
    if not g:
        return f"  {tag}: not run"
    if not g.get("ok", True):
        return f"  {tag}: could not run ({str(g.get('error'))[:120]})"
    k = g.get("sessions_per_task") or 1
    return (f"  {tag}: {g.get('n_tasks')} tasks x {k} session(s) per side, unedited vs edited skill, both fresh: "
            f"fitness {g.get('fitness_before')} → {g.get('fitness_after')}, passing sessions "
            f"{g.get('pass_before')} → {g.get('pass_after')}, tasks better/worse "
            f"{g.get('tasks_better')}/{g.get('tasks_worse')} ({'improved' if g.get('improved') else 'NOT improved'})")


def integrator_note(proposals: list[dict[str, Any]], *, line_budget: int = SKILL_LINE_BUDGET,
                    skill_lines: int | None = None) -> str:
    """The update note of the integrator turn: every proposal with its evidence."""
    head = (
        "### You are the INTEGRATOR this round\n"
        f"{len(proposals)} sub-evolvers each worked on one failure cluster, in parallel, from the skill in your "
        "working directory. Each made at most one edit and gated it on its cluster's trial tasks (development); "
        "the controller then re-tested every edit on other members of the same cluster that the sub-evolver did "
        "not use (verification). Your working directory holds the UNCHANGED skill: nothing has been merged.\n"
        "Write this round's patch from the proposals:\n"
        "1. Merge the proposals whose VERIFICATION improved, largest cluster first. Apply each as written unless "
        "two proposals touch the same passage, in which case reconcile them into one coherent instruction.\n"
        "2. A proposal that improved in development but not in verification was fitted to its trial tasks: drop "
        "it, unless its trajectories show a concrete, general reason and you can fix that in one revision.\n"
        "3. Do not add edits of your own for clusters nobody proposed for, and do not merge a proposal that "
        "changed nothing.\n"
        "4. After merging, gate the merged skill with try_skill(cluster=...) for each merged cluster; if a "
        "merged edit no longer helps in combination, take it out again.\n"
        f"5. Keep SKILL.md within {line_budget} lines"
        + (f" (it has {skill_lines} now)" if skill_lines else "")
        + ": when over, cut the passages the executors demonstrably ignore or that duplicate a helper.\n"
        "Journal Changed section: one line per proposal, `<cluster> — <edit> — merged|dropped (<reason>)`.\n"
        "If no proposal is worth merging, change nothing.\n\nProposals:\n"
    )
    blocks = []
    for p in proposals:
        b = [f"#### {p['cluster']} — {p['label']} ({p['n_tasks']} tasks)  [{p['status']}]",
             f"skill dir of the proposal: {p['skill_dir']}",
             f"journal: {p['journal']}",
             f"edit size: +{p.get('added', 0)} / -{p.get('removed', 0)} lines",
             "gates:", _gate_line("development (its trial tasks)", p.get("dev_gate")),
             _gate_line("verification (unseen cluster members)", p.get("verify_gate"))]
        if p.get("changed"):
            b.append("what it changed (its journal):\n" + (p.get("changed_text") or "(journal has no Changed section)"))
        if p.get("unresolved_text"):
            b.append("what it left unresolved:\n" + p["unresolved_text"][:600])
        if p.get("diff"):
            b.append("diff against the current skill:\n```diff\n" + p["diff"] + "\n```")
        blocks.append("\n".join(b))
    return head + "\n\n".join(blocks)


def proposal_record(p: dict[str, Any]) -> dict[str, Any]:
    """The compact, JSON-safe record kept in round_feedback['sub_evolvers']."""
    def g(d):
        if not d:
            return None
        return {k: d.get(k) for k in ("ok", "improved", "fitness_before", "fitness_after", "fitness_diff",
                                      "pass_before", "pass_after", "n_tasks", "sessions_per_task",
                                      "tasks_better", "tasks_worse", "cost_usd", "error")}
    return {"cluster": p["cluster"], "label": p["label"], "key": p["key"], "n_tasks": p["n_tasks"],
            "status": p["status"], "changed": bool(p.get("changed")), "verified": bool(p.get("verified")),
            "added": p.get("added", 0), "removed": p.get("removed", 0),
            "dev_gate": g(p.get("dev_gate")), "verify_gate": g(p.get("verify_gate")),
            "cost_usd": round(float(p.get("cost_usd") or 0.0), 4)}
