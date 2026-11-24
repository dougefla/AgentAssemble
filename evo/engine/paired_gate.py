"""Paired acceptance gate (``--gate-mode paired``).

Run 32 and the artic207 runs selected checkpoints on one session per task and
a tolerance of 0.05 pass rate, which is about one standard deviation of the
single-session noise: checkpoints were lucky draws, every later proposal
regressed to the mean and was rolled back, and the "best" skill generalised
no better than the seed. This gate replaces that rule:

* **continuous fitness** per session (:func:`session_fitness`) instead of the
  binary all-checks verdict;
* **K sessions per task** on the gate split for the candidate;
* the **champion is re-run every round** on the same tasks and its sessions are
  pooled, so the estimate of the incumbent tightens over time and the upward
  bias of the round it won in is diluted;
* a candidate replaces the champion only if the **paired** per-task difference
  is positive by ``z_accept`` standard errors (and ``min_gain``);
* a champion whose pooled estimate falls ``z_demote`` standard errors below the
  one it replaced is demoted again.

The module is pure bookkeeping: no executor, no ground truth. The loop feeds it
per-task lists of session scores read from the gate split's oracle files.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SHAPING_TAUS = 3.0            # the drift at which the shaped term reaches 0, in pose tolerances


def session_fitness(oracle: dict | None) -> float:
    """Continuous score in [0, 1] of one session's ``oracle.json`` dict.

    Per ground-truth movable joint: half for the joint being fully correct
    (every scored dimension of its report passes), half for how close the
    motion trace is to the ground truth, ``clip(1 - drift / (3 * tol), 0, 1)``
    with ``drift`` the trace drift (the rest drift when no trace resolved).
    The session score is the mean over joints; 0 without a URDF or when a
    structural check fails; the check score when the object has no joints.
    """
    if not oracle:
        return 0.0
    checks = oracle.get("checks") or {}
    for name in ("model_integrity", "valid_tree", "link_count", "link_matching", "execution",
                 "parse_pred", "parse_gt"):
        c = checks.get(name)
        if c is not None and c.get("scored", True) and not c.get("passed"):
            return 0.0
    joints = oracle.get("joints") or {}
    if not joints:
        return 1.0 if oracle.get("passed") else float(oracle.get("score") or 0.0) * 0.5
    total = 0.0
    for rep in joints.values():
        ok = 1.0 if float(rep.get("score") or 0.0) >= 1.0 - 1e-9 else 0.0
        tol = rep.get("pose_tol_m")
        drift = rep.get("trace_drift_m")
        if drift is None:
            drift = rep.get("rest_drift_m")
        if tol and drift is not None:
            shaped = max(0.0, min(1.0, 1.0 - float(drift) / (SHAPING_TAUS * float(tol))))
        else:
            shaped = ok
        total += 0.5 * ok + 0.5 * shaped
    return total / len(joints)


def session_passed(oracle: dict | None) -> bool:
    return bool(oracle and oracle.get("passed"))


def read_task_sessions(task_out_dir: Path) -> list[dict | None]:
    """The per-session oracle dicts of one task output dir (rep 0 and rep_<k>).

    Sessions dropped as infrastructure failures by the K-fold aggregation are
    skipped; a session that ran and produced no oracle is ``None`` (scores 0).
    """
    task_out_dir = Path(task_out_dir)

    def load(p: Path) -> dict | None:
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    agg = load(task_out_dir / "oracle.json")
    rep_dirs = sorted(p for p in task_out_dir.glob("rep_*") if p.is_dir())
    if not rep_dirs and (agg is None or int(agg.get("reps_requested") or agg.get("reps") or 1) <= 1):
        return [agg]
    sessions: list[dict | None] = [load(task_out_dir / "oracle.rep0.json")]
    for d in rep_dirs:
        sessions.append(load(d / "oracle.json"))
    infra = set((agg or {}).get("reps_infra_dropped") or [])
    kept = [s for i, s in enumerate(sessions) if i not in infra]
    return kept or sessions


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def paired_stats(cand: dict[str, list[float]], champ: dict[str, list[float]]) -> dict[str, Any]:
    """Mean per-task difference candidate - champion with its standard error."""
    tasks = sorted(t for t in cand if t in champ and cand[t] and champ[t])
    diffs = [mean(cand[t]) - mean(champ[t]) for t in tasks]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "diff": 0.0, "se": float("inf"), "z": 0.0}
    d = mean(diffs)
    if n > 1:
        var = sum((x - d) ** 2 for x in diffs) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("inf")
    # A floor on the standard error: with few tasks and identical outcomes the
    # sample variance collapses to 0 and any difference would look decisive.
    se = max(se, 0.5 / math.sqrt(max(n, 1)) / math.sqrt(max(1, min(len(cand[t]) for t in tasks))) * 0.2)
    return {"n": n, "diff": d, "se": se, "z": d / se if se > 0 and math.isfinite(se) else 0.0,
            "wins": sum(1 for x in diffs if x > 1e-9), "losses": sum(1 for x in diffs if x < -1e-9)}


@dataclass
class Champion:
    label: str                                  # e.g. "R3": the round whose skill this is
    skill_dir: str                              # snapshot directory
    scores: dict[str, list[float]] = field(default_factory=dict)   # task -> pooled session scores
    passes: dict[str, list[int]] = field(default_factory=dict)

    def pool(self, scores: dict[str, list[float]], passes: dict[str, list[int]] | None = None) -> None:
        for t, xs in scores.items():
            self.scores.setdefault(t, []).extend(xs)
        for t, xs in (passes or {}).items():
            self.passes.setdefault(t, []).extend(xs)

    def fitness(self) -> float:
        return mean([mean(xs) for xs in self.scores.values() if xs])

    def pass_rate(self) -> float:
        return mean([mean([float(v) for v in xs]) for xs in self.passes.values() if xs])

    def n_sessions(self) -> int:
        return sum(len(xs) for xs in self.scores.values())


@dataclass
class PairedGate:
    z_accept: float = 1.0
    z_demote: float = 1.0
    min_gain: float = 0.0
    champions: list[Champion] = field(default_factory=list)      # accepted lineage, last = current
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def champion(self) -> Champion | None:
        return self.champions[-1] if self.champions else None

    def install(self, label: str, skill_dir: str, scores, passes=None, *, round_index: int) -> dict[str, Any]:
        """First skill of a run: it is the champion by definition."""
        c = Champion(label, skill_dir)
        c.pool(scores, passes)
        self.champions.append(c)
        rec = {"round": round_index, "decision": "installed", "candidate": label, "champion": label,
               "candidate_fitness": round(c.fitness(), 4), "candidate_pass_rate": round(c.pass_rate(), 4),
               "champion_fitness": round(c.fitness(), 4), "champion_pass_rate": round(c.pass_rate(), 4),
               "champion_sessions": c.n_sessions()}
        self.history.append(rec)
        return rec

    def pool_champion(self, scores, passes=None) -> None:
        if self.champion is not None:
            self.champion.pool(scores, passes)

    def maybe_demote(self, *, round_index: int) -> dict[str, Any] | None:
        """Demote the champion if its pooled estimate fell clearly below its predecessor's."""
        if len(self.champions) < 2:
            return None
        cur, prev = self.champions[-1], self.champions[-2]
        st = paired_stats(cur.scores, prev.scores)
        if st["z"] <= -abs(self.z_demote):
            self.champions.pop()
            rec = {"round": round_index, "decision": "demoted", "candidate": cur.label, "champion": prev.label,
                   "diff": round(st["diff"], 4), "se": round(st["se"], 4), "z": round(st["z"], 2),
                   "champion_fitness": round(prev.fitness(), 4), "champion_pass_rate": round(prev.pass_rate(), 4),
                   "champion_sessions": prev.n_sessions()}
            self.history.append(rec)
            return rec
        return None

    def decide(self, label: str, skill_dir: str, scores, passes=None, *, round_index: int) -> dict[str, Any]:
        """Compare a candidate with the (already re-pooled) champion; install it if accepted."""
        champ = self.champion
        assert champ is not None, "install() the first skill before decide()"
        cand = Champion(label, skill_dir)
        cand.pool(scores, passes)
        st = paired_stats(cand.scores, champ.scores)
        accept = st["n"] > 0 and st["diff"] > self.min_gain and st["z"] >= self.z_accept
        rec = {"round": round_index, "decision": "accepted" if accept else "rejected",
               "candidate": label, "champion_before": champ.label,
               "candidate_fitness": round(cand.fitness(), 4), "candidate_pass_rate": round(cand.pass_rate(), 4),
               "champion_fitness_before": round(champ.fitness(), 4),
               "champion_pass_rate_before": round(champ.pass_rate(), 4),
               "champion_sessions_before": champ.n_sessions(),
               "diff": round(st["diff"], 4), "se": round(st["se"], 4), "z": round(st["z"], 2),
               "wins": st.get("wins"), "losses": st.get("losses"), "n_tasks": st["n"],
               "z_accept": self.z_accept, "min_gain": self.min_gain}
        if accept:
            self.champions.append(cand)
        cur = self.champion
        rec.update({"champion": cur.label, "champion_fitness": round(cur.fitness(), 4),
                    "champion_pass_rate": round(cur.pass_rate(), 4), "champion_sessions": cur.n_sessions()})
        self.history.append(rec)
        return rec

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"z_accept": self.z_accept, "z_demote": self.z_demote, "min_gain": self.min_gain,
                "champions": [{"label": c.label, "skill_dir": c.skill_dir, "scores": c.scores, "passes": c.passes}
                              for c in self.champions],
                "history": self.history}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, **defaults) -> "PairedGate":
        d = json.loads(Path(path).read_text())
        g = cls(z_accept=defaults.get("z_accept", d.get("z_accept", 1.0)),
                z_demote=defaults.get("z_demote", d.get("z_demote", 1.0)),
                min_gain=defaults.get("min_gain", d.get("min_gain", 0.0)))
        g.champions = [Champion(c["label"], c["skill_dir"], c.get("scores") or {}, c.get("passes") or {})
                       for c in d.get("champions") or []]
        g.history = d.get("history") or []
        return g


def collect_scores(results) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    """Per-task session scores / pass flags from a list of loop result tuples
    ``(task_id, instance, executor_result, oracle, out_dir, in_dir)``."""
    scores: dict[str, list[float]] = {}
    passes: dict[str, list[int]] = {}
    for tid, inst, _er, _orc, out_dir, _in_dir in results:
        sessions = read_task_sessions(Path(out_dir))
        key = f"{tid}/{inst}"
        scores[key] = [session_fitness(s) for s in sessions]
        passes[key] = [1 if session_passed(s) else 0 for s in sessions]
    return scores, passes


def artifact_fitness(art: dict | None) -> float:
    """:func:`session_fitness` of a loop run artifact (the per-task dict the
    Generator feedback is built from), for before/after comparisons."""
    if not art:
        return 0.0
    verdict = art.get("oracle_verdict")
    if verdict not in ("PASS", "FAIL"):
        return 0.0
    checks = {name: {"passed": False, "scored": True} for name in (art.get("failed_oracle_checks") or [])}
    return session_fitness({"passed": verdict == "PASS", "score": art.get("oracle_score") or 0.0,
                            "checks": checks, "joints": art.get("joint_reports") or {}})
