"""--gate-mode paired: continuous fitness, pooled champion, paired acceptance."""
from __future__ import annotations

import json
import random
from pathlib import Path

from evo.engine.paired_gate import (
    PairedGate, paired_stats, read_task_sessions, session_fitness,
)


def _oracle(passed, joints):
    return {"passed": passed, "score": 1.0 if passed else 0.8,
            "checks": {"model_integrity": {"passed": True, "scored": True}},
            "joints": {f"joint_{i}": j for i, j in enumerate(joints)}}


def test_session_fitness_is_graded() -> None:
    ok = {"score": 1.0, "trace_drift_m": 0.01, "pose_tol_m": 0.04}
    near = {"score": 0.75, "trace_drift_m": 0.06, "pose_tol_m": 0.04}     # 1.5 tol
    far = {"score": 0.5, "trace_drift_m": 0.40, "pose_tol_m": 0.04}
    assert session_fitness(None) == 0.0
    assert 0.9 < session_fitness(_oracle(True, [ok])) <= 1.0
    assert 0.2 < session_fitness(_oracle(False, [near])) < 0.3
    assert session_fitness(_oracle(False, [far])) == 0.0
    assert session_fitness(_oracle(False, [near])) > session_fitness(_oracle(False, [far]))
    two = session_fitness(_oracle(False, [ok, far]))
    assert 0.4 < two < 0.5                                              # one of two joints right
    broken = _oracle(False, [ok]); broken["checks"]["valid_tree"] = {"passed": False, "scored": True}
    assert session_fitness(broken) == 0.0


def test_read_task_sessions_reads_reps_and_drops_infra(tmp_path: Path) -> None:
    d = tmp_path / "task_001" / "0"; (d / "rep_1").mkdir(parents=True); (d / "rep_2").mkdir()
    (d / "oracle.json").write_text(json.dumps({"passed": True, "reps": 3, "reps_requested": 3, "reps_infra_dropped": [2]}))
    (d / "oracle.rep0.json").write_text(json.dumps({"passed": True, "tag": 0}))
    (d / "rep_1" / "oracle.json").write_text(json.dumps({"passed": False, "tag": 1}))
    got = read_task_sessions(d)
    assert [g["tag"] for g in got] == [0, 1]
    single = tmp_path / "task_002" / "0"; single.mkdir(parents=True)
    (single / "oracle.json").write_text(json.dumps({"passed": True, "tag": 9}))
    assert read_task_sessions(single)[0]["tag"] == 9
    empty = tmp_path / "task_003" / "0"; empty.mkdir(parents=True)
    assert read_task_sessions(empty) == [None]


def _draw(rng, p, k):
    return [1.0 if rng.random() < p else 0.0 for _ in range(k)]


def test_equal_skills_are_rarely_accepted_and_better_ones_usually() -> None:
    rng = random.Random(7)
    tasks = [f"t{i}" for i in range(20)]
    base = {t: rng.choice([0.05, 0.2, 0.5, 0.8]) for t in tasks}

    def run(delta: float) -> float:
        accepted = 0
        for _ in range(300):
            g = PairedGate(z_accept=1.0)
            g.install("R0", "s0", {t: _draw(rng, base[t], 3) for t in tasks}, round_index=0)
            g.pool_champion({t: _draw(rng, base[t], 3) for t in tasks})
            rec = g.decide("R1", "s1", {t: _draw(rng, min(1.0, base[t] + delta), 3) for t in tasks}, round_index=1)
            accepted += rec["decision"] == "accepted"
        return accepted / 300

    null_rate = run(0.0)
    assert null_rate < 0.25, null_rate
    assert run(0.20) > 0.8


def test_pooling_tightens_the_champion_and_demotion_restores_the_predecessor() -> None:
    tasks = [f"t{i}" for i in range(10)]
    g = PairedGate(z_accept=1.0, z_demote=1.0)
    g.install("R0", "s0", {t: [0.5, 0.5, 0.5] for t in tasks}, round_index=0)
    rec = g.decide("R1", "s1", {t: [0.9, 0.9, 0.7] for t in tasks}, round_index=1)   # a lucky draw
    assert rec["decision"] == "accepted" and g.champion.label == "R1"
    g.pool_champion({t: [0.0, 0.0, 0.1] for t in tasks})                             # re-run says otherwise
    dem = g.maybe_demote(round_index=2)
    assert dem is not None and g.champion.label == "R0"
    assert g.champion.n_sessions() == 30
    st = paired_stats({t: [0.5] for t in tasks}, {t: [0.5] for t in tasks})
    assert st["diff"] == 0 and st["z"] == 0


def test_state_roundtrip(tmp_path: Path) -> None:
    g = PairedGate(z_accept=1.5)
    g.install("R0", "s0", {"a/0": [0.5, 1.0]}, {"a/0": [0, 1]}, round_index=0)
    g.save(tmp_path / "paired_gate.json")
    h = PairedGate.load(tmp_path / "paired_gate.json")
    assert h.champion.label == "R0" and h.champion.scores == {"a/0": [0.5, 1.0]} and h.history[0]["decision"] == "installed"
    assert h.champion.pass_rate() == 0.5
