"""Failure clusters (EVO_FEEDBACK_FORMAT=cluster) and the cluster-gated try_skill."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evo.engine.failure_clusters import (
    build_cluster_report, cluster_runs, failure_items, select_trial_tasks, persistence_counts,
)


def _run(tag, verdict="FAIL", failed=(), details=None, joints=None, out="/tmp/x", error=None):
    r = {"tag": tag, "output_dir": out, "oracle_verdict": verdict,
         "failed_oracle_checks": list(failed), "failed_check_details": details or {},
         "joint_reports": joints or {}}
    if error:
        r["error"] = error
    return r


def _joint(**kw):
    base = {"matched": True, "pred_joint": "j", "pred_type": "REVOLUTE", "type_ok": True, "axis_ok": True,
            "scale_ok": True, "pivot_ok": True, "motion_ok": True, "limits_ok": None}
    base.update(kw)
    return base


def test_items_name_the_first_failing_dimension_and_bucket_it() -> None:
    run = _run("a", failed=["joint_axes", "kinematic_equivalence"],
               joints={"joint_0": _joint(axis_ok=False, axis_angle_deg=178.0, motion_ok=False, trace_drift_m=0.3),
                       "joint_1": _joint(pred_type="PRISMATIC", scale_ok=False, scale_ratio=0.62)})
    items = failure_items(run)
    assert [(i["mode"], i["jtype"], i["bucket"]) for i in items] == [
        ("axis", "REVOLUTE", "flipped sign (≥150°)"), ("scale", "PRISMATIC", "too small (0.5–0.95×)")]
    assert failure_items(_run("p", "PASS")) == []
    assert failure_items(_run("e", "execution_failed", error="boom"))[0]["mode"] == "execution_failed"
    pose = failure_items(_run("q", failed=["rest_pose_alignment"],
                              details={"rest_pose_alignment": "2 matched link(s) mis-posed at rest by > 0.02m (max=0.100m): link_1=0.1m"}))
    assert pose[0]["mode"] == "pose" and "≥4× tol" in pose[0]["bucket"]


def test_clusters_rank_by_tasks_and_keep_every_member() -> None:
    runs = [
        _run("a", failed=["link_scale"], joints={"j0": _joint(scale_ok=False, scale_ratio=0.8)}),
        _run("b", failed=["link_scale"], joints={"j0": _joint(scale_ok=False, scale_ratio=0.7),
                                                 "j1": _joint(scale_ok=False, scale_ratio=0.9)}),
        _run("c", failed=["joint_axes"], joints={"j0": _joint(axis_ok=False, axis_angle_deg=90.0)}),
        _run("d", "PASS"),
    ]
    clusters = cluster_runs(runs)
    assert [c["id"] for c in clusters] == ["C1", "C2"]
    assert clusters[0]["key"][0] == "scale" and clusters[0]["n_tasks"] == 2 and clusters[0]["n_items"] == 3
    persistence_counts(clusters, runs)  # same round as "previous": everything persistent
    assert clusters[0]["persistent"] == ["a", "b"]
    assert select_trial_tasks(clusters, per_cluster=1) == {"C1": ["a"], "C2": ["c"]}
    text, _ = build_cluster_report([{"round": 0, "runs": runs}], read_logs=False)
    assert "| C1 | scale · REVOLUTE · too small (0.5–0.95×) | 2 | 3 |" in text
    assert "b j1→j: size ratio 0.9" in text and "try_skill(cluster=" in text


def test_edit_ledger_rows_come_from_recorded_trials() -> None:
    hist = [
        {"round": 0, "pass_rate": 0.25, "runs": [_run("a")] * 4,
         "cluster_trials": [{"cluster": "C1", "n_tasks": 3, "pass_before": 0, "pass_after": 2,
                             "improved": True, "edit": "rim radius rule", "kept": True},
                            {"cluster": "C2", "n_tasks": 3, "pass_before": 1, "pass_after": 1,
                             "improved": False, "edit": "flip sign", "kept": False}]},
        {"round": 1, "pass_rate": 0.5, "runs": [_run("a")] * 4, "new_best": True},
    ]
    text, _ = build_cluster_report(hist, read_logs=False)
    assert "| R0 | C1 | rim radius rule | 0/3 → 2/3 improved | yes | 2/4 | new best |" in text
    assert "| R0 | C2 | flip sign | 1/3 → 1/3 not improved | reverted | 2/4 | new best |" in text


def test_cluster_try_skill_scores_and_compares(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import evo.engine.loop as L

    tasks_root = tmp_path / "tasks"; gt_root = tmp_path / "gt"
    for t in ("task_000", "task_001"):
        (tasks_root / t / "0" / "parts").mkdir(parents=True)
        (gt_root / t / "0").mkdir(parents=True)
    config = SimpleNamespace(tasks_root=tasks_root, gt_root=gt_root, executor_model="m", executor_max_turns=5,
                             headless=True, executor_backend="responses")

    async def fake_executor(**kw):
        out = kw["output_dir"]; (out / "furniture.urdf").write_text("<robot/>")
        (out / "result.json").write_text(json.dumps({"total_cost_usd": 0.05}))
        return SimpleNamespace(success=True, error=None, duration=1.0)

    class FakeOrc:
        def __init__(self, passed): self.passed = passed; self.checks = []; self.joints = {}
        def to_dict(self): return {"passed": self.passed}

    async def fake_oracle(**kw):
        return FakeOrc(kw["tag"] == "task_000/0")  # the edit fixes task_000 only

    monkeypatch.setattr(L, "run_executor", fake_executor)
    monkeypatch.setattr(L, "_run_oracle_dispatch", fake_oracle)
    monkeypatch.setattr("evo.engine.agents.check_skill_report", lambda *a, **k: {"ok": True})
    entries = [{"task_id": "task_000", "instance": "0"}, {"task_id": "task_001", "instance": "0"}]
    baseline = {"task_000/0": {"oracle_verdict": "FAIL", "failed_oracle_checks": ["link_scale"]},
                "task_001/0": {"oracle_verdict": "FAIL", "failed_oracle_checks": ["link_scale"]}}
    tool, impl, spend = L._make_try_skill_tool(
        config=config, round_dir=tmp_path / "round_0", skill_dir=tmp_path / "skill", trial_task=entries[0],
        ports=[1, 2], executor_effort=None, subagent_model=None, subagent_effort=None,
        cluster_tasks={"C1": entries}, baseline=baseline, oracle_mode="objective",
    )
    rep = asyncio.run(impl("C1"))
    assert rep["ok"] and rep["improved"] and (rep["pass_before"], rep["pass_after"]) == (0, 1)
    assert rep["tasks"][0]["before"]["failed_checks"] == ["link_scale"]
    assert spend["trials"][0]["cluster"] == "C1" and spend["cost_usd"] == pytest.approx(0.1)
    bad = asyncio.run(impl("C9"))
    assert not bad["ok"] and "unknown cluster" in bad["error"]
