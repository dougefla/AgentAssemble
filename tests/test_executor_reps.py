"""K independent executor sessions per task: aggregation, cost folding, formatting."""
from __future__ import annotations

import json
from pathlib import Path

from evo.engine.agents import ExecutorResult, _format_joint_report, _format_joint_summary
from evo.engine.loop import (
    EvoConfig, _aggregate_oracle_results, _aggregate_rep_results,
    _executor_session_summary, _task_score, _vote_fraction,
)
from evo.oracle.check_urdf import CheckResult, JointReport, OracleResult


def _session(passed_checks: dict[str, bool], joint_flags: dict[str, bool] | None, passed: bool) -> OracleResult:
    checks = [CheckResult(n, ok, f"{n} detail") for n, ok in passed_checks.items()]
    joints = None
    if joint_flags is not None:
        joints = {"joint_0": JointReport(gt_joint="joint_0", gt_child="link_1", gt_type="REVOLUTE",
                                         matched=True, pred_joint="hinge", pred_type="REVOLUTE", **joint_flags)}
    return OracleResult(passed, checks, joints=joints, cost_usd=0.0)


def test_aggregate_majority_checks_and_mean_joint_scores() -> None:
    a = _session({"axes": True, "limits": True}, dict(type_ok=True, axis_ok=True, limits_ok=True, motion_ok=True), True)
    b = _session({"axes": True, "limits": False}, dict(type_ok=True, axis_ok=True, limits_ok=False, motion_ok=False), False)
    agg = _aggregate_oracle_results([a, b, None])   # third session produced nothing
    assert agg.reps == 3
    assert agg.passed is False                        # 1 of 3 verdicts passed
    by = {c.name: c for c in agg.checks}
    assert by["axes"].passed is True and by["axes"].detail.startswith("[2/3 votes PASS]")
    assert by["limits"].passed is False and by["limits"].detail.startswith("[1/3 votes PASS]")
    assert _vote_fraction(by["axes"]) == 2 / 3 and _vote_fraction(by["limits"]) == 1 / 3
    assert abs(_task_score(agg) - 0.5) < 1e-9        # mean vote fraction over scored checks
    j = agg.joints["joint_0"]
    assert j.reps == 3
    assert abs(j.score() - (1.0 + 0.5 + 0.0) / 3) < 1e-9
    assert abs(j.score_spread - 1.0) < 1e-9
    assert j.dim_pass_frac["limits_ok"] == round(1 / 3, 3) and j.dim_pass_frac["type_ok"] == round(2 / 3, 3)
    assert j.type_ok is True and j.limits_ok is False   # majority flags
    assert j.to_dict(redact_gt=True)["reps"] == 3


def test_aggregate_all_failed_sessions_is_a_zero_result() -> None:
    agg = _aggregate_oracle_results([None, None])
    assert agg.reps == 2 and agg.passed is False and agg.score == 0.0
    assert agg.checks[0].detail.startswith("[0/2 votes PASS]")


def test_rep_results_fold_into_task_dir_and_private_copy(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"; out0 = run_dir / "round_0" / "outputs" / "task_00" / "0"
    out0.mkdir(parents=True); (out0 / "rep_1").mkdir()
    (out0 / "oracle.json").write_text("{}")
    er0 = ExecutorResult("task_00", "0", out0, success=False, error="No URDF produced", duration=10.0)
    er1 = ExecutorResult("task_00", "0", out0 / "rep_1", success=True, duration=20.0)
    o1 = _session({"axes": True}, dict(type_ok=True, axis_ok=True, limits_ok=True, motion_ok=True), True)
    tid, inst, er, orc, out, _ = _aggregate_rep_results(
        [("task_00", "0", er0, None, out0, tmp_path), ("task_00", "0", er1, o1, out0 / "rep_1", tmp_path)],
        run_dir=run_dir, iteration=0, output_stage="outputs",
    )
    assert er.success is True and er.error is None and er.duration == 30.0 and out == out0
    assert orc.reps == 2 and orc.passed is False        # 1/2 is not a strict majority
    assert (out0 / "oracle.rep0.json").exists()
    agg = json.loads((out0 / "oracle.json").read_text())
    assert agg["reps"] == 2 and agg["rep_scores"] == [None, 1.0]
    assert "gt_type" not in agg["joints"]["joint_0"]     # sanitized for the generator
    private = run_dir.parent / f"{run_dir.name}_private" / "round_0" / "outputs" / "task_00_0.oracle.json"
    assert private.exists() and json.loads(private.read_text())["joints"]["joint_0"]["gt_type"] == "REVOLUTE"


def test_session_summary_sums_rep_dirs(tmp_path: Path) -> None:
    out = tmp_path / "task"; (out / "rep_1").mkdir(parents=True); (out / "rep_2").mkdir()
    for d, cost, calls in ((out, 0.10, 10), (out / "rep_1", 0.05, 5), (out / "rep_2", 0.07, 7)):
        (d / "result.json").write_text(json.dumps({"parent_session": {
            "cost_usd": cost, "n_api_calls": calls, "output": 100, "cost_priced": True, "cost_model": "m"}}))
    sess = _executor_session_summary(out, out / "executor.log")
    assert abs(sess["cost_usd"] - 0.22) < 1e-9 and sess["n_api_calls"] == 22 and sess["output"] == 300
    assert sess["cost_priced"] is True and sess["cost_model"] == "m"


def test_config_default_and_generator_formatting() -> None:
    assert EvoConfig().executor_reps == 1
    rep = {"matched": True, "pred_joint": "h", "pred_type": "REVOLUTE", "type_ok": True, "axis_ok": True,
           "limits_ok": False, "motion_ok": False, "score": 0.5, "reps": 3, "score_spread": 1.0}
    line = _format_joint_report(rep)
    assert "score 0.50 (3 reps, spread 1.00) UNSTABLE" in line
    summary = _format_joint_summary([{"joint_reports": {"j": rep}}])
    assert "3 independent executor sessions" in summary and "UNSTABLE" in summary


def test_infra_dead_reps_are_dropped_from_the_vote(tmp_path: Path) -> None:
    """A session killed by a gateway outage after its retries must not vote
    against the skill: with one PASS session and two 408-dead ones the task is
    a PASS on 1/1 votes, not a FAIL on 1/3 (run 10 round 3, task_01/12)."""
    run_dir = tmp_path / "run"; out0 = run_dir / "round_2" / "ab" / "task_01" / "12"
    out0.mkdir(parents=True); (out0 / "rep_1").mkdir(); (out0 / "rep_2").mkdir()
    dead = "Error code: 408 - {'error': {'message': 'stream error: stream disconnected before completion'}}"
    good = _session({"axes": True}, dict(type_ok=True, axis_ok=True, limits_ok=True, motion_ok=True), True)
    er0 = ExecutorResult("task_01", "12", out0, success=True, duration=10.0)
    er1 = ExecutorResult("task_01", "12", out0 / "rep_1", success=False, error=dead, duration=1.0)
    er2 = ExecutorResult("task_01", "12", out0 / "rep_2", success=False, error=dead, duration=1.0)
    tups = [("task_01", "12", er0, good, out0, tmp_path),
            ("task_01", "12", er1, None, out0 / "rep_1", tmp_path),
            ("task_01", "12", er2, None, out0 / "rep_2", tmp_path)]
    _t, _i, er, orc, _o, _n = _aggregate_rep_results(tups, run_dir=run_dir, iteration=2, output_stage="ab")
    assert er.success is True and orc.passed is True and orc.reps == 1
    assert orc.checks[0].detail.startswith("[1/1 votes PASS]")
    agg = json.loads((out0 / "oracle.json").read_text())
    assert agg["reps_requested"] == 3 and agg["reps_infra_dropped"] == [1, 2] and agg["rep_scores"] == [1.0, None, None]
    # An agent-side failure (no URDF) is still a real failing vote.
    er1b = ExecutorResult("task_01", "12", out0 / "rep_1", success=False, error="No URDF produced", duration=1.0)
    tups[1] = ("task_01", "12", er1b, None, out0 / "rep_1", tmp_path)
    _t, _i, er, orc, _o, _n = _aggregate_rep_results(tups, run_dir=run_dir, iteration=2, output_stage="ab")
    assert orc.reps == 2 and orc.passed is False
    # Every session dead of infrastructure: nothing to drop, the task fails 0/K.
    tups[0] = ("task_01", "12", ExecutorResult("task_01", "12", out0, success=False, error=dead, duration=1.0), None, out0, tmp_path)
    tups[1] = ("task_01", "12", er1, None, out0 / "rep_1", tmp_path)
    _t, _i, er, orc, _o, _n = _aggregate_rep_results(tups, run_dir=run_dir, iteration=2, output_stage="ab")
    assert orc.reps == 3 and orc.passed is False and er.success is False


def test_retry_archive_is_rep_aware_and_keeps_running_siblings(tmp_path: Path) -> None:
    """Rep 0's retry must not move rep_1/ (a concurrently running session) into
    the archive, and rep 1's retry must archive under rep_1/ with its own
    private oracle key instead of clobbering rep 0's."""
    from evo.engine.loop import _archive_executor_code_attempt, _private_dir
    run_dir = tmp_path / "run"; out0 = run_dir / "round_0" / "outputs" / "t" / "0"
    (out0 / "rep_1").mkdir(parents=True); (out0 / "meshes").mkdir()
    (out0 / "executor.log").write_text("rep0"); (out0 / "rep_1" / "executor.log").write_text("rep1 running")
    priv = _private_dir(run_dir) / "round_0" / "outputs"; priv.mkdir(parents=True)
    (priv / "t_0.oracle.json").write_text("{}"); (priv / "t_0_rep1.oracle.json").write_text("{}")
    er0 = ExecutorResult("t", "0", out0, success=False, error="NameError: x", duration=1.0)
    arch0 = _archive_executor_code_attempt(run_dir=run_dir, iteration=0, output_stage="outputs",
                                           result=("t", "0", er0, None, out0, tmp_path), retry_number=1, error="NameError: x")
    assert (arch0 / "executor.log").read_text() == "rep0" and (arch0 / "meshes").is_dir()
    assert (out0 / "rep_1" / "executor.log").read_text() == "rep1 running"   # sibling untouched
    assert not (arch0 / "rep_1").exists()
    assert (arch0 / "private_oracle.json").exists() and (priv / "t_0_rep1.oracle.json").exists()
    er1 = ExecutorResult("t", "0", out0 / "rep_1", success=False, error="NameError: y", duration=1.0)
    arch1 = _archive_executor_code_attempt(run_dir=run_dir, iteration=0, output_stage="outputs",
                                           result=("t", "0", er1, None, out0 / "rep_1", tmp_path), retry_number=1, error="NameError: y", rep=1)
    assert arch1.parent.name == "rep_1" and arch1 != arch0 and arch0.exists()
    assert (arch1 / "executor.log").read_text() == "rep1 running" and not (out0 / "rep_1").exists()
    assert (arch1 / "private_oracle.json").exists() and not (priv / "t_0_rep1.oracle.json").exists()
    events = [json.loads(l) for l in (run_dir / "round_0" / "executor_code_retries.jsonl").read_text().splitlines()]
    assert [e["rep"] for e in events] == [0, 1]


def test_executor_code_error_veto_needs_a_majority_of_failed_sessions(tmp_path: Path) -> None:
    """One of three sessions dying on its own code (after its retry) must not
    veto the candidate; two of three, or a lone K=1 session, still does."""
    from evo.engine.loop import _candidate_executor_code_errors
    out = tmp_path / "t" / "0"; out.mkdir(parents=True)
    err = ('SIM ERROR: Traceback (most recent call last): File "<unknown>", line 18 '
           "SyntaxError: closing parenthesis ')' does not match opening parenthesis '['")
    def rep(d: Path, ok: bool) -> None:
        d.mkdir(exist_ok=True)
        (d / "result.json").write_text(json.dumps({"success": ok, "error": None if ok else err}))
        if not ok:
            (d / "executor.log").write_text(f"[12:00:00] [sim] {err}\n")
    rep(out, False); rep(out / "rep_1", True); rep(out / "rep_2", True)
    er = ExecutorResult("t", "0", out, success=True, duration=1.0)          # aggregate: some rep succeeded
    assert _candidate_executor_code_errors([("t", "0", er, None, out, tmp_path)]) == ()
    rep(out / "rep_2", False)
    er2 = ExecutorResult("t", "0", out, success=True, duration=1.0)
    assert len(_candidate_executor_code_errors([("t", "0", er2, None, out, tmp_path)])) == 1
    solo = tmp_path / "s" / "0"; solo.mkdir(parents=True); rep(solo, False)
    er3 = ExecutorResult("s", "0", solo, success=False, error=err, duration=1.0)
    assert len(_candidate_executor_code_errors([("s", "0", er3, None, solo, tmp_path)])) == 1
    solo_ok = tmp_path / "r" / "0"; solo_ok.mkdir(parents=True); rep(solo_ok, True)
    (solo_ok / "executor.log").write_text(f"[12:00:00] [sim] {err}\n[12:00:05] [success] URDF\n")
    er4 = ExecutorResult("r", "0", solo_ok, success=True, duration=1.0)      # recovered: not a veto
    assert _candidate_executor_code_errors([("r", "0", er4, None, solo_ok, tmp_path)]) == ()

