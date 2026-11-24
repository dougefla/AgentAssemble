"""Governance-layer unit tests for the evolution loop.

Covers the mechanisms added after the external review of the first
training runs:
  - graded fitness aggregation (_round_metrics)
  - generator output acceptance gate (_validate_generator_output)
  - GT-value sanitization of oracle dicts (sanitize_detail / sanitize_oracle_dict)
  - isolation audit (_scan_text_for_forbidden / _audit_round_isolation)
  - subagent oracle vote parsing + strict-majority aggregation
  - difficulty-bucket fallback for ArtVIP splits rows
  - best-skill snapshot hygiene (_save_best_skill clears stale files)
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from evo.engine.loop import (
    EvoConfig,
    ExecutorResult,
    GeneratorResult,
    TrainingSystemError,
    _adaptive_train_decision,
    _audit_round_isolation,
    _candidate_ab_decision,
    _candidate_executor_code_errors,
    _candidate_skill_contract_errors,
    _candidate_tool_contract_errors,
    _critical_ab_regressions,
    _entry_difficulty,
    _find_blocking_call_error,
    _is_candidate_tool_contract_error,
    _is_executor_code_error,
    _is_training_system_error,
    _is_skill_contract_error,
    _normalize_workflow_failure,
    _round_metrics,
    _save_best_skill,
    _scan_text_for_forbidden,
    _validate_generator_output,
    evolve_multi_task,
    evolve_single_task,
    run_evolution,
)
from evo.oracle.check_urdf import (
    CheckResult,
    OracleResult,
    sanitize_detail,
    sanitize_oracle_dict,
)
from evo.oracle.subagent import _aggregate_votes, _parse_vote


# ---------------------------------------------------------------------------
# _round_metrics — graded fitness
# ---------------------------------------------------------------------------

def _oracle(passed: bool, n_pass: int, n_total: int) -> OracleResult:
    checks = [
        CheckResult(name=f"c{i}", passed=(i < n_pass)) for i in range(n_total)
    ]
    return OracleResult(passed=passed, checks=checks)


def test_round_metrics_empty():
    assert _round_metrics([]) == (0, 0.0, 0.0)


def test_round_metrics_mixed():
    oracles = [
        _oracle(True, 10, 10),    # PASS, score 1.0
        _oracle(False, 9, 10),    # FAIL but 9/10 checks
        _oracle(False, 3, 10),    # FAIL, 3/10 checks
        None,                     # executor failed → counts 0
    ]
    n_pass, pass_rate, mean_score = _round_metrics(oracles)
    assert n_pass == 1
    assert pass_rate == pytest.approx(0.25)
    assert mean_score == pytest.approx((1.0 + 0.9 + 0.3 + 0.0) / 4)


def test_round_metrics_graded_beats_binary():
    """Two all-FAIL rounds with different check scores must be ordered."""
    worse = [_oracle(False, 2, 10)] * 4
    better = [_oracle(False, 8, 10)] * 4
    assert _round_metrics(worse)[1] == _round_metrics(better)[1] == 0.0
    assert _round_metrics(better)[2] > _round_metrics(worse)[2]


def test_candidate_ab_rejects_paired_score_collapse():
    # Hard floor: a mean per-task drop of 0.5 is rejected before any test.
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.3, baseline_mean_score=0.7,
        candidate_pass_rate=0.0, candidate_mean_score=0.2,
        paired_diffs=(-0.5,) * 5,
    )
    assert not decision.accepted and decision.reason == "score_regression"
    # Consistent small drop: significant under the one-sided paired t.
    diffs = (-0.15, -0.10, -0.20, -0.12, -0.18, -0.14, -0.16, -0.11, -0.13, -0.17)
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.3, baseline_mean_score=0.7,
        candidate_pass_rate=0.2, candidate_mean_score=0.55,
        paired_diffs=diffs,
    )
    assert not decision.accepted and decision.reason == "score_regression"
    assert decision.paired_t is not None and decision.paired_t < -decision.paired_t_critical


def test_candidate_ab_accepts_equal_result_as_non_inferior():
    # Policy change: a neutral edit passes; the best-fitness checkpoint is
    # the backstop, not strict per-round improvement.
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.3, baseline_mean_score=0.7,
        candidate_pass_rate=0.3, candidate_mean_score=0.7,
        paired_diffs=(0.0,) * 10,
    )
    assert decision.accepted and decision.reason == "non_inferior"


def test_candidate_ab_accepts_paired_improvement():
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.3, baseline_mean_score=0.6,
        candidate_pass_rate=0.5, candidate_mean_score=0.7,
        paired_diffs=(0.1,) * 10,
    )
    assert decision.accepted and decision.reason == "score_improved"


def test_candidate_ab_accepts_noisy_dip_within_noise():
    # Mixed-sign diffs, mean slightly negative, nowhere near significance.
    diffs = (-0.25, 0.25, -0.10, 0.10, -0.20, 0.15, 0.0, -0.05, 0.05, -0.10)
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.4, baseline_mean_score=0.6,
        candidate_pass_rate=0.4, candidate_mean_score=0.59,
        paired_diffs=diffs,
    )
    assert decision.accepted and decision.reason == "non_inferior"
    assert decision.paired_t is not None and decision.paired_t > -decision.paired_t_critical


def test_candidate_ab_requires_complete_sample_floor():
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.0,
        baseline_mean_score=0.2,
        candidate_pass_rate=0.0,
        candidate_mean_score=0.8,
        sample_count=3,
        minimum_samples=10,
    )
    assert not decision.accepted
    assert decision.reason == "insufficient_samples"


def test_candidate_ab_rejects_candidate_tool_error_even_if_score_improves():
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.0,
        baseline_mean_score=0.2,
        candidate_pass_rate=0.2,
        candidate_mean_score=0.8,
        candidate_tool_contract_errors=("task_01/0: KeyError",),
    )
    assert not decision.accepted
    assert decision.reason == "candidate_tool_contract_error"


def test_candidate_ab_separately_rejects_exhausted_executor_code_error():
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.0,
        baseline_mean_score=0.2,
        candidate_pass_rate=0.2,
        candidate_mean_score=0.8,
        executor_code_errors=("task_01/0: SyntaxError",),
    )
    assert not decision.accepted
    assert decision.reason == "executor_code_error"


def test_candidate_ab_rejects_critical_metric_regression():
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.0,
        baseline_mean_score=0.2,
        candidate_pass_rate=0.1,
        candidate_mean_score=0.8,
        critical_regressions=("task_01/0:joint_axes",),
    )
    assert not decision.accepted
    assert decision.reason == "critical_metric_regression"


# ---------------------------------------------------------------------------
# Within-round adaptive Train decisions
# ---------------------------------------------------------------------------

def _adaptive_result(
    tmp_path: Path,
    idx: int,
    *,
    success: bool = True,
    error: str | None = None,
    oracle: OracleResult | None = None,
    log_text: str = "",
):
    out = tmp_path / f"sample_{idx}"
    out.mkdir()
    if log_text:
        (out / "executor.log").write_text(log_text)
    executor = ExecutorResult(
        factory="task_01",
        instance=str(idx),
        output_dir=out,
        success=success,
        error=error,
    )
    return "task_01", str(idx), executor, oracle, out, tmp_path


def _failed_check_oracle(name: str) -> OracleResult:
    return OracleResult(
        passed=False,
        checks=[CheckResult(name=name, passed=False, detail="failed")],
    )


def test_adaptive_train_stops_on_one_blocking_api_error(tmp_path):
    result = _adaptive_result(
        tmp_path, 0, success=False,
        error="API Error: HTTP 400 invalid tool schema",
    )
    decision = _adaptive_train_decision([result], cumulative_train=1)
    assert decision.action == "STOP_ADDING"
    assert decision.reason == "training_system_error"
    assert "HTTP 400" in (decision.blocking_error or "")


def test_adaptive_train_stops_on_sim_startup_error(tmp_path):
    result = _adaptive_result(
        tmp_path, 0, success=False,
        error="sim worker socket (port 9876) not ready after 30s",
    )
    decision = _adaptive_train_decision([result], cumulative_train=0)
    assert decision.reason == "training_system_error"


@pytest.mark.parametrize("error", [
    "NameError: name 'SKILL_DIR' is not defined",
    "NameError: OUTPUT_DIR is not defined",
    "KeyError: 'SKILL_DIR'",
    "KeyError: \"INPUT_DIR\"",
    "KeyError: 'OUTPUT_DIR'",
    "Responses provider returned output=null",
    "Error code: 400 context_too_large",
    "Connection error.",
    "sim worker socket (port 9876) not ready after 30s",
    "sim worker exited during startup (exit code 1)",
    "FileNotFoundError: /repo/tools/clear_scene.py",
    "SKILL_DIR is runtime-owned and cannot be overwritten",
])
def test_training_runtime_errors_are_not_skill_feedback(error: str):
    assert _is_training_system_error(error)


@pytest.mark.parametrize("error", [
    "NameError: name 'SKILL_DIR' is not defined",
    "NameError: OUTPUT_DIR is not defined",
    "KeyError: 'SKILL_DIR'",
    "1 validation error for ModelResponse output Input should be a valid list "
    "[input_value=None]",
    "Responses provider returned output=null",
    "Error code: 400 context_too_large",
    "Connection error.",
])
def test_training_runtime_error_aborts_before_heldout(tmp_path, error: str):
    result = _adaptive_result(tmp_path, 0, success=False, error=error)
    assert _find_blocking_call_error(result[2], result[4]) == error
    decision = _adaptive_train_decision([result], cumulative_train=0)
    assert decision.action == "STOP_ADDING"
    assert decision.reason == "training_system_error"


def test_single_task_system_error_skips_oracle_and_generator(
    tmp_path, monkeypatch,
):
    tasks = tmp_path / "dataset" / "tasks"
    gt = tmp_path / "dataset" / "gt"
    skill = tmp_path / "skill"
    (tasks / "task_01" / "0" / "parts").mkdir(parents=True)
    (gt / "task_01" / "0").mkdir(parents=True)
    skill.mkdir()
    (skill / "SKILL.md").write_text("# smoke\n")
    called = {"oracle": False, "generator": False}

    async def fake_executor(**kwargs):
        return ExecutorResult(
            factory="task_01",
            instance="0",
            output_dir=kwargs["output_dir"],
            success=False,
            error="Error code: 400 context_too_large",
        )

    async def forbidden_oracle(**_kwargs):
        called["oracle"] = True
        raise AssertionError("oracle must not start")

    async def forbidden_generator(**_kwargs):
        called["generator"] = True
        raise AssertionError("generator must not start")

    monkeypatch.setattr("evo.engine.loop._install_skill", lambda _path: None)
    monkeypatch.setattr("evo.engine.loop.run_executor", fake_executor)
    monkeypatch.setattr("evo.engine.loop._run_oracle_dispatch", forbidden_oracle)
    monkeypatch.setattr("evo.engine.loop.run_generator", forbidden_generator)
    config = EvoConfig(
        max_rounds=1,
        tasks_root=tasks,
        gt_root=gt,
        skill_dir=skill,
        output_root=tmp_path / "logs",
    )
    with pytest.raises(TrainingSystemError, match="were not started"):
        asyncio.run(evolve_single_task("task_01", "0", config))
    assert called == {"oracle": False, "generator": False}
    failures = list((tmp_path / "logs").rglob("training_system_failure.json"))
    assert len(failures) == 1
    payload = json.loads(failures[0].read_text())
    assert payload["stage"] == "train"
    assert payload["validation_started"] is False


def test_ambiguous_legacy_sim_error_remains_candidate_tool_feedback():
    error = "SIM ERROR: ValueError: joint child link does not exist"
    assert not _is_training_system_error(error)
    assert _is_candidate_tool_contract_error(error)
    assert _is_skill_contract_error(error)


@pytest.mark.parametrize("error", [
    "SIM ERROR: TypeError: oriented_size() missing 1 required positional "
    "argument: 'candidate_id'",
    "SIM ERROR: TypeError: oriented_broad_face() missing 2 required "
    "keyword-only arguments: 'axis' and 'side'",
    "SIM ERROR: ValueError: axis must be one of +x,-x,+y,-y,+z,-z",
])
def test_executor_call_errors_are_retryable(error):
    assert _is_executor_code_error(error)
    assert not _is_candidate_tool_contract_error(error)
    assert not _is_training_system_error(error)


def test_executor_code_error_stops_addition_without_contract_repair(tmp_path):
    error = (
        "SIM ERROR: TypeError: oriented_size() missing 1 required "
        "positional argument: 'candidate_id'"
    )
    result = _adaptive_result(tmp_path, 0, success=False, error=error)
    decision = _adaptive_train_decision([result], cumulative_train=0)
    assert decision.action == "STOP_ADDING"
    assert decision.reason == "executor_code_error"


def test_candidate_helper_trace_is_a_tool_contract_error(tmp_path):
    error = (
        "SIM ERROR: Traceback (most recent call last):\n"
        "  File \"<agent_code>\", line 12, in <module>\n"
        "  File \"<string>\", line 260, in best_pca_poses\n"
        "KeyError: 0"
    )
    assert _is_candidate_tool_contract_error(error)
    assert not _is_executor_code_error(error)
    result = _adaptive_result(tmp_path, 0, success=False, error=error)
    decision = _adaptive_train_decision([result], cumulative_train=0)
    assert decision.reason == "candidate_tool_contract_error"


def test_agent_snippet_syntax_and_inspect_errors_are_executor_code_errors():
    syntax = (
        "SIM ERROR: Traceback (most recent call last):\n"
        "  File \"sim_worker.py\", line 107, in _execute\n"
        "    tree = ast.parse(code, mode=\"exec\")\n"
        "  File \"<unknown>\", line 14\n"
        "SyntaxError: '(' was never closed"
    )
    inspect_error = (
        "SIM ERROR: Traceback (most recent call last):\n"
        "  File \"<agent_code>\", line 12, in <module>\n"
        "  File \"inspect.py\", line 1096, in findsource\n"
        "OSError: could not get source code"
    )
    assert _is_executor_code_error(syntax)
    assert _is_executor_code_error(inspect_error)


@pytest.mark.parametrize("message", [
    "ASSEMBLY_SDK[sim_runtime] render_preview: renderer failed",
    "ASSEMBLY_SDK[io_failure] export_urdf: filesystem operation failed",
    "RpcError: page disconnected",
    "Failed to fetch",
    "Browser has been closed",
    "PermissionError: output is not writable",
])
def test_renderer_sdk_and_environment_errors_are_training_system_errors(message):
    assert _is_training_system_error(message)


def test_expected_sdk_contract_rejection_is_not_a_system_error():
    assert not _is_training_system_error(
        "ASSEMBLY_SDK[unknown_handle] set_part_scale: unknown part handle"
    )


def test_structured_sdk_contract_result_is_not_a_blocking_call(tmp_path):
    result = _adaptive_result(tmp_path, 0, success=True)
    (result[4] / "executor.responses.json").write_text(json.dumps({
        "items": [{
            "type": "function_call_output",
            "output": (
                'RESULT: {"ok": false, "error": '
                '{"code": "unknown_handle", '
                '"operation": "set_part_scale"}}'
            ),
        }]
    }))
    assert _find_blocking_call_error(result[2], result[4]) is None


def test_adaptive_train_detects_recovered_tool_error_in_log(tmp_path):
    result = _adaptive_result(
        tmp_path, 0, success=True,
        log_text="[tool_result] SIM ERROR: socket error: connection reset\n",
    )
    assert _find_blocking_call_error(result[2], result[4]) is not None
    decision = _adaptive_train_decision([result], cumulative_train=1)
    assert decision.reason == "training_system_error"


def test_adaptive_train_ignores_error_words_in_assistant_prose(tmp_path):
    result = _adaptive_result(
        tmp_path, 0, success=True,
        log_text=(
            "[text] If a sim error or API error occurs, stop and repair.\n"
            "[success] URDF: furniture.urdf\n"
        ),
    )
    assert _find_blocking_call_error(result[2], result[4]) is None


def test_adaptive_train_detects_responses_tool_output_error(tmp_path):
    result = _adaptive_result(tmp_path, 0, success=True)
    (result[4] / "executor.responses.json").write_text(json.dumps({
        "items": [
            {
                "type": "mcp_call",
                "name": "mcp__sim__execute_python_code",
                "output": "SIM ERROR: socket error: connection reset",
            }
        ]
    }))
    error = _find_blocking_call_error(result[2], result[4])
    assert error is not None
    assert "connection reset" in error


def test_adaptive_train_ignores_responses_image_base64_keywords(tmp_path):
    result = _adaptive_result(tmp_path, 0, success=True)
    (result[4] / "executor.responses.json").write_text(json.dumps({
        "items": [
            {
                "type": "function_call_output",
                "output": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAQuoTaBB",
                        "detail": "high",
                    }
                ],
            }
        ]
    }))
    assert _find_blocking_call_error(result[2], result[4]) is None


def test_adaptive_train_scans_text_beside_responses_image(tmp_path):
    result = _adaptive_result(tmp_path, 0, success=True)
    (result[4] / "executor.responses.json").write_text(json.dumps({
        "items": [
            {
                "type": "function_call_output",
                "output": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                    },
                    {
                        "type": "input_text",
                        "text": "SIM ERROR: socket error: connection reset",
                    },
                ],
            }
        ]
    }))
    error = _find_blocking_call_error(result[2], result[4])
    assert error == "SIM ERROR: socket error: connection reset"


def test_adaptive_train_preserves_full_blocking_error_signature(tmp_path):
    signature = "API Error: HTTP 503 " + ("upstream-detail-" * 50)
    result = _adaptive_result(
        tmp_path, 0, success=False, error=signature,
    )
    assert _find_blocking_call_error(result[2], result[4]) == signature


def test_adaptive_train_stops_on_three_same_objective_failures(tmp_path):
    results = [
        _adaptive_result(
            tmp_path, i, oracle=_failed_check_oracle("kinematic_equivalence")
        )
        for i in range(3)
    ]
    decision = _adaptive_train_decision(results, cumulative_train=3)
    assert decision.action == "STOP_ADDING"
    assert decision.reason == "objective_failure:kinematic_equivalence"
    assert decision.objective_failures == {"kinematic_equivalence": 3}


def test_adaptive_train_adds_when_failures_are_not_repeated(tmp_path):
    results = [
        _adaptive_result(
            tmp_path, 0, oracle=_failed_check_oracle("joint_axes")
        ),
        _adaptive_result(
            tmp_path, 1, oracle=_failed_check_oracle("kinematic_equivalence")
        ),
    ]
    decision = _adaptive_train_decision(results, cumulative_train=10)
    assert decision.action == "ADD_10"
    assert decision.reason == "failures_not_yet_repeated"


def test_adaptive_train_ignores_non_scored_diagnostics(tmp_path):
    diagnostic = OracleResult(
        passed=True,
        checks=[
            CheckResult(
                name="debug_diagnostic",
                passed=False,
                detail="recorded near miss",
                scored=False,
            )
        ],
    )
    results = [
        _adaptive_result(tmp_path, i, oracle=diagnostic)
        for i in range(3)
    ]
    decision = _adaptive_train_decision(results, cumulative_train=3)
    assert decision.action == "ADD_10"
    assert decision.objective_failures == {}


def test_adaptive_train_stops_on_two_same_workflow_failures(tmp_path):
    results = [
        _adaptive_result(
            tmp_path, i, success=False,
            error=f"No URDF produced for asset {100 + i}",
        )
        for i in range(2)
    ]
    assert _normalize_workflow_failure(results[0][2].error) == (
        _normalize_workflow_failure(results[1][2].error)
    )
    decision = _adaptive_train_decision(results, cumulative_train=2)
    assert decision.action == "STOP_ADDING"
    assert decision.reason.startswith("workflow_failure:")


def test_adaptive_train_stops_at_max_samples():
    decision = _adaptive_train_decision([], cumulative_train=69)
    assert decision.action == "STOP_ADDING"
    assert decision.reason == "max_train_samples"


def test_candidate_contract_error_scan_preserves_task_identity(tmp_path):
    error = (
        "SIM ERROR: Traceback (most recent call last):\n"
        "  File \"<agent_code>\", line 12, in <module>\n"
        "  File \"<string>\", line 260, in best_pca_poses\n"
        "KeyError: 0"
    )
    result = _adaptive_result(tmp_path, 0, success=False, error=error)
    errors = _candidate_tool_contract_errors([result])
    assert len(errors) == 1
    assert errors[0].startswith("task_01/0:")
    assert _candidate_skill_contract_errors([result]) == errors
    assert _candidate_executor_code_errors([result]) == ()


def test_candidate_executor_error_scan_is_disjoint(tmp_path):
    error = (
        "SIM ERROR: File \"<unknown>\", line 14\n"
        "SyntaxError: '(' was never closed"
    )
    result = _adaptive_result(tmp_path, 0, success=False, error=error)
    errors = _candidate_executor_code_errors([result])
    assert len(errors) == 1
    assert errors[0].startswith("task_01/0:")
    assert _candidate_tool_contract_errors([result]) == ()


def _paired_lists(tmp_path, base_flags, cand_flags, check="joint_axes", tag=""):
    b_root = tmp_path / f"b{tag}"; c_root = tmp_path / f"c{tag}"
    b_root.mkdir(); c_root.mkdir()
    baseline = [
        _adaptive_result(b_root, i, oracle=OracleResult(passed=f, checks=[CheckResult(check, f)]))
        for i, f in enumerate(base_flags)
    ]
    candidate = [
        _adaptive_result(c_root, i, oracle=OracleResult(passed=f, checks=[CheckResult(check, f)]))
        for i, f in enumerate(cand_flags)
    ]
    return baseline, candidate


def test_critical_regression_is_aggregate_not_per_pair(tmp_path):
    # 8 baseline passes collapsing to 1 is a real regression...
    baseline, candidate = _paired_lists(tmp_path, [True] * 8, [True] + [False] * 7)
    assert "joint_axes:8->1" in _critical_ab_regressions(baseline, candidate)


def test_critical_regression_tolerates_noise_floor(tmp_path):
    # ...but 8 -> 6 is within max(2, ceil(0.35*8)) = 3 lost passes, which
    # an unchanged skill produces from executor/judge noise alone.
    baseline, candidate = _paired_lists(tmp_path, [True] * 8, [True] * 6 + [False] * 2)
    assert _critical_ab_regressions(baseline, candidate) == ()
    # A single pair flipping never trips it (the old rule's failure mode).
    baseline, candidate = _paired_lists(tmp_path, [True], [False], tag="2")
    assert _critical_ab_regressions(baseline, candidate) == ()


def _fixture_results(tmp_path, side, name="ab_null_control.json"):
    """Real oracle.json data from a recorded replay (see the fixture's note)."""
    import json
    fx = json.loads((Path(__file__).parent / "fixtures" / name).read_text())
    root = tmp_path / side; root.mkdir()
    rows = []
    for idx, (task, checks) in enumerate(sorted(fx[side].items())):
        oracle = OracleResult(
            passed=checks.get("subagent_verdict", {}).get("passed", False),
            checks=[CheckResult(n, c["passed"], c["detail"], scored=c.get("scored", True))
                    for n, c in checks.items()],
        )
        rows.append(_adaptive_result(root, idx, oracle=oracle))
    return rows


def test_null_control_unchanged_skill_is_not_rejected(tmp_path):
    """The decisive regression test for the gate.

    Both sides are the bare v_minimal skill on the same 10 train tasks,
    executed on different days. The old gate reported 4 critical
    regressions here and rejected. A gate that rejects this is rejecting
    noise, and would reject any real candidate with equal probability.
    """
    from evo.engine.loop import _paired_task_scores
    baseline = _fixture_results(tmp_path, "baseline")
    candidate = _fixture_results(tmp_path, "candidate")
    assert _critical_ab_regressions(baseline, candidate) == ()
    paired = _paired_task_scores(baseline, candidate)
    assert len(paired) == 10
    diffs = tuple(c - b for _k, b, c in paired)
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.1, baseline_mean_score=0.475,
        candidate_pass_rate=0.5, candidate_mean_score=0.65,
        sample_count=10, minimum_samples=10, paired_diffs=diffs,
    )
    assert decision.accepted, (decision.reason, diffs)
    # And the mirror image (candidate = the lower draw) must not be rejected
    # either: 1/10 vs 5/10 was sampling noise, not a regression.
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.5, baseline_mean_score=0.65,
        candidate_pass_rate=0.1, candidate_mean_score=0.475,
        sample_count=10, minimum_samples=10,
        paired_diffs=tuple(-d for d in diffs),
    )
    assert decision.accepted, (decision.reason, decision.paired_t, decision.paired_t_critical)


def test_task_score_uses_judge_vote_fractions():
    from evo.engine.loop import _task_score
    oracle = OracleResult(passed=False, checks=[
        CheckResult("subagent_verdict", False, "[1/3 votes PASS] ..."),
        CheckResult("subagent_rest_pose", True, "3/3 votes PASS"),
        CheckResult("subagent_motion_type", False, "0/3 votes PASS"),
        CheckResult("diag", False, "n/a", scored=False),
    ])
    assert abs(_task_score(oracle) - (1/3 + 1 + 0) / 3) < 1e-9


def test_skill_content_identical_compares_files_not_paths(tmp_path):
    from evo.engine.loop import _skill_content_identical
    a = tmp_path / "a"; b = tmp_path / "b"
    for d in (a, b):
        (d / "tools").mkdir(parents=True)
        (d / "SKILL.md").write_text("# s\n"); (d / "tools" / "x.py").write_text("x=1\n")
    assert _skill_content_identical(a, b)
    (b / "SKILL.md").write_text("# changed\n")
    assert not _skill_content_identical(a, b)


def test_training_run_defers_validation_before_multi_task(
    tmp_path, monkeypatch,
):
    config = EvoConfig(
        tasks_root=tmp_path / "tasks",
        gt_root=tmp_path / "gt",
        skill_dir=tmp_path / "skill",
        output_root=tmp_path / "logs",
    )
    discovered = [
        {"task_id": "task_01", "instance": "0"},
        {"task_id": "task_02", "instance": "0"},
    ]
    split_map = {
        ("task_01", "0"): "train",
        ("task_02", "0"): "validation",
    }
    captured = {}

    async def fake_evolve(_config, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("evo.engine.loop.discover_tasks", lambda _c: discovered)
    monkeypatch.setattr(EvoConfig, "load_splits", lambda _c: split_map)
    monkeypatch.setattr("evo.engine.loop.evolve_multi_task", fake_evolve)

    run_evolution(config, split="train,validation")

    assert captured["feedback_split"] == "train"
    assert [
        (e["task_id"], e["instance"]) for e in captured["task_ids"]
    ] == [("task_01", "0")]


def test_no_generator_runs_validation_as_standalone_evaluation(
    tmp_path, monkeypatch,
):
    config = EvoConfig(
        tasks_root=tmp_path / "tasks",
        gt_root=tmp_path / "gt",
        skill_dir=tmp_path / "skill",
        output_root=tmp_path / "logs",
    )
    discovered = [
        {"task_id": "task_01", "instance": "0"},
        {"task_id": "task_02", "instance": "0"},
    ]
    split_map = {
        ("task_01", "0"): "train",
        ("task_02", "0"): "validation",
    }
    captured = {}

    async def fake_evolve(_config, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("evo.engine.loop.discover_tasks", lambda _c: discovered)
    monkeypatch.setattr(EvoConfig, "load_splits", lambda _c: split_map)
    monkeypatch.setattr("evo.engine.loop.evolve_multi_task", fake_evolve)

    run_evolution(config, split="validation", skip_generator=True)

    assert captured["feedback_split"] is None
    assert captured["skip_generator"] is True
    assert [
        (e["task_id"], e["instance"]) for e in captured["task_ids"]
    ] == [("task_02", "0")]


def test_validation_cannot_be_a_training_feedback_split(
    tmp_path, monkeypatch,
):
    config = EvoConfig(
        tasks_root=tmp_path / "tasks",
        gt_root=tmp_path / "gt",
        skill_dir=tmp_path / "skill",
    )
    monkeypatch.setattr(
        "evo.engine.loop.discover_tasks",
        lambda _c: [{"task_id": "task_02", "instance": "0"}],
    )
    monkeypatch.setattr(
        EvoConfig, "load_splits",
        lambda _c: {("task_02", "0"): "validation"},
    )
    with pytest.raises(SystemExit, match="training may only use"):
        run_evolution(config, split="validation")


# ---------------------------------------------------------------------------
# _validate_generator_output — acceptance gate
# ---------------------------------------------------------------------------

def _gen_result(tmp_path: Path, **kw) -> GeneratorResult:
    r = GeneratorResult(skill_dir=tmp_path)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_generator_gate_accepts_complete_turn(tmp_path):
    journal = tmp_path / "generator_journal.md"
    journal.write_text("## Observed\n...")
    r = _gen_result(tmp_path, files_modified=["SKILL.md"], changes_summary="done")
    ok, reason = _validate_generator_output(r, journal)
    assert ok and reason == ""


def test_generator_gate_rejects_no_edits(tmp_path):
    journal = tmp_path / "generator_journal.md"
    journal.write_text("x")
    r = _gen_result(tmp_path, files_modified=[], changes_summary="…")
    ok, reason = _validate_generator_output(r, journal)
    assert not ok and "no skill files" in reason


def test_generator_gate_rejects_missing_journal(tmp_path):
    r = _gen_result(tmp_path, files_modified=["SKILL.md"], changes_summary="…")
    ok, reason = _validate_generator_output(
        r, tmp_path / "generator_journal.md")
    assert not ok and "journal" in reason


def test_generator_gate_rejects_error_turn(tmp_path):
    journal = tmp_path / "generator_journal.md"
    journal.write_text("x")
    r = _gen_result(
        tmp_path, files_modified=["SKILL.md"],
        changes_summary="Generator error: usage limit reached",
    )
    ok, reason = _validate_generator_output(r, journal)
    assert not ok and "usage limit" in reason


def test_generator_gate_rejects_frozen_actions_contract_edit(tmp_path):
    journal = tmp_path / "generator_journal.md"
    journal.write_text("x")
    (tmp_path / "ACTIONS.md").write_text("changed\n")
    r = _gen_result(
        tmp_path,
        files_modified=["SKILL.md", "ACTIONS.md"],
        changes_summary="done",
    )
    ok, reason = _validate_generator_output(
        r,
        journal,
        current_skill_content={"ACTIONS.md": "frozen\n"},
    )
    assert not ok and "frozen SDK contract ACTIONS.md" in reason


def test_generator_gate_accepts_unchanged_actions_contract(tmp_path):
    journal = tmp_path / "generator_journal.md"
    journal.write_text("x")
    (tmp_path / "ACTIONS.md").write_text("frozen\n")
    r = _gen_result(
        tmp_path,
        files_modified=["SKILL.md"],
        changes_summary="done",
    )
    ok, reason = _validate_generator_output(
        r,
        journal,
        current_skill_content={"ACTIONS.md": "frozen\n"},
    )
    assert ok and reason == ""


def test_generator_gate_rejects_new_helper_without_tools_doc(tmp_path):
    journal = tmp_path / "generator_journal.md"
    journal.write_text("x")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("def axis(value):\n    return value\n")
    r = _gen_result(
        tmp_path,
        files_modified=["tools/geometry.py"],
        changes_summary="added helper",
    )
    ok, reason = _validate_generator_output(
        r,
        journal,
        current_skill_content={"SKILL.md": "# baseline\n"},
    )
    assert not ok
    assert "without creating TOOLS.md" in reason


def test_generator_gate_rejects_helper_change_with_stale_tools_doc(tmp_path):
    journal = tmp_path / "generator_journal.md"
    journal.write_text("x")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("def axis(value, mode=None):\n    return value\n")
    (tmp_path / "TOOLS.md").write_text("old interface\n")
    r = _gen_result(
        tmp_path,
        files_modified=["tools/geometry.py"],
        changes_summary="changed helper",
    )
    ok, reason = _validate_generator_output(
        r,
        journal,
        current_skill_content={
            "tools/geometry.py": "def axis(value):\n    return value\n",
            "TOOLS.md": "old interface\n",
        },
    )
    assert not ok
    assert "without updating TOOLS.md" in reason


def test_generator_gate_accepts_helper_and_tools_doc_change(tmp_path):
    journal = tmp_path / "generator_journal.md"
    journal.write_text("x")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "geometry.py").write_text("def axis(value, mode=None):\n    return value\n")
    (tmp_path / "TOOLS.md").write_text("new interface\n")
    r = _gen_result(
        tmp_path,
        files_modified=["tools/geometry.py", "TOOLS.md"],
        changes_summary="changed helper and docs",
    )
    ok, reason = _validate_generator_output(
        r,
        journal,
        current_skill_content={
            "tools/geometry.py": "def axis(value):\n    return value\n",
            "TOOLS.md": "old interface\n",
        },
    )
    assert ok and reason == ""


def test_candidate_ab_replays_exact_train_samples_and_rejects_regression(
    tmp_path, monkeypatch,
):
    tasks = tmp_path / "dataset" / "tasks"
    gt = tmp_path / "dataset" / "gt"
    skill = tmp_path / "skill"
    for tid in ("task_01", "task_02"):
        (tasks / tid / "0" / "parts").mkdir(parents=True)
        (gt / tid / "0").mkdir(parents=True)
    skill.mkdir()
    (skill / "SKILL.md").write_text("# baseline\n")
    (skill / "ACTIONS.md").write_text("# frozen\n")

    calls: list[tuple[str, str, str]] = []

    async def fake_executor(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "executor.log").write_text("[success] done\n")
        stage = output_dir.parents[1].name
        calls.append((stage, kwargs["factory"], kwargs["instance"]))
        return ExecutorResult(
            factory=kwargs["factory"],
            instance=kwargs["instance"],
            output_dir=output_dir,
            success=True,
        )

    async def fake_oracle(**kwargs):
        is_candidate = "ab_candidate_outputs" in kwargs["pred_dir"].parts
        return OracleResult(
            passed=False,
            checks=[CheckResult("objective", not is_candidate)],
        )

    async def fake_generator(**kwargs):
        output_skill_dir = kwargs["output_skill_dir"]
        shutil.copytree(kwargs["current_skill_dir"], output_skill_dir)
        (output_skill_dir / "SKILL.md").write_text("# candidate\n")
        (kwargs["log_file"].parent / "generator_journal.md").write_text(
            "## Observed\nnone\n"
        )
        return GeneratorResult(
            skill_dir=output_skill_dir,
            changes_summary="candidate written",
            files_modified=["SKILL.md"],
        )

    monkeypatch.setattr("evo.engine.loop._install_skill", lambda _path: None)
    monkeypatch.setattr("evo.engine.loop.run_executor", fake_executor)
    monkeypatch.setattr("evo.engine.loop._run_oracle_dispatch", fake_oracle)
    monkeypatch.setattr("evo.engine.loop.run_generator", fake_generator)

    config = EvoConfig(
        max_rounds=1,
        tasks_root=tasks,
        gt_root=gt,
        skill_dir=skill,
        output_root=tmp_path / "logs",
        parallel=2,
        adaptive_train=True,
        train_increment=2,
        max_train_samples=2,
        candidate_ab_test=True,
    )
    entries = [
        {"task_id": "task_01", "instance": "0", "split": "train"},
        {"task_id": "task_02", "instance": "0", "split": "train"},
    ]
    asyncio.run(evolve_multi_task(
        config,
        task_ids=entries,
        feedback_split="train",
        stagger_sec=0,
    ))

    record_path = next(
        (tmp_path / "logs" / "runs").glob(
            "multi_*/round_0/candidate_ab_test.json"
        )
    )
    run_dir = record_path.parents[1]
    record = json.loads(record_path.read_text())
    assert record["action"] == "REJECT_CANDIDATE"
    assert record["baseline_source"] == "round_train_results"
    assert set(record["samples"]) == {"task_01/0", "task_02/0"}

    baseline_calls = {(tid, inst) for stage, tid, inst in calls if stage == "outputs"}
    candidate_calls = {
        (tid, inst)
        for stage, tid, inst in calls
        if stage == "ab_candidate_outputs"
    }
    assert candidate_calls == baseline_calls == {
        ("task_01", "0"),
        ("task_02", "0"),
    }
    state = json.loads((run_dir / "loop_state.json").read_text())
    assert Path(state["current_skill_dir"]).resolve() == skill.resolve()


def test_skill_contract_repair_keeps_peer_trajectories_and_round_budget(
    tmp_path, monkeypatch,
):
    tasks = tmp_path / "dataset" / "tasks"
    gt = tmp_path / "dataset" / "gt"
    skill = tmp_path / "skill"
    for tid in ("task_01", "task_02"):
        (tasks / tid / "0" / "parts").mkdir(parents=True)
        (gt / tid / "0").mkdir(parents=True)
    skill.mkdir()
    (skill / "SKILL.md").write_text("# broken helper\n")
    (skill / "ACTIONS.md").write_text("# frozen\n")

    calls: list[tuple[str, str, str]] = []

    async def fake_executor(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        stage = output_dir.parents[1].name
        skill_text = (kwargs["skill_dir"] / "SKILL.md").read_text()
        calls.append((stage, kwargs["factory"], skill_text.strip()))
        if "broken" in skill_text and kwargs["factory"] == "task_01":
            error = (
                "SIM ERROR: Traceback (most recent call last):\n"
                "  File \"<agent_code>\", line 12, in <module>\n"
                "  File \"<string>\", line 260, in best_pca_poses\n"
                "KeyError: 0"
            )
            (output_dir / "executor.log").write_text(f"[tool_error] {error}\n")
            return ExecutorResult(
                factory=kwargs["factory"],
                instance=kwargs["instance"],
                output_dir=output_dir,
                success=False,
                error=error,
            )
        if kwargs["factory"] == "task_02":
            await asyncio.sleep(0.02)
        (output_dir / "executor.log").write_text("[success] done\n")
        return ExecutorResult(
            factory=kwargs["factory"],
            instance=kwargs["instance"],
            output_dir=output_dir,
            success=True,
        )

    async def fake_oracle(**_kwargs):
        return OracleResult(
            passed=False,
            checks=[CheckResult("joint_axes", True)],
        )

    async def fake_generator(**kwargs):
        output_skill_dir = kwargs["output_skill_dir"]
        shutil.copytree(kwargs["current_skill_dir"], output_skill_dir)
        (output_skill_dir / "SKILL.md").write_text("# fixed helper\n")
        (kwargs["log_file"].parent / "generator_journal.md").write_text(
            "## Contract repair\nfixed helper signature\n"
        )
        return GeneratorResult(
            skill_dir=output_skill_dir,
            changes_summary="fixed helper",
            files_modified=["SKILL.md"],
        )

    monkeypatch.setattr("evo.engine.loop._install_skill", lambda _path: None)
    monkeypatch.setattr("evo.engine.loop.run_executor", fake_executor)
    monkeypatch.setattr("evo.engine.loop._run_oracle_dispatch", fake_oracle)
    monkeypatch.setattr("evo.engine.loop.run_generator", fake_generator)

    config = EvoConfig(
        max_rounds=1,
        tasks_root=tasks,
        gt_root=gt,
        skill_dir=skill,
        output_root=tmp_path / "logs",
        parallel=2,
        adaptive_train=True,
        train_increment=2,
        max_train_samples=2,
        candidate_ab_test=True,
        candidate_ab_min_samples=2,
        max_contract_repair_attempts=3,
    )
    entries = [
        {"task_id": "task_01", "instance": "0", "split": "train"},
        {"task_id": "task_02", "instance": "0", "split": "train"},
    ]
    asyncio.run(evolve_multi_task(
        config,
        task_ids=entries,
        feedback_split="train",
        stagger_sec=0,
    ))

    run_dir = next(
        path.parent
        for path in (tmp_path / "logs" / "runs").glob("multi_*/loop_state.json")
    )
    state = json.loads((run_dir / "loop_state.json").read_text())
    assert state["round_completed"] == 1
    assert state["objective_rounds_completed"] == 1
    assert [item["round_kind"] for item in state["feedback_history"]] == [
        "contract_repair",
        "objective_evolution",
    ]
    # task_02 was already running when task_01 exposed the helper TypeError;
    # its trajectory must survive instead of being cancelled.
    broken_output_calls = [
        call for call in calls
        if call[0] == "outputs" and "broken" in call[2]
    ]
    assert {call[1] for call in broken_output_calls} == {
        "task_01", "task_02",
    }
    repair_ab = json.loads(
        (run_dir / "round_0" / "candidate_ab_test.json").read_text()
    )
    assert repair_ab["action"] == "ACCEPT_CANDIDATE"
    assert repair_ab["sample_count"] == repair_ab["minimum_samples"] == 2


def test_exhausted_executor_code_retry_does_not_consume_contract_budget(
    tmp_path, monkeypatch,
):
    tasks = tmp_path / "dataset" / "tasks"
    gt = tmp_path / "dataset" / "gt"
    skill = tmp_path / "skill"
    (tasks / "task_01" / "0" / "parts").mkdir(parents=True)
    (gt / "task_01" / "0").mkdir(parents=True)
    skill.mkdir()
    (skill / "SKILL.md").write_text("# baseline\n")
    (skill / "ACTIONS.md").write_text("# frozen\n")

    calls: list[str] = []

    async def fake_executor(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        stage = output_dir.parents[1].name
        calls.append(stage)
        if stage == "ab_candidate_outputs":
            error = (
                "SIM ERROR: Traceback (most recent call last):\n"
                "  File \"sim_worker.py\", line 107, in _execute\n"
                "    tree = ast.parse(code, mode=\"exec\")\n"
                "  File \"<unknown>\", line 14\n"
                "SyntaxError: '(' was never closed"
            )
            (output_dir / "executor.log").write_text(
                f"[tool_error] {error}\n"
            )
            return ExecutorResult(
                factory=kwargs["factory"],
                instance=kwargs["instance"],
                output_dir=output_dir,
                success=False,
                error=error,
            )
        (output_dir / "executor.log").write_text("[success] done\n")
        return ExecutorResult(
            factory=kwargs["factory"],
            instance=kwargs["instance"],
            output_dir=output_dir,
            success=True,
        )

    async def fake_oracle(**_kwargs):
        return OracleResult(
            passed=False,
            checks=[CheckResult("joint_axes", True)],
        )

    async def fake_generator(**kwargs):
        output_skill_dir = kwargs["output_skill_dir"]
        shutil.copytree(kwargs["current_skill_dir"], output_skill_dir)
        (output_skill_dir / "SKILL.md").write_text("# candidate\n")
        (kwargs["log_file"].parent / "generator_journal.md").write_text(
            "## Candidate\nupdated\n"
        )
        return GeneratorResult(
            skill_dir=output_skill_dir,
            changes_summary="candidate written",
            files_modified=["SKILL.md"],
        )

    monkeypatch.setattr("evo.engine.loop._install_skill", lambda _path: None)
    monkeypatch.setattr("evo.engine.loop.run_executor", fake_executor)
    monkeypatch.setattr("evo.engine.loop._run_oracle_dispatch", fake_oracle)
    monkeypatch.setattr("evo.engine.loop.run_generator", fake_generator)

    config = EvoConfig(
        max_rounds=1,
        tasks_root=tasks,
        gt_root=gt,
        skill_dir=skill,
        output_root=tmp_path / "logs",
        adaptive_train=True,
        train_increment=1,
        max_train_samples=1,
        candidate_ab_test=True,
        candidate_ab_min_samples=1,
        max_contract_repair_attempts=1,
        executor_code_retries=1,
    )
    asyncio.run(evolve_multi_task(
        config,
        task_ids=[{
            "task_id": "task_01", "instance": "0", "split": "train",
        }],
        feedback_split="train",
        stagger_sec=0,
    ))

    run_dir = next(
        path.parent
        for path in (tmp_path / "logs" / "runs").glob("multi_*/loop_state.json")
    )
    state = json.loads((run_dir / "loop_state.json").read_text())
    assert state["objective_rounds_completed"] == 1
    assert state["consecutive_contract_repairs"] == 0
    assert state["feedback_history"][0]["round_kind"] == "objective_evolution"
    assert calls.count("ab_candidate_outputs") == 2

    ab_record = json.loads(
        (run_dir / "round_0" / "candidate_ab_test.json").read_text()
    )
    assert ab_record["reason"] == "executor_code_error"
    assert ab_record["candidate_tool_contract_errors"] == []
    assert len(ab_record["executor_code_errors"]) == 1
    retry_dir = (
        run_dir / "round_0" / "executor_code_retry_artifacts"
        / "ab_candidate_outputs" / "task_01" / "0" / "attempt_1"
    )
    assert (retry_dir / "executor.log").is_file()
    assert (run_dir / "round_0" / "executor_code_retries.jsonl").is_file()


# ---------------------------------------------------------------------------
# sanitize_detail / sanitize_oracle_dict — GT redaction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detail,must_lose,must_keep", [
    ("task: GT=[-2.7925, 0.0], Pred=[0.0, 1.57]", "-2.7925", "Pred=[0.0, 1.57]"),
    ("cos=0.123 (GT axis=(0, 0, 1), Pred axis=(1, 0, 0))", "(0, 0, 1)",
     "Pred axis=(1, 0, 0)"),
    ("GT={'REVOLUTE': 2, 'PRISMATIC': 1}, Pred={'REVOLUTE': 3}", "'PRISMATIC'",
     "Pred={'REVOLUTE': 3}"),
    ("GT=7, Pred=6, diff=1 (tol=2)", "GT=7", "diff=1 (tol=2)"),
])
def test_sanitize_detail_redacts_gt_keeps_pred(detail, must_lose, must_keep):
    out = sanitize_detail(detail)
    assert must_lose not in out
    assert must_keep in out
    assert "GT=<redacted>" in out


def test_sanitize_detail_leaves_plain_text():
    s = "checked 3 joints, 0 type mismatches"
    assert sanitize_detail(s) == s


def test_sanitize_oracle_dict_roundtrip():
    res = OracleResult(
        passed=False,
        checks=[
            CheckResult("joint_limits", False,
                        "door: GT=[-2.79, 0], Pred=[0, 1.5]"),
            CheckResult("valid_tree", True, "ok"),
        ],
        matching={"gt_link": "pred_link"},
    )
    out = sanitize_oracle_dict(res.to_dict())
    assert "-2.79" not in json.dumps(out)
    assert out["checks"]["joint_limits"]["passed"] is False
    assert out["checks"]["valid_tree"]["detail"] == "ok"
    assert out["score"] == res.to_dict()["score"]
    assert out["matching"] == {"gt_link": "pred_link"}
    # original dict untouched
    assert "-2.79" in res.to_dict()["checks"]["joint_limits"]["detail"]


# ---------------------------------------------------------------------------
# isolation audit
# ---------------------------------------------------------------------------

def test_scan_text_for_forbidden_counts():
    text = "Read /data/gt/task_01/scene.urdf then /data/gt/task_01 again"
    hits = _scan_text_for_forbidden(text, ["/data/gt/", "/private/"])
    assert hits == {"/data/gt/": 2}


def test_audit_flags_gt_access_and_writes_log(tmp_path):
    run_dir = tmp_path / "run"
    round_dir = run_dir / "round_0" / "outputs" / "task_01" / "0"
    round_dir.mkdir(parents=True)
    (round_dir / "executor.session.jsonl").write_text(
        json.dumps({"tool": "Read", "path": "/data/gt/task_01/scene.urdf"})
    )
    hits = _audit_round_isolation(run_dir, 0, forbidden=["/data/gt/"])
    assert len(hits) == 1 and "executor.session.jsonl" in hits[0]
    assert (run_dir / "isolation_audit.log").exists()


def test_audit_clean_round_is_silent(tmp_path):
    run_dir = tmp_path / "run"
    d = run_dir / "round_0"
    d.mkdir(parents=True)
    (d / "generator.session.jsonl").write_text(
        json.dumps({"tool": "Read", "path": "/data/tasks/task_01/parts/a.obj"})
    )
    hits = _audit_round_isolation(run_dir, 0, forbidden=["/data/gt/"])
    assert hits == []
    assert not (run_dir / "isolation_audit.log").exists()


# ---------------------------------------------------------------------------
# subagent oracle: vote parsing + strict-majority aggregation
# ---------------------------------------------------------------------------

def test_parse_vote_full_schema():
    text = (
        'reasoning...\n{"rest_pose_match":"PASS","motion_type_match":"PASS",'
        '"motion_direction_match":"FAIL","verdict":"FAIL","summary":"door swings in"}'
    )
    v = _parse_vote(text)
    assert v["verdict"] == "FAIL"
    assert v["rest_pose_match"] == "PASS"
    assert v["motion_direction_match"] == "FAIL"
    assert v["summary"] == "door swings in"


def test_parse_vote_garbage_is_conservative():
    assert _parse_vote("no json here")["verdict"] == "?"
    assert _parse_vote('{"verdict": broken')["verdict"] == "?"


def test_aggregate_votes_strict_majority():
    votes = [
        {"verdict": "PASS", "summary": "ok", "rest_pose_match": "PASS"},
        {"verdict": "PASS", "summary": "ok", "rest_pose_match": "PASS"},
        {"verdict": "FAIL", "summary": "bad", "rest_pose_match": "FAIL"},
    ]
    passed, checks = _aggregate_votes(votes)
    assert passed is True
    by_name = {c.name: c for c in checks}
    assert by_name["subagent_verdict"].passed is True
    assert "2/3" in by_name["subagent_verdict"].detail
    assert by_name["subagent_rest_pose"].passed is True


def test_aggregate_votes_tie_or_minority_fails():
    # 1/2 PASS is not a strict majority.
    passed, _ = _aggregate_votes(
        [{"verdict": "PASS", "summary": "a"}, {"verdict": "FAIL", "summary": "b"}]
    )
    assert passed is False


def test_aggregate_votes_parse_failures_count_against():
    votes = [
        {"verdict": "PASS", "summary": "ok"},
        {"verdict": "?", "summary": "judge crashed"},
        {"verdict": "?", "summary": "judge crashed"},
    ]
    passed, checks = _aggregate_votes(votes)
    assert passed is False
    assert "1/3" in checks[0].detail


def test_aggregate_votes_empty_fails():
    passed, checks = _aggregate_votes([])
    assert passed is False
    assert checks[0].name == "subagent_verdict"


def test_aggregate_votes_sub_checks_only_when_reported():
    votes = [{"verdict": "PASS", "summary": "old-style judge, no sub-keys"}]
    _, checks = _aggregate_votes(votes)
    assert [c.name for c in checks] == ["subagent_verdict"]


# ---------------------------------------------------------------------------
# difficulty fallback + best-skill snapshot hygiene
# ---------------------------------------------------------------------------

def test_entry_difficulty_explicit_field_wins():
    assert _entry_difficulty({"difficulty": "hard", "n_joints": 1}) == "hard"


@pytest.mark.parametrize("n_joints,expected", [
    (0, "easy"), (1, "easy"), (2, "medium"), (3, "medium"), (4, "hard"),
])
def test_entry_difficulty_derived_from_n_joints(n_joints, expected):
    assert _entry_difficulty({"n_joints": n_joints}) == expected


def test_entry_difficulty_default_medium():
    assert _entry_difficulty({}) == "medium"


def test_save_best_skill_clears_stale_files(tmp_path):
    skill_v1 = tmp_path / "v1"
    skill_v1.mkdir()
    (skill_v1 / "SKILL.md").write_text("v1")
    (skill_v1 / "obsolete_tool.py").write_text("x")
    dest = tmp_path / "best"
    _save_best_skill(skill_v1, dest)

    skill_v2 = tmp_path / "v2"
    skill_v2.mkdir()
    (skill_v2 / "SKILL.md").write_text("v2")
    _save_best_skill(skill_v2, dest)

    assert (dest / "SKILL.md").read_text() == "v2"
    assert not (dest / "obsolete_tool.py").exists()


# --- the round must be fully recorded before the target-reached break ------

def _multi_task_source() -> str:
    import inspect
    from evo.engine import loop

    return inspect.getsource(loop.evolve_multi_task)


def test_target_reached_break_comes_after_the_round_is_recorded() -> None:
    """A round that hits the target still has to leave a record behind.

    The break used to sit immediately after the pass-rate check, so
    `round_<n>/` held nothing but `outputs/`: no round_feedback, no
    all_artifacts, no loop-state checkpoint. The experiment report is built
    from those files, so a run that passed 100% rendered a report claiming
    0% with zero samples.
    """
    src = _multi_task_source()
    target = src.index("target_reached = fb_pass_rate")
    record = src.index("feedback_history.append(round_feedback)")
    artifacts = src.index('"all_artifacts.json"')
    checkpoint = src.index("_save_loop_state(")
    brk = src.index("if target_reached:\n            break")
    assert target < record < brk, "round_feedback must be written before the break"
    assert target < artifacts < brk, "all_artifacts must be written before the break"
    assert target < checkpoint < brk, "loop state must be checkpointed before the break"


def test_best_skill_is_checkpointed_before_the_target_break() -> None:
    """Reaching the target must still snapshot the skill that got there."""
    src = _multi_task_source()
    save = src.index("_save_best_skill(current_skill, best_skill_dir)")
    target = src.index("target_reached = fb_pass_rate")
    assert save < target


def test_round_spend_is_accounted_before_the_target_break() -> None:
    """A round that reaches the target still cost money."""
    src = _multi_task_source()
    spend = src.index("_round_spend(results)")
    target = src.index("target_reached = fb_pass_rate")
    assert spend < target


# --- what the Generator is shown of a failed check ---------------------------

def test_generator_detail_strips_billing_suffix_not_reasoning() -> None:
    from evo.engine.loop import _generator_facing_detail

    raw = ("[0/3 votes PASS] Prediction provides only one handle swivel, leaving "
           "the spout unarticulated. | $0.3103 / 56s / 3 votes / responses:gpt-6.1-sol")
    out = _generator_facing_detail(raw)
    assert out.endswith("leaving the spout unarticulated.")
    assert "$" not in out and "votes /" not in out
    # Sub-check tallies carry no suffix and must survive untouched.
    assert _generator_facing_detail("0/3 votes PASS") == "0/3 votes PASS"


def test_generator_detail_redacts_gt_and_caps_only_runaway_text() -> None:
    from evo.engine.loop import _GENERATOR_DETAIL_CAP, _generator_facing_detail

    assert "GT=<redacted>" in _generator_facing_detail("limits differ, GT=[-1.57, 0.0]")
    typical = "x" * 520  # the longest real verdict observed
    assert _generator_facing_detail(typical) == typical
    assert len(_generator_facing_detail("y" * 2000)) <= _GENERATOR_DETAIL_CAP + 2


def test_neutral_trade_between_foundational_checks_is_not_rejected(tmp_path):
    """Run 3, round 1, real data: motion_direction 4->1, rest_pose 3->5,
    paired continuous score exactly 0.000. The first calibration (a uniform
    35% per-check tolerance) rejected this on the check with a 70% measured
    flip rate. Foundational checks trade against each other at net zero;
    that is not a collapse."""
    from evo.engine.loop import _paired_task_scores
    baseline = _fixture_results(tmp_path, "baseline", "ab_neutral_trade.json")
    candidate = _fixture_results(tmp_path, "candidate", "ab_neutral_trade.json")
    assert _critical_ab_regressions(baseline, candidate) == ()
    paired = _paired_task_scores(baseline, candidate)
    diffs = tuple(c - b for _k, b, c in paired)
    decision = _candidate_ab_decision(
        baseline_pass_rate=0.1, baseline_mean_score=0.4,
        candidate_pass_rate=0.0, candidate_mean_score=0.35,
        sample_count=10, minimum_samples=10, paired_diffs=diffs,
    )
    assert decision.accepted and decision.reason == "non_inferior", decision


def test_critical_total_collapse_across_checks_is_rejected(tmp_path):
    # Two foundational checks each losing most passes: total 16 -> 5.
    b_root = tmp_path / "b"; c_root = tmp_path / "c"; b_root.mkdir(); c_root.mkdir()
    def rows(root, a_flags, b_flags):
        return [_adaptive_result(root, i, oracle=OracleResult(passed=False, checks=[
            CheckResult("subagent_rest_pose", a), CheckResult("subagent_motion_type", b)]))
            for i, (a, b) in enumerate(zip(a_flags, b_flags))]
    baseline = rows(b_root, [True] * 8, [True] * 8)
    candidate = rows(c_root, [True] * 3 + [False] * 5, [True] * 2 + [False] * 6)
    flags = _critical_ab_regressions(baseline, candidate)
    assert any(f.startswith("foundational_total:16->5") for f in flags), flags


def test_transient_errors_cover_provider_and_renderer_blips() -> None:
    from evo.engine.loop import _is_transient_error, _is_training_system_error
    for text in (
        "Connection error.",
        "Error getting response; filtered.input=[...]",
        "simulation render failed; cause: page disconnected",
        "HTTP 503 service unavailable",
    ):
        assert _is_transient_error(text), text
        # Still a training-system error once retries are exhausted.
        assert _is_training_system_error(text) or "response" in text.lower(), text
    # Gateway stream drop seen in run 10 (2026-09-24), previously scored as a failure.
    assert _is_transient_error("Error code: 408 - {'error': {'message': 'stream error: "
                               "stream disconnected before completion: stream closed before "
                               "response.completed', 'type': 'invalid_request_error'}}")
    assert not _is_transient_error("NameError: name 'os' is not defined")
    assert not _is_transient_error("No URDF produced")


def test_transient_status_codes_need_a_whole_token_in_an_error_context() -> None:
    from evo.engine.loop import _is_transient_error
    # Genuine gateway / provider failures, as the SDK and proxies phrase them.
    for text in (
        "Error code: 502 - {'error': {'message': 'Bad Gateway'}}",
        "HTTP 504 gateway timeout",
        "openai.APIStatusError: status 429",
        "upstream returned 503",
        "Error code: 408 - stream disconnected before completion",
    ):
        assert _is_transient_error(text), text
    # Run 22 round 2 (2026-10-08): 26 SUCCESSFUL sessions were re-run because the
    # executor had printed mesh bounds containing 0.504 / -0.429 and a traceback
    # had "line 502". None of these are gateway errors.
    for text in (
        "[[-0.299869 -0.203593  0.      ]\n [ 0.300162  0.225085  0.55043 ]] [0.600031 0.428678 0.504]",
        "0 [[-0.896732 -0.691091 -0.389086]] obb [0.04100106 0.4291028 1.99270164]",
        "SIM ERROR: Traceback (most recent call last):\n"
        '  File "/x/.venv/lib/python3.11/site-packages/errors.py", line 502, in set_box_aspect\n'
        "ValueError: output array is read-only",
        "center [-0.00315924 -0.00328353 0.02009257] verts 14420 extent 0.429 error 0.0503",
        "RESULT: {\"span\": [504, 429, 502]}",
    ):
        assert not _is_transient_error(text), text


def test_sim_port_range_shifts_past_a_foreign_listener() -> None:
    import socket
    from evo.engine.loop import EvoConfig, _shift_to_free_port_range
    cfg = EvoConfig(); cfg.parallel = 3; cfg.sim_base_port = 47100
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 47102)); blocker.listen(1)   # inside 47100-47103
        base = _shift_to_free_port_range(cfg)
    assert base >= 47103 and cfg.sim_base_port == base
    cfg.sim_base_port = 47200
    assert _shift_to_free_port_range(cfg) == 47200   # free block is left alone


def test_sim_port_range_shifts_past_another_runs_live_worker(monkeypatch) -> None:
    # 2026-11-03: two runs started in the same second both took 9876+, the
    # second reclaimed the first's workers as "stale" and one executor talked to
    # the other run's sim session. A worker whose parent is alive is foreign.
    import socket
    from evo.engine import loop
    cfg = loop.EvoConfig(); cfg.parallel = 2; cfg.sim_base_port = 47300
    with socket.socket() as worker:
        worker.bind(("127.0.0.1", 47301)); worker.listen(1)
        monkeypatch.setattr(loop, "_is_orphaned_sim_worker", lambda pid: False)   # live parent
        assert loop._shift_to_free_port_range(cfg) >= 47302
        cfg.sim_base_port = 47300
        monkeypatch.setattr(loop, "_is_orphaned_sim_worker", lambda pid: True)    # orphan: reclaimable
        assert loop._shift_to_free_port_range(cfg) == 47300


def test_executor_code_retry_only_when_the_session_produced_no_urdf(tmp_path) -> None:
    from evo.engine.agents import ExecutorResult
    from evo.engine.loop import _should_retry_executor_code
    slip = 'SIM ERROR: Traceback (most recent call last):\n  File "<agent_code>", line 2\n    def_placeholder = None\nIndentationError: unexpected indent'
    failed = ExecutorResult("t", "0", tmp_path, success=False, error="No URDF produced", duration=1.0)
    recovered = ExecutorResult("t", "0", tmp_path, success=True, duration=1.0)
    assert _should_retry_executor_code(failed, slip, 0, 1) is True
    assert _should_retry_executor_code(failed, slip, 1, 1) is False          # budget spent
    assert _should_retry_executor_code(recovered, slip, 0, 1) is False       # URDF exported: keep it
    assert _should_retry_executor_code(failed, "Connection error.", 0, 1) is False   # not a code slip


def test_skill_preflight_timeout_default_and_env(monkeypatch, tmp_path) -> None:
    from evo.engine import skill_preflight as sp
    monkeypatch.delenv(sp.PREFLIGHT_TIMEOUT_ENV, raising=False)
    assert sp._default_preflight_timeout() == 120.0
    monkeypatch.setenv(sp.PREFLIGHT_TIMEOUT_ENV, "0.2")
    assert sp._default_preflight_timeout() == 0.2
    skill = tmp_path / "skill"; (skill / "tools").mkdir(parents=True)
    (skill / "SKILL.md").write_text("# s\n"); (skill / "ACTIONS.md").write_text("# a\n")
    (skill / "tools" / "preflight.py").write_text("import time; time.sleep(2)\n")
    result = sp.preflight_skill(skill)          # env cap applies when no explicit timeout
    assert not result.ok and any("exceeded 0.2s" in e for e in result.errors)



def test_preflight_rejection_reruns_only_the_generator(tmp_path, monkeypatch):
    """A candidate that fails the skill preflight costs one more Generator turn,
    not a re-evaluation of the tranche: the second turn sees the reason, the
    rejected attempt is kept as skill.rejected1, and A/B proceeds."""
    tasks = tmp_path / "dataset" / "tasks"
    gt = tmp_path / "dataset" / "gt"
    skill = tmp_path / "skill"
    for tid in ("task_01", "task_02"):
        (tasks / tid / "0" / "parts").mkdir(parents=True)
        (gt / tid / "0").mkdir(parents=True)
    skill.mkdir()
    (skill / "SKILL.md").write_text("# baseline\n")
    (skill / "ACTIONS.md").write_text("# frozen\n")

    calls: list[tuple[str, str, str]] = []

    async def fake_executor(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "executor.log").write_text("[success] done\n")
        calls.append((output_dir.parents[1].name, kwargs["factory"], kwargs["instance"]))
        return ExecutorResult(factory=kwargs["factory"], instance=kwargs["instance"],
                              output_dir=output_dir, success=True)

    async def fake_oracle(**kwargs):
        # one failing task, otherwise the loop skips the Generator (target reached)
        ok = kwargs["pred_dir"].parents[1].name == "task_01"
        return OracleResult(passed=ok, checks=[CheckResult("objective", ok)])

    generator_calls: list[dict] = []

    async def fake_generator(**kwargs):
        generator_calls.append(dict(kwargs))
        output_skill_dir = kwargs["output_skill_dir"]
        shutil.copytree(kwargs["current_skill_dir"], output_skill_dir)
        (output_skill_dir / "tools").mkdir()
        if len(generator_calls) == 1:
            # relative import outside a package: the preflight import step fails
            (output_skill_dir / "tools" / "review.py").write_text("from .other import x\n")
            (output_skill_dir / "TOOLS.md").write_text("### `tools/review.py::nothing()`\n")
            files = ["tools/review.py", "TOOLS.md"]
        else:
            (output_skill_dir / "SKILL.md").write_text("# candidate without tools\n")
            shutil.rmtree(output_skill_dir / "tools")
            files = ["SKILL.md"]
        (kwargs["log_file"].parent / "generator_journal.md").write_text("## Observed\nnone\n")
        return GeneratorResult(skill_dir=output_skill_dir, changes_summary="candidate written",
                               files_modified=files)

    monkeypatch.setattr("evo.engine.loop._install_skill", lambda _path: None)
    monkeypatch.setattr("evo.engine.loop.run_executor", fake_executor)
    monkeypatch.setattr("evo.engine.loop._run_oracle_dispatch", fake_oracle)
    monkeypatch.setattr("evo.engine.loop.run_generator", fake_generator)

    config = EvoConfig(max_rounds=1, tasks_root=tasks, gt_root=gt, skill_dir=skill,
                       output_root=tmp_path / "logs", parallel=2, adaptive_train=True,
                       train_increment=2, max_train_samples=2, candidate_ab_test=True,
                       generator_repair_retries=2)
    entries = [{"task_id": "task_01", "instance": "0", "split": "train"},
               {"task_id": "task_02", "instance": "0", "split": "train"}]
    asyncio.run(evolve_multi_task(config, task_ids=entries, feedback_split="train", stagger_sec=0))

    round_dir = next(d for d in (tmp_path / "logs" / "runs").glob("multi_*/round_0")
                     if not d.parent.name.endswith("_private"))
    assert len(generator_calls) == 2
    assert generator_calls[0]["repair_note"] is None
    assert "preflight failed" in generator_calls[1]["repair_note"]
    assert (round_dir / "skill.rejected1" / "tools" / "review.py").exists()
    assert (round_dir / "candidate_ab_test.json").exists()            # A/B ran on the repaired candidate
    # the tranche was evaluated exactly once per task (no round re-run)
    assert [c for c in calls if c[0] == "outputs"].count(("outputs", "task_01", "0")) == 1
    assert sum(1 for c in calls if c[0] == "outputs") == 2
    assert not (round_dir.parent / "round_1").exists()


def test_each_update_gets_its_own_ab_verdict(tmp_path, monkeypatch):
    """updates_per_round=2: the Generator runs twice in one round; update 0 is
    accepted and becomes the baseline of update 1 (no re-evaluation), update 1 is
    rejected; the round's lineage state is the accepted update."""
    tasks = tmp_path / "dataset" / "tasks"
    gt = tmp_path / "dataset" / "gt"
    skill = tmp_path / "skill"
    for tid in ("task_01", "task_02"):
        (tasks / tid / "0" / "parts").mkdir(parents=True)
        (gt / tid / "0").mkdir(parents=True)
    skill.mkdir()
    (skill / "SKILL.md").write_text("# baseline\n")
    (skill / "ACTIONS.md").write_text("# frozen\n")

    calls: list[tuple[str, str]] = []

    async def fake_executor(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "executor.log").write_text("[success] done\n")
        stage = output_dir.parents[2].name if output_dir.parents[2].name.startswith("update_") else output_dir.parents[1].name
        calls.append((stage, kwargs["factory"]))
        return ExecutorResult(factory=kwargs["factory"], instance=kwargs["instance"],
                              output_dir=output_dir, success=True)

    async def fake_oracle(**kwargs):
        parts = kwargs["pred_dir"].parts
        tid = kwargs["pred_dir"].parents[1].name
        if "update_0" in parts:        # first candidate: both tasks pass
            ok = True
        elif "update_1" in parts:      # second candidate: regresses task_02
            ok = tid == "task_01"
        else:                          # round evaluation: only task_01 passes
            ok = tid == "task_01"
        return OracleResult(passed=ok, checks=[CheckResult("objective", ok)])

    generator_calls: list[dict] = []

    async def fake_generator(**kwargs):
        generator_calls.append(dict(kwargs))
        out = kwargs["output_skill_dir"]
        shutil.copytree(kwargs["current_skill_dir"], out)
        (out / "SKILL.md").write_text(f"# candidate {len(generator_calls)}\n")
        (kwargs["log_file"].parent / "generator_journal.md").write_text("## Observed\nnone\n")
        return GeneratorResult(skill_dir=out, changes_summary=f"update {len(generator_calls)}",
                               files_modified=["SKILL.md"])

    monkeypatch.setattr("evo.engine.loop._install_skill", lambda _path: None)
    monkeypatch.setattr("evo.engine.loop.run_executor", fake_executor)
    monkeypatch.setattr("evo.engine.loop._run_oracle_dispatch", fake_oracle)
    monkeypatch.setattr("evo.engine.loop.run_generator", fake_generator)

    config = EvoConfig(max_rounds=1, tasks_root=tasks, gt_root=gt, skill_dir=skill,
                       output_root=tmp_path / "logs", parallel=2, adaptive_train=True,
                       train_increment=2, max_train_samples=2, candidate_ab_test=True,
                       updates_per_round=2)
    entries = [{"task_id": "task_01", "instance": "0", "split": "train"},
               {"task_id": "task_02", "instance": "0", "split": "train"}]
    asyncio.run(evolve_multi_task(config, task_ids=entries, feedback_split="train", stagger_sec=0))

    run_dir = next(d for d in (tmp_path / "logs" / "runs").glob("multi_*") if not d.name.endswith("_private"))
    round_dir = run_dir / "round_0"
    assert len(generator_calls) == 2
    assert generator_calls[0]["update_note"] and "update 1 of 2" in generator_calls[0]["update_note"]
    assert "update 2 of 2" in generator_calls[1]["update_note"] and "ACCEPT" in generator_calls[1]["update_note"]
    # update 1 starts from the accepted update 0
    assert Path(generator_calls[1]["current_skill_dir"]).resolve() == (round_dir / "update_0" / "skill").resolve()
    rec0 = json.loads((round_dir / "update_0" / "candidate_ab_test.json").read_text())
    rec1 = json.loads((round_dir / "update_1" / "candidate_ab_test.json").read_text())
    assert rec0["action"] == "ACCEPT_CANDIDATE" and rec0["baseline_source"] == "round_train_results"
    assert rec1["action"] == "REJECT_CANDIDATE" and rec1["baseline_source"] == "previous_update_replay"
    # evaluation happened once; each update replayed the two assets once
    assert sum(1 for s, _ in calls if s == "outputs") == 2
    assert sum(1 for s, _ in calls if s == "update_0") == 2 and sum(1 for s, _ in calls if s == "update_1") == 2
    assert (round_dir / "skill" / "SKILL.md").read_text() == "# candidate 1\n"
    state = json.loads((run_dir / "loop_state.json").read_text())
    assert Path(state["current_skill_dir"]).resolve() == (round_dir / "update_0" / "skill").resolve()
    assert [u["action"] for u in state["feedback_history"][-1]["updates"]] == ["ACCEPT_CANDIDATE", "REJECT_CANDIDATE"]
    assert "## Update 0" in (round_dir / "generator_journal.md").read_text()


def test_try_skill_tool_is_a_fixed_task_smoke_test(tmp_path, monkeypatch):
    from evo.engine.loop import _make_try_skill_tool
    tasks = tmp_path / "dataset" / "tasks"; gt = tmp_path / "dataset" / "gt"
    (tasks / "task_01" / "0" / "parts").mkdir(parents=True); (gt / "task_01" / "0").mkdir(parents=True)
    (tasks / "task_01" / "0" / "thumbnail.png").write_bytes(b"png")
    skill = tmp_path / "skill"; skill.mkdir(); (skill / "SKILL.md").write_text("# s\n")
    seen: list[dict] = []
    scripted: list[dict] = []

    async def fake_executor(**kwargs):
        seen.append(kwargs)
        out = kwargs["output_dir"]; out.mkdir(parents=True, exist_ok=True)
        step = scripted.pop(0)
        if step.get("urdf"):
            (out / "furniture.urdf").write_text('<robot name="r"><link name="base"/></robot>')
        if step.get("failed_txt"):
            (out / "FAILED.txt").write_text(step["failed_txt"])
        (out / "result.json").write_text(json.dumps({"total_cost_usd": 0.07}))
        return ExecutorResult(factory=kwargs["factory"], instance=kwargs["instance"], output_dir=out,
                              success=bool(step.get("urdf")), error=step.get("error"), duration=12.5)

    oracle_calls: list[dict] = []

    async def fake_oracle(**kwargs):
        oracle_calls.append(kwargs)
        return OracleResult(passed=True, checks=[])

    monkeypatch.setattr("evo.engine.loop.run_executor", fake_executor)
    monkeypatch.setattr("evo.engine.loop._run_oracle_dispatch", fake_oracle)
    config = EvoConfig(tasks_root=tasks, gt_root=gt, skill_dir=skill, output_root=tmp_path / "logs", parallel=2)
    tool, impl, spend = _make_try_skill_tool(
        config=config, round_dir=tmp_path / "round_0", skill_dir=skill,
        trial_task={"task_id": "task_01", "instance": "0"}, ports=[9000],
        executor_effort="low", subagent_model=None, subagent_effort=None, max_calls=3)
    assert tool.name == "try_skill"

    # 1. the executor gave up: raw FAILED.txt text, no cause classification, no oracle
    scripted.append({"urdf": False, "error": "No URDF produced", "failed_txt": "sim worker: MemoryError in split()"})
    rep = asyncio.run(impl())
    assert rep["ok"] and rep["task"] == "task_01/0" and rep["trial"] == 1 and not rep["clean"]
    assert rep["executor_success"] is False and rep["executor_error"] == "No URDF produced"
    assert rep["failed_txt"] == "sim worker: MemoryError in split()"
    assert rep["trajectory"].endswith("task_01_0_1/executor.log") and "Read executor.log" in rep["next"]
    assert "problems" not in rep and "oracle" not in rep and oracle_calls == []
    assert (tmp_path / "round_0" / "generator_trials" / "task_01_0_1" / "smoke_test.json").exists()

    # 2. the executor finished
    scripted.append({"urdf": True})
    rep = asyncio.run(impl())
    assert rep["clean"] and rep["executor_error"] is None and rep["failed_txt"] is None and rep["urdf"] == ["furniture.urdf"]
    assert "check_skill" in rep["next"] and rep["trial"] == 2

    # 3. a URDF plus FAILED.txt is still a failure the Generator must look at
    scripted.append({"urdf": True, "failed_txt": "gave up after 3 attempts"})
    rep = asyncio.run(impl())
    assert not rep["clean"] and rep["executor_success"] is True and rep["failed_txt"].startswith("gave up")
    assert all(k["factory"] == "task_01" and k["instance"] == "0" and k["sim_port"] == 9000 for k in seen)
    assert spend == {"calls": 3, "cost_usd": pytest.approx(0.21), "last_clean": False, "clean_calls": 1}
    assert "safety limit" in asyncio.run(impl())["error"]

    # a candidate that fails the contract check is refused before any executor runs
    (skill / "tools").mkdir(); (skill / "tools" / "bad.py").write_text("import nope_xyz\n")
    tool2, impl2, spend2 = _make_try_skill_tool(
        config=config, round_dir=tmp_path / "round_1", skill_dir=skill,
        trial_task={"task_id": "task_01", "instance": "0"}, ports=[9000],
        executor_effort="low", subagent_model=None, subagent_effort=None, max_calls=2)
    refused = asyncio.run(impl2())
    assert not refused["ok"] and "acceptance check" in refused["error"] and refused["check_skill"]["errors"]
    assert spend2["calls"] == 0 and len(seen) == 3


def test_fixed_batch_draws_the_same_tasks_every_round():
    from evo.engine.loop import _fixed_round_batch, _sample_batch
    import random

    pool = [{"task_id": f"t{i}", "instance": "0"} for i in range(30)]
    meta = {(f"t{i}", "0"): {"difficulty": ["easy", "medium", "hard"][i % 3]} for i in range(30)}
    ratio = {"easy": 3, "medium": 3, "hard": 2}
    a = _fixed_round_batch(pool, meta, 8, ratio, seed=7)
    b = _fixed_round_batch(pool, meta, 8, ratio, seed=7)
    assert a == b and len(a) == 8
    assert sorted(meta[(e["task_id"], e["instance"])]["difficulty"] for e in a) == ["easy"] * 3 + ["hard"] * 2 + ["medium"] * 3
    # the rotating sampler, by contrast, moves on to unseen tasks next round
    seen: dict = {}
    rng = random.Random(7)
    r1 = _sample_batch(pool, meta, 8, ratio, seen, rng)
    r2 = _sample_batch(pool, meta, 8, ratio, seen, rng)
    assert not {(e["task_id"], e["instance"]) for e in r1} & {(e["task_id"], e["instance"]) for e in r2}


def test_memory_cap_messages_are_executor_code_errors_not_transient() -> None:
    from evo.engine.loop import _execution_error_kind, _is_training_system_error, _is_transient_error
    interrupted = (
        "MemoryError: this snippet pushed the sim session to 6.21 GB resident, over the 6.0 GB "
        "per-session memory cap, and was interrupted; its result is discarded. All session variables were "
        "cleared to free memory; re-create what you still need. Usual cause: trimesh.load(path, "
        "force='mesh') decodes every texture image ...\nTraceback (most recent call last):\n"
        '  File "<agent_code>", line 3, in <module>\n'
        "evo.engine.sim_worker.SimMemoryLimitExceeded: session resident memory 6.21 GB exceeds the 6.0 GB per-session memory cap"
    )
    assert _execution_error_kind(interrupted) == "executor_code_error"
    assert not _is_transient_error(interrupted) and not _is_training_system_error(interrupted)
    killed = (
        "MemoryError: the sim worker for this session was killed: it stayed at 7.9 GB resident, above "
        "the 6.0 GB per-session memory cap, for 10.0 s while running your snippet and could not be interrupted."
    )
    assert not _is_transient_error(killed) and not _is_training_system_error(killed)
