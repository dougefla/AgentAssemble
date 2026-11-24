"""The Evaluator's feedback digest: deterministic, ranked, GT-free."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from evo.engine.feedback_digest import (
    MODE_ORDER,
    build_digest,
    classify_run,
    feature_from_journal,
    feature_ledger,
    persistence,
    stability,
    summarize_executor_log,
)


def _run(tag, verdict="FAIL", failed=(), details=None, joints=None, reps=1, out="/tmp/x", error=None):
    r = {
        "tag": tag, "output_dir": out, "oracle_verdict": verdict,
        "failed_oracle_checks": list(failed), "failed_check_details": details or {},
        "joint_reports": joints or {}, "executor_reps": reps,
    }
    if error:
        r["error"] = error
    return r


REST = "2 matched link(s) mis-posed at rest by > 0.02m (max={mx}m): link_1={mx}m"
KIN = "1 invalid independent joint trace(s); link_1: trace drift={d}m (forward, tol=0.02m)"


def test_modes_are_exclusive_and_ordered_by_cause() -> None:
    assert classify_run(_run("a", "PASS")) == ("pass", 0.0)
    assert classify_run(_run("b", "execution_failed", error="boom"))[0] == "execution_failed"
    assert classify_run(_run("c", failed=["valid_tree", "rest_pose_alignment"]))[0] == "structure"
    mode, sev = classify_run(_run("d", failed=["joint_axes", "rest_pose_alignment"],
                                  joints={"j0": {"axis_ok": False, "axis_angle_deg": 25.0}}))
    assert (mode, sev) == ("axis", 25.0)
    # a wrongly scaled link is also mis-posed; it counts once, under scale
    mode, sev = classify_run(_run("e", failed=["link_scale", "rest_pose_alignment", "kinematic_equivalence"],
                                  details={"link_scale": "checked 2 matched link(s), 1 off-size: link_1: size ratio 0.500 (tol ±5%)",
                                           "rest_pose_alignment": REST.format(mx="0.100")}))
    assert mode == "scale" and abs(sev - 0.693) < 1e-3
    mode, sev = classify_run(_run("f", failed=["rest_pose_alignment", "kinematic_equivalence"],
                                  details={"rest_pose_alignment": REST.format(mx="0.118"),
                                           "kinematic_equivalence": KIN.format(d="0.118")}))
    assert mode == "pose" and abs(sev - 5.9) < 1e-6
    mode, sev = classify_run(_run("g", failed=["rest_pose_alignment", "kinematic_equivalence"],
                                  details={"rest_pose_alignment": REST.format(mx="0.021"),
                                           "kinematic_equivalence": KIN.format(d="0.021")}))
    assert mode == "near_miss" and abs(sev - 1.05) < 1e-6
    mode, sev = classify_run(_run("h", failed=["joint_pivot_alignment"],
                                  joints={"j0": {"pivot_ok": False, "pivot_distance_m": 0.044}}))
    assert mode == "pivot" and abs(sev - 4.4) < 1e-9
    mode, sev = classify_run(_run("i", failed=["kinematic_equivalence"],
                                  details={"kinematic_equivalence": KIN.format(d="0.358")}))
    assert mode == "motion" and abs(sev - 17.9) < 1e-6
    assert classify_run(_run("j", failed=["joint_limits"]))[0] == "limits"
    assert MODE_ORDER[0] == "execution_failed" and MODE_ORDER[-1] == "pass"


def test_stability_needs_several_sessions_and_reads_the_vote_prefix() -> None:
    assert stability(_run("a", failed=["rest_pose_alignment"])) is None
    stable = _run("b", reps=3, failed=["rest_pose_alignment"],
                  details={"rest_pose_alignment": "[0/3 votes PASS] " + REST.format(mx="0.1")})
    flaky = _run("c", reps=3, failed=["rest_pose_alignment"],
                 details={"rest_pose_alignment": "[1/3 votes PASS] " + REST.format(mx="0.1")})
    assert stability(stable) == "stable" and stability(flaky) == "unstable"
    spread = _run("d", reps=3, failed=["kinematic_equivalence"],
                  joints={"j0": {"motion_ok": False, "score_spread": 0.67}})
    assert stability(spread) == "unstable"


def test_executor_summary_lists_helpers_inspection_and_the_final_claim(tmp_path: Path) -> None:
    log = tmp_path / "executor.log"
    log.write_text(
        "[10:00:00] [tool_call] Read: {\"file_path\":\"/d/thumbnail.png\"}\n"
        "[10:00:00] [tool_result] [Read] <image>\n"
        "[10:00:01] [tool_call] mcp__sim__execute_python_code: from tools.reference_overlay import render_reference_overlay, render_reference_\n"
        "[10:00:01] [tool_call] r = tools.geometry.fit_part(report)\n"
        "[10:00:02] [tool_call] mcp__sim__execute_python_code: print(1)\n"
        "[10:00:03] [tool_call] Read: {\"file_path\":\"/d/overlay.PNG\"}\n"
        "[10:00:04] [text] Completed the assembly; the overlay check passed.\n"
        "[10:00:04] [text] - furniture.urdf\n"
    )
    s = summarize_executor_log(log)
    assert "reference_overlay.render_reference_overlay" in s
    assert "render_reference_," not in s            # truncated prefix dropped
    assert "geometry.fit_part" in s
    assert "sim calls 2" in s and "images inspected 2" in s
    assert 'final claim: "Completed the assembly; the overlay check passed."' in s
    assert summarize_executor_log(tmp_path / "missing.log") == "(executor.log unavailable)"


def test_feature_from_journal_takes_the_bold_name_under_changed(tmp_path: Path) -> None:
    j = tmp_path / "generator_journal.md"
    j.write_text("## Observed\nx\n\n## Changed\nONE feature: **root-anchored rest pose**, SKILL.md only.\n\n## Verified\ny\n")
    assert feature_from_journal(j) == "root-anchored rest pose"
    j.write_text("## Changed\nAdded a camera registration helper that does many things.\n")
    assert feature_from_journal(j) == "Added a camera registration helper that does many things"
    j.write_text("## Changed\nAdded ONE feature, the surface-constrained rest-placement workflow, in SKILL.md only.\n")
    assert feature_from_journal(j) == "surface-constrained rest-placement workflow, in SKILL.md only"
    assert feature_from_journal(tmp_path / "none.md") is None


def _history(tmp_path: Path):
    """R0 seed 6, R1 8 (new best), R2 7 (kept), R3 12 (new best), R4 9 (rolled back)."""
    def fb(k, n_pass, fit, rolled=False, changed=True):
        runs = [_run(f"task_{i:03d}/0", "PASS" if i < n_pass else "FAIL",
                     failed=() if i < n_pass else ["rest_pose_alignment", "kinematic_equivalence"],
                     details={} if i < n_pass else {"rest_pose_alignment": REST.format(mx="0.100"),
                                                    "kinematic_equivalence": KIN.format(d="0.100")},
                     out=str(tmp_path / f"round_{k}" / "outputs" / f"task_{i:03d}" / "0"))
                for i in range(10)]
        d = {"round": k, "pass_rate": n_pass / 10, "selection_fitness": fit, "runs": runs,
             "per_check_pass_rate": {}, "generator_changed": changed}
        if rolled:
            d["skill_rolled_back"] = True
        return d
    hist = [fb(0, 6, 0.14), fb(1, 8, 0.19), fb(2, 7, 0.16), fb(3, 12 - 4, 0.28), fb(4, 5, 0.12, rolled=True)]
    for k, name in enumerate(["scale-baked export", "root-anchored rest pose", "camera registration", "motion check"]):
        d = tmp_path / f"round_{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "generator_journal.md").write_text(f"## Changed\n**{name}**\n")
    return hist


def test_ledger_measures_each_feature_against_the_checkpoint(tmp_path: Path) -> None:
    rows = feature_ledger(_history(tmp_path), tmp_path)
    assert [r["outcome"] for r in rows] == ["seed", "new best", "kept", "new best", "rolled back"]
    assert [r["feature"] for r in rows][1:] == [
        "scale-baked export", "root-anchored rest pose", "camera registration", "motion check"]
    # R2 (7) is compared with the checkpoint R1 (8), R4 (5) with the checkpoint R3 (8)
    assert [r["delta"] for r in rows] == [None, 2, -1, 0, -3]
    assert rows[4]["written_in_round"] == 3


def test_persistence_counts_consecutive_rounds_in_the_same_mode(tmp_path: Path) -> None:
    hist = _history(tmp_path)
    # task_009 failed as "pose" in every round; task_006 passed in R3 and fails again in R4
    assert persistence("task_009/0", "pose", hist) == 4
    assert persistence("task_006/0", "pose", hist) == 0
    assert persistence("task_009/0", "scale", hist) == 0
    assert persistence("task_009/0", "pose", hist[:1]) == 0


def test_digest_ranks_modes_shows_flips_and_points_to_the_full_list(tmp_path: Path) -> None:
    hist = _history(tmp_path)
    # make the last round's picture richer: one scale failure with a larger error than the pose ones,
    # one task that flipped PASS -> FAIL and a task that is now a near miss
    last = hist[-1]["runs"]
    last[5]["failed_oracle_checks"] = ["link_scale", "rest_pose_alignment"]
    last[5]["failed_check_details"] = {"link_scale": "checked 2 matched link(s), 1 off-size: link_1: size ratio 0.400 (tol ±5%)"}
    last[9]["failed_check_details"] = {"rest_pose_alignment": REST.format(mx="0.025"),
                                       "kinematic_equivalence": KIN.format(d="0.025")}
    out_dir = Path(last[6]["output_dir"]); out_dir.mkdir(parents=True)
    (out_dir / "executor.log").write_text("[1:1:1] [text] Overlay matched the reference.\n")
    digest = build_digest(hist, run_dir=tmp_path, full_list_path=tmp_path / "round_4" / "feedback_tasks.md",
                          n_representatives=2)
    # pose = task_006/007/008; only task_008 was "pose" in R3 too (006/007 passed there, 009 is now a near miss)
    assert "| pose | 3 | 1 |" in digest and "| scale | 1 | 0 |" in digest and "| near_miss | 1 | 0 |" in digest
    assert "| pass | 5 | — | -3 |" in digest
    # representatives: at most 2 per mode, persistent ones first, executor claim shown
    pose = digest.split("- **pose**")[1].split("- **near_miss**")[0]
    assert pose.count("  - task_") == 2 and "error 5.0× tol" in pose
    first, second = [l for l in pose.splitlines() if l.startswith("  - task_")]
    assert first.startswith("  - task_008/0  error 5.0× tol, same mode for 5 rounds")
    assert "new this round" in second   # 006/007 tie on error; either is a fresh failure
    assert 'final claim: "Overlay matched the reference."' in pose
    assert "(edit under test: motion check)" in digest
    assert "PASS→FAIL (3): task_005/0 (link_scale, rest_pose_alignment), task_006/0" in digest
    assert "FAIL→PASS (0): none" in digest
    assert "| R4 | motion check | 5/10 | -3 | rolled back |" in digest
    assert str(tmp_path / "round_4" / "feedback_tasks.md") in digest
    assert "GT=" not in digest   # nothing beyond the sanitised details reaches the Evolver
    assert build_digest([]) == "(no evaluation yet)"


def test_generator_prompt_carries_the_digest_and_writes_both_files(tmp_path: Path, monkeypatch) -> None:
    from evo.engine.agents import run_generator
    from evo.engine.responses_backend import ResponsesRunOutcome

    current = tmp_path / "current"; current.mkdir()
    (current / "SKILL.md").write_text("# Before\n")
    run_dir = tmp_path / "run"
    output = run_dir / "round_1" / "skill"
    log_file = run_dir / "round_1" / "generator.log"
    log_file.parent.mkdir(parents=True)
    (run_dir / "round_0").mkdir()
    (run_dir / "round_0" / "generator_journal.md").write_text("## Changed\n**scale-baked export**\n")
    captured: dict = {}

    async def fake_run_responses_agent(**kwargs):
        captured.update(kwargs)
        (output / "SKILL.md").write_text("# After\n")
        (output.parent / "generator_journal.md").write_text("Verified\n")
        kwargs["artifact_path"].write_text("{}")
        return ResponsesRunOutcome(
            final_output="updated", response_id="resp", artifact_path=kwargs["artifact_path"],
            usage_summary={"cost_usd": 0.0, "input": 0, "output": 0, "cache_read": 0,
                           "cache_write_5m": 0, "cache_write_1h": 0, "cache_misses": 0, "n_api_calls": 0},
        )
    monkeypatch.setattr("evo.engine.responses_backend.run_responses_agent", fake_run_responses_agent)

    def runs(k, n_pass):
        return [_run(f"task_{i:03d}/0", "PASS" if i < n_pass else "FAIL",
                     failed=() if i < n_pass else ["rest_pose_alignment"],
                     details={} if i < n_pass else {"rest_pose_alignment": REST.format(mx="0.090")},
                     out=str(run_dir / f"round_{k}" / "outputs" / f"task_{i:03d}" / "0")) for i in range(4)]
    history = [{"round": 0, "pass_rate": 0.25, "selection_fitness": 0.25, "runs": runs(0, 1), "generator_changed": True},
               {"round": 1, "pass_rate": 0.5, "selection_fitness": 0.5, "runs": runs(1, 2)}]
    asyncio.run(run_generator(
        current_skill_dir=current, current_skill_content={"SKILL.md": "# Before\n"},
        feedback_history=history, output_skill_dir=output, log_file=log_file, sim_port=None,
    ))
    prompt = captured["prompt"]
    assert "### Evaluator digest of this round" in prompt
    assert "#### 1. Failure modes" in prompt and "| pose | 2 |" in prompt
    assert "FAIL→PASS (1): task_001/0 (rest_pose_alignment)" in prompt
    assert "| R1 | scale-baked export | 2/4 | +1 | new best |" in prompt
    full = run_dir / "round_1" / "feedback_tasks.md"
    assert full.exists() and "task_003/0  [FAIL]" in full.read_text()
    assert str(full) in prompt
    assert (run_dir / "round_1" / "feedback_digest.md").read_text().strip() in prompt
    # the flat list itself is no longer inline
    assert prompt.count("output_dir:") == 2   # only the two pose representatives
