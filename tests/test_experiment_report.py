from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from evo.reporting import experiment_report as report_module
from evo.reporting.experiment_report import generate_experiment_report


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _build_run(tmp_path: Path) -> Path:
    run = tmp_path / "multi_123"
    round_dir = run / "round_0"
    output = round_dir / "outputs" / "task_01" / "0"
    snapshot = output / "skill_snapshot"
    candidate = round_dir / "skill"
    snapshot.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (snapshot / "SKILL.md").write_text("# Baseline\nold\n")
    (candidate / "SKILL.md").write_text("# Candidate\nnew\n")
    (output / "render_assembled.png").write_bytes(_PNG_1X1)
    (output / "furniture.urdf").write_text("<robot name='test'/>")
    (output / "oracle.json").write_text(json.dumps({
        "passed": False,
        "checks": [
            {"name": "valid_tree", "passed": True},
            {"name": "joint_origins", "passed": False},
        ],
    }))
    (round_dir / "all_artifacts.json").write_text(json.dumps([{
        "tag": "task_01/0",
        "output_dir": str(output),
        "oracle_verdict": "FAIL",
        "failed_oracle_checks": ["joint_origins"],
        "cost_summary": {"n_api_calls": 3, "output_tokens": 120},
    }]))
    (round_dir / "generator_journal.md").write_text(
        "**Observed**\nOrigin failed.\n\n"
        "**Investigated**\nRead the sweep.\n\n"
        "**Changed**\nUpdated `SKILL.md`.\n\n"
        "**Verified**\nSelf-test passed.\n\n"
        "**Unresolved**\nTight tolerance.\n"
    )
    (round_dir / "generator.result.json").write_text(json.dumps({
        "backend": "responses",
        "model": "gpt-test",
        "files_modified": ["SKILL.md"],
        "usage": {"n_api_calls": 2, "output": 40},
        "error": None,
    }))
    (round_dir / "generator.responses.json").write_text(json.dumps({
        "backend": "openai-agents-responses",
        "model": "openai/gpt-test",
        "provider_base_url": "https://openrouter.ai/api/v1",
        "effort": "high",
        "service_tier": "default",
        "usage": {
            "requests": 2,
            "input_tokens": 500,
            "input_tokens_details": {"cached_tokens": 400},
            "output_tokens": 40,
        },
        "items": [
            {"type": "function_call", "name": "Read"},
            {"type": "function_call_output", "output": "ok"},
        ],
    }))
    (run / "loop_state.json").write_text(json.dumps({
        "round_completed": 0,
        "best_fitness": 0.0005,
        "saved_at": 123.0,
        "feedback_history": [{
            "round": 0,
            "pass_rate": 0.0,
            "mean_oracle_score": 0.5,
            "selection_fitness": 0.0005,
            "n_failures": 1,
            "per_check_pass_rate": {
                "valid_tree": "1/1",
                "joint_origins": "0/1",
            },
            "batch_cost_summary": {
                "total_api_calls": 3,
                "total_output_tokens": 120,
            },
            "adaptive_train_decision": {
                "action": "STOP_ADDING",
                "reason": "objective_failure:joint_origins",
            },
        }],
    }))
    return run


def test_generate_portable_report_from_persisted_run(tmp_path):
    run = _build_run(tmp_path)
    report_dir = tmp_path / "report"

    index = generate_experiment_report(
        run, report_dir, create_videos=False, summarize_skill_changes=False
    )

    assert index == report_dir / "index.html"
    page = index.read_text()
    assert "multi_123" in page
    assert "Origin failed." in page
    assert "openai/gpt-test" in page
    assert "joint_origins" in page
    assert "Updated <code>SKILL.md</code>." in page
    assert "Agent summary" in page
    assert "artifact_fallback" in page
    assert (report_dir / "assets/media/r00-task_01-0.png").is_file()
    assert (report_dir / "assets/diffs/round-1.diff").is_file()
    data = json.loads((report_dir / "report-data.json").read_text())
    assert data["round_count"] == 1
    assert data["total_attempts"] == 1
    assert data["rounds"][0]["per_check"]["valid_tree"] == 1.0
    assert data["rounds"][0]["trajectory"]["tool_calls"] == 1

    # Generated reports are safely replaceable and deterministic in shape.
    second = generate_experiment_report(
        run, report_dir, create_videos=False, summarize_skill_changes=False
    )
    assert second.is_file()


def test_skill_change_agent_summary_is_persisted(tmp_path, monkeypatch):
    run = _build_run(tmp_path)
    report_dir = tmp_path / "report"

    def fake_summary(rounds, **_kwargs):
        assert rounds[0]["number"] == 1
        return {1: "新增几何诊断流程，并收紧关节原点校验步骤。"}

    monkeypatch.setattr(
        report_module, "_request_skill_change_summaries", fake_summary
    )
    index = generate_experiment_report(
        run, report_dir, create_videos=False, summarize_skill_changes=True
    )
    page = index.read_text()
    data = json.loads((report_dir / "report-data.json").read_text())
    assert "新增几何诊断流程" in page
    assert data["rounds"][0]["skill_diff"]["summary_source"] == "agent"
    assert data["skill_summary_agent"]["status"] == "complete"


def test_refuses_to_replace_unmarked_directory(tmp_path):
    run = _build_run(tmp_path)
    output = tmp_path / "someones-files"
    output.mkdir()
    (output / "keep.txt").write_text("do not delete")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        generate_experiment_report(
            run, output, create_videos=False, summarize_skill_changes=False
        )

    assert (output / "keep.txt").read_text() == "do not delete"


def test_motion_frames_group_by_camera_and_ping_pong(tmp_path):
    from evo.reporting.experiment_report import _motion_frames, _motion_frames_by_camera
    for cam in ("", "_az225"):
        for i, pct in enumerate(("000", "033", "067", "100")):
            (tmp_path / f"lid_{i:02d}_{pct}pct{cam}.png").write_bytes(b"x")
    (tmp_path / "lid_00_000pct_crop.png").write_bytes(b"x")
    groups = _motion_frames_by_camera(tmp_path)
    assert set(groups) == {"", "_az225"}
    names = [p.name for p in groups["_az225"]]
    assert names == ["lid_00_000pct_az225.png", "lid_01_033pct_az225.png", "lid_02_067pct_az225.png",
                     "lid_03_100pct_az225.png", "lid_02_067pct_az225.png", "lid_01_033pct_az225.png"]
    assert [p.name for p in _motion_frames(tmp_path)][0] == "lid_00_000pct.png"


def test_sweep_frame_is_the_union_envelope(monkeypatch):
    from evo.oracle import render_lightphyx as rl
    boxes = iter([((0, 0, 0), (1, 1, 1)), ((0, 0, 0), (3, 1, 1))])
    monkeypatch.setattr(rl, "_mesh_entries", lambda *a, **k: ["entry"])
    monkeypatch.setattr(rl, "_world_aabb", lambda entries: next(boxes))
    monkeypatch.setattr("evo.oracle.render_motion.build_pose", lambda *a, **k: {})
    center, extent = rl.sweep_frame({}, {}, None, None, [{}, {}])
    assert center == (1.5, 0.5, 0.5)
    assert abs(extent - (9 + 1 + 1) ** 0.5) < 1e-9
