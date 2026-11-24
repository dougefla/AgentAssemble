"""The gate split (val) decides checkpoints and rollbacks; the Generator sees only its counts."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from evo.engine.agents import ExecutorResult, GeneratorResult
from evo.engine.loop import EvoConfig, evolve_multi_task
from evo.oracle.check_urdf import CheckResult, OracleResult


def _setup(tmp_path: Path):
    tasks = tmp_path / "dataset" / "tasks"
    gt = tmp_path / "dataset" / "gt"
    skill = tmp_path / "skill"
    for tid in ("task_01", "task_02", "task_11", "task_12"):
        (tasks / tid / "0" / "parts").mkdir(parents=True)
        (gt / tid / "0").mkdir(parents=True)
    skill.mkdir()
    (skill / "SKILL.md").write_text("# baseline\n")
    (skill / "ACTIONS.md").write_text("# frozen\n")
    return tasks, gt, skill


def _run(tmp_path, monkeypatch, *, gate_mode: str, val_pass_by_round: dict[int, bool],
         train_pass_by_round: dict[int, bool], rounds: int = 2, regression_tol: float = 0.05):
    tasks, gt, skill = _setup(tmp_path)
    calls: list[tuple[Path, str]] = []
    prompts: list[list[dict]] = []

    async def fake_executor(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "executor.log").write_text("[success] done\n")
        calls.append((output_dir, kwargs["factory"]))
        return ExecutorResult(factory=kwargs["factory"], instance=kwargs["instance"],
                              output_dir=output_dir, success=True)

    async def fake_oracle(**kwargs):
        pred = kwargs["pred_dir"]
        rnd = int(next(p for p in pred.parts if p.startswith("round_")).split("_")[1])
        is_val = pred.parent.parent.parent.parent.name.endswith("_val")
        ok = val_pass_by_round[rnd] if is_val else train_pass_by_round[rnd]
        return OracleResult(passed=ok, checks=[CheckResult("objective", ok)])

    async def fake_generator(**kwargs):
        prompts.append(json.loads(json.dumps(kwargs["feedback_history"])))
        out = kwargs["output_skill_dir"]
        shutil.copytree(kwargs["current_skill_dir"], out)
        (out / "SKILL.md").write_text(f"# candidate {len(prompts)}\n")
        (kwargs["log_file"].parent / "generator_journal.md").write_text(
            f"## Changed\n**feature {len(prompts)}**\n")
        return GeneratorResult(skill_dir=out, changes_summary="c", files_modified=["SKILL.md"])

    monkeypatch.setattr("evo.engine.loop._install_skill", lambda _p: None)
    monkeypatch.setattr("evo.engine.loop.run_executor", fake_executor)
    monkeypatch.setattr("evo.engine.loop._run_oracle_dispatch", fake_oracle)
    monkeypatch.setattr("evo.engine.loop.run_generator", fake_generator)
    monkeypatch.setattr("evo.engine.loop._validate_generator_output", lambda *a, **k: (True, ""))
    monkeypatch.setattr("evo.engine.loop.preflight_skill",
                        lambda _d: type("P", (), {"ok": True, "errors": [], "to_dict": lambda self: {}})())

    config = EvoConfig(max_rounds=rounds, tasks_root=tasks, gt_root=gt, skill_dir=skill,
                       output_root=tmp_path / "logs", parallel=2, adaptive_train=False, batch_size=0)
    train = [{"task_id": "task_01", "instance": "0", "split": "train"},
             {"task_id": "task_02", "instance": "0", "split": "train"}]
    val = [{"task_id": "task_11", "instance": "0", "split": "val"},
           {"task_id": "task_12", "instance": "0", "split": "val"}]
    asyncio.run(evolve_multi_task(config, task_ids=train, feedback_split="train", stagger_sec=0,
                                  gate_tasks=val, gate_mode=gate_mode, gate_min_drop=2,
                                  regression_tol=regression_tol))
    run_dir = next((tmp_path / "logs" / "runs").glob("multi_*/loop_state.json")).parent
    state = json.loads((run_dir / "loop_state.json").read_text())
    return run_dir, state, calls, prompts


def test_select_mode_rolls_back_on_the_val_split_even_when_train_improves(tmp_path, monkeypatch):
    # round 0: train fails, val passes (checkpoint val 2/2); round 1: train passes, val fails.
    run_dir, state, calls, prompts = _run(
        tmp_path, monkeypatch, gate_mode="select",
        train_pass_by_round={0: False, 1: True}, val_pass_by_round={0: True, 1: False})
    fh = state["feedback_history"]
    g0, g1 = fh[0]["gate"], fh[1]["gate"]
    assert (g0["split"], g0["pass"], g0["n"], g0["decision"]) == ("val", 2, 2, "new_best")
    assert (g1["pass"], g1["checkpoint_pass"], g1["decision"]) == (0, 2, "rolled_back")
    assert fh[1].get("skill_rolled_back") is True
    assert fh[1]["pass_rate"] == 1.0            # train went up and still did not count
    assert state["best_gate_pass"] == 2
    assert Path(state["current_skill_dir"]) == (run_dir / "best_skill").resolve()
    # val sessions live outside the run dir, train sessions inside it
    val_dirs = {d for d, _ in calls if d.parent.parent.parent.parent.name.endswith("_val")}
    assert len(val_dirs) == 4 and all(run_dir.name + "_val" in str(d) for d in val_dirs)
    assert all(str(run_dir) in str(d) for d, _ in calls if d not in val_dirs)
    # the Generator's feedback carries the val counts but no val task
    for history in prompts:
        for fb in history:
            assert {r["tag"] for r in fb["runs"]} <= {"task_01/0", "task_02/0"}
            assert "val" in json.dumps(fb["gate"]) and "task_11" not in json.dumps(fb)
    assert not any("task_11" in str(p) for p in (run_dir).rglob("*"))


def test_log_mode_records_the_val_curve_but_selects_on_train(tmp_path, monkeypatch):
    run_dir, state, calls, prompts = _run(
        tmp_path, monkeypatch, gate_mode="log",
        train_pass_by_round={0: False, 1: True}, val_pass_by_round={0: True, 1: False})
    fh = state["feedback_history"]
    assert fh[0]["gate"]["pass"] == 2 and fh[1]["gate"]["pass"] == 0
    assert fh[1]["gate"]["decision"] == "new_best"     # train 2/2 beat the train checkpoint
    assert not fh[1].get("skill_rolled_back")
    assert state["best_gate_pass"] == 0                # val passes of the (train-chosen) checkpoint


def test_digest_ledger_shows_val_counts(tmp_path, monkeypatch):
    from evo.engine.feedback_digest import build_digest
    run_dir, state, _, _ = _run(
        tmp_path, monkeypatch, gate_mode="select",
        train_pass_by_round={0: False, 1: True}, val_pass_by_round={0: True, 1: False})
    d = build_digest(state["feedback_history"], run_dir=run_dir, read_logs=False)
    assert "| val pass | val Δ vs checkpoint |" in d
    assert "| R1 | feature 1 | 2/2 | +2 | 0/2 | -2 | rolled back |" in d
    assert "decided on the val split" in d
