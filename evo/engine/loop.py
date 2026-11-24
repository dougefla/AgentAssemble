#!/usr/bin/env python3
"""Co-evolution loop for the /assemble skill.

Alternates between:
  - Surrogate iterations: verifier checks predicted URDFs → generator refines skill
  - Oracle iterations: deterministic check against GT → escalate if surrogate missed errors

Data layout (sanitized):
  tasks/<task_id>/<instance>/parts/   ← executor sees this (no GT, no semantic names)
  gt/<task_id>/<instance>/scene.urdf  ← oracle sees this
  mapping.json                        ← loop sees this (task_id ↔ category)

Usage:
    # Single-task overfit test
    python -m evo.engine.loop --single task_01 0 --max-rounds 3 -v

    # Multi-task evolution
    python -m evo.engine.loop --split train

    # Dry run
    python -m evo.engine.loop --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import random
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Load the shared provider env and then <repo>/.env BEFORE importing the
# evo.engine modules below: several of
# them snapshot env-derived constants at import time (paths.DATA_ROOT/
# GT_ROOT and similar). If we defer this to main(),
# a direct `python -m evo.engine.loop` run resolves those against the defaults
# before .env is read. The `evo` CLI works either way (it loads .env in the
# parent and the subprocess inherits it); this makes the direct path correct too.
from evo.engine.paired_gate import PairedGate, artifact_fitness, collect_scores
from evo.engine.dedup import hardlink_duplicate_meshes
from evo.engine.sub_evolvers import SUB_TRIAL_REPS, is_verified

GATE_EXTEND_MIN_Z = 0.5       # paired gate: a rejected candidate at least this far ahead gets one more batch
from evo.engine.sub_evolvers import (
    cooled_clusters, integrator_note, journal_section, proposal_record, select_clusters,
    skill_diff, sub_evolver_count, sub_note,
)
from evo.engine.envfile import load_env_chain as _load_env_chain  # noqa: E402
_load_env_chain()

from evo.engine.agents import (
    ExecutorResult,
    GeneratorResult,
    _summarize_session,
    run_executor,
    run_generator,
    setup_auth_env,
)
from evo.engine.sim_manager import SimStartupError
from evo.engine.lightphyx_manager import SharedLightphyxServer, recycle_shared_pool
from evo.engine.skill_preflight import SkillPreflightResult, preflight_skill
from evo.engine.envfile import load_env_chain
from evo.engine.model_defaults import DEFAULT_AGENT_EFFORT, DEFAULT_AGENT_MODEL
from evo.engine.paths import data_root as _resolved_data_root, gt_root as _resolved_gt_root
from evo.oracle.template_fill import finalize_pred_dir
from evo.oracle.check_urdf import (
    CheckResult, JointReport, OracleResult, Thresholds, run_oracle,
    sanitize_detail, sanitize_oracle_dict,
)
from evo.oracle import ORACLES
# Subagent-oracle path — opus 4.7 visual judge over rest+motion frames.
# Lazily imported inside the dispatcher to keep `--oracle-mode script`
# runs free of claude_agent_sdk + sub-process render dependencies.

log = logging.getLogger(__name__)
_mesh_dedup_index: dict[tuple[int, str], str] = {}


# ---------------------------------------------------------------------------
# Loop-level logger
# ---------------------------------------------------------------------------

class LoopLogger:
    """Writes loop orchestration decisions to loop.log (separate from agent logs).

    Captures:
      - Config used for the run
      - Which skill version is active each round
      - Oracle per-check breakdown (full values for post-hoc human review)
      - Feedback being assembled and passed to each agent
      - Decisions (converged / escalate / refine)
      - Per-phase timing

    Not read by any agent — this is for human debugging only.
    """

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "w")
        self._round: int | None = None

    def _write(self, tag: str, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        prefix = f"[{ts}]"
        if self._round is not None:
            prefix += f" [round {self._round}]"
        prefix += f" [{tag}]"
        for line in str(text).splitlines() or [""]:
            self.f.write(f"{prefix} {line}\n")
        self.f.flush()

    def round(self, n: int) -> None:
        self._round = n

    def info(self, text: str) -> None:
        self._write("info", text)

    def config(self, config: Any) -> None:
        self._write("config", json.dumps(asdict(config), default=str, indent=2))

    def phase(self, name: str, elapsed: float | None = None) -> None:
        msg = name if elapsed is None else f"{name} ({elapsed:.1f}s)"
        self._write("phase", msg)

    def decision(self, text: str) -> None:
        self._write("decision", text)

    def oracle(self, result: OracleResult) -> None:
        self._write("oracle", f"score={result.score:.2f} passed={result.passed}")
        for c in result.checks:
            status = "PASS" if c.passed else "FAIL"
            self._write("oracle", f"  [{status}] {c.name}: {c.detail}")

    def feedback(self, feedback: dict[str, Any]) -> None:
        self._write("feedback", json.dumps(feedback, indent=2, default=str))

    def skill_version(self, skill_dir: Path) -> None:
        self._write("skill", f"active version: {skill_dir}")

    def error(self, text: str) -> None:
        self._write("error", text)

    def close(self) -> None:
        self.f.close()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Defaults come from evo/engine/paths.py — env-var driven, with the bundled
# `dataset/infinitest` sample as the out-of-the-box fallback.
DEFAULT_TASKS_ROOT = _resolved_data_root()
DEFAULT_GT_ROOT = _resolved_gt_root()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvoConfig:
    max_rounds: int = 5
    executor_model: str = DEFAULT_AGENT_MODEL
    generator_model: str = DEFAULT_AGENT_MODEL
    executor_backend: str = "responses"   # responses | claude | codex
    generator_backend: str = "responses"  # responses | claude | codex
    executor_max_turns: int = 60
    tasks_root: Path = DEFAULT_TASKS_ROOT     # sanitized inputs (executor sees)
    gt_root: Path = DEFAULT_GT_ROOT           # GT URDFs (oracle only)
    # Default starter: v_io's input/output contract plus a minimal frozen SDK
    # reference, so executors reach geometry through the SDK (no texture decode).
    skill_dir: Path = PROJECT_ROOT / "skills" / "v_io_sdk"
    output_root: Path = PROJECT_ROOT / "logs"
    seed: int = 42
    split: str | None = "train"

    # Parallel sim-worker execution (per-round, across tasks). Each worker
    # pins its own sim worker to port `sim_base_port + i`. parallel=1
    # preserves the original sequential behavior.
    parallel: int = 1
    headless: bool = False
    sim_base_port: int = 9876
    # Resident-memory cap per sim worker (MB); the worker interrupts a snippet
    # that crosses it and kills itself if the snippet cannot be interrupted.
    # Run 18 (164 GB) and run 23 (112 GB) each lost a machine to one worker.
    sim_worker_max_rss_mb: int = 6144

    # Per-round sampling. batch_size=0 disables sampling and runs the full
    # filtered task set every round (legacy behavior). batch_ratio is the
    # (easy, medium, hard) target counts within each batch — must sum to
    # batch_size; the sampler also prefers un-seen instances first, falling
    # back to already-seen if a difficulty bucket exhausts its pool.
    batch_size: int = 0
    batch_ratio: dict = field(
        default_factory=lambda: {"easy": 3, "medium": 3, "hard": 2}
    )
    # fixed_batch=True (with adaptive_train=False and batch_size>0) draws the
    # stratified batch once from `seed` and evaluates the SAME tasks every
    # round, so per-round pass rates compare the skill, not the sample.
    fixed_batch: bool = False

    # Adaptive Train sampling within each outer round. The feedback split is
    # evaluated in sequential tranches; after each tranche the controller
    # either adds another tranche or stops collecting Train evidence.
    # Validation is a separate evaluation-only run, never an inner-round stage.
    adaptive_train: bool = True
    train_increment: int = 10
    max_train_samples: int = 69
    objective_failure_stop_count: int = 3
    workflow_failure_stop_count: int = 2
    candidate_ab_test: bool = True
    candidate_ab_min_samples: int = 10
    max_contract_repair_attempts: int = 3
    # A candidate rejected before A/B (no journal, frozen contract edited,
    # preflight import/signature/smoke failure) costs only the Generator turn:
    # re-run the Generator in the same round with the reason, up to this many
    # extra times, before falling back to a contract-repair round.
    generator_repair_retries: int = 2
    # Number of independent Generator updates per round. Each update is one
    # self-contained change, A/B-tested on the round's assets against the
    # state left by the previous accepted update, so every change gets its
    # own verdict instead of riding along with the others.
    updates_per_round: int = 1
    # Safety ceiling on `try_skill` smoke tests per Generator turn (one real
    # executor session each, always on the first Train task of the batch; the
    # Generator is told to repeat until the test is clean). 0 disables the tool.
    generator_trials_per_turn: int = 10
    # Executor-authored Python/call mistakes are stochastic model failures,
    # not evidence that the Generator's helper implementation is broken.
    # Retry those samples immediately before classifying the final result.
    executor_code_retries: int = 1
    # Transient provider/renderer errors (connection blips, gateway 5xx, a lost
    # lightphyx page) retry the task in place with a short backoff before they
    # can count as a training-system failure. Applies in adaptive mode too —
    # the legacy retry_wait_min policy did not, so one blip aborted a run.
    transient_retries: int = 6
    # Independent executor sessions per task per evaluation. The executor is
    # the dominant noise source (the same skill on the same 20 tasks scored
    # 5/20 and 10/20 on two days); K sessions are aggregated into one result
    # per task: checks by strict majority, joint scores by mean, so the A/B
    # pairing and the Generator see K-fold less variance at K-fold cost.
    executor_reps: int = 1

    @property
    def mapping_path(self) -> Path:
        # Dataset-local metadata is authoritative.  The tracked mapping is a
        # compatibility fallback and can describe a different dataset build.
        dataset_mapping = self.tasks_root.parent / "mapping.json"
        if dataset_mapping.exists():
            return dataset_mapping
        return PROJECT_ROOT / "evo" / "data" / "mapping.json"

    def load_mapping(self) -> dict:
        """Load task_id ↔ category mapping."""
        if self.mapping_path.exists():
            return json.loads(self.mapping_path.read_text())
        return {}

    @property
    def splits_path(self) -> Path:
        """Return the selected split manifest next to tasks/ + gt/.

        ``EVO_SPLITS_PATH`` is an explicit override. ``splits.json`` is the
        canonical manifest for ArtVIP Assembly 115. A legacy frozen 50/30/50
        manifest is only used when the canonical file is absent.
        """
        override = os.environ.get("EVO_SPLITS_PATH")
        if override:
            return Path(override).expanduser()
        canonical = self.tasks_root.parent / "splits.json"
        if canonical.exists():
            return canonical
        return self.tasks_root.parent / "splits_experiment_50_30_50_seed42.json"

    def _read_split_rows(self) -> list[dict]:
        """Read and validate that the manifest belongs to this dataset root."""
        if not self.splits_path.exists():
            return []
        try:
            rows = json.loads(self.splits_path.read_text())
        except Exception as e:
            log.warning(f"Malformed splits file {self.splits_path}: {e}")
            return []
        if not isinstance(rows, list):
            log.warning(f"Malformed splits file {self.splits_path}: expected a row list")
            return []
        discovered = {
            (task_dir.name, instance_dir.name)
            for task_dir in self.tasks_root.glob("task_*") if task_dir.is_dir()
            for instance_dir in task_dir.iterdir() if instance_dir.is_dir()
        }
        manifest = {
            (str(row.get("task_id")), str(row.get("instance")))
            for row in rows if isinstance(row, dict)
            and "task_id" in row and "instance" in row
        }
        if discovered and manifest != discovered:
            missing = sorted(discovered - manifest)[:5]
            extra = sorted(manifest - discovered)[:5]
            raise ValueError(
                "split manifest does not match tasks_root; refusing a silent "
                f"cross-dataset split: splits={self.splits_path}, "
                f"tasks_root={self.tasks_root}, discovered={len(discovered)}, "
                f"manifest={len(manifest)}, missing={missing}, extra={extra}"
            )
        return rows

    def load_splits(self) -> dict[tuple[str, str], str]:
        """Return {(task_id, instance) -> split_name} or empty dict.

        Returns {} if splits.json doesn't exist or is malformed (malformed
        is loudly warned — callers then report "no splits found", which
        would otherwise be misleading).
        """
        rows = self._read_split_rows()
        return {
            (r["task_id"], r["instance"]): r["split"]
            for r in rows
            if "task_id" in r and "instance" in r and "split" in r
        }

    def load_split_metadata(self) -> dict[tuple[str, str], dict]:
        """Return {(task_id, instance) -> full row dict from splits.json}.

        Includes split, difficulty, group, challenges, joint counts, etc.
        Returns {} if splits.json doesn't exist or is malformed.
        """
        rows = self._read_split_rows()
        return {
            (r["task_id"], r["instance"]): r
            for r in rows if "task_id" in r and "instance" in r
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvoConfig:
        with open(path) as f:
            try:
                raw = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise SystemExit(f"Malformed config YAML {path}: {e}") from e
        evo = raw.get("evolution", {})
        agents = raw.get("agents", {})
        data = raw.get("data", {})
        paths = raw.get("paths", {})
        execution = raw.get("execution", {})
        return cls(
            max_rounds=evo.get("max_rounds", 5),
            executor_model=agents.get("executor_model", DEFAULT_AGENT_MODEL),
            generator_model=agents.get("generator_model", DEFAULT_AGENT_MODEL),
            executor_backend=agents.get("executor_backend", "responses"),
            generator_backend=agents.get("generator_backend", "responses"),
            executor_max_turns=evo.get("executor_max_turns", 60),
            tasks_root=_resolve_path(
                data.get("tasks_root", str(DEFAULT_TASKS_ROOT)),
                fallback=DEFAULT_TASKS_ROOT,
            ),
            gt_root=_resolve_path(
                data.get("gt_root", str(DEFAULT_GT_ROOT)),
                fallback=DEFAULT_GT_ROOT,
            ),
            skill_dir=Path(paths.get("skill_dir", str(cls.skill_dir))),
            output_root=Path(paths.get("output_root", str(cls.output_root))),
            seed=evo.get("seed", 42),
            split=data.get("splits", "train"),
            parallel=int(execution.get("max_parallel", 1)),
            # YAML keys kept for config compat (legacy Blender-era names).
            headless=execution.get("blender_mode", "mcp") == "headless",
            sim_base_port=int(execution.get("mcp_port", 9876)),
            sim_worker_max_rss_mb=int(execution.get("sim_worker_max_rss_mb", 6144)),
            batch_size=int(evo.get("batch_size", 0)),
            batch_ratio=dict(evo.get("batch_ratio",
                                      {"easy": 3, "medium": 3, "hard": 2})),
            fixed_batch=bool(evo.get("fixed_batch", False)),
            adaptive_train=bool(evo.get("adaptive_train", True)),
            train_increment=int(evo.get("train_increment", 10)),
            max_train_samples=int(evo.get("max_train_samples", 69)),
            objective_failure_stop_count=int(
                evo.get("objective_failure_stop_count", 3)
            ),
            workflow_failure_stop_count=int(
                evo.get("workflow_failure_stop_count", 2)
            ),
            candidate_ab_test=bool(evo.get("candidate_ab_test", True)),
            candidate_ab_min_samples=int(
                evo.get("candidate_ab_min_samples", 10)
            ),
            max_contract_repair_attempts=int(
                evo.get("max_contract_repair_attempts", 3)
            ),
            generator_repair_retries=int(
                evo.get("generator_repair_retries", 2)
            ),
            updates_per_round=int(evo.get("updates_per_round", 1)),
            generator_trials_per_turn=int(evo.get("generator_trials_per_turn", 10)),
            executor_code_retries=int(evo.get("executor_code_retries", 1)),
            transient_retries=int(evo.get("transient_retries", 6)),
            executor_reps=int(evo.get("executor_reps", 1)),
        )


# ---------------------------------------------------------------------------
# Task discovery from sanitized dataset
# ---------------------------------------------------------------------------

def discover_tasks(config: EvoConfig) -> list[dict[str, str]]:
    """Discover all valid tasks from the sanitized dataset.

    Returns list of {"task_id": "task_01", "instance": "0"}.
    """
    tasks = []
    tasks_root = config.tasks_root
    if not tasks_root.exists():
        return tasks

    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        for inst_dir in sorted(task_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            # Verify this instance has parts
            parts_dir = inst_dir / "parts"
            if parts_dir.exists() and any(parts_dir.glob("*.obj")):
                # Verify GT exists
                gt_dir = config.gt_root / task_id / inst_dir.name
                if gt_dir.exists() and (gt_dir / "scene.urdf").exists():
                    tasks.append({"task_id": task_id, "instance": inst_dir.name})

    return tasks


# ---------------------------------------------------------------------------
# Single-task evolution loop (overfit test)
# ---------------------------------------------------------------------------

async def evolve_single_task(
    task_id: str,
    instance: str,
    config: EvoConfig,
    verbose: bool = True,
    resume_dir: Path | None = None,
    executor_effort: str | None = None,
    generator_effort: str | None = None,
    subagent_model: str | None = None,
    subagent_effort: str | None = None,
    generator_auth: str | None = None,
    oracle_mode: str = "subagent",
    oracle_effort: str = DEFAULT_AGENT_EFFORT,
    oracle_votes: int = 3,
    oracle_model: str | None = None,
    regression_tol: float = 0.05,
    skip_generator: bool = False,
) -> None:
    """Run the full co-evolution loop on a single task.

    This is the overfit test: can the system improve the skill until
    it passes the oracle on ONE specific task?

    Args:
        resume_dir: optional run dir to resume from (loads loop_state.json).
    """
    # Resolve paths: executor sees tasks/, oracle sees gt/
    input_dir = config.tasks_root / task_id / instance / "parts"
    gt_dir = config.gt_root / task_id / instance

    if not input_dir.exists():
        log.error(f"Input dir not found: {input_dir}")
        return
    if not gt_dir.exists():
        log.error(f"GT dir not found: {gt_dir}")
        return

    # Resolve category from mapping (for display only, not passed to executor)
    mapping = config.load_mapping()
    category = mapping.get("task_to_category", {}).get(task_id, "unknown")

    start_iteration = 0
    current_skill = config.skill_dir
    feedback_history: list[dict[str, Any]] = []
    best_score = 0.0
    if resume_dir is not None:
        state = _load_loop_state(resume_dir)
        if state is None:
            raise SystemExit(
                f"--resume {resume_dir} requested but {LOOP_STATE_FILENAME} "
                f"is missing or unreadable."
            )
        run_dir = resume_dir
        start_iteration = int(state["round_completed"]) + 1
        current_skill = Path(state["current_skill_dir"])
        feedback_history = state.get("feedback_history", []) or []
        best_score = float(
            state.get("best_fitness", state.get("best_pass_rate", 0.0))
        )
        if start_iteration >= config.max_rounds:
            print(f"Resume: already completed all {config.max_rounds} rounds; "
                  f"nothing to do.")
            return
        print(f"Resuming run at {run_dir} from round {start_iteration+1}")
    else:
        run_dir = config.output_root / "runs" / f"{task_id}_{instance}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
    active_preflight = preflight_skill(current_skill)
    (run_dir / "active_skill_preflight.json").write_text(
        json.dumps(active_preflight.to_dict(), indent=2)
    )
    if not active_preflight.ok:
        raise SkillContractError(
            "active skill failed model-free contract preflight before Train: "
            + "; ".join(active_preflight.errors[:3])
        )
    log_path = run_dir / "evolution.jsonl"
    gen_add_dirs = _generator_add_dirs(run_dir, config)

    # Loop-level log (orchestration decisions, full oracle values, feedback)
    lg = LoopLogger(run_dir / "loop.log")
    lg.info(f"Task: {task_id}/{instance} (category={category})")
    lg.info(f"Input: {input_dir}")
    lg.info(f"GT: {gt_dir}")
    lg.info(f"Run dir: {run_dir}")
    lg.config(config)

    print(f"{'='*60}")
    print(f"Single-task overfit test")
    print(f"{'='*60}")
    print(f"Task:     {task_id}/{instance} (category: {category})")
    print(f"Input:    {input_dir}")
    print(f"GT:       {gt_dir}")
    print(f"Skill:    {current_skill}")
    print(f"Rounds:   {config.max_rounds}")
    print(f"Output:   {run_dir}")
    print(f"{'='*60}\n")

    try:
      for iteration in range(start_iteration, config.max_rounds):
        lg.round(iteration)
        print(f"\n{'─'*40}")
        print(f"  Round {iteration + 1}/{config.max_rounds}")
        print(f"{'─'*40}\n")

        # Fresh browsers each round: Chromium's GPU memory only grows within
        # a server's lifetime, and nothing renders at a round boundary.
        if iteration > start_iteration and recycle_shared_pool(f"round {iteration + 1}"):
            lg.info("LightPhyx pool recycled (fresh browsers for this round)")

        # 1. Execute skill on task
        round_dir = run_dir / f"round_{iteration}"
        round_dir.mkdir(parents=True, exist_ok=True)
        output_dir = round_dir / "output"
        # Ensure executor loads the latest skill version
        _install_skill(current_skill)
        lg.skill_version(current_skill)
        lg.phase("executor start")
        print(f"[1/3] Executing skill...")
        # ArtVIP Assembly 115 publishes the solver-visible reference as
        # thumbnail.png. Keep render_thumbnail.png as a legacy fallback.
        exec_thumbnail = input_dir.parent / "thumbnail.png"
        if not exec_thumbnail.exists():
            exec_thumbnail = input_dir.parent / "render_thumbnail.png"
        if not exec_thumbnail.exists():
            exec_thumbnail = (
                config.tasks_root.parent / "gt_renders" / task_id / instance
                / "render_assembled.png"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        if exec_thumbnail.exists():
            try:
                import shutil as _shutil
                _shutil.copy2(exec_thumbnail, output_dir / "render_thumbnail.png")
            except OSError:
                pass
        exec_result = await run_executor(
            factory=task_id,       # anonymized name (executor never sees real category)
            instance=instance,
            input_dir=input_dir,
            output_dir=output_dir,
            skill_dir=current_skill,
            thumbnail_path=exec_thumbnail if exec_thumbnail.exists() else None,
            model=config.executor_model,
            max_turns=config.executor_max_turns,
            log_file=round_dir / "executor.log",
            sim_port=config.sim_base_port,
            sim_headless=config.headless,
            effort=executor_effort,
            subagent_model=subagent_model,
            subagent_effort=subagent_effort,
            backend=config.executor_backend,
        )

        lg.phase("executor done", elapsed=exec_result.duration)

        if not exec_result.success:
            print(f"  FAILED: {exec_result.error}")
            lg.error(f"executor failed: {exec_result.error}")
            if _is_training_system_error(exec_result.error):
                failure_path = _write_training_system_failure(
                    round_dir,
                    stage="train",
                    error=exec_result.error or "unknown training-system error",
                )
                lg.decision(
                    "training-system failure → abort before Oracle/Generator"
                )
                raise TrainingSystemError(
                    "Train hit a training-system failure; Oracle and Generator "
                    "were not started. Fix the runtime and rerun the current "
                    f"round. Details: {failure_path}"
                )
            lg.decision("executor failed → skip oracle/verifier, go straight to generator")
            feedback_history.append({
                "round": iteration,
                "pass_rate": 0.0,
                "n_failures": 1,
                "diagnostics": f"Skill execution failed: {exec_result.error}",
                "suggested_fixes": "Fix execution errors before improving assembly logic.",
            })
            lg.feedback(feedback_history[-1])
            if skip_generator:
                print("\n[3/3] Generator stage SKIPPED (--no-generator).")
                lg.decision("generator skipped (--no-generator)")
                _log_iteration(log_path, iteration, "execution_fail", task_id, instance,
                               exec_result, None, None, 0.0)
                continue
            # Still generate improved skill from failure feedback
            print(f"\n[3/3] Generating improved skill...")
            lg.phase("generator start (on execution failure)")
            new_skill_dir = round_dir / "skill"
            skill_content = _read_skill_content(current_skill)
            gen_model = config.generator_model
            if config.generator_backend == "claude" and generator_auth == "local":
                gen_ctx = _auth_swap("local")
                gen_model = "claude-opus-4-7"
            else:
                gen_ctx = contextlib.nullcontext()
            with gen_ctx:
                gen_result = await run_generator(
                    current_skill_dir=current_skill,
                    current_skill_content=skill_content,
                    feedback_history=feedback_history,
                    output_skill_dir=new_skill_dir,
                    model=gen_model,
                    log_file=round_dir / "generator.log",
                    effort=generator_effort,
                    sim_port=config.sim_base_port + config.parallel,
                    sim_headless=config.headless,
                    backend=config.generator_backend,
                    add_dirs=gen_add_dirs,
                    dataset_tasks_root=config.tasks_root,
                )
            ok, reason = _validate_generator_output(
                gen_result,
                round_dir / "generator_journal.md",
                current_skill_content=skill_content,
            )
            if ok:
                candidate_preflight = preflight_skill(new_skill_dir)
                (round_dir / "candidate_skill_preflight.json").write_text(
                    json.dumps(candidate_preflight.to_dict(), indent=2)
                )
                if not candidate_preflight.ok:
                    ok = False
                    reason = (
                        "candidate skill preflight failed: "
                        + "; ".join(candidate_preflight.errors[:3])
                    )
            if ok:
                current_skill = new_skill_dir
            else:
                lg.decision(f"generator output rejected: {reason}")
                print(f"  Generator output REJECTED ({reason}); keeping skill")
            lg.phase("generator done")
            lg.info(f"generator modified files: {gen_result.files_modified}")
            print(f"  Summary: {gen_result.changes_summary[:200]}")
            _log_iteration(log_path, iteration, "execution_fail", task_id, instance,
                           exec_result, None, gen_result, 0.0)
            continue

        print(f"  OK ({exec_result.duration:.0f}s), URDF: {exec_result.urdf_path}")

        # 2. Run oracle (compares against GT — executor never sees this)
        #    Oracle results written to isolated file, never fed back to agents
        print(f"\n[2/3] Oracle check...")
        lg.phase("oracle start")
        oracle_start = time.time()
        oracle_result = await _run_oracle_dispatch(
            mode=oracle_mode,
            tag=f"{task_id}/{instance}",
            gt_dir=gt_dir,
            pred_dir=output_dir,
            tasks_root=config.tasks_root,
            effort=oracle_effort,
            votes=oracle_votes,
            oracle_model=oracle_model,
        )
        score = oracle_result.score
        lg.phase("oracle done", elapsed=time.time() - oracle_start)
        lg.oracle(oracle_result)
        # Generator-reachable copy is sanitized (GT values redacted); the
        # full dict goes to the run's private sibling for human review.
        full = oracle_result.to_dict()
        (round_dir / "oracle.json").write_text(
            json.dumps(sanitize_oracle_dict(full), indent=2)
        )
        try:
            private = _private_dir(run_dir) / f"round_{iteration}"
            private.mkdir(parents=True, exist_ok=True)
            (private / "oracle.json").write_text(json.dumps(full, indent=2))
        except OSError:
            pass

        print(f"  {'PASS' if oracle_result.passed else 'FAIL'} (score={score:.2f})")
        if verbose:
            for c in oracle_result.checks:
                status = "PASS" if c.passed else "FAIL"
                print(f"    [{status}] {c.name}: {c.detail}")

        if oracle_result.passed:
            print(f"\n*** CONVERGED at round {iteration + 1}! ***")
            lg.decision(f"CONVERGED at round {iteration + 1} (oracle passed)")
            _save_best_skill(current_skill, run_dir / "best_skill")
            _log_iteration(log_path, iteration, "converged", task_id, instance,
                           exec_result, oracle_result, None, score)
            lg.close()
            return

        # Track best + regression gate (same policy as multi-task mode).
        best_dir = run_dir / "best_skill"
        if score > best_score + 1e-9:
            best_score = score
            lg.decision(f"new best score {score:.2f} → save checkpoint")
            _save_best_skill(current_skill, best_dir)
        elif (
            regression_tol >= 0
            and best_dir.is_dir()
            and score < best_score - regression_tol
            and Path(current_skill).resolve() != best_dir.resolve()
        ):
            lg.decision(
                f"regression gate: score {score:.2f} < best {best_score:.2f} "
                f"− tol {regression_tol:.2f} → fork next generator from best"
            )
            print(f"  REGRESSION GATE: reverting to best checkpoint "
                  f"({best_score:.2f})")
            current_skill = best_dir

        # 3. Build feedback artifact directly from oracle + executor outputs.
        #    The verifier role is dropped — the generator now reads the
        #    predicted URDF and a filtered executor.log itself, instead of
        #    digesting verifier diagnostics.  GT URDF and oracle CHECK
        #    VALUES are still excluded; only check NAMES leak through.
        run_artifact = _build_run_artifact(
            tag=f"{task_id}/{instance}",
            er=exec_result,
            orc=oracle_result,
            executor_log_path=round_dir / "executor.log",
            output_dir=output_dir,
            input_dir=input_dir,
        )
        failed_checks = [
            c.name for c in oracle_result.checks
            if c.scored and not c.passed
        ]
        # In single-task mode each check is binary on the one task.
        per_check_pass = {
            c.name: f"{1 if c.passed else 0}/1"
            for c in oracle_result.checks
        }
        new_feedback = {
            "round": iteration,
            "pass_rate": score,
            "n_failures": len(failed_checks),
            "failed_oracle_checks": failed_checks,
            "per_check_pass_rate": per_check_pass,
            "runs": [run_artifact],
        }
        feedback_history.append(new_feedback)
        lg.feedback(new_feedback)

        # 4. Generate improved skill
        #    loop.py reads skill content, passes as dict — generator writes to isolated dir
        if skip_generator:
            print("\n[3/3] Generator stage SKIPPED (--no-generator).")
            lg.decision("generator skipped (--no-generator)")
            _log_iteration(log_path, iteration, "no_generator", task_id, instance,
                           exec_result, oracle_result, None, score)
        else:
            print(f"\n[3/3] Generating improved skill...")
            lg.phase("generator start")
            gen_start = time.time()
            new_skill_dir = round_dir / "skill"
            skill_content = _read_skill_content(current_skill)
            gen_model = config.generator_model
            if config.generator_backend == "claude" and generator_auth == "local":
                gen_ctx = _auth_swap("local")
                gen_model = "claude-opus-4-7"
            else:
                gen_ctx = contextlib.nullcontext()
            with gen_ctx:
                gen_result = await run_generator(
                    current_skill_dir=current_skill,
                    current_skill_content=skill_content,
                    feedback_history=feedback_history,
                    output_skill_dir=new_skill_dir,
                    model=gen_model,
                    log_file=round_dir / "generator.log",
                    effort=generator_effort,
                    sim_port=config.sim_base_port + config.parallel,
                    sim_headless=config.headless,
                    backend=config.generator_backend,
                    add_dirs=gen_add_dirs,
                    dataset_tasks_root=config.tasks_root,
                )
            lg.phase("generator done", elapsed=time.time() - gen_start)
            lg.info(f"generator modified files: {gen_result.files_modified}")

            ok, reason = _validate_generator_output(
                gen_result,
                round_dir / "generator_journal.md",
                current_skill_content=skill_content,
            )
            if ok:
                candidate_preflight = preflight_skill(new_skill_dir)
                (round_dir / "candidate_skill_preflight.json").write_text(
                    json.dumps(candidate_preflight.to_dict(), indent=2)
                )
                if not candidate_preflight.ok:
                    ok = False
                    reason = (
                        "candidate skill preflight failed: "
                        + "; ".join(candidate_preflight.errors[:3])
                    )
            if ok:
                current_skill = new_skill_dir
            else:
                lg.decision(f"generator output rejected: {reason}")
                print(f"  Generator output REJECTED ({reason}); keeping skill")
            print(f"  Modified: {gen_result.files_modified}")
            print(f"  Summary: {gen_result.changes_summary[:300]}")

            _log_iteration(log_path, iteration, "surrogate", task_id, instance,
                           exec_result, oracle_result, gen_result, score)

        # Round complete — checkpoint state for resume.
        _save_loop_state(
            run_dir,
            round_completed=iteration,
            current_skill_dir=current_skill,
            feedback_history=feedback_history,
            best_fitness=best_score,
            best_skill_dir=run_dir / "best_skill" if (run_dir / "best_skill").exists() else None,
            seen_by_diff=None,
        )

    except SimStartupError as e:
        print(f"\n*** FATAL: sim worker failed to start: {e} ***")
        lg.error(f"SimStartupError: {e}")
        lg.decision("FATAL: sim worker startup failed. Aborting run.")
        lg.close()
        sys.exit(1)

    print(f"\nMax rounds reached. Best score: {best_score:.2f}")
    lg.decision(f"Max rounds reached. Best score: {best_score:.2f}")
    lg.close()


# ---------------------------------------------------------------------------
# Multi-task evolution loop
# ---------------------------------------------------------------------------

def _fixed_round_batch(
    pool: list[dict[str, str]],
    metadata: dict[tuple[str, str], dict],
    batch_size: int,
    ratio: dict[str, int],
    seed: int,
) -> list[dict[str, str]]:
    """The stratified batch drawn from a fresh RNG(seed) with nothing marked
    seen: identical on every call, hence identical every round and on resume."""
    return _sample_batch(
        pool=pool, metadata=metadata, batch_size=batch_size, ratio=ratio,
        seen={}, rng=random.Random(seed),
    )


def _entry_difficulty(row: dict) -> str:
    """Difficulty bucket for one splits.json row.

    The ArtVIP splits rows don't carry an explicit `difficulty` field, so
    without a fallback every entry lands in "medium" and the stratified
    sampler silently shrinks each batch to the medium quota. Derive a
    bucket from joint count when the field is absent.
    """
    diff = row.get("difficulty")
    if diff:
        return str(diff)
    n_joints = row.get("n_joints")
    if isinstance(n_joints, (int, float)):
        if n_joints <= 1:
            return "easy"
        if n_joints <= 3:
            return "medium"
        return "hard"
    return "medium"


def _sample_batch(
    pool: list[dict[str, str]],
    metadata: dict[tuple[str, str], dict],
    batch_size: int,
    ratio: dict[str, int],
    seen: dict[str, set[tuple[str, str]]],
    rng: random.Random,
) -> list[dict[str, str]]:
    """Stratified-by-difficulty batch sampler that prefers un-seen instances.

    `pool`     — full filtered task pool (e.g. train split).
    `metadata` — splits.json row keyed by (task_id, instance), used to read
                 the per-instance `difficulty`.
    `ratio`    — target counts per difficulty bucket (e.g. {easy:3, med:3, hard:2}).
                 Sum should equal batch_size; if not, total is whatever the
                 ratio implies.
    `seen`     — mutable {difficulty: set of (tid, inst) pairs already drawn}.
                 Updated in place.

    Strategy per bucket:
      1. Take all `pool` rows with that difficulty.
      2. Partition into un-seen (not in `seen[diff]`) and seen.
      3. Shuffle each partition.
      4. Drain unseen first, then dip into seen if quota not yet met.
      5. Add the picks to `seen[diff]`.
    """
    by_diff: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in pool:
        row = metadata.get((entry["task_id"], entry["instance"]), {})
        by_diff[_entry_difficulty(row)].append(entry)

    chosen: list[dict[str, str]] = []
    for diff, n_target in ratio.items():
        if n_target <= 0:
            continue
        bucket = by_diff.get(diff, [])
        seen_set = seen.setdefault(diff, set())
        unseen = [b for b in bucket if (b["task_id"], b["instance"]) not in seen_set]
        already = [b for b in bucket if (b["task_id"], b["instance"]) in seen_set]
        rng.shuffle(unseen); rng.shuffle(already)
        picks = unseen[:n_target]
        if len(picks) < n_target:
            picks.extend(already[:n_target - len(picks)])
        for p in picks:
            seen_set.add((p["task_id"], p["instance"]))
        chosen.extend(picks)

    return chosen


# ---------------------------------------------------------------------------
# Adaptive Train sampling (within one outer round)
# ---------------------------------------------------------------------------

_BLOCKING_CALL_ERROR_PATTERNS = (
    "api error", "api_error", "responses error", "response error",
    "error code: 400", "error code: 401", "error code: 403",
    "error code: 404", "error code: 408", "error code: 409",
    "error code: 422", "error code: 429", "error code: 500",
    "error code: 502", "error code: 503", "error code: 504",
    "http 400", "http 401", "http 403", "http 404", "http 408",
    "http 409", "http 422", "http 429", "http 500", "http 502",
    "http 503", "http 504", "rate limit", "rate_limit", "quota",
    "bad request", "unauthorized", "forbidden", "service unavailable",
    "internal server error", "gateway timeout", "mcp error", "mcp_error",
    "tool error", "tool_error", "tool call error", "tool_call_error",
    "error running tool", "error invoking tool", "tool execution failed",
    "unknown tool",
    "socket error", "connection refused", "connection reset",
    "connection closed", "connection error", "connection timeout",
    "sim error", "sim worker failed", "sim worker exited",
    "sim worker socket", "failed to start sim worker", "not ready after",
    "timed out", "timeout error", "request timeout",
    "up_bad_request", "up_error", "上游错误",
)


@dataclass(frozen=True)
class AdaptiveTrainDecision:
    """One ADD_10 / STOP_ADDING decision for a Train tranche."""

    action: str
    reason: str
    cumulative_train: int
    objective_failures: dict[str, int] = field(default_factory=dict)
    workflow_failures: dict[str, int] = field(default_factory=dict)
    blocking_error: str | None = None


class TrainingSystemError(RuntimeError):
    """A runtime/API invariant failed; formal training must not continue."""


class SkillContractError(RuntimeError):
    """An evolvable skill repeatedly failed its executable contract gate."""


_TRAINING_SYSTEM_ERROR_PATTERNS = (
    "api error", "api_error", "responses error", "response error",
    "emptyresponsesoutputerror", "output=null", "modelresponse",
    "context_too_large", "exceeds the context window",
    "assembly_sdk[sim_runtime]", "assembly_sdk[io_failure]",
    "rpcerror", "rpc error", "page disconnected", "failed to fetch",
    "browser has been closed", "browser closed", "permissionerror",
    "permission error",
    "connection error", "connection refused", "connection reset",
    "connection closed", "connection timeout", "socket error",
    "sim worker exited", "sim worker socket", "not ready after",
    "rate limit", "rate_limit", "quota", "up_bad_request", "up_error",
    "上游错误",
)


def _is_training_system_error(error: str | None) -> bool:
    """Separate runtime/API defects from assembly-skill quality defects."""
    text = (error or "").lower()
    if any(pattern in text for pattern in _TRAINING_SYSTEM_ERROR_PATTERNS):
        return True
    if re.search(r"name ['\"]?(?:skill_dir|output_dir)['\"]? is not defined", text):
        return True
    if re.search(r"(?:skill_dir|output_dir) is not defined", text):
        return True
    if re.search(
        r"keyerror:\s*['\"](?:skill_dir|input_dir|output_dir)['\"]",
        text,
    ):
        return True
    if (
        "filenotfounderror" in text
        and re.search(r"[/\\]tools[/\\][^\n]+\.py", text)
    ):
        # Missing executor helper source means the runtime globals/snapshot are
        # broken. It is not evidence that the assembly strategy needs edits.
        return True
    if "runtime-owned" in text and any(
        name in text for name in ("skill_dir", "input_dir", "output_dir")
    ):
        return True
    if "output directory is unset" in text:
        return True
    if re.search(
        r"(?:http|error code|status(?:_code)?)\s*[:=]?\s*['\"]?[45]\d\d\b",
        text,
    ):
        return True
    return False


_EXECUTION_EXCEPTION_RE = re.compile(
    r"\b(?:syntaxerror|indentationerror|taberror|typeerror|valueerror|"
    r"attributeerror|indexerror|keyerror|nameerror|assertionerror|"
    r"importerror|modulenotfounderror|oserror|runtimeerror|"
    r"zerodivisionerror|memoryerror)\b"
)


def _has_candidate_helper_frame(text: str) -> bool:
    """Whether a traceback entered Generator-authored helper source."""
    return any(marker in text for marker in (
        "/tools/",
        "\\tools\\",
        'file "<string>"',
    ))


def _should_retry_executor_code(
    executor_result: ExecutorResult, blocking_error: str | None,
    attempt: int, budget: int,
) -> bool:
    """Re-run a session for an Executor-authored code slip only when the slip
    actually cost the session its output.

    `_find_blocking_call_error` reports recovered errors too. Runs 11-12
    (2026-09-25/29) discarded 53 sessions for a mid-session SyntaxError /
    IndentationError / empty-array ValueError that the executor had already
    fixed in its next call: every one of them had exported a URDF and 22 had
    passed the oracle. Re-running those replaced a valid measurement with a
    fresh sample (~17% of sessions at gpt-6-astra). A session that produced no
    URDF after such a slip is still retried once.
    """
    if attempt >= budget or executor_result.success:
        return False
    return _is_executor_code_error(blocking_error)


def _is_executor_code_error(error: str | None) -> bool:
    """Identify stochastic Python/call mistakes authored by the Executor.

    The sim worker parses Executor-authored snippets as ``<unknown>`` and
    executes them as ``<agent_code>``. A traceback that subsequently enters
    ``<string>`` or ``tools/`` belongs to an evolved helper instead and is
    handled by :func:`_is_candidate_tool_contract_error`.
    """
    text = (error or "").lower()
    if not text or _is_training_system_error(text):
        return False

    helper_frame = _has_candidate_helper_frame(text)
    syntax_error = re.search(
        r"\b(?:syntaxerror|indentationerror|taberror)\b", text
    )
    if syntax_error and not helper_frame and any(marker in text for marker in (
        'file "<unknown>"',
        'file "<agent_code>"',
        "ast.parse(code",
        "sim error",
    )):
        return True

    # Python reports signature binding failures at the call site without
    # entering the called helper. They are therefore Executor call mistakes
    # unless the traceback was already executing another evolved helper.
    signature_error = re.search(
        r"missing \d+ required|missing a required|unexpected keyword|"
        r"required positional argument|required keyword-only argument|"
        r"got multiple values for argument|takes \d+ positional arguments?",
        text,
    )
    if signature_error and not helper_frame:
        return True

    invalid_call_argument = re.search(
        r"invalid (?:argument|keyword)|unknown keyword|must be one of",
        text,
    )
    if invalid_call_argument and not helper_frame:
        return True

    if (
        "oserror: could not get source code" in text
        and "inspect.py" in text
        and not helper_frame
    ):
        return True

    # Attribute/index/name mistakes that fail directly in the authored
    # snippet are also retryable. Do not steal exceptions raised from helper
    # frames; those are deterministic candidate-tool defects.
    return bool(
        _EXECUTION_EXCEPTION_RE.search(text)
        and 'file "<agent_code>"' in text
        and not helper_frame
    )


def _is_candidate_tool_contract_error(error: str | None) -> bool:
    """Identify deterministic failures inside Generator-authored helpers."""
    text = (error or "").lower()
    if (
        not text
        or _is_training_system_error(text)
        or _is_executor_code_error(text)
    ):
        return False
    exception = _EXECUTION_EXCEPTION_RE.search(text)
    if not exception:
        return False
    if _has_candidate_helper_frame(text):
        return True
    # Legacy executor adapters sometimes returned only the terminal exception
    # line. Preserve contract feedback for those ambiguous records, while the
    # strongly identifiable syntax/signature cases above remain retryable.
    return "sim error" in text


def _is_skill_contract_error(error: str | None) -> bool:
    """Backward-compatible alias for candidate tool contract failures."""
    return _is_candidate_tool_contract_error(error)


def _execution_error_kind(error: str | None) -> str | None:
    """Return the mutually exclusive controller class for one call error."""
    if not error:
        return None
    if _is_training_system_error(error):
        return "training_system_error"
    if _is_executor_code_error(error):
        return "executor_code_error"
    if _is_candidate_tool_contract_error(error):
        return "candidate_tool_contract_error"
    return "blocking_call_error"


def _write_training_system_failure(
    round_dir: Path,
    *,
    stage: str,
    error: str,
) -> Path:
    """Persist a machine-readable failure before aborting the experiment."""
    path = round_dir / "training_system_failure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": stage,
        "error": error,
        "action": "ABORT_EXPERIMENT",
        "validation_started": False if stage == "train" else None,
        "requires_system_fix": True,
    }, indent=2, ensure_ascii=False))
    return path


def _normalize_workflow_failure(error: str | None) -> str:
    """Normalize one executor failure into a stable comparison signature."""
    text = (error or "unknown workflow failure").strip().lower()
    text = re.sub(r"/[^\s:]+", "<path>", text)
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", text)
    return re.sub(r"\s+", " ", text)[:240]


def _responses_tool_output_texts(value: Any) -> list[str]:
    """Extract only textual Responses tool output that can contain errors.

    Function tool outputs may contain ``input_image`` entries with complete
    data URLs. Binary/base64 data is opaque and must never be fed to keyword
    error classifiers: arbitrary encodings can accidentally contain strings
    such as ``quota``. The Agents SDK formats text tool results as strings or
    as ``input_text``/``text`` entries; MCP results can also wrap those values
    in ``error``, ``output``, ``content``, or ``message`` fields.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        texts: list[str] = []
        for entry in value:
            texts.extend(_responses_tool_output_texts(entry))
        return texts
    if not isinstance(value, dict):
        return []

    output_type = str(value.get("type", "")).lower()
    if output_type in {"input_image", "image", "input_file", "file"}:
        return []
    if output_type in {"input_text", "text"}:
        text = value.get("text")
        return [text] if isinstance(text, str) else []

    texts = []
    for key in ("error", "output", "content", "message"):
        if key in value:
            texts.extend(_responses_tool_output_texts(value[key]))
    return texts


def _find_blocking_call_error(
    executor_result: ExecutorResult,
    output_dir: Path,
) -> str | None:
    """Return a blocking API/tool/MCP/sim-worker error, including recovered ones.

    A skill can recover far enough to emit a URDF after a bad tool call. Inspect
    the structured result, explicitly tagged log records, and Responses tool
    outputs so the controller can distinguish runtime/API failures (abort the
    experiment) from assembly-script tool failures (Skill feedback). Prompts
    and assistant prose are deliberately excluded to avoid treating a mere
    discussion of an error as an actual call failure.
    """
    candidates: list[str] = []
    if executor_result.error:
        candidates.append(executor_result.error)

    log_path = output_dir / "executor.log"
    if log_path.exists():
        try:
            for line in log_path.read_text(errors="replace").splitlines():
                # These are records emitted by the executor, not free-form
                # assistant text or the initial prompt.
                if re.search(
                    r"\[(?:api_error|error|tool_error|tool_result|sim)\]",
                    line,
                    flags=re.IGNORECASE,
                ):
                    candidates.append(line)
        except OSError:
            pass

    responses_path = output_dir / "executor.responses.json"
    if responses_path.exists():
        try:
            payload = json.loads(responses_path.read_text(errors="replace"))
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", "")).lower()
                # The Agents SDK serializes actual tool results as call output
                # items. Restricting the scan to them avoids matching tool-call
                # arguments, system instructions, or assistant narration.
                if "output" not in item_type and item_type != "mcp_call":
                    continue
                tool_output = item.get("error")
                if tool_output is None:
                    tool_output = item.get("output")
                candidates.extend(_responses_tool_output_texts(tool_output))
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    for candidate in candidates:
        lowered = candidate.lower()
        # System failures are a strict superset of the generic call-pattern
        # list below. Check them first so legacy SDK messages such as
        # "ModelResponse.output ... input_value=None" and bare NameError
        # runtime-global failures abort before held-out evaluation starts.
        if _is_training_system_error(candidate):
            return candidate.strip()
        has_error_status = re.search(
            r"(?:http|error code|status(?:_code)?)\s*[:=]?\s*['\"]?[45]\d\d\b",
            lowered,
        )
        if has_error_status or any(
            pattern in lowered for pattern in _BLOCKING_CALL_ERROR_PATTERNS
        ):
            # Preserve the complete signature in the decision log, as required
            # by the policy. The source artifacts remain available separately.
            return candidate.strip()
    return None


def _adaptive_train_decision(
    results: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
    *,
    cumulative_train: int,
    train_increment: int = 10,
    max_train_samples: int = 69,
    objective_failure_stop_count: int = 3,
    workflow_failure_stop_count: int = 2,
) -> AdaptiveTrainDecision:
    """Decide whether the current round should add another Train tranche.

    Priority is deliberately simple:
      1. one runtime/API error → abort; one candidate helper error → repair;
         one exhausted Executor-code retry → record separately;
      2. a repeated workflow failure → stop;
      3. a repeated objective check failure → stop;
      4. max Train samples reached → stop;
      5. otherwise add the next tranche.
    """
    objective_counts: Counter[str] = Counter()
    workflow_counts: Counter[str] = Counter()

    for _tid, _inst, executor, oracle, output_dir, _input_dir in results:
        blocking = _find_blocking_call_error(executor, output_dir)
        if blocking is not None:
            reason = _execution_error_kind(blocking) or "blocking_call_error"
            return AdaptiveTrainDecision(
                action="STOP_ADDING",
                reason=reason,
                cumulative_train=cumulative_train,
                blocking_error=blocking,
            )
        if not executor.success:
            workflow_counts[_normalize_workflow_failure(executor.error)] += 1
        if oracle is not None:
            for check in oracle.checks:
                if check.scored and not check.passed:
                    objective_counts[check.name] += 1

    if workflow_counts:
        signature, count = workflow_counts.most_common(1)[0]
        if count >= workflow_failure_stop_count:
            return AdaptiveTrainDecision(
                action="STOP_ADDING",
                reason=f"workflow_failure:{signature}",
                cumulative_train=cumulative_train,
                objective_failures=dict(objective_counts),
                workflow_failures=dict(workflow_counts),
            )

    if objective_counts:
        check_name, count = objective_counts.most_common(1)[0]
        if count >= objective_failure_stop_count:
            return AdaptiveTrainDecision(
                action="STOP_ADDING",
                reason=f"objective_failure:{check_name}",
                cumulative_train=cumulative_train,
                objective_failures=dict(objective_counts),
                workflow_failures=dict(workflow_counts),
            )

    if cumulative_train >= max_train_samples:
        return AdaptiveTrainDecision(
            action="STOP_ADDING",
            reason="max_train_samples",
            cumulative_train=cumulative_train,
            objective_failures=dict(objective_counts),
            workflow_failures=dict(workflow_counts),
        )

    return AdaptiveTrainDecision(
        action=f"ADD_{train_increment}",
        reason="failures_not_yet_repeated",
        cumulative_train=cumulative_train,
        objective_failures=dict(objective_counts),
        workflow_failures=dict(workflow_counts),
    )


def _archive_executor_code_attempt(
    *,
    run_dir: Path,
    iteration: int,
    output_stage: str,
    result: tuple[
        str, str, ExecutorResult, OracleResult | None, Path, Path
    ],
    retry_number: int,
    error: str,
    bucket: str = "executor_code_retry_artifacts",
    error_kind: str = "executor_code_error",
    event_name: str = "executor_code_retries.jsonl",
    rep: int = 0,
) -> Path:
    """Preserve a failed attempt before an immediate retry (code or transient).

    With K concurrent executor sessions per task, rep 0 owns the task dir and
    reps k>0 live in its rep_<k>/ subdirs: rep 0's archive moves everything
    *except* those sibling subdirs, and rep k archives under rep_<k>/attempt_n
    with its own private oracle key, so retries never sweep up or collide with
    a session that is still running.
    """
    tid, inst, executor, _oracle, output_dir, _input_dir = result
    archive_dir = (
        run_dir
        / f"round_{iteration}"
        / bucket
        / output_stage
        / tid
        / inst
    )
    if rep > 0:
        archive_dir = archive_dir / f"rep_{rep}"
    archive_dir = archive_dir / f"attempt_{retry_number}"
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    if output_dir.exists():
        siblings = [c for c in output_dir.iterdir()
                    if c.is_dir() and c.name.startswith("rep_")] if rep == 0 else []
        if siblings:
            archive_dir.mkdir(parents=True)
            for child in output_dir.iterdir():
                if child not in siblings:
                    shutil.move(str(child), str(archive_dir / child.name))
        else:
            shutil.move(str(output_dir), str(archive_dir))
    else:
        archive_dir.mkdir(parents=True)

    # A recovered call error may still have reached the oracle. Keep that
    # private result with the failed attempt so a retry cannot leave stale GT
    # comparison data under the canonical sample key.
    private_oracle = (
        _private_dir(run_dir)
        / f"round_{iteration}"
        / output_stage
        / f"{tid}_{inst}{'' if rep == 0 else f'_rep{rep}'}.oracle.json"
    )
    if private_oracle.exists():
        shutil.move(
            str(private_oracle),
            str(archive_dir / "private_oracle.json"),
        )

    event_path = run_dir / f"round_{iteration}" / event_name
    with event_path.open("a") as handle:
        handle.write(json.dumps({
            "task": f"{tid}/{inst}",
            "rep": rep,
            "output_stage": output_stage,
            "retry_number": retry_number,
            "error_kind": error_kind,
            "error": error,
            "duration_sec": executor.duration,
            "archived_output": str(archive_dir),
        }, ensure_ascii=False, default=str) + "\n")
    return archive_dir


def _aggregate_oracle_results(
    oracles: list[OracleResult | None],
) -> OracleResult:
    """Fold K independent sessions of one task into one OracleResult.

    - each check passes by strict majority over ALL K sessions (a session
      that produced no oracle counts as failing every check); its detail is
      prefixed `[k/K votes PASS]` so `_vote_fraction` reads the fraction and
      the A/B task score stays continuous;
    - `passed` is the strict majority of per-session verdicts;
    - every GT joint gets one JointReport: majority flags, mean/spread of the
      per-session scores, per-dimension pass fractions.
    """
    K = len(oracles)
    valid = [o for o in oracles if o is not None]
    if not valid:
        return OracleResult(False, [CheckResult("execution", False, f"[0/{K} votes PASS] no session produced a URDF")], reps=K)
    order: list[str] = []
    for o in valid:
        for c in o.checks:
            if c.name not in order:
                order.append(c.name)
    checks = []
    for name in order:
        hits = [next((c for c in o.checks if c.name == name), None) for o in valid]
        present = [c for c in hits if c is not None]
        n_pass = sum(1 for c in present if c.passed)
        scored = any(c.scored for c in present)
        failing = next((c for c in present if not c.passed), None)
        detail = f"[{n_pass}/{K} votes PASS] " + ((failing or present[0]).detail if present else "")
        checks.append(CheckResult(name, n_pass * 2 > K, detail, scored=scored))
    passed = sum(1 for o in valid if o.passed) * 2 > K
    joints: dict[str, JointReport] | None = None
    names = []
    for o in valid:
        for jn in (o.joints or {}):
            if jn not in names:
                names.append(jn)
    if names:
        joints = {}
        for jn in names:
            reps = [(o.joints or {}).get(jn) for o in valid]
            got = [r for r in reps if r is not None]
            base = got[0]
            scores = [r.score() for r in got] + [0.0] * (K - len(got))
            frac = {}
            for d in JointReport.DIMS:
                vals = [getattr(r, d) for r in got if getattr(r, d) is not None]
                if vals:
                    frac[d] = sum(1 for v in vals if v) / K
            agg = JointReport(
                gt_joint=base.gt_joint, gt_child=base.gt_child, gt_type=base.gt_type,
                matched=sum(1 for r in got if r.matched) * 2 > K,
                pred_joint=base.pred_joint, pred_child=base.pred_child, pred_type=base.pred_type,
                axis_angle_deg=base.axis_angle_deg, pivot_distance_m=base.pivot_distance_m,
                limit_ratio=base.limit_ratio, trace_drift_m=base.trace_drift_m,
                scale_ratio=base.scale_ratio, rest_drift_m=base.rest_drift_m,
                pose_tol_m=base.pose_tol_m, pose_cause=base.pose_cause,
                rot_err_deg=base.rot_err_deg, pos_err_m=base.pos_err_m,
                reps=K, score_mean=sum(scores) / K, score_spread=max(scores) - min(scores),
                dim_pass_frac={d: round(v, 3) for d, v in frac.items()},
            )
            for d in JointReport.DIMS:
                setattr(agg, d, (frac[d] > 0.5) if d in frac else None)
            joints[jn] = agg
    cost = sum((o.cost_usd or 0.0) for o in valid) if any(o.cost_usd is not None for o in valid) else None
    return OracleResult(passed=passed, checks=checks, matching=valid[0].matching,
                        cost_usd=cost, joints=joints, reps=K)


def _aggregate_rep_results(
    tups: list[tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]],
    *, run_dir: Path, iteration: int, output_stage: str,
) -> tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]:
    """Combine K per-session tuples into the one tuple the round consumes.

    The task dir (rep 0's) stays the canonical output_dir. Rep 0's own
    oracle.json is kept as oracle.rep0.json and the aggregate takes its place,
    so every downstream reader of <task>/oracle.json sees the K-fold result.
    """
    import dataclasses as _dc
    tid, inst, er0, _o0, out0, in0 = tups[0]
    executors = [t[2] for t in tups]
    oracles = [t[3] for t in tups]
    ok = [e for e in executors if e.success]
    # A session that died of infrastructure (transient error, retries
    # exhausted) is no evidence about the skill: drop it from the vote as long
    # as some other session produced an oracle. A session that failed on its
    # own (no URDF, agent code error) still counts as a failing vote.
    infra = [i for i, t in enumerate(tups)
             if t[3] is None and not t[2].success and _is_transient_error(t[2].error)]
    vote = [o for i, o in enumerate(oracles) if i not in infra]
    if not any(o is not None for o in vote):
        vote, infra = oracles, []
    exec_agg = _dc.replace(
        er0,
        success=bool(ok),
        error=None if ok else (er0.error or next((e.error for e in executors if e.error), None)),
        duration=sum(e.duration for e in executors),
    )
    oracle_agg = _aggregate_oracle_results(vote)
    try:
        rep0 = out0 / "oracle.json"
        if rep0.exists():
            rep0.replace(out0 / "oracle.rep0.json")
        full = oracle_agg.to_dict()
        full["rep_scores"] = [None if o is None else round(o.score, 3) for o in oracles]
        full["reps_requested"] = len(tups)
        full["reps_infra_dropped"] = infra
        (out0 / "oracle.json").write_text(json.dumps(sanitize_oracle_dict(full), indent=2))
        private = _private_dir(run_dir) / f"round_{iteration}" / output_stage
        private.mkdir(parents=True, exist_ok=True)
        (private / f"{tid}_{inst}.oracle.json").write_text(json.dumps(full, indent=2))
    except OSError:
        pass
    return tid, inst, exec_agg, oracle_agg, out0, in0


def _first_training_system_error(
    results: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
) -> str | None:
    """Return the first infrastructure/runtime failure in evaluated results."""
    for _tid, _inst, executor, _oracle, output_dir, _input_dir in results:
        error = _find_blocking_call_error(executor, output_dir)
        if error is not None and _is_training_system_error(error):
            return error
    return None


def _apply_dataset_variant(tasks_root: Path) -> dict:
    """Read ``<dataset>/dataset_variant.json`` (parent of tasks/) and apply run-wide settings.

    ``scored_joint_dims`` narrows JointReport.DIMS and ``scored_checks`` narrows
    the object-level checks that count for score/pass (views where the template
    gives the structure, so structural checks must not inflate the score).
    """
    marker = Path(tasks_root).parent / "dataset_variant.json"
    if not marker.is_file():
        return {}
    try:
        variant = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(f"unreadable {marker}: {exc}")
        return {}
    dims = variant.get("scored_joint_dims")
    if dims:
        JointReport.configure_dims(dims)
        log.info(f"dataset {variant.get('dataset')}: scored joint dims = {JointReport.DIMS}")
        from evo.engine import failure_clusters as _fc
        _fc.DIAGNOSTIC_DIMS = {d for d in ("scale_ok",) if d not in dims}
    checks = variant.get("scored_checks")
    if checks:
        from evo.oracle.check_urdf import configure_scored_checks
        chosen = configure_scored_checks(checks)
        log.info(f"dataset {variant.get('dataset')}: scored checks = {chosen}")
    return variant


async def _run_oracle_dispatch(
    *,
    mode: str,
    tag: str,
    gt_dir: Path,
    pred_dir: Path,
    tasks_root: Path,
    effort: str,
    votes: int = 3,
    oracle_model: str | None = None,
) -> OracleResult:
    """Pick the oracle implementation and run it.

    `mode='script'` keeps the legacy URDF text-comparison path (free, fast,
    deterministic). It's CPU-bound + synchronous so we run it in a thread
    via `asyncio.to_thread` to avoid blocking the event loop while N
    parallel workers all return at roughly the same time.

    `mode='subagent'` first renders pred rest + motion frames (subprocess
    renderer — also pushed to a worker thread), then awaits `votes`
    concurrent judges combined by strict majority. The judges run on the same
    Responses provider as the executor and generator; `oracle_model` (None =
    EVO_ORACLE_MODEL, then the module default) picks the judge model.
    """
    if mode not in ORACLES:
        raise ValueError(f"unknown oracle mode {mode!r}; known: {sorted(ORACLES)}")

    # Joint-fill view: an executor that answered by filling parts/template.urdf
    # hands back a URDF without visual origins. Derive them (FK of the filled
    # joints against the world-baked meshes) so every oracle sees a standard
    # URDF. SDK exports already carry visual origins and are left untouched.
    try:
        finalized = finalize_pred_dir(pred_dir, mesh_dir=tasks_root / tag / "parts")
    except (ValueError, FileNotFoundError) as exc:
        return OracleResult(False, [CheckResult("finalize_template", False, str(exc))])
    if finalized is not None:
        log.info(f"[{tag}] finalized filled template -> {finalized}")

    if mode in ("script", "objective"):
        return await asyncio.to_thread(
            run_oracle, gt_dir, pred_dir, Thresholds(),
        )

    from evo.oracle.subagent import run_subagent_oracle_async

    # 4 canonical azimuths. Oracle renders both GT and Pred itself into
    # <pred_dir>/oracle_render/ using the same mesh source (original
    # parts/*.obj) → zero material drift between sides. See
    # evo/oracle/render.py.
    azs = [45, 135, 225, 315]

    return await run_subagent_oracle_async(
        gt_dir, pred_dir,
        tag=tag,
        azs=azs,
        effort=effort,
        votes=votes,
        model=oracle_model,
    )


async def _execute_and_score(
    tid: str,
    inst: str,
    config: EvoConfig,
    run_dir: Path,
    iteration: int,
    current_skill: Path,
    port_queue: asyncio.Queue[int],
    executor_effort: str | None = None,
    subagent_model: str | None = None,
    subagent_effort: str | None = None,
    oracle_mode: str = "subagent",
    oracle_effort: str = DEFAULT_AGENT_EFFORT,
    oracle_votes: int = 3,
    oracle_model: str | None = None,
    output_stage: str = "outputs",
    rep: int = 0,
) -> tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]:
    """One (task, instance) round: pull a port → run executor → run oracle.

    Returns (tid, inst, exec_result, oracle_result, output_dir, input_dir).

    Concurrency model: the port queue is
    sized to config.parallel, so awaiting `port_queue.get()` doubles as
    the concurrency gate. The port is always returned, even on exception.
    """
    input_dir = config.tasks_root / tid / inst / "parts"
    gt_dir = config.gt_root / tid / inst
    output_dir = run_dir / f"round_{iteration}" / output_stage / tid / inst
    if rep > 0:
        # Extra independent sessions live under the task dir so every existing
        # consumer (report, generator paths, valcurve) keeps reading rep 0.
        output_dir = output_dir / f"rep_{rep}"
    # Reference image (see _execute_and_score's sibling logic).
    exec_thumbnail = input_dir.parent / "thumbnail.png"
    if not exec_thumbnail.exists():
        exec_thumbnail = input_dir.parent / "render_thumbnail.png"
    if not exec_thumbnail.exists():
        exec_thumbnail = (
            config.tasks_root.parent / "gt_renders" / tid / inst / "render_assembled.png"
        )

    # Mirror the reference thumbnail into output_dir so the generator
    # has a single self-contained directory per run instead of having
    # to walk back to the dataset path.
    output_dir.mkdir(parents=True, exist_ok=True)
    if exec_thumbnail.exists():
        try:
            import shutil as _shutil
            _shutil.copy2(exec_thumbnail, output_dir / "render_thumbnail.png")
        except OSError:
            pass

    port = await port_queue.get()
    try:
        exec_result = await run_executor(
            factory=tid, instance=inst,
            input_dir=input_dir, output_dir=output_dir,
            skill_dir=current_skill,
            thumbnail_path=exec_thumbnail if exec_thumbnail.exists() else None,
            model=config.executor_model,
            max_turns=config.executor_max_turns,
            log_file=output_dir / "executor.log",
            sim_port=port,
            sim_headless=config.headless,
            effort=executor_effort,
            subagent_model=subagent_model,
            subagent_effort=subagent_effort,
            backend=config.executor_backend,
        )
    finally:
        port_queue.put_nowait(port)

    oracle_result: OracleResult | None = None
    if exec_result.success:
        oracle_result = await _run_oracle_dispatch(
            mode=oracle_mode,
            tag=f"{tid}/{inst}",
            gt_dir=gt_dir,
            pred_dir=output_dir,
            tasks_root=config.tasks_root,
            effort=oracle_effort,
            votes=oracle_votes,
            oracle_model=oracle_model,
        )
        # Persist per-task oracle output so later audit / re-scoring with a
        # different oracle build can compare without re-running checks.
        # output_dir is generator-reachable → it gets the SANITIZED dict
        # (GT values redacted); the full dict goes to the run's private
        # sibling dir, which is excluded from generator add_dirs and
        # flagged by the isolation audit if it ever shows up in a session.
        full = oracle_result.to_dict()
        try:
            (output_dir / "oracle.json").write_text(
                json.dumps(sanitize_oracle_dict(full), indent=2)
            )
            private = (
                _private_dir(run_dir) / f"round_{iteration}" / output_stage
            )
            private.mkdir(parents=True, exist_ok=True)
            rep_suffix = "" if rep == 0 else f"_rep{rep}"
            (private / f"{tid}_{inst}{rep_suffix}.oracle.json").write_text(
                json.dumps(full, indent=2)
            )
        except OSError:
            pass

    return tid, inst, exec_result, oracle_result, output_dir, input_dir


async def evolve_multi_task(
    config: EvoConfig,
    task_ids: list[dict[str, str]] | None = None,
    verbose: bool = False,
    resume_dir: Path | None = None,
    retry_wait_min: int = 0,
    max_retries: int = 3,
    stagger_sec: int = 0,
    generator_auth: str | None = None,
    executor_effort: str | None = None,
    generator_effort: str | None = None,
    subagent_model: str | None = None,
    subagent_effort: str | None = None,
    oracle_mode: str = "subagent",
    oracle_effort: str = DEFAULT_AGENT_EFFORT,
    oracle_votes: int = 3,
    oracle_model: str | None = None,
    regression_tol: float = 0.05,
    feedback_split: str | None = None,
    bootstrap_generator: bool = False,
    skip_generator: bool = False,
    generate_report: bool = False,
    gate_tasks: list[dict[str, str]] | None = None,
    gate_mode: str = "select",
    gate_min_drop: int = 3,
    gate_reps: int = 3,
    gate_z: float = 1.0,
    gate_min_gain: float = 0.0,
    gate_reroll: bool = True,
    gate_extend: bool = True,
) -> None:
    """Run co-evolution on a set of tasks (aggregate objective).

    Gate split (`gate_tasks`, normally the `val` split): evaluated with the
    round's skill after every train batch, one session per task, into the
    sibling directory `<run_dir>_val` that the Generator cannot read.
    `gate_mode="select"` makes checkpoint selection and the regression gate
    use the gate split's pass count (rollback when it falls `gate_min_drop`
    or more tasks below the checkpoint's); `gate_mode="log"` only records the
    curve and leaves selection on the train fitness. Either way the Generator
    sees only the scalar summary (`round_feedback["gate"]`), never the gate
    split's per-task results.

    Tasks within a round run concurrently up to `config.parallel` workers,
    each pinned to its own sim worker on `config.sim_base_port + i`.
    Generator runs serially at end-of-round.

    Fitness is the mean oracle score (fraction of checks passed) over the
    FEEDBACK split — graded, so a high check-pass fraction beats a low one even
    when both fail the binary verdict, and held-out splits never influence
    selection. The binary pass rate is still reported (and drives the
    ≥0.9 early stop). `regression_tol` arms the rollback gate: when a
    round's fitness drops more than this below the best checkpoint, the
    next generator forks from the best skill instead of compounding edits
    on a regressed one. Negative disables.

    Args:
        task_ids: list of {"task_id": ..., "instance": ...}. If None, discover all.
        resume_dir: if set, load `loop_state.json` from this dir and
            continue from `round_completed + 1`.  The current_skill is
            taken from the state, NOT from `config.skill_dir`.  Earlier
            rounds' artifacts (round_*/skill, generator.log, ...) are
            preserved.  A round that crashed mid-execution is restarted
            from scratch (its directory is overwritten).
        retry_wait_min: minutes to sleep after a transient executor error
            before retrying.  Useful for Claude Code's 5-hour usage
            window — set to ~305 (5h + buffer).  0 disables auto-retry.
        max_retries: max retry attempts per task per round.
        generate_report: write ``<run_dir>/report/index.html`` when this
            formal training process ends. Report failures are logged but never
            change the experiment outcome.
    """
    if task_ids is None:
        task_ids = discover_tasks(config)

    if not task_ids:
        print("No tasks found.")
        return

    adaptive_train_enabled = config.adaptive_train and feedback_split is not None
    if config.adaptive_train and feedback_split is None and not skip_generator:
        print("Warning: adaptive Train sampling requires a named feedback split; "
              "falling back to the legacy full/batch path.")
    if adaptive_train_enabled:
        if config.train_increment <= 0:
            raise ValueError("train_increment must be > 0")
        if config.max_train_samples <= 0:
            raise ValueError("max_train_samples must be > 0")
        if config.objective_failure_stop_count <= 0:
            raise ValueError("objective_failure_stop_count must be > 0")
        if config.workflow_failure_stop_count <= 0:
            raise ValueError("workflow_failure_stop_count must be > 0")
        if config.candidate_ab_min_samples <= 0:
            raise ValueError("candidate_ab_min_samples must be > 0")
        if config.max_contract_repair_attempts <= 0:
            raise ValueError("max_contract_repair_attempts must be > 0")
    if config.executor_code_retries < 0:
        raise ValueError("executor_code_retries must be >= 0")

    # Legacy fixed batch sampling state. Adaptive Train sampling uses a
    # shuffled feedback-split order and does not consult batch_size.
    splits_meta = (
        config.load_split_metadata()
        if config.batch_size > 0 and not adaptive_train_enabled
        else {}
    )
    seen_by_diff: dict[str, set[tuple[str, str]]] = {}
    rng = random.Random(config.seed)

    if oracle_mode == "subagent":
        from evo.oracle.subagent import oracle_backend, oracle_model as _oracle_model

        judge_backend = oracle_backend()
        print(f"Subagent oracle: {judge_backend}:"
              f"{_oracle_model(judge_backend, oracle_model)} "
              f"effort={oracle_effort}, votes={oracle_votes} "
              f"(strict-majority aggregation)")

    # Resume bookkeeping --------------------------------------------------
    start_iteration = 0
    objective_rounds_completed = 0
    consecutive_contract_repairs = 0
    current_skill = config.skill_dir
    feedback_history: list[dict[str, Any]] = []
    best_fitness = 0.0
    run_cost_usd = 0.0
    best_skill_dir: Path | None = None
    seen_train: set[tuple[str, str]] = set()
    best_gate_pass: int | None = None     # gate-split passes of the checkpoint
    gate_split_name = (gate_tasks[0].get("split") or "gate") if gate_tasks else None
    if gate_tasks and gate_mode not in ("select", "log", "paired"):
        raise ValueError(f"gate_mode must be 'select', 'log' or 'paired', got {gate_mode!r}")
    # --gate-mode paired: continuous fitness, K sessions per gate task, the
    # champion re-run and pooled every round, acceptance by a paired test
    # (evo/engine/paired_gate.py). State lives beside the gate outputs.
    paired_mode = bool(gate_tasks) and gate_mode == "paired"
    paired_gate: PairedGate | None = (
        PairedGate(z_accept=gate_z, z_demote=gate_z, min_gain=gate_min_gain) if paired_mode else None
    )

    if resume_dir is not None:
        state = _load_loop_state(resume_dir)
        if state is None:
            raise SystemExit(
                f"--resume {resume_dir} requested but {LOOP_STATE_FILENAME} "
                f"is missing or unreadable.  Did this run reach end-of-round 0?"
            )
        run_dir = resume_dir
        start_iteration = int(state["round_completed"]) + 1
        objective_rounds_completed = int(
            state.get("objective_rounds_completed", start_iteration)
        )
        consecutive_contract_repairs = int(
            state.get("consecutive_contract_repairs", 0)
        )
        current_skill = Path(state["current_skill_dir"])
        feedback_history = state.get("feedback_history", []) or []
        # Pre-graded-fitness states only have best_pass_rate; use it as the
        # baseline so resume keeps a meaningful (if conservative) checkpoint.
        best_fitness = float(
            state.get("best_fitness", state.get("best_pass_rate", 0.0))
        )
        best_path_str = state.get("best_skill_dir")
        best_skill_dir = Path(best_path_str) if best_path_str else None
        best_gate_pass = state.get("best_gate_pass")
        if paired_mode:
            _pg_path = run_dir.with_name(run_dir.name + "_val") / "paired_gate.json"
            if _pg_path.is_file():
                paired_gate = PairedGate.load(_pg_path, z_accept=gate_z, z_demote=gate_z, min_gain=gate_min_gain)
        for d, pairs in (state.get("seen_by_diff") or {}).items():
            seen_by_diff[d] = {tuple(p) for p in pairs}
        seen_train = {
            tuple(pair) for pair in (state.get("seen_train") or [])
        }
        if objective_rounds_completed >= config.max_rounds:
            print(
                "Resumed run already completed "
                f"{objective_rounds_completed} objective rounds "
                f"≥ max_rounds={config.max_rounds}; nothing to do."
            )
            return
        print(
            f"Resuming run at {run_dir} from attempt {start_iteration+1}, "
            f"objective round {objective_rounds_completed+1} "
            f"(best fitness so far: {best_fitness:.3f})"
        )
    else:
        run_dir = config.output_root / "runs" / f"multi_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # machine-readable for drivers that launch the loop and need its run dir
        print(f"Run dir: {run_dir}", flush=True)

    active_preflight = preflight_skill(current_skill)
    (run_dir / "active_skill_preflight.json").write_text(
        json.dumps(active_preflight.to_dict(), indent=2)
    )
    if not active_preflight.ok:
        raise SkillContractError(
            "active skill failed model-free contract preflight before Train: "
            + "; ".join(active_preflight.errors[:3])
        )

    print(f"{'='*60}")
    if adaptive_train_enabled:
        n_train = sum(1 for e in task_ids if e.get("split") == feedback_split)
        n_deferred = len(task_ids) - n_train
        print(
            f"Multi-task evolution ({n_train} {feedback_split} tasks; "
            f"{n_deferred} non-feedback tasks deferred); adaptive Train "
            f"{config.train_increment}→…→{config.max_train_samples}"
        )
        print(
            "Adaptive stop gates: one runtime/API error aborts; one "
            "candidate helper error completes the tranche then repairs; "
            f"Executor code errors retry {config.executor_code_retries}x; transient provider/renderer errors retry {config.transient_retries}x (30 s → 10 min backoff); executor reps {config.executor_reps}; "
            f"objective repeat ≥{config.objective_failure_stop_count}; "
            f"workflow repeat ≥{config.workflow_failure_stop_count}"
        )
        if config.candidate_ab_test and not skip_generator:
            print(
                "Generator candidate A/B: replay exact completed Train "
                "assets; accept unless clearly worse (paired non-inferiority, "
                "one-sided t at alpha 0.10) or a foundational check collapses"
            )
    elif config.batch_size > 0:
        print(f"Multi-task evolution ({len(task_ids)} tasks in pool, "
              f"batch_size={config.batch_size} per round, "
              f"ratio={dict(config.batch_ratio)})")
        if not splits_meta:
            print(f"Warning: batch sampling enabled but splits.json missing at "
                  f"{config.splits_path}; difficulty unknown, all rows treated as 'medium'.")
    else:
        print(f"Multi-task evolution ({len(task_ids)} tasks, full pool every round)")
    if config.parallel > 1:
        ports = f"{config.sim_base_port}–{config.sim_base_port + config.parallel - 1}"
        print(f"Parallel: {config.parallel} workers (sim ports {ports}), "
              f"headless={config.headless}")
    else:
        print(f"Sequential (parallel=1, headless={config.headless})")
    print(f"{'='*60}\n")

    # Bootstrap: let Generator design initial tools BEFORE any executor
    # runs. Use when starting from a blank skill (no tools/) so executors
    # don't waste a round flailing without infrastructure. Output goes
    # to <run_dir>/bootstrap/skill, which becomes the skill for round 0.
    # Generator file access is scoped to what its job requires: this run's
    # artifacts, the (GT-free) task inputs, the reference renders that the
    # executor sees anyway, and the oracle SOURCE (the metric is public by
    # design — the GT answer files under gt/ are not). Bash remains an open
    # channel; the post-round isolation audit covers it.
    gen_add_dirs = _generator_add_dirs(run_dir, config)

    if bootstrap_generator and start_iteration == 0:
        bootstrap_dir = run_dir / "bootstrap"
        bootstrap_skill_dir = bootstrap_dir / "skill"
        print(">>> Bootstrap phase: Generator writes initial tools <<<")
        gen_model = config.generator_model
        if config.generator_backend == "claude" and generator_auth == "local":
            boot_ctx = _auth_swap("local")
            gen_model = "claude-opus-4-7"
        else:
            boot_ctx = contextlib.nullcontext()
        bootstrap_base_content = _read_skill_content(current_skill)
        with boot_ctx:
            bootstrap_result = await run_generator(
                current_skill_dir=current_skill,
                current_skill_content=bootstrap_base_content,
                feedback_history=[],  # no executor data yet
                output_skill_dir=bootstrap_skill_dir,
                model=gen_model,
                log_file=bootstrap_dir / "generator.log",
                effort=generator_effort,
                sim_port=config.sim_base_port + config.parallel,
                sim_headless=config.headless,
                backend=config.generator_backend,
                add_dirs=gen_add_dirs,
                dataset_tasks_root=config.tasks_root,
            )
        bootstrap_ok, bootstrap_reason = _validate_generator_output(
            bootstrap_result,
            bootstrap_dir / "generator_journal.md",
            current_skill_content=bootstrap_base_content,
        )
        if not bootstrap_ok:
            error_type = (
                TrainingSystemError
                if bootstrap_result.changes_summary.startswith("Generator error:")
                else SkillContractError
            )
            raise error_type(f"bootstrap Generator rejected: {bootstrap_reason}")
        bootstrap_preflight = preflight_skill(bootstrap_skill_dir)
        (bootstrap_dir / "skill_preflight.json").write_text(
            json.dumps(bootstrap_preflight.to_dict(), indent=2)
        )
        if not bootstrap_preflight.ok:
            raise SkillContractError(
                "bootstrap skill failed model-free contract preflight: "
                + "; ".join(bootstrap_preflight.errors[:3])
            )
        current_skill = bootstrap_skill_dir
        print(f">>> Bootstrap done: skill at {bootstrap_skill_dir} <<<\n")

    iteration = start_iteration
    try:
      while objective_rounds_completed < config.max_rounds:
        print(
            f"\n--- Objective round {objective_rounds_completed + 1}/"
            f"{config.max_rounds} (attempt {iteration + 1}) ---\n"
        )
        # Fresh browsers each round (same reason as evolve_single_task):
        # Chromium's GPU memory only grows within a server's lifetime, and
        # nothing renders at a round boundary.
        if iteration > start_iteration and recycle_shared_pool(f"round {iteration + 1}"):
            print("LightPhyx pool recycled (fresh browsers for this round)", flush=True)
        round_contract_failure = False

        # Activate the current skill version once per round (idempotent).
        evaluated_skill = current_skill
        _install_skill(current_skill)

        # Legacy per-round batch selection. In adaptive mode the feedback
        # split is ordered below and consumed one Train tranche at a time.
        if adaptive_train_enabled:
            round_batch: list[dict[str, str]] = []
        elif config.batch_size > 0 and config.fixed_batch:
            round_batch = _fixed_round_batch(
                pool=task_ids, metadata=splits_meta,
                batch_size=config.batch_size, ratio=config.batch_ratio,
                seed=config.seed,
            )
            picked = ", ".join(f"{e['task_id']}/{e['instance']}" for e in round_batch)
            print(f"Fixed batch ({len(round_batch)}, same tasks every round): {picked}")
        elif config.batch_size > 0:
            round_batch = _sample_batch(
                pool=task_ids,
                metadata=splits_meta,
                batch_size=config.batch_size,
                ratio=config.batch_ratio,
                seen=seen_by_diff,
                rng=rng,
            )
            picked = ", ".join(f"{e['task_id']}/{e['instance']}" for e in round_batch)
            print(f"Sampled batch ({len(round_batch)}): {picked}")
        else:
            round_batch = task_ids

        # Port pool: one slot per parallel worker.
        port_queue: asyncio.Queue[int] = asyncio.Queue()
        for i in range(config.parallel):
            port_queue.put_nowait(config.sim_base_port + i)

        async def _execute_rep_with_retry(
            entry: dict[str, str],
            *,
            rep: int = 0,
            skill_dir: Path | None = None,
            output_stage: str = "outputs",
            run_root: Path | None = None,
        ):
            """Run one executor session with separate model-code and transient retries.

            `run_root` redirects every artifact of this session (outputs and
            the private oracle copy) to another run directory; the gate split
            uses it to keep its results out of the Generator's read scope.

            Executor-authored syntax/call mistakes retry immediately and are
            independent of the rate-limit retry budget. Transient provider
            failures retain the legacy optional sleep/retry policy.
            """
            transient_attempt = 0
            executor_code_attempt = 0
            _run_root = run_root or run_dir
            while True:
                tup = await _execute_and_score(
                    tid=entry["task_id"], inst=entry["instance"],
                    config=config, run_dir=_run_root, iteration=iteration,
                    current_skill=skill_dir or current_skill,
                    port_queue=port_queue,
                    executor_effort=executor_effort,
                    subagent_model=subagent_model,
                    subagent_effort=subagent_effort,
                    oracle_mode=oracle_mode,
                    oracle_effort=oracle_effort,
                    oracle_votes=oracle_votes,
                    oracle_model=oracle_model,
                    output_stage=output_stage,
                    rep=rep,
                )
                tid, inst, er, _orc, out_dir, _in = tup
                blocking_error = _find_blocking_call_error(er, out_dir)
                if _should_retry_executor_code(
                    er, blocking_error, executor_code_attempt, config.executor_code_retries,
                ):
                    executor_code_attempt += 1
                    archived = _archive_executor_code_attempt(
                        run_dir=_run_root,
                        iteration=iteration,
                        output_stage=output_stage,
                        result=tup,
                        retry_number=executor_code_attempt,
                        error=blocking_error or "unknown Executor code error",
                        rep=rep,
                    )
                    print(
                        f"  [executor-code-retry] {tid}/{inst} "
                        f"(attempt {executor_code_attempt}/"
                        f"{config.executor_code_retries}); archived="
                        f"{archived}: {(blocking_error or '')[:120]}",
                        flush=True,
                    )
                    continue
                transient_error = next(
                    (e for e in (blocking_error, er.error) if _is_transient_error(e)),
                    None,
                )
                if transient_error and transient_attempt < config.transient_retries:
                    transient_attempt += 1
                    wait_s = (
                        retry_wait_min * 60 if retry_wait_min > 0
                        else _TRANSIENT_BACKOFF_S[
                            min(transient_attempt, len(_TRANSIENT_BACKOFF_S)) - 1
                        ]
                    )
                    archived = _archive_executor_code_attempt(
                        run_dir=_run_root,
                        iteration=iteration,
                        output_stage=output_stage,
                        result=tup,
                        retry_number=transient_attempt,
                        error=transient_error,
                        bucket="transient_retry_artifacts",
                        error_kind="transient_error",
                        event_name="transient_retries.jsonl",
                        rep=rep,
                    )
                    print(
                        f"  [transient-retry] {tid}/{inst} "
                        f"(attempt {transient_attempt}/{config.transient_retries}); "
                        f"sleeping {wait_s}s; archived={archived}: "
                        f"{transient_error[:120]}",
                        flush=True,
                    )
                    await asyncio.sleep(wait_s)
                    continue
                if (
                    er.success
                    or adaptive_train_enabled
                    or retry_wait_min <= 0
                    or transient_attempt >= max_retries
                    or not _is_transient_error(er.error)
                ):
                    return tup
                transient_attempt += 1
                wait_s = retry_wait_min * 60
                print(
                    f"  [retry] {tid}/{inst} hit transient error "
                    f"(attempt {transient_attempt}/{max_retries}); sleeping "
                    f"{retry_wait_min} min: {er.error[:120]}",
                    flush=True,
                )
                await asyncio.sleep(wait_s)

        async def _execute_with_retry(
            entry: dict[str, str],
            *,
            skill_dir: Path | None = None,
            output_stage: str = "outputs",
            run_root: Path | None = None,
        ):
            """Run config.executor_reps independent sessions of one task and
            aggregate them into a single result tuple.

            The K sessions run concurrently; the port pool (config.parallel
            slots) is the only concurrency bound, so --parallel 60 with 20
            tasks x 3 reps runs all 60 sessions at once. Rep 0 writes the task
            dir and rep k>0 writes rep_<k>/; the retry archive is rep-aware and
            skips sibling subdirs, so a retry can never sweep up a session that
            is still running. Reps of one task start `stagger_sec` apart.
            """
            reps = max(1, int(config.executor_reps))

            async def _one(k: int):
                if stagger_sec > 0 and k > 0:
                    await asyncio.sleep(k * stagger_sec)
                return await _execute_rep_with_retry(
                    entry, rep=k, skill_dir=skill_dir, output_stage=output_stage,
                    run_root=run_root,
                )

            tups = list(await asyncio.gather(*[_one(k) for k in range(reps)]))
            if reps == 1:
                return tups[0]
            return _aggregate_rep_results(tups, run_dir=run_root or run_dir,
                                          iteration=iteration, output_stage=output_stage)

        async def _execute_with_stagger(
            idx: int,
            entry: dict[str, str],
            *,
            skill_dir: Path | None = None,
            output_stage: str = "outputs",
            run_root: Path | None = None,
        ):
            """Delay this worker's start by `idx * stagger_sec` (off by
            default).  The stagger existed for a legacy `uvx blender-mcp`
            handshake race; the sim worker bootstraps in ~1 s and the port
            pool bounds concurrency, so a non-zero value only serialises
            the round: with 69 workers at 20 s the last session started
            23 min after the first while sessions themselves took ~1 min
            (run 20).  Keep it available for debugging bootstrap races."""
            if stagger_sec > 0 and idx > 0:
                await asyncio.sleep(idx * stagger_sec)
            return await _execute_with_retry(
                entry, skill_dir=skill_dir, output_stage=output_stage,
                run_root=run_root,
            )

        ResultTuple = tuple[
            str, str, ExecutorResult, OracleResult | None, Path, Path
        ]

        def _print_completed(
            result: ResultTuple, completed: int, total: int, label: str,
        ) -> None:
            tid, inst, _er, orc, _out_dir, _in_dir = result
            prefix = f"  [{label} {completed}/{total}]"
            if orc:
                scored_checks = [c for c in orc.checks if c.scored]
                n_chk = len(scored_checks)
                n_chk_pass = sum(1 for c in scored_checks if c.passed)
                tag = "PASS" if orc.passed else "FAIL"
                reps = getattr(orc, "reps", 1)
                rep_note = f", {reps} reps majority" if reps > 1 else ""
                print(
                    f"{prefix} {tid}/{inst}: {tag} "
                    f"({n_chk_pass}/{n_chk} checks = {orc.score:.0%}{rep_note})",
                    flush=True,
                )
            else:
                print(f"{prefix} {tid}/{inst}: NO_ORACLE", flush=True)

        async def _run_entries(
            entries: list[dict[str, str]],
            *,
            label: str,
            apply_train_policy: bool = False,
            cumulative_before: int = 0,
            skill_dir: Path | None = None,
            output_stage: str = "outputs",
            run_root: Path | None = None,
        ) -> tuple[list[ResultTuple], AdaptiveTrainDecision | None]:
            """Execute one tranche/pool and optionally watch blocking errors.

            Only genuine API/MCP/sim-worker/runtime failures cancel pending
            tasks. Generator-written helper TypeError/ValueError failures stop
            *additional* Train collection after the already-started tranche
            completes, preserving peer trajectories for contract repair.
            Objective and workflow repetition are also checked only after the
            complete tranche.
            """
            tasks = [
                asyncio.create_task(_execute_with_stagger(
                    i,
                    entry,
                    skill_dir=skill_dir,
                    output_stage=output_stage,
                    run_root=run_root,
                ))
                for i, entry in enumerate(entries)
            ]
            completed_results: list[ResultTuple] = []
            stop_decision: AdaptiveTrainDecision | None = None
            try:
                for future in asyncio.as_completed(tasks):
                    result = await future
                    completed_results.append(result)
                    _print_completed(
                        result, len(completed_results), len(entries), label
                    )
                    if not apply_train_policy:
                        continue
                    blocking_error = _find_blocking_call_error(
                        result[2], result[4]
                    )
                    if (
                        blocking_error is not None
                        and not _is_training_system_error(blocking_error)
                    ):
                        # Skill/helper failures are useful Generator evidence.
                        # Do not throw away already-running peer trajectories.
                        continue
                    if blocking_error is not None:
                        # The failing call itself is not an effective Train
                        # result; completed non-blocking peers still are.
                        effective_completed = sum(
                            _find_blocking_call_error(r[2], r[4]) is None
                            for r in completed_results
                        )
                        stop_decision = AdaptiveTrainDecision(
                            action="STOP_ADDING",
                            reason="training_system_error",
                            cumulative_train=(
                                cumulative_before + effective_completed
                            ),
                            blocking_error=blocking_error,
                        )
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        break
            finally:
                await asyncio.gather(*tasks, return_exceptions=True)
            return completed_results, stop_decision

        # Stage A — adaptively collect Train evidence, or preserve the legacy
        # fixed/full-pool behavior when adaptive Train is disabled.
        results: list[ResultTuple] = []
        adaptive_decision: AdaptiveTrainDecision | None = None
        decision_events_path = (
            run_dir / f"round_{iteration}" / "adaptive_train_decisions.jsonl"
        )

        if adaptive_train_enabled:
            train_pool = [
                e for e in task_ids if e.get("split") == feedback_split
            ]
            heldout_pool = [
                e for e in task_ids if e.get("split") != feedback_split
            ]
            if not train_pool:
                raise RuntimeError(
                    f"adaptive Train found no entries in feedback split "
                    f"{feedback_split!r}"
                )
            unseen_train = [
                entry for entry in train_pool
                if (entry["task_id"], entry["instance"]) not in seen_train
            ]
            recycled_train = [
                entry for entry in train_pool
                if (entry["task_id"], entry["instance"]) in seen_train
            ]
            # Round-local RNG keeps ordering reproducible across resume: a
            # resumed round N gets the same order as an uninterrupted round N.
            order_rng = random.Random(config.seed + iteration)
            order_rng.shuffle(unseen_train)
            order_rng.shuffle(recycled_train)
            train_order = unseen_train + recycled_train
            train_limit = min(config.max_train_samples, len(train_order))
            cumulative_train = 0
            cursor = 0
            tranche_index = 0
            decision_events_path.parent.mkdir(parents=True, exist_ok=True)

            while cumulative_train < train_limit:
                tranche_index += 1
                tranche = train_order[
                    cursor:cursor + min(
                        config.train_increment,
                        train_limit - cumulative_train,
                    )
                ]
                if not tranche:
                    break
                print(
                    f"Train tranche {tranche_index}: requesting "
                    f"{len(tranche)} assets "
                    f"(completed before tranche={cumulative_train})"
                )
                tranche_results, early_decision = await _run_entries(
                    tranche,
                    label=f"train-{tranche_index}",
                    apply_train_policy=True,
                    cumulative_before=cumulative_train,
                )
                results.extend(tranche_results)
                completed_keys = {(r[0], r[1]) for r in tranche_results}
                round_batch.extend(
                    e for e in tranche
                    if (e["task_id"], e["instance"]) in completed_keys
                )
                effective_tranche_results = [
                    r for r in tranche_results
                    if _find_blocking_call_error(r[2], r[4]) is None
                ]
                seen_train.update(
                    (result[0], result[1])
                    for result in effective_tranche_results
                )
                cumulative_train += len(effective_tranche_results)
                cursor += len(tranche)

                adaptive_decision = early_decision or _adaptive_train_decision(
                    tranche_results,
                    cumulative_train=cumulative_train,
                    train_increment=config.train_increment,
                    max_train_samples=train_limit,
                    objective_failure_stop_count=(
                        config.objective_failure_stop_count
                    ),
                    workflow_failure_stop_count=(
                        config.workflow_failure_stop_count
                    ),
                )
                if (
                    adaptive_decision.blocking_error
                    and _is_training_system_error(
                        adaptive_decision.blocking_error
                    )
                ):
                    adaptive_decision = AdaptiveTrainDecision(
                        action="STOP_ADDING",
                        reason="training_system_error",
                        cumulative_train=adaptive_decision.cumulative_train,
                        objective_failures=(
                            adaptive_decision.objective_failures
                        ),
                        workflow_failures=(
                            adaptive_decision.workflow_failures
                        ),
                        blocking_error=adaptive_decision.blocking_error,
                    )
                event = {
                    "round": iteration,
                    "tranche": tranche_index,
                    "requested": len(tranche),
                    "completed": len(tranche_results),
                    "effective_completed": len(effective_tranche_results),
                    **asdict(adaptive_decision),
                }
                with decision_events_path.open("a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
                print(
                    f"Adaptive Train decision: {adaptive_decision.action} "
                    f"({adaptive_decision.reason}); "
                    f"cumulative={cumulative_train}/{train_limit}"
                )
                if adaptive_decision.reason == "training_system_error":
                    failure_path = _write_training_system_failure(
                        decision_events_path.parent,
                        stage="train",
                        error=adaptive_decision.blocking_error or "unknown",
                    )
                    raise TrainingSystemError(
                        "Train hit a training-system failure; Validation and "
                        "Generator were not started. Fix the runtime and rerun "
                        f"the current round. Details: {failure_path}"
                    )
                if (
                    adaptive_decision.reason
                    == "candidate_tool_contract_error"
                ):
                    round_contract_failure = True
                if adaptive_decision.action == "STOP_ADDING":
                    break

            # Validation is deliberately outside the training loop. Do not
            # spend held-out calls after Train finds an actionable failure;
            # hand Train evidence to Generator immediately. A separate
            # validation --no-generator --max-rounds 1 process owns full
            # validation and writes into a different run directory.
            if heldout_pool:
                print(
                    f"Held-out evaluation deferred ({len(heldout_pool)} "
                    "assets); run it separately with --split validation "
                    "--no-generator --max-rounds 1"
                )
        else:
            results, _ = await _run_entries(round_batch, label="batch")

        # Per-entry split tag (set in run_evolution when --split was
        # given). Used to (a) filter generator feedback to one split and
        # (b) split pass-rate reporting by split.
        entry_split: dict[tuple[str, str], str | None] = {
            (e["task_id"], e["instance"]): e.get("split")
            for e in task_ids
        }

        # Aggregate. Selection metrics (fitness, early stop, best/rollback)
        # come from the FEEDBACK split only — held-out splits are evaluated
        # and logged but must not influence which skill survives, or the
        # "held-out" label is a lie at the model-selection level.
        fb_results = [
            r for r in results
            if feedback_split is None
            or entry_split.get((r[0], r[1])) == feedback_split
        ]
        n_pass, pass_rate, mean_score = _round_metrics(
            [o for _, _, _, o, _, _ in results]
        )
        fb_n_pass, fb_pass_rate, fb_mean_score = _round_metrics(
            [o for _, _, _, o, _, _ in fb_results]
        )
        # Model selection is lexicographic in practice: binary object
        # success dominates, while the old equal-weight check score is only
        # a tiny tie-breaker when two rounds have the same pass rate.
        fitness = _selection_fitness(fb_pass_rate, fb_mean_score)
        print(f"\nPass rate (binary, whole batch): {n_pass}/{len(results)} "
              f"({pass_rate:.1%});  mean oracle score: {mean_score:.3f}")
        if feedback_split is not None:
            print(f"Feedback split ({feedback_split}): "
                  f"{fb_n_pass}/{len(fb_results)} ({fb_pass_rate:.1%});  "
                  f"mean diagnostic score: {fb_mean_score:.3f};  "
                  f"selection fitness: {fitness:.6f}")

        # Per-check pass aggregation over the feedback split.  Useful
        # signal even when binary pass_rate is 0 — tells the generator
        # which check is the bottleneck (e.g. "rest_pose_alignment 2/8 vs
        # joint_axes 7/8" → focus on origin placement).  Names only,
        # NOT the underlying GT values. Held-out results stay out: the
        # generator must not steer on test-split check rates.
        per_check_pass: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [n_pass, n_total]
        for _, _, _, o, _, _ in fb_results:
            if not o:
                continue
            for c in o.checks:
                per_check_pass[c.name][1] += 1
                if c.passed:
                    per_check_pass[c.name][0] += 1
        if per_check_pass:
            print("Per-check pass rates (feedback split):")
            for name in sorted(per_check_pass):
                p, t = per_check_pass[name]
                pct = (p / t) if t else 0.0
                print(f"    {name:<28} {p}/{t}  ({pct:.0%})")

        # Stage A2 — gate split. Run the same skill on the gate split (val)
        # so that selection can be made on objects the Generator never
        # learns from. Artifacts go to <run_dir>_val, outside the
        # Generator's read scope; only the scalars below reach feedback.
        gate_record: dict[str, Any] | None = None
        paired_accept: bool | None = None
        gate_spend_results: list = []
        if paired_mode and not skip_generator:
            val_root = run_dir.with_name(run_dir.name + "_val")
            label_r = f"R{iteration}"
            print(f"\nGate split '{gate_split_name}' (paired): {len(gate_tasks)} task(s) x "
                  f"{max(1, gate_reps)} session(s), z_accept={gate_z}, min_gain={gate_min_gain}", flush=True)
            _saved_reps = config.executor_reps
            config.executor_reps = max(1, int(gate_reps))
            try:
                gate_results, _ = await _run_entries(
                    gate_tasks, label=gate_split_name, run_root=val_root,
                )
                gate_spend_results += gate_results
                cand_scores, cand_pass = collect_scores(gate_results)
                champ = paired_gate.champion
                snap = val_root / "champions" / label_r
                if champ is None:
                    _save_best_skill(Path(current_skill), snap)
                    rec = paired_gate.install(label_r, str(snap), cand_scores, cand_pass, round_index=iteration)
                    paired_accept = True
                elif _skill_content_identical(Path(current_skill), Path(champ.skill_dir)):
                    # The Generator produced nothing new (or the loop is back on
                    # the champion): these sessions are more evidence about it.
                    paired_gate.pool_champion(cand_scores, cand_pass)
                    rec = {"round": iteration, "decision": "champion_rerun", "candidate": champ.label,
                           "champion": champ.label, "champion_fitness": round(champ.fitness(), 4),
                           "champion_pass_rate": round(champ.pass_rate(), 4),
                           "champion_sessions": champ.n_sessions()}
                    paired_gate.history.append(rec)
                else:
                    if gate_reroll:
                        champ_results, _ = await _run_entries(
                            gate_tasks, label=f"{gate_split_name}-champion({champ.label})",
                            run_root=val_root, skill_dir=Path(champ.skill_dir),
                            output_stage="champion_outputs",
                        )
                        gate_spend_results += champ_results
                        c_scores, c_pass = collect_scores(champ_results)
                        paired_gate.pool_champion(c_scores, c_pass)
                        demoted = paired_gate.maybe_demote(round_index=iteration)
                        if demoted is not None:
                            print(f"PAIRED GATE: champion {demoted['candidate']} demoted, back to "
                                  f"{demoted['champion']} (diff {demoted['diff']:+.3f}, z {demoted['z']:+.2f})")
                            best_skill_dir = run_dir / "best_skill"
                            _save_best_skill(Path(paired_gate.champion.skill_dir), best_skill_dir)
                    _save_best_skill(Path(current_skill), snap)
                    rec = paired_gate.decide(label_r, str(snap), cand_scores, cand_pass, round_index=iteration)
                    # Sequential extension: a candidate that is ahead but not by
                    # z_accept is neither confirmed nor refuted by K sessions.
                    # Run one more batch on both sides, pool, and decide again.
                    if (rec["decision"] == "rejected" and gate_reroll and gate_extend
                            and rec["diff"] > 0 and rec["z"] >= GATE_EXTEND_MIN_Z):
                        print(f"PAIRED GATE {label_r}: inconclusive (diff {rec['diff']:+.4f}, z {rec['z']:+.2f}); "
                              f"extending with {max(1, gate_reps)} more session(s) per task on both sides", flush=True)
                        ext_results, _ = await _run_entries(
                            gate_tasks, label=f"{gate_split_name}-extension", run_root=val_root,
                            output_stage="outputs_ext")
                        ext_champ, _ = await _run_entries(
                            gate_tasks, label=f"{gate_split_name}-champion({champ.label})-extension",
                            run_root=val_root, skill_dir=Path(champ.skill_dir),
                            output_stage="champion_outputs_ext")
                        gate_spend_results += ext_results + ext_champ
                        e_scores, e_pass = collect_scores(ext_results)
                        for t, xs in e_scores.items():
                            cand_scores.setdefault(t, []).extend(xs)
                        for t, xs in e_pass.items():
                            cand_pass.setdefault(t, []).extend(xs)
                        c_scores, c_pass = collect_scores(ext_champ)
                        paired_gate.pool_champion(c_scores, c_pass)
                        first = paired_gate.history.pop()
                        rec = paired_gate.decide(label_r, str(snap), cand_scores, cand_pass, round_index=iteration)
                        rec["extended"] = True
                        rec["first_look"] = {"diff": first["diff"], "z": first["z"]}
                    paired_accept = rec["decision"] == "accepted"
            finally:
                config.executor_reps = _saved_reps
            paired_gate.save(val_root / "paired_gate.json")
            g_pass = sum(1 for xs in cand_pass.values() if sum(xs) * 2 > len(xs))
            gate_record = {
                "split": gate_split_name, "mode": "paired", "reps": max(1, int(gate_reps)),
                "pass": g_pass, "n": len(gate_results), **rec,
            }
            try:
                gate_dir = val_root / f"round_{iteration}"
                gate_dir.mkdir(parents=True, exist_ok=True)
                (gate_dir / "gate_results.json").write_text(json.dumps({
                    "summary": gate_record, "candidate_scores": cand_scores, "candidate_passes": cand_pass,
                }, indent=2))
            except OSError:
                pass
            print(f"PAIRED GATE {label_r}: {rec['decision']}"
                  + (f"  candidate {rec.get('candidate_fitness')} vs champion "
                     f"{rec.get('champion_fitness_before')} ({rec.get('champion_before')}, "
                     f"{rec.get('champion_sessions_before')} sessions): diff {rec.get('diff'):+.4f} "
                     f"± {rec.get('se')} (z {rec.get('z'):+.2f}, wins/losses {rec.get('wins')}/{rec.get('losses')})"
                     if "diff" in rec and rec["decision"] in ("accepted", "rejected") else "")
                  + f"  -> champion {rec['champion']} fitness {rec['champion_fitness']} "
                    f"pass {rec['champion_pass_rate']:.1%} ({rec['champion_sessions']} sessions)", flush=True)
            fitness = float(rec["champion_fitness"])
        elif gate_tasks and not skip_generator:
            val_root = run_dir.with_name(run_dir.name + "_val")
            print(f"\nGate split '{gate_split_name}': {len(gate_tasks)} task(s), "
                  f"1 session each, mode={gate_mode}", flush=True)
            gate_results, _ = await _run_entries(
                gate_tasks, label=gate_split_name, run_root=val_root,
            )
            g_pass, g_rate, g_mean = _round_metrics(
                [o for _, _, _, o, _, _ in gate_results]
            )
            gate_fitness = _selection_fitness(g_rate, g_mean)
            gate_record = {
                "split": gate_split_name, "mode": gate_mode,
                "pass": g_pass, "n": len(gate_results),
                "pass_rate": round(g_rate, 4), "mean_score": round(g_mean, 4),
                "fitness": round(gate_fitness, 6),
                "checkpoint_pass": best_gate_pass,
                "min_drop": gate_min_drop, "decision": "kept",
            }
            try:
                gate_dir = val_root / f"round_{iteration}"
                gate_dir.mkdir(parents=True, exist_ok=True)
                (gate_dir / "gate_results.json").write_text(json.dumps({
                    "summary": gate_record,
                    "tasks": [
                        _build_run_artifact(
                            tag=f"{tid}/{inst}", er=er, orc=orc,
                            executor_log_path=out_dir / "executor.log",
                            output_dir=out_dir, input_dir=in_dir,
                        )
                        for tid, inst, er, orc, out_dir, in_dir in gate_results
                    ],
                }, indent=2))
            except OSError:
                pass
            print(f"Gate split ({gate_split_name}): {g_pass}/{len(gate_results)} "
                  f"({g_rate:.1%});  mean score {g_mean:.3f};  "
                  f"checkpoint {best_gate_pass if best_gate_pass is not None else '-'}")
            if gate_mode == "select":
                fitness = gate_fitness

        # Regression trigger: gate-split pass count in select mode, else the
        # train-fitness tolerance of the legacy rule.
        if paired_mode and gate_record is not None:
            regression_triggered = paired_accept is False
        elif gate_record is not None and gate_mode == "select":
            regression_triggered = (
                best_gate_pass is not None
                and gate_record["pass"] <= best_gate_pass - gate_min_drop
            )
        else:
            regression_triggered = (
                regression_tol >= 0 and fitness < best_fitness - regression_tol
            )

        # Best checkpoint. This runs BEFORE the target check on purpose:
        # the round that reaches the target is usually the best round, and
        # it still has to be snapshotted and counted. While this sat after
        # the `break`, a run that hit the target left run_dir/best_skill
        # missing and reported "Best selection fitness: 0.000000".
        new_best_this_round = False
        if paired_mode and gate_record is not None and paired_accept is not True:
            # Paired mode: only an accepted candidate becomes the checkpoint;
            # the fitness tracked is the champion's pooled gate fitness.
            best_fitness = fitness
        if (paired_accept is True) if (paired_mode and gate_record is not None) else (fitness > best_fitness + 1e-9):
            best_fitness = fitness
            best_skill_dir = run_dir / "best_skill"
            _save_best_skill(current_skill, best_skill_dir)
            new_best_this_round = True
            print(f"New best fitness {fitness:.3f} → checkpoint saved")
            if gate_record is not None and not paired_mode:
                best_gate_pass = gate_record["pass"]
                gate_record["decision"] = "new_best"

        # Account for spend before the target check: the round that reaches
        # the target still cost money, and `break` skips everything below.
        _exec_usd, _oracle_usd = _round_spend(results)
        if gate_spend_results:  # paired gate: candidate and champion sessions on the gate split
            _g_exec, _g_oracle = _round_spend(gate_spend_results)
            _exec_usd += _g_exec
            _oracle_usd += _g_oracle
        run_cost_usd += _exec_usd + _oracle_usd
        print(
            f"Round spend: ${_exec_usd + _oracle_usd:.2f} "
            f"(executor ${_exec_usd:.2f} + oracle ${_oracle_usd:.2f}) "
            f"over {len(results)} task(s); run total ${run_cost_usd:.2f}"
        )
        # Sessions export their part meshes next to every URDF (97% of a run
        # directory): hardlink the identical copies so a run does not fill the disk.
        try:
            _n_l, _b_l = hardlink_duplicate_meshes(
                [run_dir, run_dir.with_name(run_dir.name + "_val")], _mesh_dedup_index)
            if _n_l:
                print(f"  Mesh dedup: hardlinked {_n_l} identical file(s), freed {_b_l / 1e9:.2f} GB")
        except Exception as exc:  # noqa: BLE001 - never fail a round over disk housekeeping
            log.warning("mesh dedup failed: %s", exc)

        # Deliberately NOT a break. Everything below — round_feedback.json,
        # all_artifacts.json, the per-split rates, the cost tables and the
        # loop-state checkpoint — is the round's record, and the experiment
        # report is built from it. Breaking here left round_<n>/ holding
        # nothing but outputs/, so a run that passed 100% rendered a report
        # claiming 0%. Finish the round, skip only the rewrite (there is
        # nothing left to improve), and break at the bottom.
        target_reached = fb_pass_rate >= 0.9
        if target_reached:
            print(f"\n*** Target reached! ***")

        # Regression gate — only meaningful when another round follows. If
        # this round regressed past tolerance, the generator forks from the
        # best checkpoint instead; without it one bad edit compounds for
        # the rest of the run (observed: 0.667 → 0.278 over three rounds
        # with no recovery path). A round that just set a new best cannot
        # match this condition, since best_fitness now equals fitness.
        skill_rolled_back = False
        if (
            regression_triggered
            and best_skill_dir is not None
            and Path(best_skill_dir).is_dir()
            and Path(current_skill).resolve() != Path(best_skill_dir).resolve()
            # A path difference is not a skill difference: after every
            # candidate is refused, best_skill is a byte-for-byte copy of
            # the skill still in use, and "rolling back" to it is a no-op
            # that also told the Generator its edits were reverted and
            # made the A/B replay the baseline a second time.
            and not _skill_content_identical(Path(current_skill), Path(best_skill_dir))
        ):
            if paired_mode:
                print(f"PAIRED GATE: candidate rejected; next generator forks from the "
                      f"champion {paired_gate.champion.label}")
            elif gate_record is not None and gate_mode == "select":
                print(f"REGRESSION GATE: {gate_split_name} {gate_record['pass']}/"
                      f"{gate_record['n']} ≤ checkpoint {best_gate_pass} − "
                      f"{gate_min_drop} tasks; next generator forks from the "
                      "best checkpoint")
            else:
                print(f"REGRESSION GATE: fitness {fitness:.3f} < best "
                      f"{best_fitness:.3f} − tol {regression_tol:.2f}; next "
                      f"generator forks from the best checkpoint")
            if gate_record is not None and not paired_mode:
                gate_record["decision"] = "rolled_back"
            current_skill = Path(best_skill_dir)
            skill_rolled_back = True

        # Stage B — bundle per-task artifacts for the generator. The
        # generator sees ONLY the feedback_split (or all entries if no
        # split was set). Held-out splits are still oracle-evaluated and
        # logged below as `split_pass_rates`, but never reach the
        # generator prompt — the loop's job is to keep them honest.
        run_artifacts: list[dict[str, Any]] = []
        all_artifacts: list[dict[str, Any]] = []
        for tid, inst, er, orc, out_dir, in_dir in results:
            a = _build_run_artifact(
                tag=f"{tid}/{inst}",
                er=er,
                orc=orc,
                executor_log_path=out_dir / "executor.log",
                output_dir=out_dir,
                input_dir=in_dir,
            )
            split = entry_split.get((tid, inst))
            if split is not None:
                a["split"] = split
            all_artifacts.append(a)
            if feedback_split is None or split == feedback_split:
                run_artifacts.append(a)

        # Per-split pass rates for visibility (e.g. catch test regressions
        # while train improves). All artifacts go into the run log; only
        # `run_artifacts` (filtered) is shown to the generator.
        split_pass_rates: dict[str, str] = {}
        if any("split" in a for a in all_artifacts):
            by_split: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for a in all_artifacts:
                s = a.get("split") or "_unsplit"
                by_split[s][1] += 1
                if a.get("oracle_verdict") == "PASS":
                    by_split[s][0] += 1
            for s, (p, t) in by_split.items():
                pct = (p / t) if t else 0.0
                split_pass_rates[s] = f"{p}/{t} ({pct:.0%})"
            if split_pass_rates:
                print("Per-split pass rates: " + ", ".join(
                    f"{s}={v}" for s, v in sorted(split_pass_rates.items())
                ))

        # Aggregate cost across the batch — read off run_artifacts
        # (each artifact has its own cost_summary). Surfacing this lets
        # the generator detect cost regressions ("output tokens up 50%
        # but pass rate flat → last edit over-engineered").
        n_runs = max(len(run_artifacts), 1)
        _costs = [a.get("cost_summary") or {} for a in run_artifacts]
        batch_cost_summary = {
            "total_usd": round(sum(c.get("cost_usd", 0) for c in _costs), 4),
            "mean_usd_per_task": round(
                sum(c.get("cost_usd", 0) for c in _costs) / n_runs, 4),
            "executor_usd": round(
                sum(c.get("executor_usd", 0) for c in _costs), 4),
            "oracle_usd": round(
                sum(c.get("oracle_usd") or 0 for c in _costs), 4),
            "total_output_tokens": sum(c.get("output_tokens", 0) for c in _costs),
            "total_api_calls": sum(c.get("n_api_calls", 0) for c in _costs),
            "total_cache_misses": sum(c.get("cache_misses", 0) for c in _costs),
        }

        # Generator's own cost from the previous round (if any) — surfaces
        # the fact that the rewrite itself isn't free, and a generator
        # turn that takes 50K output tokens to add three lines is wasteful.
        prev_generator_cost: dict[str, Any] | None = None
        if iteration > 0:
            prev_round_dir = run_dir / f"round_{iteration - 1}"
            prev_gen_result = prev_round_dir / "generator.result.json"
            prev_gen_jsonl = prev_round_dir / "generator.session.jsonl"
            ps: dict[str, Any] | None = None
            if prev_gen_result.exists():
                try:
                    ps = json.loads(prev_gen_result.read_text()).get("usage")
                except Exception:
                    ps = None
            if ps is None and prev_gen_jsonl.exists():
                ps = _summarize_session(prev_gen_jsonl)
            if ps is not None:
                prev_generator_cost = {
                    "round": iteration - 1,
                    "cost_usd": round(ps["cost_usd"], 4),
                    "output_tokens": ps["output"],
                    "n_api_calls": ps["n_api_calls"],
                }

        n_failures = sum(
            1 for a in run_artifacts
            if a.get("oracle_verdict") in ("FAIL", "execution_failed", "no_oracle_result")
        )
        round_feedback: dict[str, Any] = {
            # Feedback-split metrics only — the generator must never see
            # (or be implicitly steered by) held-out split performance.
            "round": iteration, "pass_rate": fb_pass_rate,
            "mean_oracle_score": round(fb_mean_score, 3),
            "selection_fitness": round(fitness, 6),
            "n_failures": n_failures,
            "per_check_pass_rate": {} if _feedback_format() == "score" else {
                name: f"{p}/{t}" for name, (p, t) in per_check_pass.items()
            },
            "batch_cost_summary": batch_cost_summary,
            "runs": run_artifacts,
        }
        if adaptive_decision is not None:
            round_feedback["adaptive_train_decision"] = asdict(
                adaptive_decision
            )
        if skill_rolled_back:
            round_feedback["skill_rolled_back"] = True
        if new_best_this_round:
            round_feedback["new_best"] = True
        if gate_record is not None:
            round_feedback["gate"] = gate_record
            # The on-disk summary was written before the decision was known.
            try:
                gp = run_dir.with_name(run_dir.name + "_val") / f"round_{iteration}" / "gate_results.json"
                gd = json.loads(gp.read_text()); gd["summary"] = gate_record  # noqa: E702
                gp.write_text(json.dumps(gd, indent=2))
            except (OSError, ValueError):
                pass
        if prev_generator_cost is not None:
            round_feedback["prev_generator_cost"] = prev_generator_cost
        if split_pass_rates:
            # Logged but NOT in `runs` — only the loop / human sees it,
            # not the generator. Keeps test feedback truly held-out.
            round_feedback["split_pass_rates"] = split_pass_rates
        feedback_history.append(round_feedback)

        # Snapshot ALL artifacts (including held-out splits) to disk for
        # post-hoc analysis. Loop_state.json + generator prompts only
        # ever contain `run_artifacts`, which respects the split filter.
        try:
            (run_dir / f"round_{iteration}" / "all_artifacts.json").write_text(
                json.dumps(all_artifacts, indent=2)
            )
        except OSError:
            pass
        if _feedback_format() == "score":
            _hide_oracle_files(run_dir, iteration)

        # Stage C — generator (serial; one global skill rewrite per round).
        # Skipped entirely when --no-generator is set (e.g. evaluation-only
        # runs where we just want executor+oracle on each batch), and when
        # the round already reached the target.
        candidate_preflight: SkillPreflightResult | None = None
        if skip_generator or target_reached:
            print(
                "Generator stage SKIPPED "
                + ("(--no-generator)." if skip_generator
                   else "(target reached; nothing left to improve).")
            )
        else:
            # If `generator_auth` is set and differs from the loop's auth,
            # temporarily swap the auth env so the generator's claude
            # subprocess captures different creds while leaving the next
            # batch's executor env untouched.
            round_dir_path = run_dir / f"round_{iteration}"
            n_updates = max(1, int(getattr(config, "updates_per_round", 1)))
            stacked_results = None
            stacked_pass_rate = stacked_mean_score = None
            update_records: list[dict] = []
            no_more_updates = False
            for update_index in range(n_updates):
                upd = round_dir_path if n_updates == 1 else round_dir_path / f"update_{update_index}"
                upd.mkdir(parents=True, exist_ok=True)
                ab_stage = "ab_candidate_outputs" if n_updates == 1 else f"update_{update_index}/ab_candidate_outputs"
                ab_baseline_stage = "ab_baseline_outputs" if n_updates == 1 else f"update_{update_index}/ab_baseline_outputs"
                update_note = None
                if n_updates > 1:
                    done = "\n".join(
                        f"  - update {r['update']}: {r['action']} ({r['reason']}; paired mean_diff "
                        f"{(r['paired_mean_diff'] or 0):+.3f}) — {r['summary'][:120]}"
                        for r in update_records
                    ) or "  (none yet)"
                    update_note = (
                        f"### Updates this round: this is update {update_index + 1} of {n_updates}\n"
                        "The skill evolves through independent updates, each A/B-tested on the same "
                        "assets before the next one starts; your working directory already contains "
                        "every update accepted so far this round. Make exactly ONE self-contained "
                        "change this turn (one tool, one rule, or one fix — the smallest edit that "
                        "addresses the dominant remaining failure) and leave other ideas for later "
                        "updates. If nothing else is worth changing, say so and modify no file.\n"
                        f"Updates so far this round:\n{done}"
                    )
                new_skill_dir = upd / "skill"
                gen_model = config.generator_model
                if config.generator_backend == "claude" and generator_auth == "local":
                    gen_ctx = _auth_swap("local")
                    gen_model = "claude-opus-4-7"  # local OAuth has no [次] prefix
                else:
                    gen_ctx = contextlib.nullcontext()
                # Generator gets a sim worker on a dedicated port (one beyond the
                # executor pool). Lets the generator's subagents test newly
                # written tools against the real sim runtime.
                gen_sim_port = config.sim_base_port + config.parallel
                generator_base_skill = current_skill
                skill_content = _read_skill_content(current_skill)
                repair_note: str | None = None
                # A candidate rejected before A/B invalidates nothing about this
                # round's evaluation: the Generator simply gets another turn with
                # the reason, instead of the whole tranche being re-run.
                for repair_attempt in range(config.generator_repair_retries + 1):
                    if repair_attempt:
                        # keep the rejected attempt's artifacts for forensics
                        for name in ("generator.log", "generator_journal.md",
                                     "generator.responses.json", "generator.result.json",
                                     "candidate_skill_preflight.json"):
                            src = upd / name
                            if src.exists():
                                src.replace(upd / f"{Path(name).stem}.rejected{repair_attempt}{Path(name).suffix}")
                        if new_skill_dir.exists():
                            rejected_dir = upd / f"skill.rejected{repair_attempt}"
                            shutil.rmtree(rejected_dir, ignore_errors=True)
                            new_skill_dir.replace(rejected_dir)
                        print(
                            f"  [generator-repair] attempt {repair_attempt}/"
                            f"{config.generator_repair_retries}: {repair_note[:160] if repair_note else ''}",
                            flush=True,
                        )
                    # The executor pool is idle during the Generator turn: lend it a
                    # port so the Generator can smoke-test its candidate with one
                    # real executor session (`try_skill`), always on the first
                    # Train task of this round's batch (the sorted pool when the
                    # batch is the pool).
                    trial_entry = (round_batch or task_ids)[0]
                    trial_key = f"{trial_entry['task_id']}/{trial_entry['instance']}"
                    # Cluster feedback: the Generator gates each cluster edit on
                    # that cluster's trial tasks, scored by the same oracle as the
                    # batch and compared with this round's artifacts (the baseline).
                    cluster_task_entries: dict[str, list[dict]] | None = None
                    cluster_trial_keys: dict[str, list[str]] | None = None
                    round_clusters: list[dict[str, Any]] = []
                    trial_ports = [config.sim_base_port]
                    if _feedback_format() == "cluster":
                        from evo.engine.failure_clusters import (
                            TRIAL_TASKS_PER_CLUSTER, cluster_runs, persistence_counts, select_trial_tasks,
                        )
                        _clusters = cluster_runs(run_artifacts)
                        round_clusters = _clusters
                        _prev_runs = feedback_history[-2].get("runs") if len(feedback_history) >= 2 else None
                        persistence_counts(_clusters, _prev_runs)
                        cluster_trial_keys = select_trial_tasks(_clusters)
                        _entry_by_key = {f"{e['task_id']}/{e['instance']}": e for e in (round_batch or task_ids)}
                        cluster_task_entries = {
                            cid: [_entry_by_key[k] for k in keys if k in _entry_by_key]
                            for cid, keys in cluster_trial_keys.items()
                        }
                        trial_ports = [config.sim_base_port + i for i in range(TRIAL_TASKS_PER_CLUSTER)]
                    trial_tool, _trial_impl, trial_spend = _make_try_skill_tool(
                        config=config, round_dir=upd, skill_dir=new_skill_dir,
                        trial_task=trial_entry,
                        ports=trial_ports,
                        executor_effort=executor_effort, subagent_model=subagent_model,
                        subagent_effort=subagent_effort,
                        max_calls=config.generator_trials_per_turn,
                        current_skill_content=skill_content,
                        cluster_tasks=cluster_task_entries,
                        baseline={a["tag"]: a for a in run_artifacts} if cluster_task_entries is not None else None,
                        oracle_mode=oracle_mode, oracle_effort=oracle_effort,
                        oracle_votes=oracle_votes, oracle_model=oracle_model,
                        baseline_skill=Path(current_skill),
                    )
                    # Sub-evolvers (EVO_SUB_EVOLVERS=K): one proposal per failure
                    # cluster in parallel, each verified on unseen members of its
                    # cluster; the Generator turn below then integrates them.
                    effective_update_note = update_note
                    if config.generator_trials_per_turn <= 0:
                        # Ablation without the regression strategy: no edit gate either.
                        effective_update_note = ((update_note + "\n\n") if update_note else "") + (
                            "NOTE FOR THIS RUN: the `try_skill` tool is NOT available. Do not call it and do "
                            "not wait for an edit gate. Make your edits directly from the feedback; they are "
                            "used as written in the next round, without any trial run."
                        )
                    if (
                        sub_evolver_count() > 0 and cluster_task_entries
                        and repair_attempt == 0 and update_index == 0
                    ):
                        _sub_note, _sub_records, _sub_cost = await _run_sub_evolver_stage(
                            config=config, upd=upd, current_skill=current_skill,
                            skill_content=skill_content, feedback_history=feedback_history,
                            clusters=round_clusters, cluster_trial_keys=cluster_trial_keys,
                            entry_by_key={f"{e['task_id']}/{e['instance']}": e for e in (round_batch or task_ids)},
                            run_artifacts=run_artifacts, trial_entry=trial_entry,
                            gen_model=gen_model, generator_effort=generator_effort,
                            gen_sim_port=gen_sim_port, gen_add_dirs=gen_add_dirs,
                            executor_effort=executor_effort, subagent_model=subagent_model,
                            subagent_effort=subagent_effort, oracle_mode=oracle_mode,
                            oracle_effort=oracle_effort, oracle_votes=oracle_votes,
                            oracle_model=oracle_model, gen_ctx=gen_ctx,
                        )
                        round_feedback["sub_evolvers"] = _sub_records
                        run_cost_usd += _sub_cost
                        if _sub_note:
                            effective_update_note = ((update_note + "\n\n") if update_note else "") + _sub_note
                        print(f"  Sub-evolvers: {len(_sub_records)} proposal(s), "
                              f"{sum(1 for r in _sub_records if r['verified'])} verified, "
                              f"trial spend ${_sub_cost:.2f}; run total ${run_cost_usd:.2f}", flush=True)
                    with gen_ctx:
                        # The Generator makes one long call per round; a provider blip
                        # here used to abort the whole run (run 8, round 4). Retry the
                        # turn on transient errors with the executor's backoff.
                        # run_generator re-stages output_skill_dir itself, so a retry
                        # starts from a clean copy of the current skill.
                        for gen_attempt in range(config.transient_retries + 1):
                            gen_result = await run_generator(
                                current_skill_dir=current_skill,
                                current_skill_content=skill_content,
                                feedback_history=feedback_history,
                                output_skill_dir=new_skill_dir,
                                model=gen_model,
                                log_file=upd / "generator.log",
                                effort=generator_effort,
                                sim_port=gen_sim_port,
                                sim_headless=config.headless,
                                backend=config.generator_backend,
                                add_dirs=gen_add_dirs,
                                dataset_tasks_root=config.tasks_root,
                                repair_note=repair_note,
                                update_note=effective_update_note,
                                extra_tools=[trial_tool] if config.generator_trials_per_turn > 0 else None,
                                trial_task=trial_key if config.generator_trials_per_turn > 0 else None,
                                cluster_trial_tasks=cluster_trial_keys,
                                same_tasks_each_round=(
                                    not adaptive_train_enabled
                                    and (config.batch_size == 0 or config.fixed_batch)
                                ),
                            )
                            summary = gen_result.changes_summary or ""
                            if trial_spend.get("trials"):
                                # Edit ledger: every gated trial of this turn, labelled
                                # with the edit line the journal gives for its cluster.
                                _edits = _journal_cluster_edits(upd / "generator_journal.md")
                                for _t in trial_spend["trials"]:
                                    _e = _edits.get(_t.get("cluster"))
                                    if _e:
                                        _t["edit"], _t["kept"] = _e
                                round_feedback.setdefault("cluster_trials", []).extend(trial_spend["trials"])
                                trial_spend["trials"] = []
                            if trial_spend["calls"]:
                                run_cost_usd += trial_spend["cost_usd"]
                                round_feedback.setdefault("generator_trials", []).append(
                                    {"update": update_index, "task": trial_key,
                                     "calls": trial_spend["calls"], "clean_calls": trial_spend["clean_calls"],
                                     "last_clean": trial_spend["last_clean"],
                                     "cost_usd": round(trial_spend["cost_usd"], 4)})
                                print(f"  Generator ran {trial_spend['calls']} try_skill smoke test(s) on "
                                      f"{trial_key}, last one "
                                      f"{'clean' if trial_spend['last_clean'] else 'with problems'}, "
                                      f"${trial_spend['cost_usd']:.2f}; run total ${run_cost_usd:.2f}", flush=True)
                                trial_spend["calls"] = 0
                                trial_spend["cost_usd"] = 0.0
                                trial_spend["clean_calls"] = 0
                                trial_spend["last_clean"] = None
                            if not (
                                summary.startswith("Generator error:")
                                and _is_transient_error(summary)
                                and gen_attempt < config.transient_retries
                            ):
                                break
                            wait_s = _TRANSIENT_BACKOFF_S[min(gen_attempt, len(_TRANSIENT_BACKOFF_S) - 1)]
                            print(
                                f"  [transient-retry] generator (attempt {gen_attempt + 1}/"
                                f"{config.transient_retries}); sleeping {wait_s}s: {summary[:120]}",
                                flush=True,
                            )
                            await asyncio.sleep(wait_s)
                    # Acceptance gate: a generator turn that died mid-flight (usage
                    # limit, max_turns) leaves an unchanged or half-written skill
                    # and no journal. Adopting it silently breaks the journal chain
                    # and burns a whole round evaluating a non-edit; keep the
                    # current skill instead so the next round retries from a known
                    # state.
                    ok, reason = _validate_generator_output(
                        gen_result,
                        upd / "generator_journal.md",
                        current_skill_content=skill_content,
                    )
                    if ok:
                        candidate_preflight = preflight_skill(new_skill_dir)
                        preflight_path = upd / "candidate_skill_preflight.json"
                        preflight_path.write_text(json.dumps(
                            candidate_preflight.to_dict(), indent=2
                        ))
                        round_feedback["candidate_skill_preflight"] = (
                            candidate_preflight.to_dict()
                        )
                        if not candidate_preflight.ok:
                            ok = False
                            reason = (
                                "candidate skill preflight failed: "
                                + "; ".join(candidate_preflight.errors[:3])
                            )
                    if ok or (gen_result.changes_summary or "").startswith("Generator error:"):
                        break
                    if reason == "generator modified no skill files":
                        # The Generator is told it may change nothing when no edit
                        # is justified; that is a verdict, not a failure to repair.
                        no_more_updates = update_index > 0
                        print("Generator chose to change nothing this round; skill unchanged", flush=True)
                        break
                    repair_note = reason
                    round_feedback.setdefault("generator_repairs", []).append(reason)
                    print(f"Generator output REJECTED before A/B ({reason[:200]})", flush=True)
                if not ok and repair_note is not None and reason.startswith("candidate skill preflight failed"):
                    round_contract_failure = True
                if no_more_updates:
                    print(f"Generator: no further update this round (update {update_index})", flush=True)
                    break
                if ok and config.candidate_ab_test and round_batch:
                    # Replay the exact assets that produced this round's Train
                    # score. In the common case Generator edited the evaluated
                    # skill, so the already-computed round metrics are the B side.
                    # If the regression gate changed Generator's base checkpoint,
                    # replay that checkpoint too; comparing a candidate derived
                    # from best_skill against a different regressed skill would
                    # not be a valid A/B test.
                    sample_keys = [
                        f"{entry['task_id']}/{entry['instance']}"
                        for entry in round_batch
                    ]
                    original_keys = {(r[0], r[1]) for r in results}
                    requested_keys = {
                        (entry["task_id"], entry["instance"])
                        for entry in round_batch
                    }
                    can_reuse_round = (
                        Path(generator_base_skill).resolve()
                        == Path(evaluated_skill).resolve()
                        and original_keys == requested_keys
                    )

                    if stacked_results is not None:
                        # the base is the previous accepted update of this round;
                        # its A/B replay on these exact assets is the baseline
                        baseline_source = "previous_update_replay"
                        baseline_pass_rate = stacked_pass_rate
                        baseline_mean_score = stacked_mean_score
                        baseline_comparison_results = stacked_results
                    elif can_reuse_round:
                        baseline_source = "round_train_results"
                        baseline_pass_rate = fb_pass_rate
                        baseline_mean_score = fb_mean_score
                        baseline_comparison_results = results
                    else:
                        baseline_source = "replayed_generator_base"
                        print(
                            "Candidate A/B: replaying Generator base on "
                            f"{len(round_batch)} exact Train assets"
                        )
                        baseline_ab_results, _ = await _run_entries(
                            round_batch,
                            label="ab-baseline",
                            apply_train_policy=True,
                            skill_dir=generator_base_skill,
                            output_stage=ab_baseline_stage,
                        )
                        ab_error = _first_training_system_error(
                            baseline_ab_results
                        )
                        if ab_error is not None:
                            failure_path = _write_training_system_failure(
                                run_dir / f"round_{iteration}",
                                stage="candidate_ab_baseline",
                                error=ab_error,
                            )
                            raise TrainingSystemError(
                                "Candidate A/B baseline hit a training-system "
                                "failure; candidate was not accepted or rejected. "
                                f"Fix the runtime and rerun the current round. "
                                f"Details: {failure_path}"
                            )
                        _, baseline_pass_rate, baseline_mean_score = (
                            _round_metrics([
                                o for _, _, _, o, _, _ in baseline_ab_results
                            ])
                        )
                        baseline_comparison_results = baseline_ab_results

                    # The replay re-renders every asset on browser pages that have
                    # already served the whole evaluation tranche plus the
                    # Generator's own renders. At 30 workers those pages started
                    # dying ("page disconnected") on the first A/B render of every
                    # session (run 16, 2026-10-02: 286 retries, 0/20 replays
                    # finished). Nothing renders at this point, so swap in fresh
                    # browsers exactly as at a round boundary.
                    if recycle_shared_pool("candidate A/B replay"):
                        print("LightPhyx pool recycled (fresh browsers for the A/B replay)", flush=True)
                    print(
                        "Candidate A/B: testing Generator candidate on "
                        f"{len(round_batch)} exact Train assets"
                    )
                    candidate_results, _ = await _run_entries(
                        round_batch,
                        label="ab-candidate",
                        apply_train_policy=True,
                        skill_dir=new_skill_dir,
                        output_stage=ab_stage,
                    )
                    ab_error = _first_training_system_error(candidate_results)
                    if ab_error is not None:
                        failure_path = _write_training_system_failure(
                            run_dir / f"round_{iteration}",
                            stage="candidate_ab_candidate",
                            error=ab_error,
                        )
                        raise TrainingSystemError(
                            "Generator candidate A/B hit a training-system "
                            "failure; candidate was not accepted or rejected. "
                            f"Fix the runtime and rerun the current round. "
                            f"Details: {failure_path}"
                        )
                    _, candidate_pass_rate, candidate_mean_score = _round_metrics(
                        [o for _, _, _, o, _, _ in candidate_results]
                    )
                    candidate_tool_errors = (
                        _candidate_tool_contract_errors(candidate_results)
                    )
                    candidate_executor_errors = (
                        _candidate_executor_code_errors(candidate_results)
                    )
                    critical_regressions = _critical_ab_regressions(
                        baseline_comparison_results,
                        candidate_results,
                    )
                    paired_scores = _paired_task_scores(
                        baseline_comparison_results, candidate_results
                    )
                    paired_joint_scores = _paired_joint_scores(
                        baseline_comparison_results, candidate_results
                    )
                    paired_unit = "joint" if paired_joint_scores else "task"
                    paired_for_gate = paired_joint_scores or paired_scores
                    feedback_pool_size = sum(
                        1 for entry in task_ids
                        if feedback_split is None
                        or entry.get("split") == feedback_split
                    )
                    minimum_ab_samples = min(
                        config.candidate_ab_min_samples,
                        max(1, feedback_pool_size),
                    )
                    ab_decision = _candidate_ab_decision(
                        baseline_pass_rate=baseline_pass_rate,
                        baseline_mean_score=baseline_mean_score,
                        candidate_pass_rate=candidate_pass_rate,
                        candidate_mean_score=candidate_mean_score,
                        sample_count=len(candidate_results),
                        minimum_samples=minimum_ab_samples,
                        candidate_tool_contract_errors=candidate_tool_errors,
                        executor_code_errors=candidate_executor_errors,
                        critical_regressions=critical_regressions,
                        paired_diffs=tuple(c - b for _k, b, c in paired_for_gate),
                    )
                    ab_record = {
                        "samples": sample_keys,
                        "baseline_source": baseline_source,
                        **asdict(ab_decision),
                        "candidate_outputs": str(round_dir_path / ab_stage),
                        "paired_unit": paired_unit,
                        "paired_scores": [
                            {"task": k, "baseline": round(b, 4), "candidate": round(c, 4)}
                            for k, b, c in paired_scores
                        ],
                        "paired_joint_scores": [
                            {"joint": k, "baseline": round(b, 4), "candidate": round(c, 4)}
                            for k, b, c in paired_joint_scores
                        ],
                    }
                    round_feedback["candidate_ab_test"] = ab_record
                    # The replays cost as much as the round itself; count them.
                    ab_exec, ab_oracle = _round_spend(candidate_results)
                    if baseline_source == "replayed_generator_base":
                        extra_exec, extra_oracle = _round_spend(baseline_ab_results)
                        ab_exec += extra_exec
                        ab_oracle += extra_oracle
                    run_cost_usd += ab_exec + ab_oracle
                    print(
                        f"A/B spend: ${ab_exec + ab_oracle:.2f} (executor ${ab_exec:.2f}"
                        f" + oracle ${ab_oracle:.2f}); run total ${run_cost_usd:.2f}"
                    )
                    try:
                        (upd / "candidate_ab_test.json").write_text(json.dumps(ab_record, indent=2))
                    except OSError:
                        pass
                    print(
                        f"Candidate A/B: {ab_decision.action}; baseline="
                        f"{ab_decision.baseline_fitness:.6f}, candidate="
                        f"{ab_decision.candidate_fitness:.6f}; "
                        f"reason={ab_decision.reason}; paired n={ab_decision.paired_n} {paired_unit}s "
                        f"mean_diff={ab_decision.paired_mean_diff:+.3f}"
                        + (f" t={ab_decision.paired_t:.2f} (crit -{ab_decision.paired_t_critical:.2f})"
                           if ab_decision.paired_t is not None else "")
                    )
                    update_records.append({
                        "update": update_index, "skill": str(new_skill_dir),
                        "summary": (gen_result.changes_summary or "")[:200],
                        **{k: ab_record[k] for k in ("action", "reason", "paired_n", "paired_mean_diff", "paired_t")},
                    })
                    if ab_decision.accepted:
                        current_skill = new_skill_dir
                        stacked_results = candidate_results
                        stacked_pass_rate = candidate_pass_rate
                        stacked_mean_score = candidate_mean_score
                    else:
                        current_skill = generator_base_skill
                        if (
                            ab_decision.reason
                            == "candidate_tool_contract_error"
                        ):
                            round_contract_failure = True
                        round_feedback["generator_output_rejected"] = (
                            f"candidate_ab:{ab_decision.reason}"
                        )
                elif ok and config.candidate_ab_test:
                    current_skill = generator_base_skill
                    reason = "candidate A/B has no completed Train samples"
                    print(f"Generator output REJECTED ({reason})")
                    round_feedback["generator_output_rejected"] = reason
                elif ok:
                    current_skill = new_skill_dir
                    round_feedback["generator_changed"] = True
                else:
                    print(f"Generator output REJECTED ({reason}); keeping "
                          f"previous skill for next round")
                    round_feedback["generator_output_rejected"] = reason
                _install_skill(current_skill)
                print(f"Generator: {gen_result.changes_summary[:300]}")
                if (
                    not ok
                    and gen_result.changes_summary.startswith("Generator error:")
                ):
                    failure_path = _write_training_system_failure(
                        run_dir / f"round_{iteration}",
                        stage="generator",
                        error=gen_result.changes_summary,
                    )
                    raise TrainingSystemError(
                        "Generator backend failed; the outer round was not "
                        "advanced. Fix the training system and rerun the current "
                        f"round. Details: {failure_path}"
                    )

            if n_updates > 1:
                round_feedback["updates"] = update_records
                # lineage state and journal chain in the places the rest of the
                # loop (resume, reports, valcurve manifests) expect them
                final_skill = round_dir_path / "skill"
                shutil.rmtree(final_skill, ignore_errors=True)
                shutil.copytree(current_skill, final_skill, ignore=shutil.ignore_patterns("__pycache__"))
                journals = sorted(round_dir_path.glob("update_*/generator_journal.md"))
                if journals:
                    (round_dir_path / "generator_journal.md").write_text(
                        "\n\n".join(f"## Update {j.parent.name.split('_')[1]}\n\n{j.read_text()}" for j in journals)
                    )
                print("Round updates: " + "; ".join(f"u{r['update']}={r['action'].split('_')[0]}" for r in update_records), flush=True)


        # Isolation audit: GT paths must never appear in any agent session.
        # Catches the observed failure mode where a generator read GT URDFs
        # and baked their values into a tool as constants.
        _audit_round_isolation(
            run_dir, iteration,
            forbidden=[str(config.gt_root.resolve()) + "/",
                       str(_private_dir(run_dir))],
        )

        if round_contract_failure:
            consecutive_contract_repairs += 1
            round_feedback["round_kind"] = "contract_repair"
            round_feedback["objective_round"] = objective_rounds_completed
        else:
            objective_rounds_completed += 1
            consecutive_contract_repairs = 0
            round_feedback["round_kind"] = "objective_evolution"
            round_feedback["objective_round"] = objective_rounds_completed - 1

        # Round complete — checkpoint state so a 5-hour-window break or a
        # process kill can resume from here on next launch.
        _save_loop_state(
            run_dir,
            round_completed=iteration,
            current_skill_dir=current_skill,
            feedback_history=feedback_history,
            best_fitness=best_fitness,
            best_skill_dir=best_skill_dir,
            seen_by_diff=seen_by_diff,
            objective_rounds_completed=objective_rounds_completed,
            consecutive_contract_repairs=consecutive_contract_repairs,
            seen_train=seen_train,
            best_gate_pass=best_gate_pass,
        )

        if (
            round_contract_failure
            and consecutive_contract_repairs
            >= config.max_contract_repair_attempts
        ):
            failure_path = run_dir / f"round_{iteration}" / "skill_contract_failure.json"
            failure_path.write_text(json.dumps({
                "action": "STOP_AND_FIX_SKILL",
                "attempts": consecutive_contract_repairs,
                "max_attempts": config.max_contract_repair_attempts,
                "blocking_error": (
                    adaptive_decision.blocking_error
                    if adaptive_decision is not None else None
                ),
                "candidate_preflight": (
                    candidate_preflight.to_dict()
                    if candidate_preflight is not None else None
                ),
            }, indent=2))
            raise SkillContractError(
                "skill contract repair did not stabilize within "
                f"{config.max_contract_repair_attempts} attempts; "
                f"formal training stopped at {failure_path}"
            )

        if target_reached:
            break

        iteration += 1

    except SimStartupError as e:
        print(f"\n*** FATAL: sim worker failed to start: {e} ***")
        sys.exit(1)
    finally:
        if generate_report and not skip_generator and any(run_dir.glob("round_*")):
            try:
                from evo.reporting.experiment_report import (
                    generate_experiment_report,
                )

                report_path = generate_experiment_report(run_dir)
                print(f"\nExperiment report: {report_path}")
            except Exception as report_error:
                # Reporting is post-hoc observability. Never turn a completed
                # training run into a failed run because ffmpeg or one damaged
                # historical artifact could not be summarized.
                print(
                    "\nWarning: experiment report generation failed: "
                    f"{report_error}",
                    file=sys.stderr,
                )

    print(f"\nBest selection fitness (binary pass rate first, diagnostic "
          f"tie-break second): {best_fitness:.6f}")
    print(f"Total measured spend: ${run_cost_usd:.2f} "
          f"(executor + subagent oracle; priced by evo/engine/pricing.py)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Transient-error detection
# ---------------------------------------------------------------------------

# Two provider outages on 2026-09-18 each lasted 15-20 min and exhausted a
# 30/60/120 s budget; six attempts now ride out ~28 min before aborting.
_TRANSIENT_BACKOFF_S = (30, 60, 120, 300, 600, 600)

_TRANSIENT_PATTERNS = (
    # Provider connectivity blips: the Agents SDK surfaces these as a bare
    # "Connection error." and run 7 aborted a 10-round experiment on one.
    "connection error", "connection reset", "connection closed",
    "connection refused", "connection timeout", "timed out", "timeout",
    "error getting response", "bad gateway", "service unavailable",
    "gateway timeout", "up_error", "上游错误",
    # Renderer / sim-worker transients (a lost lightphyx page kills one task,
    # the pool survives): retry the task instead of scoring the outage.
    "page disconnected", "simulation render failed", "browser has been closed",
    "browser closed", "failed to fetch", "sim worker exited", "sim worker socket",
    "closed without replying",
    "rate limit", "rate_limit", "ratelimit",
    "usage limit", "quota", "5-hour", "5 hour",
    "session limit", "session_limit", "session has expired",
    "insufficient credit", "insufficient balance",
    "insufficient_quota", "余额不足",
    "overloaded",
    # Gateway stream drops (run 10, 2026-09-24: cliproxyapi answered 408
    # "stream disconnected before completion" during a DNS outage; the two
    # sessions were scored as failures and flipped a K=3 vote).
    "request timeout", "stream disconnected", "stream closed before",
    "stream error",
    # Gateway / upstream short outages (linkapi 504 burst on 2026-05-04
    # took out 18 calls across rounds 4–5 + every generator after round 1).
    # The HTTP status codes themselves (408/429/502/503/504) are matched by
    # _TRANSIENT_STATUS_RE below, as whole tokens in an error context only.
    "server_error", "gateway timeout", "bad gateway",
    "service unavailable",
    # linkapi-specific upstream-side failures: proxy returns 400 with
    # `up_bad_request` / `up_error` / `请求失败` when the routed upstream
    # provider is flaky or doesn't support the request type (e.g.
    # vision payload routed to a non-vision model). These look terminal
    # but in practice clear after a short wait.
    "up_bad_request", "up_error", "上游错误",
    "请求失败",
    # SDK transport-level fallback: when the upstream error doesn't make
    # it into result.error verbatim, the SDK only surfaces a generic
    # exit-code message. Treat it as transient — better to spend a
    # couple extra retries than to lose every batch slot.
    "command failed with exit code",
)



def _port_is_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _is_orphaned_sim_worker(pid: str) -> bool:
    """True for a sim worker whose parent run has exited (reclaimable)."""
    import subprocess
    out = subprocess.run(["ps", "-p", pid, "-o", "ppid=,command="], capture_output=True,
                         text=True, check=False).stdout.strip().lower()
    if not out:
        return True                      # already gone
    ppid, _, cmd = out.partition(" ")
    is_worker = "sim_worker" in cmd or "blender" in cmd
    return is_worker and ppid.strip() == "1"


def _shift_to_free_port_range(config: "EvoConfig", *, max_shift: int = 2000) -> int:
    """Move `config.sim_base_port` up until workers + generator ports are all free.

    Every worker slot pins `sim_base_port + i` and the generator uses
    `sim_base_port + parallel`; a single foreign listener inside that range
    (run 12, 2026-09-29: ZeroTier on 9993 inside 9976-9996) killed every
    session that drew the slot and aborted the run as a training-system
    failure. Only an orphaned sim worker (its parent run is gone, so its
    parent pid is 1) is reclaimable by SimManager; a worker whose parent is
    alive belongs to another run on this machine and its port is foreign.
    Before this rule two runs started in the same second on one host
    (2026-11-03) shared 9876+: the second killed the first's workers and one
    executor talked to the other run's sim session.
    """
    import subprocess
    span = int(config.parallel) + 1
    base = int(config.sim_base_port)
    for candidate in range(base, base + max_shift):
        busy = [port for port in range(candidate, candidate + span) if not _port_is_free(port)]
        foreign = []
        for port in busy:
            holder = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True,
                                    text=True, check=False).stdout.split()
            if not holder or any(not _is_orphaned_sim_worker(pid) for pid in holder):
                foreign.append(port)
        if not foreign:
            if candidate != base:
                print(f"sim ports: {base}-{base + span - 1} has a foreign listener; "
                      f"using {candidate}-{candidate + span - 1} instead", flush=True)
                config.sim_base_port = candidate
            return candidate
    raise SystemExit(f"no free block of {span} sim ports found above {base}")

# HTTP status codes that mean "try again": only as a whole token (not inside a
# float such as 0.504 or -0.429, not a traceback's "line 502") and only on a
# line that also talks about an error / status / gateway. Run 22 round 2
# (2026-10-08) re-ran 26 successful sessions because bare "502"/"504"/"429"
# substrings matched mesh bounds the executor had printed.
_TRANSIENT_STATUS_RE = re.compile(r"(?<!line )(?<![\d.])(?:408|429|502|503|504)(?![\d.])")
_TRANSIENT_STATUS_CONTEXT_RE = re.compile(
    r"error|status|http|gateway|timeout|time-out|too many|unavailable|overloaded|"
    r"upstream|retry|rate.?limit|请求|上游"
)


def _mentions_transient_status(low: str) -> bool:
    """True when some line names a retryable HTTP status in an error context."""
    for line in low.splitlines():
        if _TRANSIENT_STATUS_RE.search(line) and _TRANSIENT_STATUS_CONTEXT_RE.search(line):
            return True
    return False


def _is_transient_error(err: str | None) -> bool:
    if not err:
        return False
    low = err.lower()
    return any(p in low for p in _TRANSIENT_PATTERNS) or _mentions_transient_status(low)


# ---------------------------------------------------------------------------
# Auth swap (executor proxy ↔ generator local)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _auth_swap(target_mode: str):
    """Temporarily reconfigure auth env for one agent invocation.

    The bundled claude CLI captures `ANTHROPIC_BASE_URL` /
    `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_MODEL` at subprocess spawn time.
    Swapping to `target_mode` before a run_generator call makes that
    one subprocess use a different auth, then restoring on exit lets
    the next executor batch keep its proxy creds intact.

    Safe to call when the loop's parent auth is the same as `target_mode`
    — `setup_auth_env` is idempotent.
    """
    saved = {
        k: os.environ.get(k)
        for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL")
    }
    setup_auth_env(target_mode)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Loop-level checkpoint / resume
# ---------------------------------------------------------------------------

LOOP_STATE_FILENAME = "loop_state.json"


def _save_loop_state(
    run_dir: Path,
    *,
    round_completed: int,
    current_skill_dir: Path,
    feedback_history: list[dict[str, Any]],
    best_fitness: float,
    best_skill_dir: Path | None,
    seen_by_diff: dict[str, set[tuple[str, str]]] | None = None,
    objective_rounds_completed: int | None = None,
    consecutive_contract_repairs: int = 0,
    seen_train: set[tuple[str, str]] | None = None,
    best_gate_pass: int | None = None,
) -> None:
    """Persist enough state to resume an interrupted multi-round run.

    Writes atomically (tmp + rename) so a SIGKILL mid-write can't corrupt
    the file.  feedback_history is JSON-serializable in our schema (no
    Path objects, no sets).  seen_by_diff sets are converted to sorted
    lists for stable JSON.
    """
    seen_serializable = None
    if seen_by_diff is not None:
        seen_serializable = {
            d: sorted([list(p) for p in pairs]) for d, pairs in seen_by_diff.items()
        }
    payload = {
        "round_completed": round_completed,
        "current_skill_dir": str(current_skill_dir.resolve()),
        "feedback_history": feedback_history,
        "best_fitness": best_fitness,
        "best_skill_dir": str(best_skill_dir.resolve()) if best_skill_dir else None,
        "seen_by_diff": seen_serializable,
        "saved_at": time.time(),
        "best_gate_pass": best_gate_pass,
    }
    if objective_rounds_completed is not None:
        payload["objective_rounds_completed"] = objective_rounds_completed
        payload["consecutive_contract_repairs"] = consecutive_contract_repairs
        payload["seen_train"] = sorted(
            [list(pair) for pair in (seen_train or set())]
        )
    tmp = run_dir / (LOOP_STATE_FILENAME + ".tmp")
    final = run_dir / LOOP_STATE_FILENAME
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(final)


def _load_loop_state(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / LOOP_STATE_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _round_metrics(
    oracles: list[OracleResult | None],
) -> tuple[int, float, float]:
    """Aggregate one round's oracle results → (n_pass, pass_rate, mean_score).

    `mean_score` is the graded fitness: mean fraction of checks passed,
    with failed executions / missing oracles counting 0. Unlike the binary
    pass rate it has a gradient — a URDF passing most checks scores higher
    than one passing few even though both FAIL the gate.
    """
    n = len(oracles)
    if n == 0:
        return 0, 0.0, 0.0
    n_pass = sum(1 for o in oracles if o is not None and o.passed)
    mean_score = sum((o.score if o is not None else 0.0) for o in oracles) / n
    return n_pass, n_pass / n, mean_score


def _selection_fitness(pass_rate: float, mean_score: float) -> float:
    """Binary success first, diagnostic check score only as a tie-breaker.

    The previous training path selected checkpoints directly by mean fraction
    of passed checks.  That let a structurally wrong object with vacuous
    axis/limit checks look competitive.  With batches capped at 50, the
    largest diagnostic contribution here (0.001) is far smaller than one
    additional binary success (at least 0.02), while still distinguishing
    equally unsuccessful early rounds.
    """
    pass_rate = max(0.0, min(1.0, pass_rate))
    mean_score = max(0.0, min(1.0, mean_score))
    return pass_rate + (1.0 - pass_rate) * 0.001 * mean_score


@dataclass(frozen=True)
class CandidateABDecision:
    """Objective acceptance decision for one Generator candidate replay."""

    accepted: bool
    action: str
    baseline_fitness: float
    candidate_fitness: float
    baseline_pass_rate: float
    candidate_pass_rate: float
    baseline_mean_score: float
    candidate_mean_score: float
    reason: str
    sample_count: int
    minimum_samples: int
    candidate_tool_contract_errors: tuple[str, ...] = ()
    executor_code_errors: tuple[str, ...] = ()
    critical_regressions: tuple[str, ...] = ()
    paired_n: int = 0
    paired_mean_diff: float = 0.0
    paired_t: float | None = None
    paired_t_critical: float | None = None


# One-sided Student t critical values at alpha = 0.10, by degrees of freedom.
# Small-n replays (the default tranche is 10 tasks) sit squarely where the
# normal approximation is too generous.
_T_CRIT_ONE_SIDED_10 = {
    1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440, 7: 1.415,
    8: 1.397, 9: 1.383, 10: 1.372, 11: 1.363, 12: 1.356, 13: 1.350,
    14: 1.345, 15: 1.341, 16: 1.337, 17: 1.333, 18: 1.330, 19: 1.328,
    20: 1.325, 21: 1.323, 22: 1.321, 23: 1.319, 24: 1.318, 25: 1.316,
    26: 1.315, 27: 1.314, 28: 1.313, 29: 1.311, 30: 1.310,
}
# A mean per-task score drop this large is rejected regardless of the test:
# at n=2 or 3 the t-test has almost no power, and a collapse should not
# ride through on a wide confidence interval.
_AB_HARD_FLOOR = 0.25


def _t_critical_one_sided_10(df: int) -> float:
    return _T_CRIT_ONE_SIDED_10.get(max(1, df), 1.282)


def _candidate_ab_decision(
    *,
    baseline_pass_rate: float,
    baseline_mean_score: float,
    candidate_pass_rate: float,
    candidate_mean_score: float,
    sample_count: int = 1,
    minimum_samples: int = 1,
    candidate_tool_contract_errors: tuple[str, ...] = (),
    executor_code_errors: tuple[str, ...] = (),
    critical_regressions: tuple[str, ...] = (),
    paired_diffs: tuple[float, ...] = (),
) -> CandidateABDecision:
    """Accept a complete, executable candidate unless it is clearly worse.

    This is a non-inferiority gate, and deliberately so. The previous rule
    demanded strict improvement and rejected on any single (task, check)
    pair that flipped from pass to fail. Measured on an unchanged skill
    replayed twice on the same ten tasks, pairs flip pass->fail 21% of the
    time from executor and judge noise alone, so that rule rejected a
    skill compared with ITSELF (4 "regressions") and, with more passing
    pairs to lose, rejected good skills more often than bad ones. Four
    candidates in a row were refused; none could ever have been accepted.

    `paired_diffs` are per-task (candidate - baseline) continuous scores —
    the mean vote fraction across the oracle's checks — on the same assets.
    They carry far more information than the binary pass rate and are the
    primary evidence here. The candidate is rejected only when that paired
    mean is significantly negative (one-sided t, alpha 0.10) or below a
    hard floor; a neutral candidate is accepted, and the best-fitness
    checkpoint remains the backstop against slow drift.
    """
    baseline_fitness = _selection_fitness(baseline_pass_rate, baseline_mean_score)
    candidate_fitness = _selection_fitness(candidate_pass_rate, candidate_mean_score)

    n = len(paired_diffs)
    mean_d = sum(paired_diffs) / n if n else 0.0
    t_stat: float | None = None
    t_crit: float | None = None
    uniform_drop = False
    if n >= 2:
        var = sum((d - mean_d) ** 2 for d in paired_diffs) / (n - 1)
        sd = math.sqrt(var)
        if sd > 1e-12:
            t_stat = mean_d / (sd / math.sqrt(n))
            t_crit = _t_critical_one_sided_10(n - 1)
        else:
            uniform_drop = mean_d < 0

    if sample_count < minimum_samples:
        accepted, reason = False, "insufficient_samples"
    elif candidate_tool_contract_errors:
        accepted, reason = False, "candidate_tool_contract_error"
    elif executor_code_errors:
        accepted, reason = False, "executor_code_error"
    elif critical_regressions:
        accepted, reason = False, "critical_metric_regression"
    elif n and mean_d < -_AB_HARD_FLOOR:
        accepted, reason = False, "score_regression"
    elif n and t_stat is not None and t_crit is not None and t_stat < -t_crit:
        accepted, reason = False, "score_regression"
    elif n and uniform_drop:
        accepted, reason = False, "score_regression"
    elif not n and candidate_mean_score + _AB_HARD_FLOOR < baseline_mean_score:
        # Legacy callers without paired scores: only a collapse is rejected.
        accepted, reason = False, "score_regression"
    else:
        improved = mean_d > 1e-12 if n else candidate_mean_score > baseline_mean_score + 1e-12
        accepted, reason = True, ("score_improved" if improved else "non_inferior")

    return CandidateABDecision(
        accepted=accepted,
        action="ACCEPT_CANDIDATE" if accepted else "REJECT_CANDIDATE",
        baseline_fitness=baseline_fitness,
        candidate_fitness=candidate_fitness,
        baseline_pass_rate=baseline_pass_rate,
        candidate_pass_rate=candidate_pass_rate,
        baseline_mean_score=baseline_mean_score,
        candidate_mean_score=candidate_mean_score,
        reason=reason,
        sample_count=sample_count,
        minimum_samples=minimum_samples,
        candidate_tool_contract_errors=tuple(candidate_tool_contract_errors),
        executor_code_errors=tuple(executor_code_errors),
        critical_regressions=tuple(critical_regressions),
        paired_n=n,
        paired_mean_diff=mean_d,
        paired_t=t_stat,
        paired_t_critical=t_crit,
    )


_CRITICAL_AB_CHECKS = frozenset({
    # subagent oracle
    "subagent_verdict",
    "subagent_rest_pose",
    "subagent_motion_type",
    "subagent_motion_direction",
    # objective oracle
    "model_integrity",
    "valid_tree",
    "link_count",
    "link_matching",
    "topology",
    "joint_types",
    "joint_axes",
    "rest_pose_alignment",
    "joint_pivot_alignment",
    "kinematic_equivalence",
})


def _candidate_tool_contract_errors(
    results: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
) -> tuple[str, ...]:
    """Return deterministic candidate-helper failures with task identity."""
    errors = []
    for tid, inst, executor, _oracle, output_dir, _input_dir in results:
        error = _find_blocking_call_error(executor, output_dir)
        if _is_candidate_tool_contract_error(error):
            errors.append(f"{tid}/{inst}: {error}")
    return tuple(errors)


def _candidate_executor_code_errors(
    results: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
) -> tuple[str, ...]:
    """Return Executor-authored errors that sank a task after immediate retries.

    With K independent sessions per task, one session's own code slip (an
    empty-array numpy reduction, a typo) is a failing vote, not evidence
    against the candidate skill: run 13 round 2 vetoed a 12/20-vs-9/20
    candidate on exactly that. A task counts here only when a strict majority
    of its sessions ended without a URDF on an Executor code error. A session
    that recovered and exported is never counted.
    """
    errors = []
    for tid, inst, executor, _oracle, output_dir, _input_dir in results:
        rep_dirs = [output_dir] + sorted(
            d for d in output_dir.glob("rep_*") if d.is_dir()
        ) if output_dir.is_dir() else [output_dir]
        failed: list[str] = []
        for rep_dir in rep_dirs:
            success = executor.success if rep_dir == output_dir and len(rep_dirs) == 1 else None
            result_path = rep_dir / "result.json"
            if result_path.exists():
                try:
                    success = bool(json.loads(result_path.read_text()).get("success"))
                except (OSError, ValueError):
                    pass
            if success:
                continue
            rep_exec = executor if rep_dir == output_dir else dataclasses.replace(executor, error=None)
            error = _find_blocking_call_error(rep_exec, rep_dir)
            if _is_executor_code_error(error):
                failed.append(error)
        if failed and 2 * len(failed) > len(rep_dirs):
            errors.append(f"{tid}/{inst}: {failed[0]}")
    return tuple(errors)


def _candidate_skill_contract_errors(
    results: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
) -> tuple[str, ...]:
    """Backward-compatible alias for candidate-tool contract errors."""
    return _candidate_tool_contract_errors(results)


_VOTE_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)\s*votes")


def _vote_fraction(check: CheckResult) -> float:
    """A check's score in [0, 1]: its judge vote fraction, else pass/fail."""
    m = _VOTE_FRACTION.search(check.detail or "")
    if m and int(m.group(2)) > 0:
        return int(m.group(1)) / int(m.group(2))
    return 1.0 if check.passed else 0.0


def _task_score(oracle: OracleResult | None) -> float | None:
    """Mean vote fraction over scored checks; None when nothing was scored."""
    if oracle is None:
        return None
    scored = [c for c in oracle.checks if c.scored]
    if not scored:
        return None
    return sum(_vote_fraction(c) for c in scored) / len(scored)


def _paired_task_scores(
    baseline: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
    candidate: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
) -> list[tuple[str, float, float]]:
    """(task, baseline_score, candidate_score) for every asset on both sides."""
    cand = {(t, i): o for t, i, _e, o, _o, _in in candidate}
    out = []
    for t, i, _e, o, _o, _in in baseline:
        b = _task_score(o)
        c = _task_score(cand.get((t, i)))
        if b is not None and c is not None:
            out.append((f"{t}/{i}", b, c))
    return out


def _paired_joint_scores(
    baseline: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
    candidate: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
) -> list[tuple[str, float, float]]:
    """(task::gt_joint, baseline_score, candidate_score) for every GT joint
    resolved on both sides.

    The script oracle reports one JointReport per GT joint; pairing on joints
    instead of tasks turns a 20-object tranche into ~47 samples and lets the
    non-inferiority t-test see the dimension that actually changed. Oracles
    without joint reports (the subagent judge) yield no pairs, and the caller
    falls back to task-level pairing.
    """
    cand = {(t, i): o for t, i, _e, o, _o, _in in candidate}
    out = []
    for t, i, _e, o, _o, _in in baseline:
        bj = getattr(o, "joints", None) or {}
        cj = getattr(cand.get((t, i)), "joints", None) or {}
        for name in bj:
            if name in cj:
                out.append((f"{t}/{i}::{name}", bj[name].score(), cj[name].score()))
    return out


def _critical_ab_regressions(
    baseline: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
    candidate: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
) -> tuple[str, ...]:
    """Foundational checks that genuinely collapsed, beyond the noise floor.

    Aggregated, never per (task, check) pair: a single pair flips pass->fail
    21% of the time with no change to the skill, so the original per-pair
    rule rejected an unchanged skill compared with itself.

    Two triggers, both calibrated against real replays:
      - the TOTAL pass count across foundational checks drops by more than
        max(2, 25% of the baseline total). Trades between foundational
        checks net to zero here on purpose: a candidate that moved
        motion_direction 4->1 while lifting rest_pose 3->5, with a paired
        continuous score of exactly 0.000, was rejected by a per-check
        tolerance — on the one check measured at a 70% flip rate. Noise
        differs seven-fold between checks; a single per-check tolerance
        cannot be right for all of them.
      - any single foundational check with a solid base (>= 5 passes)
        collapses to <= 1. That is the "broke something that worked" case
        the guard exists for, and no observed replay noise produces it.
    """
    cand = {(t, i): o for t, i, _e, o, _o, _in in candidate}
    passes: dict[str, list[int]] = {}  # check -> [baseline, candidate]
    for t, i, _e, base_oracle, _o, _in in baseline:
        if base_oracle is None:
            continue
        cand_oracle = cand.get((t, i))
        cand_checks = {c.name: c for c in (cand_oracle.checks if cand_oracle else [])}
        for check in base_oracle.checks:
            if not (check.scored and check.name in _CRITICAL_AB_CHECKS):
                continue
            tally = passes.setdefault(check.name, [0, 0])
            tally[0] += int(check.passed)
            tally[1] += int(cand_checks.get(check.name, CheckResult(check.name, False, "missing")).passed)
    regressions = []
    total_b = sum(pb for pb, _pc in passes.values())
    total_c = sum(pc for _pb, pc in passes.values())
    if total_b - total_c > max(2, math.ceil(0.25 * total_b)):
        regressions.append(f"foundational_total:{total_b}->{total_c}")
    for name, (pb, pc) in sorted(passes.items()):
        if pb >= 5 and pc <= 1:
            regressions.append(f"{name}:{pb}->{pc}")
    return tuple(regressions)


def _validate_generator_output(
    gen_result: GeneratorResult,
    journal_path: Path,
    *,
    current_skill_content: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Acceptance check for a generator turn → (ok, reason_if_not).

    Rejects turns that errored out, modified nothing, skipped the journal,
    changed the frozen SDK action contract, or changed evolved helper source
    without synchronizing its mutable ``TOOLS.md`` interface contract.
    """
    if (gen_result.changes_summary or "").startswith("Generator error:"):
        return False, gen_result.changes_summary[:200]
    if not gen_result.files_modified:
        return False, "generator modified no skill files"
    if not journal_path.exists():
        return False, f"generator wrote no journal at {journal_path}"
    if current_skill_content is not None:
        for relative_path in ("ACTIONS.md",):
            expected = current_skill_content.get(relative_path)
            if expected is None:
                continue
            candidate = gen_result.skill_dir / relative_path
            try:
                actual = candidate.read_text()
            except OSError:
                return False, (
                    f"generator removed frozen SDK contract {relative_path}"
                )
            if actual != expected:
                return False, (
                    f"generator modified frozen SDK contract {relative_path}"
                )

        baseline_tool_paths = {
            relative_path
            for relative_path in current_skill_content
            if relative_path.startswith("tools/")
            and relative_path.endswith(".py")
            and relative_path != "tools/preflight.py"
        }
        candidate_tool_content: dict[str, str] = {}
        try:
            for path in gen_result.skill_dir.glob("tools/**/*.py"):
                relative_path = path.relative_to(
                    gen_result.skill_dir
                ).as_posix()
                if relative_path == "tools/preflight.py":
                    continue
                candidate_tool_content[relative_path] = path.read_text()
        except OSError as exc:
            return False, f"could not inspect evolved tool source: {exc}"

        tool_source_changed = any(
            current_skill_content.get(relative_path)
            != candidate_tool_content.get(relative_path)
            for relative_path in (
                baseline_tool_paths | set(candidate_tool_content)
            )
        )
        if tool_source_changed:
            tools_doc = gen_result.skill_dir / "TOOLS.md"
            try:
                candidate_tools_doc = tools_doc.read_text()
            except OSError:
                return False, (
                    "generator changed evolved tool source without creating "
                    "TOOLS.md"
                )
            if candidate_tools_doc == current_skill_content.get("TOOLS.md"):
                return False, (
                    "generator changed evolved tool source without updating "
                    "TOOLS.md"
                )
    return True, ""

async def _run_sub_evolver_stage(
    *, config: "EvoConfig", upd: Path, current_skill: Path, skill_content: dict[str, str],
    feedback_history: list[dict[str, Any]], clusters: list[dict[str, Any]],
    cluster_trial_keys: dict[str, list[str]] | None, entry_by_key: dict[str, dict],
    run_artifacts: list[dict[str, Any]], trial_entry: dict, gen_model: str,
    generator_effort: str | None, gen_sim_port: int, gen_add_dirs: list[str] | None,
    executor_effort: str | None, subagent_model: str | None, subagent_effort: str | None,
    oracle_mode: str, oracle_effort: str, oracle_votes: int, oracle_model: str | None,
    gen_ctx,
) -> tuple[str | None, list[dict[str, Any]], float]:
    """Run K sub-evolvers in parallel (one failure cluster each), verify each
    proposal on unseen members of its cluster, and return the integrator note,
    the per-proposal records and the executor spend of all gates."""
    k = sub_evolver_count()
    chosen = select_clusters(clusters, cluster_trial_keys or {}, k,
                             cooled=cooled_clusters(feedback_history))
    if not chosen:
        return None, [], 0.0
    baseline = {a["tag"]: a for a in run_artifacts}
    per_sub = 4   # trial sessions a sub-evolver may run at once
    print(f"  Sub-evolvers: {len(chosen)} cluster(s): "
          + "; ".join(f"{c['id']} {c['label']} ({c['n_tasks']})" for c in chosen), flush=True)

    async def one(i: int, c: dict[str, Any]) -> dict[str, Any]:
        sub_dir = upd / "sub_evolvers" / c["id"]
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_skill = sub_dir / "skill"
        ports = [config.sim_base_port + per_sub * (i + 1) + j for j in range(per_sub)]
        dev_entries = [entry_by_key[t] for t in c["dev"] if t in entry_by_key]
        tool, _impl, spend = _make_try_skill_tool(
            config=config, round_dir=sub_dir, skill_dir=sub_skill, trial_task=trial_entry,
            ports=ports, executor_effort=executor_effort, subagent_model=subagent_model,
            subagent_effort=subagent_effort, max_calls=min(2, config.generator_trials_per_turn or 2),
            current_skill_content=skill_content, cluster_tasks={c["id"]: dev_entries},
            baseline=baseline, oracle_mode=oracle_mode, oracle_effort=oracle_effort,
            oracle_votes=oracle_votes, oracle_model=oracle_model,
            baseline_skill=Path(current_skill), reps=SUB_TRIAL_REPS,
        )
        p: dict[str, Any] = {"cluster": c["id"], "label": c["label"], "key": c["key"],
                             "n_tasks": c["n_tasks"], "skill_dir": str(sub_skill),
                             "journal": str(sub_dir / "generator_journal.md"), "status": "no change",
                             "changed": False, "verified": False, "cost_usd": 0.0}
        try:
            with gen_ctx:
                res = await run_generator(
                    current_skill_dir=current_skill, current_skill_content=skill_content,
                    feedback_history=feedback_history, output_skill_dir=sub_skill,
                    model=gen_model, log_file=sub_dir / "generator.log", effort=generator_effort,
                    sim_port=gen_sim_port + 1 + i, sim_headless=config.headless,
                    backend=config.generator_backend, add_dirs=gen_add_dirs,
                    dataset_tasks_root=config.tasks_root, update_note=sub_note(c),
                    extra_tools=[tool], trial_task=f"{trial_entry['task_id']}/{trial_entry['instance']}",
                    cluster_trial_tasks={c["id"]: c["dev"]}, same_tasks_each_round=True,
                )
        except Exception as exc:  # noqa: BLE001 - one sub-evolver must not sink the round
            p["status"] = f"sub-evolver failed: {type(exc).__name__}: {str(exc)[:160]}"
            p["cost_usd"] = spend.get("cost_usd", 0.0)
            return p
        p["cost_usd"] += spend.get("cost_usd", 0.0)
        trials = [t for t in spend.get("trials") or [] if t.get("cluster") == c["id"]]
        if trials:
            last = trials[-1]
            p["dev_gate"] = {"ok": True, **{kk: last.get(kk) for kk in (
                "improved", "fitness_before", "fitness_after", "pass_before", "pass_after", "n_tasks", "cost_usd")}}
        if (res.changes_summary or "").startswith("Generator error:"):
            p["status"] = "sub-evolver error: " + (res.changes_summary or "")[:160]
            return p
        changed = sub_skill.is_dir() and not _skill_content_identical(sub_skill, Path(current_skill))
        p["changed"] = changed
        p["changed_text"] = journal_section(sub_dir / "generator_journal.md", "Changed")[:900]
        p["unresolved_text"] = journal_section(sub_dir / "generator_journal.md", "Unresolved")[:900]
        if not changed:
            p["status"] = "no change (the sub-evolver found no edit that helps this cluster)"
            return p
        pre = preflight_skill(sub_skill)
        if not pre.ok:
            p["status"] = "rejected: skill preflight failed: " + "; ".join(pre.errors[:2])[:200]
            p["changed"] = False
            return p
        p["diff"], p["added"], p["removed"] = skill_diff(Path(current_skill), sub_skill)
        verify_entries = [entry_by_key[t] for t in c["verify"] if t in entry_by_key]
        if verify_entries:
            _vtool, vimpl, vspend = _make_try_skill_tool(
                config=config, round_dir=sub_dir / "verify", skill_dir=sub_skill, trial_task=trial_entry,
                ports=ports, executor_effort=executor_effort, subagent_model=subagent_model,
                subagent_effort=subagent_effort, max_calls=1, current_skill_content=skill_content,
                cluster_tasks={c["id"]: verify_entries}, baseline=baseline, oracle_mode=oracle_mode,
                oracle_effort=oracle_effort, oracle_votes=oracle_votes, oracle_model=oracle_model,
                baseline_skill=Path(current_skill), reps=SUB_TRIAL_REPS,
            )
            try:
                vrep = await vimpl(c["id"])
            except Exception as exc:  # noqa: BLE001
                vrep = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            p["verify_gate"] = {kk: vrep.get(kk) for kk in (
                "ok", "improved", "fitness_before", "fitness_after", "fitness_diff", "pass_before", "pass_after",
                "n_tasks", "sessions_per_task", "tasks_better", "tasks_worse", "cost_usd", "error")}
            p["cost_usd"] += vspend.get("cost_usd", 0.0)
            p["verified"] = is_verified(vrep)
        else:
            # No unseen member to verify on: fall back to the development gate.
            p["verified"] = bool((p.get("dev_gate") or {}).get("improved"))
        p["status"] = ("verified on unseen cluster members" if p["verified"]
                       else "NOT verified on unseen cluster members")
        return p

    proposals = list(await asyncio.gather(*(one(i, c) for i, c in enumerate(chosen))))
    try:
        n_lines = len((Path(current_skill) / "SKILL.md").read_text().splitlines())
    except OSError:
        n_lines = None
    note = integrator_note(proposals, skill_lines=n_lines)
    records = [proposal_record(p) for p in proposals]
    try:
        (upd / "sub_evolvers" / "proposals.json").write_text(json.dumps(records, indent=1))
        (upd / "sub_evolvers" / "integrator_note.md").write_text(note)
    except OSError:
        pass
    return note, records, sum(float(p.get("cost_usd") or 0.0) for p in proposals)


def _make_try_skill_tool(
    *,
    config: "EvoConfig",
    round_dir: Path,
    skill_dir: Path,
    trial_task: dict,
    ports: list[int],
    executor_effort: str | None,
    subagent_model: str | None,
    subagent_effort: str | None,
    max_calls: int = 10,
    current_skill_content: dict[str, str] | None = None,
    cluster_tasks: dict[str, list[dict]] | None = None,
    baseline: dict[str, dict[str, Any]] | None = None,
    oracle_mode: str = "objective",
    oracle_effort: str = DEFAULT_AGENT_EFFORT,
    oracle_votes: int = 3,
    oracle_model: str | None = None,
    baseline_skill: Path | None = None,
    reps: int = 1,
):
    """`try_skill()` for the Generator: a smoke test — one real executor
    session with the skill in its working directory on ONE fixed Train task
    (``trial_task``, the first of the round's batch). It reports only whether
    the executor finished or failed, the raw error, and where the trajectory
    is; the Generator diagnoses causes itself. The oracle is not run.

    Returns ``(tool, impl, spend)``: the function tool to register, the plain
    coroutine behind it (for tests), and a dict accumulating executor cost
    and the verdict of the last call.

    Cluster mode (``cluster_tasks`` given, EVO_FEEDBACK_FORMAT=cluster): the
    tool takes ``cluster="C3"`` and runs that cluster's trial tasks (up to
    ``len(ports)`` in parallel), scores every output with the batch oracle
    (``oracle_mode``) and compares it with ``baseline`` — this round's batch
    artifact of the same task — so the Generator gets a per-edit verdict:
    passes before → after and whether the cluster improved. Every call is
    recorded in ``spend["trials"]`` for the edit ledger.
    """
    from agents import function_tool

    port_queue: asyncio.Queue[int] = asyncio.Queue()
    for port in ports:
        port_queue.put_nowait(port)
    spend: dict[str, Any] = {"calls": 0, "cost_usd": 0.0, "last_clean": None, "clean_calls": 0}
    if cluster_tasks is not None:
        spend["trials"] = []
    trials_dir = round_dir / "generator_trials"
    tid, inst = trial_task["task_id"], trial_task["instance"]
    key = f"{tid}/{inst}"
    # Fresh baseline for the per-edit gate. The trial tasks of a cluster were
    # picked BECAUSE their round session failed, so comparing a new session
    # with that stored one credits every edit with regression to the mean.
    # The unedited skill (``baseline_skill``: the skill the turn forked from,
    # NOT ``skill_dir``, which may already hold the edit) is snapshotted and
    # re-run on the same trial tasks; "before" is those fresh sessions.
    # ``reps`` sessions per task on each side. EVO_TRIAL_FRESH_BASELINE=0
    # restores the old rule.
    reps = max(1, int(reps))
    fresh_baseline_on = (cluster_tasks is not None and baseline_skill is not None
                         and os.environ.get("EVO_TRIAL_FRESH_BASELINE", "1") != "0")
    baseline_skill_dir = round_dir / "generator_trials_baseline_skill"
    if fresh_baseline_on:
        try:
            if baseline_skill_dir.exists():
                shutil.rmtree(baseline_skill_dir)
            shutil.copytree(baseline_skill, baseline_skill_dir)
        except OSError as exc:
            log.warning(f"[try_skill] no baseline skill snapshot ({exc}); using stored sessions as 'before'")
            fresh_baseline_on = False
    fresh_baseline: dict[str, list[dict[str, Any]]] = {}     # tag -> artifacts of the unedited skill's fresh sessions

    async def _run_one(tid: str, inst: str, *, skill: Path | None = None, label: str = "") -> tuple[Path, ExecutorResult, float]:
        input_dir = config.tasks_root / tid / inst / "parts"
        n = sum(1 for d in trials_dir.glob(f"{tid}_{inst}_{label}*") if d.is_dir()) if trials_dir.exists() else 0
        output_dir = trials_dir / f"{tid}_{inst}_{label}{n + 1}"
        output_dir.mkdir(parents=True, exist_ok=True)
        thumbnail = input_dir.parent / "thumbnail.png"
        if not thumbnail.exists():
            thumbnail = input_dir.parent / "render_thumbnail.png"
        if thumbnail.exists():
            try:
                shutil.copy2(thumbnail, output_dir / "render_thumbnail.png")
            except OSError:
                pass
        port = await port_queue.get()
        try:
            exec_result = await run_executor(
                factory=tid, instance=inst, input_dir=input_dir, output_dir=output_dir,
                skill_dir=skill or skill_dir,
                thumbnail_path=thumbnail if thumbnail.exists() else None,
                model=config.executor_model, max_turns=config.executor_max_turns,
                log_file=output_dir / "executor.log", sim_port=port,
                sim_headless=config.headless, effort=executor_effort,
                subagent_model=subagent_model, subagent_effort=subagent_effort,
                backend=config.executor_backend,
            )
        finally:
            port_queue.put_nowait(port)
        try:
            cost = float(json.loads((output_dir / "result.json").read_text()).get("total_cost_usd") or 0.0)
        except (OSError, ValueError, json.JSONDecodeError):
            cost = 0.0
        return output_dir, exec_result, cost

    def _verdict_summary(a: dict[str, Any]) -> dict[str, Any]:
        """The generator-facing part of a run artifact: verdict, failed checks, joint marks."""
        from evo.engine.agents import _format_joint_report
        return {
            "verdict": a.get("oracle_verdict"),
            "failed_checks": list(a.get("failed_oracle_checks") or []),
            "joints": {j: _format_joint_report(rep) for j, rep in sorted((a.get("joint_reports") or {}).items())},
            "error": a.get("error"),
        }

    async def impl_cluster(cluster: str) -> dict:
        if spend["calls"] >= max_calls:
            return {"ok": False, "error": f"try_skill safety limit of {max_calls} calls per turn reached; "
                                         f"finish with check_skill and the journal"}
        from evo.engine.agents import check_skill_report
        contract = check_skill_report(skill_dir, current_skill_content)
        if not contract["ok"]:
            return {"ok": False, "error": "skill fails the acceptance check; fix these before trying it",
                    "check_skill": contract}
        cluster = (cluster or "").strip()
        if cluster:
            entries = (cluster_tasks or {}).get(cluster)
            if not entries:
                return {"ok": False, "error": f"unknown cluster {cluster!r}; clusters with trial tasks: "
                                             f"{sorted((cluster_tasks or {}).keys())}"}
        else:
            entries = [trial_task]
        spend["calls"] += 1
        call_no = spend["calls"]
        need_base = [e for e in entries if fresh_baseline_on and cluster
                     and f"{e['task_id']}/{e['instance']}" not in fresh_baseline]
        n_e, n_b = len(entries) * reps, len(need_base) * reps
        gathered = await asyncio.gather(
            *(_run_one(e["task_id"], e["instance"]) for e in entries for _ in range(reps)),
            *(_run_one(e["task_id"], e["instance"], skill=baseline_skill_dir, label="base")
              for e in need_base for _ in range(reps)))
        per_task: list[dict[str, Any]] = []
        total_cost = 0.0

        async def _score(e: dict, output_dir: Path, exec_result) -> dict[str, Any]:
            tag = f"{e['task_id']}/{e['instance']}"
            orc = None
            if exec_result.success:
                try:
                    orc = await _run_oracle_dispatch(
                        mode=oracle_mode, tag=tag, gt_dir=config.gt_root / e["task_id"] / e["instance"],
                        pred_dir=output_dir, tasks_root=config.tasks_root, effort=oracle_effort,
                        votes=oracle_votes, oracle_model=oracle_model,
                    )
                    try:
                        (output_dir / "oracle.json").write_text(json.dumps(orc.to_dict(), indent=2))
                    except (OSError, AttributeError):
                        pass
                except Exception as exc:  # noqa: BLE001 - reported to the generator
                    log.warning(f"[try_skill {tag}] oracle failed: {exc}")
            return _build_run_artifact(
                tag=tag, er=exec_result, orc=orc, executor_log_path=output_dir / "executor.log",
                output_dir=output_dir, input_dir=config.tasks_root / e["task_id"] / e["instance"] / "parts",
            )

        for bi, e in enumerate(need_base):
            b_tag = f"{e['task_id']}/{e['instance']}"
            for b_dir, b_res, b_cost in gathered[n_e + bi * reps: n_e + (bi + 1) * reps]:
                total_cost += b_cost
                fresh_baseline.setdefault(b_tag, []).append(await _score(e, b_dir, b_res))
        pass_before = pass_after = 0
        failed_before = failed_after = 0
        fit_before = fit_after = 0.0
        task_diffs: list[float] = []
        for ei, e in enumerate(entries):
            tag = f"{e['task_id']}/{e['instance']}"
            runs = gathered[ei * reps: (ei + 1) * reps]
            after_arts = []
            for output_dir, exec_result, cost in runs:
                total_cost += cost
                after_arts.append(await _score(e, output_dir, exec_result))
            output_dir, exec_result, cost = runs[0]
            after_art = after_arts[0]
            base_arts = fresh_baseline.get(tag) or [(baseline or {}).get(tag) or {}]
            after = _verdict_summary(after_art)
            before = _verdict_summary(base_arts[0])
            before["source"] = (f"{len(base_arts)} fresh session(s) of the unedited skill" if tag in fresh_baseline
                                else "this round's session")
            failed_txt = output_dir / "FAILED.txt"
            if failed_txt.exists():
                try:
                    after["failed_txt"] = failed_txt.read_text()[:400]
                except OSError:
                    pass
            _fb = sum(artifact_fitness(a) for a in base_arts) / len(base_arts)
            _fa = sum(artifact_fitness(a) for a in after_arts) / len(after_arts)
            pass_before += sum(a.get("oracle_verdict") == "PASS" for a in base_arts)
            pass_after += sum(a.get("oracle_verdict") == "PASS" for a in after_arts)
            fit_before += _fb
            fit_after += _fa
            task_diffs.append(_fa - _fb)
            before["fitness"], after["fitness"] = round(_fb, 3), round(_fa, 3)
            before["sessions"], after["sessions"] = len(base_arts), len(after_arts)
            failed_before += len(before["failed_checks"]) if before["verdict"] == "FAIL" else 99
            failed_after += len(after["failed_checks"]) if after["verdict"] == "FAIL" else (0 if after["verdict"] == "PASS" else 99)
            per_task.append({
                "task": tag, "before": before, "after": after,
                "output_dir": str(output_dir), "trajectory": str(output_dir / "executor.log"),
                "executor_error": (exec_result.error or "")[:600] or None,
                "cost_usd": round(cost, 4), "duration_sec": round(exec_result.duration, 1),
            })
        # Graded verdict: the mean continuous fitness of the trial tasks (joint
        # correctness + how close each motion trace is), not only whole-object
        # passes, which stay 0/3 through most real improvements.
        n_trial = max(1, len(per_task))
        fit_before /= n_trial
        fit_after /= n_trial
        # With equal session counts on both sides passes are comparable; when the
        # baseline is the single stored session, scale it.
        _pb = pass_before * (reps if not fresh_baseline_on or not cluster else 1)
        improved = (fit_after > fit_before + 0.03) and pass_after >= _pb or pass_after > _pb and fit_after >= fit_before - 0.03
        spend["cost_usd"] += total_cost
        clean = all(p["after"]["verdict"] in ("PASS", "FAIL") for p in per_task)
        spend["last_clean"] = clean
        if clean:
            spend["clean_calls"] += 1
        record = {
            "call": call_no, "cluster": cluster or "smoke", "tasks": [p["task"] for p in per_task],
            "n_tasks": len(per_task), "pass_before": pass_before, "pass_after": pass_after,
            "failed_checks_before": failed_before, "failed_checks_after": failed_after,
            "fitness_before": round(fit_before, 3), "fitness_after": round(fit_after, 3),
            "fitness_diff": round(fit_after - fit_before, 3), "sessions_per_task": reps,
            "tasks_better": sum(1 for d in task_diffs if d > 0.03), "tasks_worse": sum(1 for d in task_diffs if d < -0.03),
            "improved": improved, "cost_usd": round(total_cost, 4),
        }
        spend["trials"].append(record)
        report: dict[str, Any] = {
            "ok": True, "cluster": cluster or "smoke", "improved": improved,
            "pass_before": pass_before, "pass_after": pass_after, "n_tasks": len(per_task),
            "fitness_before": round(fit_before, 3), "fitness_after": round(fit_after, 3),
            "fitness_diff": record["fitness_diff"], "sessions_per_task": reps,
            "tasks_better": record["tasks_better"], "tasks_worse": record["tasks_worse"],
            "fitness_note": "mean graded fitness of the trial tasks in [0, 1]: per joint, half for every scored "
                            "dimension passing and half for 1 - trace drift / (3 x tolerance); `sessions_per_task` "
                            "session(s) per task and side, so a change below ~0.05 is noise. `before` is a FRESH session of the unedited skill on "
                            "the same tasks (not the failed session that put the task in the cluster), so the "
                            "difference is the edit's effect plus session noise, without regression to the mean",
            "tasks": per_task, "cost_usd": round(total_cost, 4),
            "next": (
                "the cluster improved: keep this edit, record it in the journal's Changed section "
                "as `<cluster> — <edit> — kept`, and move to the next cluster or finish with check_skill."
                if improved else
                "the cluster did NOT improve: read the after-trajectories (executor.log) to see what the "
                "executor did with your edit, then either revise the edit once and gate again, or revert "
                "it (Edit the files back) and record `<cluster> — <edit> — reverted` in the journal."
            ),
        }
        try:
            trials_dir.mkdir(parents=True, exist_ok=True)
            (trials_dir / f"trial_{call_no:02d}_{cluster or 'smoke'}.json").write_text(json.dumps(report, indent=2))
        except OSError:
            pass
        return report

    async def impl() -> dict:
        if cluster_tasks is not None:
            return await impl_cluster("")
        if spend["calls"] >= max_calls:
            return {"ok": False, "error": f"try_skill safety limit of {max_calls} sessions per turn reached; "
                                         f"finish with check_skill and the journal"}
        # No point spending an executor session on a candidate the controller
        # would reject anyway: run the contract check first and report it.
        from evo.engine.agents import check_skill_report
        contract = check_skill_report(skill_dir, current_skill_content)
        if not contract["ok"]:
            return {"ok": False, "error": "skill fails the acceptance check; fix these before trying it",
                    "check_skill": contract}
        spend["calls"] += 1
        input_dir = config.tasks_root / tid / inst / "parts"
        n = sum(1 for d in trials_dir.glob(f"{tid}_{inst}_*") if d.is_dir()) if trials_dir.exists() else 0
        output_dir = trials_dir / f"{tid}_{inst}_{n + 1}"
        output_dir.mkdir(parents=True, exist_ok=True)
        thumbnail = input_dir.parent / "thumbnail.png"
        if not thumbnail.exists():
            thumbnail = input_dir.parent / "render_thumbnail.png"
        if thumbnail.exists():
            try:
                shutil.copy2(thumbnail, output_dir / "render_thumbnail.png")
            except OSError:
                pass
        port = await port_queue.get()
        try:
            exec_result = await run_executor(
                factory=tid, instance=inst, input_dir=input_dir, output_dir=output_dir,
                skill_dir=skill_dir,
                thumbnail_path=thumbnail if thumbnail.exists() else None,
                model=config.executor_model, max_turns=config.executor_max_turns,
                log_file=output_dir / "executor.log", sim_port=port,
                sim_headless=config.headless, effort=executor_effort,
                subagent_model=subagent_model, subagent_effort=subagent_effort,
                backend=config.executor_backend,
            )
        finally:
            port_queue.put_nowait(port)
        try:
            cost = float(json.loads((output_dir / "result.json").read_text()).get("total_cost_usd") or 0.0)
        except (OSError, ValueError, json.JSONDecodeError):
            cost = 0.0
        spend["cost_usd"] += cost
        failed_txt = output_dir / "FAILED.txt"
        failed_reason = None
        if failed_txt.exists():
            try:
                failed_reason = failed_txt.read_text()[:800]
            except OSError:
                failed_reason = "(unreadable FAILED.txt)"
        clean = bool(exec_result.success) and failed_reason is None
        spend["last_clean"] = clean
        if clean:
            spend["clean_calls"] += 1
        report: dict = {
            "ok": True, "clean": clean, "task": key, "trial": n + 1,
            "output_dir": str(output_dir), "executor_success": bool(exec_result.success),
            "executor_error": (exec_result.error or "")[:800] or None,
            "failed_txt": failed_reason,
            "duration_sec": round(exec_result.duration, 1), "cost_usd": round(cost, 4),
            "urdf": sorted(p.name for p in output_dir.glob("*.urdf")),
            "renders": sorted(p.name for p in output_dir.glob("*.png"))[:40],
            "trajectory": str(output_dir / "executor.log"),
            "next": ("the executor finished (URDF written). Read executor.log if you want to see how it "
                     "used the skill; then finish with check_skill and the journal. This says nothing "
                     "about effectiveness — the round's evaluation measures that." if clean else
                     "the executor failed. Read executor.log (every tool call and result is there) and "
                     "work out why; fix the skill, then call try_skill() again."),
        }
        (output_dir / "smoke_test.json").write_text(json.dumps(report, indent=2))
        return report

    if cluster_tasks is not None:
        @function_tool(name_override="try_skill")
        async def try_skill_cluster(cluster: str = "") -> str:
            """Gate an edit: run the trial tasks of failure cluster `cluster`
            (e.g. "C3", ids from the cluster report; up to 3 tasks in
            parallel, a few cents and 2-4 minutes) with the skill in your
            working directory and score them with the batch oracle. Returns
            JSON with `improved`, `pass_before`, `pass_after` and, per task,
            the verdict, failed checks and joint marks before (this round's
            batch) and after (your edit), plus `output_dir` / `trajectory`
            of the new session. With no argument it is a plain smoke test on
            the first Train task. Runs check_skill first and refuses while
            the skill fails the acceptance check. Keep an edit only when its
            cluster improved."""
            return json.dumps(await impl_cluster(cluster), indent=2)

        return try_skill_cluster, impl_cluster, spend

    @function_tool(name_override="try_skill")
    async def try_skill() -> str:
        """Smoke-test the skill in your working directory: ONE real executor
        session on a fixed Train task (always the same one, the first of this
        round's list), about 2 minutes and a few cents. Takes no arguments.
        Reports whether the executor finished (a URDF was written) or failed,
        with its raw error / FAILED.txt text and the path of executor.log,
        which holds every tool call and result of the session. It does NOT
        classify causes and does NOT score the output: you read the log and
        work out what went wrong. Returns JSON with `clean`, `executor_success`,
        `executor_error`, `failed_txt`, `output_dir`, `trajectory` and
        `next`. Call it after your
        change; if it is not clean, fix and call it again until it is. It runs
        check_skill first and refuses while the skill fails the acceptance
        check."""
        return json.dumps(await impl(), indent=2)

    return try_skill, impl, spend



def _feedback_format() -> str:
    """The Evolver feedback format of this process (see agents.FEEDBACK_FORMAT)."""
    from evo.engine.agents import FEEDBACK_FORMAT
    return FEEDBACK_FORMAT


_JOURNAL_EDIT_RE = re.compile(r"^\s*(?:[-*]\s*)?(C\d+)\s*[—:\-–]+\s*(.+?)\s*[—:\-–]+\s*(kept|reverted)\b", re.I | re.M)


def _journal_cluster_edits(journal: Path) -> dict[str, tuple[str, bool]]:
    """`C3 — <edit> — kept|reverted` lines of a journal → {cluster: (edit, kept)}."""
    try:
        text = journal.read_text()
    except OSError:
        return {}
    out: dict[str, tuple[str, bool]] = {}
    for m in _JOURNAL_EDIT_RE.finditer(text):
        out[m.group(1).upper()] = (m.group(2).strip(), m.group(3).lower() == "kept")
    return out


def _hide_oracle_files(run_dir: Path, iteration: int) -> int:
    """Score-only feedback ablation: move every oracle*.json of this round's outputs
    to the run's _private sibling (mirrored path) before the Evolver turn, so the
    per-check / per-joint diagnosis is not readable from the output dirs."""
    round_dir = run_dir / f"round_{iteration}"
    dest_root = _private_dir(run_dir) / f"round_{iteration}" / "hidden_from_evolver"
    n = 0
    for fp in sorted((round_dir / "outputs").rglob("oracle*.json")):
        dest = dest_root / fp.relative_to(round_dir)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fp.replace(dest)
            n += 1
        except OSError as exc:
            log.warning(f"could not hide {fp}: {exc}")
    print(f"  Score-only feedback: moved {n} oracle file(s) out of the Evolver's reach")
    return n


def _private_dir(run_dir: Path) -> Path:
    """Sibling dir for run artifacts the generator must never read
    (full oracle dicts with GT values). Kept OUTSIDE run_dir so granting
    the generator run_dir access doesn't grant these, and so the path
    string doubles as an isolation-audit tripwire.
    """
    return run_dir.with_name(run_dir.name + "_private")


def _generator_add_dirs(run_dir: Path, config: EvoConfig) -> list[str]:
    """File-tool access scope for the generator (Read/Glob/Grep/Edit).

    Includes: this run's artifacts, GT-free task inputs, the reference
    renders the executor already sees, the oracle source (the metric is
    intentionally public), and /tmp. Excludes: the dataset's gt/ answer
    files and the run's _private sibling.
    """
    return [
        str(run_dir),
        str(config.tasks_root),
        str(config.tasks_root.parent / "gt_renders"),
        str(PROJECT_ROOT / "evo" / "oracle"),
        "/tmp",
    ]


def _scan_text_for_forbidden(
    text: str, forbidden: list[str],
) -> dict[str, int]:
    """Count occurrences of each forbidden substring in `text`."""
    return {s: text.count(s) for s in forbidden if s and s in text}


def _audit_round_isolation(
    run_dir: Path,
    iteration: int,
    forbidden: list[str],
) -> list[str]:
    """Scan this round's agent session transcripts for GT-path mentions.

    Isolation is enforced by what the prompts contain plus the generator's
    add_dirs scope — but Bash is an open channel, so this audit is the
    backstop: a GT path appearing ANYWHERE in a session transcript (it is
    never part of any prompt we build) means an agent discovered and
    touched ground truth, and the round's fitness can't be trusted.
    Findings are printed loudly and appended to isolation_audit.log.
    """
    hits: list[str] = []
    round_dir = run_dir / f"round_{iteration}"
    for pattern in ("*.responses.json", "*.session.jsonl", "agent-*.jsonl"):
        for fp in sorted(round_dir.rglob(pattern)):
            try:
                text = fp.read_text()
            except OSError:
                continue
            for sub, count in _scan_text_for_forbidden(text, forbidden).items():
                hits.append(f"{fp.relative_to(run_dir)}: {count}x {sub!r}")
    if hits:
        print("\n*** ISOLATION AUDIT FAILED — GT paths referenced in agent "
              "sessions this round ***")
        for h in hits:
            print(f"    {h}")
        print("    Treat this round's fitness (and any skill edits derived "
              "from it) as contaminated.")
        try:
            with open(run_dir / "isolation_audit.log", "a") as f:
                f.write(f"round {iteration}:\n")
                for h in hits:
                    f.write(f"  {h}\n")
        except OSError:
            pass
    return hits


def _executor_session_summary(output_dir: Path, executor_log_path: Path) -> dict:
    """Token/cost summary for one executor run.

    Responses runs persist the already-priced aggregate in result.json (see
    evo.engine.pricing); legacy Claude runs fall back to the session JSONL.
    """
    result_json = output_dir / "result.json"
    if result_json.exists():
        try:
            sess = json.loads(result_json.read_text()).get("parent_session")
            if sess is not None:
                # K>1 independent sessions: rep_<k>/ siblings are the same task
                # and the same money; fold their numeric fields in.
                for extra in sorted(output_dir.glob("rep_*/result.json")):
                    try:
                        more = json.loads(extra.read_text()).get("parent_session") or {}
                    except Exception:
                        continue
                    sess = dict(sess)
                    for key, value in more.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            sess[key] = sess.get(key, 0) + value
                return sess
        except Exception:
            pass
    session_jsonl = (
        executor_log_path.parent / f"{executor_log_path.stem}.session.jsonl"
    )
    return _summarize_session(session_jsonl if session_jsonl.exists() else None)


def _round_spend(
    results: list[
        tuple[str, str, ExecutorResult, OracleResult | None, Path, Path]
    ],
) -> tuple[float, float]:
    """(executor_usd, oracle_usd) over EVERY task scored this round.

    Deliberately wider than `batch_cost_summary`, which covers only the
    feedback split because that is what the generator is shown. Held-out
    splits are evaluated too, and they cost the same money.
    """
    executor_usd = oracle_usd = 0.0
    for _tid, _inst, _er, orc, output_dir, _in_dir in results:
        sess = _executor_session_summary(output_dir, output_dir / "executor.log")
        executor_usd += float(sess.get("cost_usd") or 0.0)
        if orc is not None:
            oracle_usd += float(getattr(orc, "cost_usd", None) or 0.0)
    return executor_usd, oracle_usd


# The subagent oracle appends " | $0.31 / 56s / 3 votes / responses:model"
# to the verdict for humans reading oracle.json. It is billing metadata, not
# diagnosis, and it sits at the END — so a naive truncation kept it and cut
# the reasoning. Measured on 88 real details: the verdict's prose runs
# p50≈430 / max≈520 chars, and 97% of them exceeded the old 280 cap, which
# sliced mid-sentence through exactly the "what is right vs wrong" half.
_ORACLE_BILLING_SUFFIX = re.compile(
    r"\s*\|\s*(?:\$[\d.]+\s*/\s*)?\d+s\s*/\s*\d+\s*votes\s*/\s*\S+\s*$"
)
_GENERATOR_DETAIL_CAP = 600  # safety net only; nothing observed comes close


def _generator_facing_detail(detail: str) -> str:
    """One check's reasoning as the Generator should see it."""
    text = _ORACLE_BILLING_SUFFIX.sub("", sanitize_detail(detail)).strip()
    if len(text) > _GENERATOR_DETAIL_CAP:
        text = text[:_GENERATOR_DETAIL_CAP].rstrip() + " …"
    return text


def _build_run_artifact(
    *, tag: str,
    er: ExecutorResult,
    orc: OracleResult | None,
    executor_log_path: Path,
    output_dir: Path,
    input_dir: Path,
) -> dict[str, Any]:
    """Build the per-run artifact dict the generator consumes.

    The generator gets PATHS, not embedded content — it Reads the URDF /
    log / images itself when it decides to. Keeps the prompt small and
    makes the generator's investigation visible in its own tool calls.
    """
    a: dict[str, Any] = {
        "tag": tag,
        "output_dir": str(output_dir.resolve()),
        "input_dir": str(input_dir.resolve()),
    }
    if not er.success:
        a["oracle_verdict"] = "execution_failed"
        a["error"] = (er.error or "")[:300]
    elif orc is None:
        a["oracle_verdict"] = "no_oracle_result"
    else:
        a["oracle_verdict"] = "PASS" if orc.passed else "FAIL"
        a["failed_oracle_checks"] = [
            c.name for c in orc.checks if c.scored and not c.passed
        ]
        a["diagnostic_oracle_checks"] = [
            c.name for c in orc.checks if not c.scored and not c.passed
        ]
        # WHY each check failed, not just which. The judge already writes a
        # specific diagnosis ("prediction provides only one handle swivel,
        # leaving the spout unarticulated"), sanitize_detail strips the GT
        # constants from it, and outputs/*/oracle.json has held the sanitized
        # copy all along — but nothing ever put it in front of the Generator,
        # so it could only see that a check named `motion_direction` failed
        # and had to guess at the cause.
        a["failed_check_details"] = {
            c.name: _generator_facing_detail(c.detail)
            for c in orc.checks
            if not c.passed and c.detail
        }
        # Structured per-joint outcomes (script oracle only). GT type / child
        # are redacted; error magnitudes are what the Generator can act on.
        if getattr(orc, "joints", None):
            a["joint_reports"] = {
                name: rep.to_dict(redact_gt=True) for name, rep in orc.joints.items()
            }
        if getattr(orc, "reps", 1) > 1:
            a["executor_reps"] = orc.reps
        try:
            from evo.engine.paired_gate import session_fitness
            a["fitness"] = round(session_fitness(orc.to_dict()), 4)
        except Exception:  # the score is informational; never fail a round on it
            pass
        if _feedback_format() == "score":
            # Ablation (no structured feedback): the Evolver gets the verdict and the score only.
            for k in ("failed_oracle_checks", "diagnostic_oracle_checks", "failed_check_details", "joint_reports"):
                a.pop(k, None)

    sess = _executor_session_summary(output_dir, executor_log_path)
    # The subagent oracle spends more per task than the executor does
    # (`votes` judges each read every rendered frame), so a cost summary
    # that counted only the executor understated the run several-fold.
    oracle_cost = getattr(orc, "cost_usd", None) if orc else None
    a["cost_summary"] = {
        "cost_usd": round(sess["cost_usd"] + (oracle_cost or 0.0), 4),
        "executor_usd": round(sess["cost_usd"], 4),
        "oracle_usd": round(oracle_cost, 4) if oracle_cost is not None else None,
        "n_api_calls": sess["n_api_calls"],
        "output_tokens": sess["output"],
        "cache_read_tokens": sess["cache_read"],
        "cache_misses": sess["cache_misses"],
    }

    return a


def _install_skill(skill_dir: Path) -> None:
    """Point the skills/current symlink at this skill version (best-effort).

    The loop itself never reads skills/current — executors receive the skill
    content INLINE in their prompt (see _load_skill_as_executor_context), so
    correctness doesn't depend on this link. It exists for humans and for the
    interactive `/assemble` skill (~/.claude/skills/assemble -> skills/current)
    to track the latest version. Last writer wins if two runs race over it;
    failure to update never aborts a run.
    """
    current_link = PROJECT_ROOT / "skills" / "current"
    target = skill_dir.resolve()
    try:
        if current_link.is_symlink() or current_link.exists():
            current_link.unlink()
        current_link.symlink_to(target)
    except OSError as e:
        log.warning(f"Could not update {current_link} -> {target}: {e}")
        return
    log.info(f"Installed skill: {current_link} -> {target}")


def _resolve_path(p: str, fallback: Path | None = None) -> Path:
    """Resolve a path from a YAML string.

    Supports `~` expansion and `${VAR}` env-var references — so configs can
    write `gt_root: ${EVO_GT_ROOT}` instead of hardcoding host-specific paths.
    Absolute paths stay absolute; relative paths resolve from PROJECT_ROOT.

    If the expanded string still contains `${...}` (env var was unset),
    return `fallback` if provided, else the literal expansion. This avoids
    producing a literal-`${VAR}` path on disk.
    """
    expanded = os.path.expandvars(os.path.expanduser(p))
    if "${" in expanded and fallback is not None:
        return fallback
    path = Path(expanded)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _read_skill_content(skill_dir: Path) -> dict[str, str]:
    """Read all skill files into a dict for embedding in generator prompt.

    Returns {"SKILL.md": "...", "ACTIONS.md": "...", "tools/create_joint.py": "...", ...}
    """
    content: dict[str, str] = {}
    for fp in sorted(skill_dir.rglob("*")):
        if fp.is_file() and fp.suffix in (".md", ".py"):
            rel = str(fp.relative_to(skill_dir))
            try:
                content[rel] = fp.read_text()
            except Exception:
                pass
    return content


def _skill_content_identical(a: Path, b: Path) -> bool:
    """True when two skill directories hold the same evolvable files."""
    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in {".md", ".py"}
        }
    try:
        return snapshot(a) == snapshot(b)
    except OSError:
        return False


def _save_best_skill(skill_dir: Path, dest: Path) -> None:
    # Clear the previous snapshot first — dirs_exist_ok alone would leave
    # files the newer skill deleted, resurrecting them on rollback.
    if dest.exists():
        shutil.rmtree(dest)
    # A candidate that imported its own tools/ leaves __pycache__ behind;
    # the checkpoint holds the evolvable sources, not compiled artefacts.
    shutil.copytree(
        skill_dir, dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    log.info(f"Saved best skill to {dest}")


def _log_iteration(
    log_path: Path,
    iteration: int,
    phase: str,
    task_id: str,
    instance: str,
    exec_result: ExecutorResult | None,
    oracle_result: OracleResult | None,
    gen_result: GeneratorResult | None,
    score: float,
) -> None:
    record: dict[str, Any] = {
        "iteration": iteration,
        "phase": phase,
        "task": f"{task_id}/{instance}",
        "score": score,
        "timestamp": time.time(),
    }
    if exec_result:
        record["executor"] = {"success": exec_result.success, "duration": exec_result.duration}
    if oracle_result:
        record["oracle"] = oracle_result.to_dict()
    if gen_result:
        record["generator"] = {"files": gen_result.files_modified,
                                "summary": gen_result.changes_summary[:500]}

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Public API (used by the evo CLI / `python -m evo.engine.loop`)
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> EvoConfig:
    """Thin alias for EvoConfig.from_yaml — public config entry point."""
    return EvoConfig.from_yaml(path)


def run_evolution(
    config: EvoConfig,
    single: tuple[str, str] | None = None,
    include: list[tuple[str, str]] | None = None,
    split: str | None = None,
    verbose: bool = False,
    resume_dir: Path | None = None,
    retry_wait_min: int = 0,
    max_retries: int = 3,
    stagger_sec: int = 0,
    generator_auth: str | None = None,
    executor_effort: str | None = None,
    generator_effort: str | None = None,
    subagent_model: str | None = None,
    subagent_effort: str | None = None,
    oracle_mode: str = "subagent",
    oracle_effort: str = DEFAULT_AGENT_EFFORT,
    oracle_votes: int = 3,
    oracle_model: str | None = None,
    regression_tol: float = 0.05,
    bootstrap_generator: bool = False,
    skip_generator: bool = False,
    generate_report: bool = True,
    gate_split: str | None = None,
    gate_mode: str = "select",
    gate_min_drop: int = 3,
    gate_reps: int = 3,
    gate_z: float = 1.0,
    gate_min_gain: float = 0.0,
    gate_reroll: bool = True,
    gate_extend: bool = True,
    gate_limit: int | None = None,
) -> None:
    """Public entry point used by the evo CLI.

    Dispatches to evolve_single_task (overfit one task) or evolve_multi_task
    (full evolution across config.parallel workers).

    Filters compose (intersection) when both are set:
      `split`   — keep only instances whose splits.json entry matches.
                  Requires <tasks_root>/../splits.json to exist.
      `include` — keep only the listed (task_id, instance) pairs.

    `single` short-circuits both filters. When `split` is omitted, the
    experiment split configured in EvoConfig is used.
    """
    if split is None:
        split = config.split
    if single is not None:
        task_id, instance = single
        if bootstrap_generator:
            raise SystemExit(
                "--bootstrap-generator is not supported with --single "
                "(multi-task mode only)."
            )
        asyncio.run(evolve_single_task(
            task_id, instance, config, verbose=verbose, resume_dir=resume_dir,
            executor_effort=executor_effort,
            generator_effort=generator_effort,
            subagent_model=subagent_model,
            subagent_effort=subagent_effort,
            generator_auth=generator_auth,
            oracle_mode=oracle_mode,
            oracle_effort=oracle_effort,
            oracle_votes=oracle_votes,
            oracle_model=oracle_model,
            regression_tol=regression_tol,
            skip_generator=skip_generator,
        ))
        return

    discovered = discover_tasks(config)
    selected = list(discovered)

    # Split filter. Training runs execute only the FIRST split and defer every
    # additional split to a separate evaluation-only process. --no-generator
    # makes all requested splits evaluation-only and disables adaptive Train.
    feedback_split: str | None = None
    deferred_splits: set[str] = set()
    if split is not None:
        wanted_splits = [s.strip() for s in split.split(",") if s.strip()]
        if not wanted_splits:
            raise SystemExit(f"--split={split!r} parsed to empty list.")
        splits = config.load_splits()
        if not splits:
            raise SystemExit(
                f"--split={split} requested but no splits.json found at "
                f"{config.splits_path}. Run scripts/dataset/build_artvip_splits.py first."
            )
        wanted_set = set(wanted_splits)
        for t in selected:
            t["split"] = splits.get((t["task_id"], t["instance"]))
        requested = [t for t in selected if t.get("split") in wanted_set]
        if not requested:
            raise SystemExit(
                f"--split={split} matched zero discovered tasks; aborting."
            )
        if skip_generator:
            selected = requested
            breakdown = ", ".join(
                f"{s}={sum(1 for t in selected if t['split']==s)}"
                for s in wanted_splits
            )
            print(
                f"--split={split}: evaluation-only on {len(selected)} of "
                f"{len(discovered)} tasks ({breakdown}); adaptive Train and "
                "Generator disabled"
            )
        else:
            feedback_split = wanted_splits[0]
            if feedback_split != "train":
                raise SystemExit(
                    f"--split={split}: training may only use the 'train' "
                    "split. Use --no-generator for standalone validation/test "
                    "evaluation."
                )
            deferred_splits = set(wanted_splits[1:])
            selected = [
                t for t in requested if t.get("split") == feedback_split
            ]
            print(
                f"--split={split}: training on {len(selected)} "
                f"{feedback_split} tasks"
            )
            if deferred_splits:
                deferred_breakdown = ", ".join(
                    f"{s}={sum(1 for t in requested if t['split']==s)}"
                    for s in wanted_splits[1:]
                )
                print(
                    "  Deferred to standalone evaluation: "
                    f"{deferred_breakdown}"
                )

    if include:
        wanted = set(include)
        sel_set = {(t["task_id"], t["instance"]) for t in selected}
        missing = wanted - sel_set
        if missing:
            print(f"Warning: --include pairs not in current selection: {sorted(missing)}")
        selected = [t for t in selected
                    if (t["task_id"], t["instance"]) in wanted]
        if not selected:
            raise SystemExit("--include filter selected zero tasks; aborting.")
        listed = ", ".join(f"{t['task_id']}/{t['instance']}" for t in selected)
        print(f"--include filter: running on {len(selected)} task(s): {listed}")

    gate_tasks: list[dict[str, str]] | None = None
    if gate_split is not None:
        if split is None or skip_generator:
            raise SystemExit("--gate-split needs --split train and a Generator "
                             "(it is the selection split of a training run).")
        if gate_split == feedback_split:
            raise SystemExit("--gate-split must differ from the training split.")
        gate_tasks = [t for t in discovered if t.get("split") == gate_split]
        if not gate_tasks:
            raise SystemExit(f"--gate-split={gate_split} matched zero tasks.")
        if gate_limit:
            gate_tasks = gate_tasks[:gate_limit]
        print(f"--gate-split={gate_split}: {len(gate_tasks)} task(s) evaluated "
              f"every round, mode={gate_mode}, rollback when ≥ {gate_min_drop} "
              "fewer pass than the checkpoint")

    task_ids = selected if (split is not None or include) else None
    asyncio.run(evolve_multi_task(
        config, task_ids=task_ids, verbose=verbose,
        resume_dir=resume_dir,
        retry_wait_min=retry_wait_min,
        max_retries=max_retries,
        stagger_sec=stagger_sec,
        generator_auth=generator_auth,
        executor_effort=executor_effort,
        generator_effort=generator_effort,
        subagent_model=subagent_model,
        subagent_effort=subagent_effort,
        oracle_mode=oracle_mode,
        oracle_effort=oracle_effort,
        oracle_votes=oracle_votes,
        oracle_model=oracle_model,
        regression_tol=regression_tol,
        feedback_split=feedback_split,
        bootstrap_generator=bootstrap_generator,
        skip_generator=skip_generator,
        generate_report=generate_report,
        gate_tasks=gate_tasks,
        gate_mode=gate_mode,
        gate_min_drop=gate_min_drop,
        gate_reps=gate_reps,
        gate_z=gate_z,
        gate_min_gain=gate_min_gain,
        gate_reroll=gate_reroll,
        gate_extend=gate_extend,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Co-evolution loop for /assemble skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Overfit test on one task
  python -m evo.engine.loop --single task_10 0 --max-rounds 3 -v

  # Multi-task evolution (all discovered tasks)
  python -m evo.engine.loop --max-rounds 5

  # Auto-wait through Claude Code's 5-hour usage window:
  python -m evo.engine.loop --max-rounds 5 --retry-wait 305 --max-retries 3

  # Resume an interrupted run from its loop_state.json:
  python -m evo.engine.loop --resume runs/multi_1777825428 --max-rounds 5

  # Dry run — show config + discovered tasks
  python -m evo.engine.loop --dry-run
        """,
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "evo" / "configs" / "default.yaml")
    parser.add_argument(
        "--data-root", type=Path,
        help="Override the sanitized dataset root (the parent of tasks/ and "
             "gt/ — same semantics as EVO_DATASET_ROOT). Sets BOTH "
             "config.tasks_root=<root>/tasks and config.gt_root=<root>/gt.",
    )
    parser.add_argument("--single", nargs=2, metavar=("TASK_ID", "INSTANCE"),
                        help="Overfit test on a single task (e.g., --single task_01 0)")
    parser.add_argument(
        "--include", default=None,
        help="Comma-separated list of `task_id/instance` pairs (e.g. "
             "`task_02/0,task_06/0`). When set, evolve_multi_task only "
             "operates on the listed pairs. Pairs not present in the "
             "discovered task set are reported and skipped.",
    )
    parser.add_argument(
        "--split", default=None,
        help="Filter to one or more splits from <tasks_root>/../splits.json. "
             "Training executes only the FIRST split (which must be train) "
             "and defers additional splits. Run validation independently with "
             "--split validation --no-generator --max-rounds 1. With "
             "--no-generator, all requested splits run as a full-pool "
             "evaluation. Composes with --include (intersection).",
    )
    parser.add_argument(
        "--oracle-mode", choices=("script", "objective", "subagent"), default="subagent",
        help="`subagent` (default) — render GT + pred rest/motion frames at 4 "
             "canonical azimuths and ask --oracle-votes LLM judges to compare "
             "them visually (see --oracle-model). `objective` — evo/oracle/check_urdf.py (10 "
             "deterministic checks against the GT URDF; `script` is an alias).",
    )
    parser.add_argument(
        "--oracle-effort", choices=("low", "medium", "high", "max"),
        default=DEFAULT_AGENT_EFFORT,
        help="Effort for the subagent oracle (--oracle-mode subagent). "
             f"Ignored in script mode. Default {DEFAULT_AGENT_EFFORT}.",
    )
    parser.add_argument(
        "--oracle-model", default=None,
        help="Judge model for the subagent oracle. Default: EVO_ORACLE_MODEL, "
             "then evo.oracle.subagent.DEFAULT_ORACLE_MODEL for the active "
             "judge backend (EVO_ORACLE_BACKEND, default 'responses' — the "
             "same provider the executor and generator use). Ignored in "
             "script mode.",
    )
    parser.add_argument(
        "--oracle-votes", type=int, default=3,
        help="Independent judge calls per task for the subagent oracle, "
             "combined by strict majority (rerun experiments showed a "
             "single call flips on ~50%% of cases). Ignored in script "
             "mode. 1 restores single-call behavior. Default 3.",
    )
    parser.add_argument(
        "--gate-split", default=None,
        help="Split (normally 'val') evaluated with the round's skill after "
             "every train batch, one session per task, into <run_dir>_val. "
             "With --gate-mode select it decides checkpoints and rollbacks; "
             "the Generator only ever sees its pass count.",
    )
    parser.add_argument(
        "--gate-mode", choices=("select", "log", "paired"), default="select",
        help="select: checkpoint/rollback on the gate split's pass count. "
             "log: record the gate curve, keep selection on the train fitness. "
             "paired: --gate-reps sessions per gate task, continuous fitness, the "
             "champion re-run and pooled every round, and a candidate replaces it "
             "only when the paired per-task gain is >= --gate-z standard errors "
             "(evo/engine/paired_gate.py).",
    )
    parser.add_argument("--gate-reps", type=int, default=3,
                        help="paired mode: executor sessions per gate task (default 3).")
    parser.add_argument("--gate-z", type=float, default=1.0,
                        help="paired mode: accept a candidate at this many standard errors of "
                             "paired gain; the same threshold demotes a champion (default 1.0).")
    parser.add_argument("--gate-min-gain", type=float, default=0.0,
                        help="paired mode: minimum mean paired fitness gain to accept (default 0).")
    parser.add_argument("--gate-extend", action=argparse.BooleanOptionalAction, default=True,
                        help="Paired gate: when a candidate is ahead with 0.5 <= z < --gate-z, run one more batch "
                             "of sessions on both sides and decide on the pooled data instead of rejecting.")
    parser.add_argument("--gate-reroll", action=argparse.BooleanOptionalAction, default=True,
                        help="paired mode: re-run the champion on the gate split every round "
                             "and pool its sessions (default on).")
    parser.add_argument(
        "--gate-min-drop", type=int, default=3,
        help="select mode: roll back when the gate split passes this many "
             "tasks fewer than the checkpoint (or more). Default 3.",
    )
    parser.add_argument(
        "--gate-limit", type=int, default=None,
        help="Use only the first N gate-split tasks (smoke tests).",
    )
    parser.add_argument(
        "--regression-tol", type=float, default=0.05,
        help="Rollback gate: when a round's fitness (mean oracle score on "
             "the feedback split) drops more than this below the best "
             "checkpoint, the next generator forks from the best skill "
             "instead of the regressed one. Negative disables. Default 0.05.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Per-round batch size (0 disables sampling, runs full filtered "
             "pool every round). Default from config (default.yaml: 0).",
    )
    parser.add_argument(
        "--batch-ratio", default=None,
        help="Difficulty ratio in `easy:medium:hard` form. Sum should equal "
             "--batch-size. Default 3:3:2.",
    )
    parser.add_argument(
        "--fixed-batch", action=argparse.BooleanOptionalAction, default=None,
        help="With --no-adaptive-train and --batch-size N: evaluate the same "
             "stratified batch of N Train tasks every round (drawn once from "
             "--seed) instead of rotating through unseen tasks.",
    )
    parser.add_argument(
        "--adaptive-train", action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable within-round Train increments. Enabled by "
             "default: run 10, then ADD_10 or STOP_ADDING up to 69.",
    )
    parser.add_argument(
        "--train-increment", type=int, default=None,
        help="Train assets requested per adaptive tranche (default 10).",
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Maximum Train assets per outer round (default 69).",
    )
    parser.add_argument(
        "--objective-failure-stop-count", type=int, default=None,
        help="Stop adding when one objective check fails this many times "
             "within the current tranche (default 3).",
    )
    parser.add_argument(
        "--workflow-failure-stop-count", type=int, default=None,
        help="Stop adding when one normalized workflow failure repeats this "
             "many times within the current tranche (default 2).",
    )
    parser.add_argument(
        "--candidate-ab-test", action=argparse.BooleanOptionalAction,
        default=None,
        help="After Generator edits the skill, replay the exact completed "
             "Train assets; require a complete, executable, strictly better "
             "candidate with no critical-check regression. Enabled by default.",
    )
    parser.add_argument(
        "--candidate-ab-min-samples", type=int, default=None,
        help="Minimum completed paired Train samples required to accept a "
             "Generator candidate (default 10, clipped to pool size).",
    )
    parser.add_argument(
        "--max-contract-repair-attempts", type=int, default=None,
        help="Maximum consecutive skill-contract repair attempts before "
             "formal training stops for an explicit fix (default 3).",
    )
    parser.add_argument(
        "--executor-code-retries", type=int, default=None,
        help="Immediate retries for Executor-authored Python syntax/call "
             "errors. These do not consume Generator contract-repair "
             "attempts (default 1).",
    )
    parser.add_argument(
        "--transient-retries", type=int, default=None,
        help="Retries per task for transient provider/renderer errors "
             "(connection blips, gateway 5xx, lost renderer page) with a "
             "30/60/120/300/600/600 s backoff before they abort the run. Default 6.",
    )
    parser.add_argument(
        "--executor-reps", type=int, default=None,
        help="Independent executor sessions per task per evaluation (default 1). "
             "K>1 aggregates them: checks by strict majority, joint scores by "
             "mean; rep 0 writes to the task dir, reps 1..K-1 to rep_<k>/.",
    )
    parser.add_argument("--max-rounds", type=int, help="Override max rounds")
    parser.add_argument(
        "--executor-model", default=None,
        help="Exact Agent SDK model for assembly executors. Overrides auth defaults.",
    )
    parser.add_argument(
        "--generator-model", default=None,
        help="Exact Agent SDK model for the skill generator. Overrides auth defaults.",
    )
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Per-executor agent turn cap (overrides "
                             "config.evolution.executor_max_turns, default 60). "
                             "Bump to 100+ if Phase 5 frequently runs out of budget.")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Concurrent sim workers in --multi mode "
                             "(overrides config.execution.max_parallel). "
                             "Each worker pins to its own port.")
    parser.add_argument("--headless", action="store_true",
                        help="Launch the sim worker headless (accepted for compatibility).")
    parser.add_argument("--sim-base-port", type=int, default=None,
                        help="First sim-worker port to use; workers occupy "
                             "[base, base+parallel-1]. Default 9876.")
    parser.add_argument("--sim-worker-max-rss-mb", type=int, default=None,
                        help="Resident-memory cap per sim worker in MB (default 6144, "
                             "config execution.sim_worker_max_rss_mb). A snippet that "
                             "crosses it is interrupted; 0 disables the guard.")
    parser.add_argument(
        "--auth", choices=("local", "proxy"), default="proxy",
        help="Auth selection for legacy Claude backends. Responses uses the "
             "OPENAI_API_KEY or ANTHROPIC_AUTH_TOKEN from .env without "
             "mutating it. 'local' — use Claude Code's native creds (keychain "
             "OAuth / API key from `claude setup-token`).  'proxy' — use "
             "ANTHROPIC_BASE_URL / AUTH_TOKEN exported in the shell "
             "(source .env.local first).  The sim MCP tool is an in-process "
             "SDK server, so both routes register it.",
    )
    parser.add_argument(
        "--generator-auth", choices=("local", "proxy"), default=None,
        help="Override the auth mode used ONLY for run_generator calls. "
             "Use this to keep executors on a paid proxy (--auth proxy) "
             "while routing the generator's heavier reasoning step to "
             "the local Claude Code subscription, which is more stable "
             "for long thinking + tool-use chains. Default: same as --auth.",
    )
    parser.add_argument(
        "--executor-effort", choices=("low", "medium", "high", "max"),
        default=None,
        help="Reasoning effort for run_executor (SDK option `effort`). "
             "low = fastest, max = deepest. Executor mostly executes "
             "pre-canned pipeline steps, so 'low' is usually enough on its "
             "own; the shared default keeps it level with the other agents. "
             f"Default {DEFAULT_AGENT_EFFORT}.",
    )
    parser.add_argument(
        "--generator-effort", choices=("low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Reasoning effort for run_generator. Generator must induce "
             "principles that generalize across the batch — favor 'max' or "
             "'high' when iterating on the skill. "
             f"Default {DEFAULT_AGENT_EFFORT}.",
    )
    parser.add_argument(
        "--subagent-model", default=None,
        help="Force every executor-spawned validation subagent to this exact "
             "model, overriding skill YAML frontmatter. Intended for "
             "controlled model-comparison experiments.",
    )
    parser.add_argument(
        "--subagent-effort", choices=("low", "medium", "high", "max"),
        default=None,
        help="Force reasoning effort for every executor-spawned subagent "
             "(validate-assembly, validate-joints), overriding skill YAML "
             "frontmatter. Default = use frontmatter, then inherit.",
    )
    parser.add_argument(
        "--bootstrap-generator", action="store_true",
        help="Run Generator once BEFORE round 0 to populate initial "
             "tools/ from a blank starter skill. Output lands in "
             "<run_dir>/bootstrap/skill which becomes the skill for "
             "round 0 executors. Point --skill-dir at a starter skill with no tools/.",
    )
    parser.add_argument(
        "--no-generator", dest="skip_generator", action="store_true",
        help="Skip the post-round Generator stage entirely. Use for "
             "standalone full-pool evaluation; this also disables adaptive "
             "Train split semantics. Example: --split validation "
             "--no-generator --max-rounds 1.",
    )
    parser.add_argument(
        "--no-report", dest="generate_report", action="store_false",
        default=True,
        help="Do not generate <run_dir>/report/index.html and rendered MP4s "
             "when a formal multi-task training process ends.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from a previous run dir (loads loop_state.json and "
             "continues at round_completed+1).  The run dir must already "
             "contain at least one completed round (round_0/ + state file).",
    )
    parser.add_argument(
        "--retry-wait", type=int, default=0,
        help="Minutes to sleep on a transient executor error (Claude Code "
             "5-hour usage cap, rate limit, 429, etc.) before retrying the "
             "same task.  Use ~305 (5h + buffer) for the local-Claude "
             "subscription window.  Default 0 = no auto-retry.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Maximum retry attempts per task per round when --retry-wait > 0. "
             "Default 3.",
    )
    parser.add_argument(
        "--stagger-sec", type=int, default=0,
        help="Seconds between successive workers' starts within a round "
             "(worker idx waits idx * N s; reps of one task wait k * N s). "
             "Default 0: all workers start at once, bounded only by the port "
             "pool.  The delay was insurance for a legacy `uvx blender-mcp` "
             "handshake race the sim worker does not have; at 20 s it made "
             "every 69-task round take ~24 min for ~1-min sessions.  Set >0 "
             "only to debug bootstrap races.",
    )
    parser.add_argument(
        "--skill-dir", type=Path, default=None,
        help="Override config.skill_dir (the initial skill the loop "
             "evolves from). Default skills/v_io_sdk; e.g. "
             "--skill-dir skills/v_io for the SDK-less starter.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--executor-backend", choices=("responses", "claude", "codex"), default=None,
        help="Model backend for the executor (default from config: responses).",
    )
    parser.add_argument(
        "--generator-backend", choices=("responses", "claude", "codex"), default=None,
        help="Model backend for the generator (default from config: responses).",
    )
    args = parser.parse_args()
    # Shared provider credentials and repo-local dataset/runtime settings.
    load_env_chain()

    # Defense in depth: every Claude CLI spawned by this process inherits the
    # hard kill-switch. Individual ClaudeAgentOptions also carry the same gate
    # so programmatic callers that bypass main() remain protected.
    os.environ["CLAUDE_CODE_DISABLE_FAST_MODE"] = "1"

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    config = EvoConfig.from_yaml(args.config)
    if args.executor_backend:
        config.executor_backend = args.executor_backend
    if args.generator_backend:
        config.generator_backend = args.generator_backend
    if args.data_root:
        config.tasks_root = args.data_root / "tasks"
        config.gt_root = args.data_root / "gt"
    _apply_dataset_variant(config.tasks_root)
    if args.max_rounds:
        config.max_rounds = args.max_rounds
    if args.max_turns is not None:
        config.executor_max_turns = args.max_turns
    if args.parallel is not None:
        config.parallel = args.parallel
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.batch_ratio is not None:
        try:
            e, m, h = (int(x) for x in args.batch_ratio.split(":"))
            config.batch_ratio = {"easy": e, "medium": m, "hard": h}
        except Exception:
            raise SystemExit(
                f"--batch-ratio expects 'easy:medium:hard' integers, got: "
                f"{args.batch_ratio!r}"
            )
    if args.adaptive_train is not None:
        config.adaptive_train = args.adaptive_train
    if args.fixed_batch is not None:
        config.fixed_batch = args.fixed_batch
    if args.train_increment is not None:
        config.train_increment = args.train_increment
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.objective_failure_stop_count is not None:
        config.objective_failure_stop_count = args.objective_failure_stop_count
    if args.workflow_failure_stop_count is not None:
        config.workflow_failure_stop_count = args.workflow_failure_stop_count
    if args.candidate_ab_test is not None:
        config.candidate_ab_test = args.candidate_ab_test
    if args.candidate_ab_min_samples is not None:
        config.candidate_ab_min_samples = args.candidate_ab_min_samples
    if args.max_contract_repair_attempts is not None:
        config.max_contract_repair_attempts = args.max_contract_repair_attempts
    if args.executor_code_retries is not None:
        config.executor_code_retries = args.executor_code_retries
    if args.transient_retries is not None:
        config.transient_retries = args.transient_retries
    if args.executor_reps is not None:
        config.executor_reps = max(1, int(args.executor_reps))
    if args.headless:
        config.headless = True
    if args.sim_base_port is not None:
        config.sim_base_port = args.sim_base_port
    if args.sim_worker_max_rss_mb is not None:
        config.sim_worker_max_rss_mb = args.sim_worker_max_rss_mb
    # SimWorkerManager reads the cap from the environment so every spawn site
    # (executor, generator, try_skill, validation) inherits it unchanged.
    from evo.engine.sim_worker import MEMORY_LIMIT_ENV
    os.environ[MEMORY_LIMIT_ENV] = str(config.sim_worker_max_rss_mb)
    if args.skill_dir is not None:
        skill_dir = args.skill_dir.resolve()
        if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").is_file():
            raise SystemExit(
                f"--skill-dir {skill_dir} must be a directory containing SKILL.md"
            )
        config.skill_dir = skill_dir

    # Auth + matching default model. The Responses backend consumes the
    # already-loaded API env directly; do not let Claude's local-auth helper
    # strip the proxy token needed by OpenAI Agents SDK.
    uses_claude = (
        config.executor_backend == "claude"
        or config.generator_backend == "claude"
    )
    if uses_claude:
        auth_info = setup_auth_env(args.auth)
    else:
        # Report exactly what the client will use — resolved by the backend
        # itself, so this banner cannot drift from the real configuration.
        from evo.engine.responses_backend import (
            _responses_api_key,
            _responses_base_url,
        )

        responses_base_url = _responses_base_url()
        auth_info = {
            "mode": "responses",
            "base_url": responses_base_url,
            "api_key": "(redacted)" if _responses_api_key(responses_base_url)
            else "(unset)",
        }
    print(f"auth: {args.auth}  ({auth_info})")
    print("Fast mode: disabled; Responses uses service_tier=default")
    if config.executor_backend == "claude":
        config.executor_model = (
            "claude-opus-4-7" if args.auth == "local"
            else os.environ.get("ANTHROPIC_MODEL") or "claude-opus-4-6"
        )
    if config.generator_backend == "claude":
        config.generator_model = (
            "claude-opus-4-7" if args.auth == "local"
            else os.environ.get("ANTHROPIC_MODEL") or "claude-opus-4-6"
        )
    if args.executor_model:
        config.executor_model = args.executor_model
    if args.generator_model:
        config.generator_model = args.generator_model

    if args.dry_run:
        tasks = discover_tasks(config)
        mapping = config.load_mapping()
        print(f"Tasks dir:  {config.tasks_root}")
        print(f"GT dir:     {config.gt_root}")
        print(f"Mapping:    {config.mapping_path} ({'found' if mapping else 'missing'})")
        print(f"Skill:      {config.skill_dir}")
        print(f"Rounds:     {config.max_rounds}")
        print(f"Parallel:   {config.parallel} (headless={config.headless}, "
              f"port_base={config.sim_base_port})")
        print(f"Adaptive:   {config.adaptive_train} "
              f"(increment={config.train_increment}, "
              f"max_train={config.max_train_samples}, "
              f"objective_stop={config.objective_failure_stop_count}, "
              f"workflow_stop={config.workflow_failure_stop_count})")
        print(f"Candidate A/B: {config.candidate_ab_test}")
        print(f"Tasks:      {len(tasks)}")
        if tasks:
            print(f"\nFirst 10 tasks:")
            for t in tasks[:10]:
                cat = mapping.get("task_to_category", {}).get(t["task_id"], "?")
                print(f"  {t['task_id']}/{t['instance']} ({cat})")
        return

    single = tuple(args.single) if args.single else None
    include = None
    if args.include:
        wanted = [p.strip() for p in args.include.split(",") if p.strip()]
        try:
            include = [tuple(p.split("/", 1)) for p in wanted]
        except Exception:
            raise SystemExit(
                f"--include expects 'task_id/instance' pairs, got: {args.include!r}"
            )

    use_lightphyx = os.environ.get("EVO_RENDER_BACKEND", "lightphyx") == "lightphyx"
    # Envs held on the shared pool at once: one per sim worker, the
    # Generator's own worker (port sim_base_port + parallel), and the oracle
    # renders running in this process. Sizing it to `parallel` alone left the
    # last two to overflow onto extra pages stacked on full browsers.
    from evo.oracle.render_lightphyx import ORACLE_RENDER_SIMS, configure_render_pool

    configure_render_pool(ORACLE_RENDER_SIMS)
    workers = 1 if single is not None else config.parallel
    render_capacity = workers + 1 + ORACLE_RENDER_SIMS
    _shift_to_free_port_range(config)
    with SharedLightphyxServer(render_capacity, enabled=use_lightphyx):
        run_evolution(
            config, single=single, include=include,
            split=args.split if args.split is not None else config.split,
            resume_dir=args.resume,
            retry_wait_min=args.retry_wait,
            max_retries=args.max_retries,
            stagger_sec=args.stagger_sec,
            generator_auth=args.generator_auth,
            executor_effort=args.executor_effort,
            generator_effort=args.generator_effort,
            subagent_model=args.subagent_model,
            subagent_effort=args.subagent_effort,
            oracle_mode=args.oracle_mode,
            oracle_effort=args.oracle_effort,
            oracle_votes=args.oracle_votes,
            oracle_model=args.oracle_model,
            regression_tol=args.regression_tol,
            bootstrap_generator=args.bootstrap_generator,
            skip_generator=args.skip_generator,
            generate_report=args.generate_report,
            gate_split=args.gate_split,
            gate_mode=args.gate_mode,
            gate_min_drop=args.gate_min_drop,
            gate_reps=args.gate_reps,
            gate_z=args.gate_z,
            gate_min_gain=args.gate_min_gain,
            gate_reroll=args.gate_reroll,
            gate_extend=args.gate_extend,
            gate_limit=args.gate_limit,
        )


if __name__ == "__main__":
    main()
