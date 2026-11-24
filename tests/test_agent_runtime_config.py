"""Regression tests for the fixed ArtVIP experiment runtime contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import inspect
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import yaml

from evo.engine.agents import (
    ASSEMBLY_SDK_DIR,
    _restore_frozen_sdk,
    _snapshot_frozen_sdk,
    CLAUDE_FAST_MODE_DISABLED_ENV,
    CLAUDE_FAST_MODE_DISABLED_SETTINGS,
    _load_skill_as_executor_context,
    run_executor,
    run_generator,
)
from evo.engine.envfile import env_chain, load_env_chain
from evo.engine import pricing
from evo.engine.model_defaults import DEFAULT_AGENT_EFFORT, DEFAULT_AGENT_MODEL
from evo.engine.loop import EvoConfig
from evo.engine.responses_backend import (
    SimResponsesMCPServer,
    DEFAULT_SERVICE_TIER,
    EmptyResponsesOutputError,
    GuardedOpenAIResponsesModel,
    MAX_TEXT_TOOL_OUTPUT_CHARS,
    ResponsesRunOutcome,
    _bounded_text,
    _inject_runtime_globals,
    _provider_model_name,
    _resolve_scoped,
    _responses_api_key,
    _responses_base_url,
    _settings,
    make_file_tools,
    run_responses_agent,
)
from agents.models.openai_responses import OpenAIResponsesModel


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("skill, agent", [
    ("v_minimal", "validate-assembly"), ("v_minimal", "validate-joints"),
    ("v_beginner", "validate-joints"),
])
def test_internal_validators_match_the_shared_agent_defaults(
    skill: str, agent: str,
) -> None:
    # Frontmatter cannot import model_defaults, so assert the two agree —
    # this is the only thing stopping the validators from silently drifting
    # off the model the rest of the loop runs on.
    text = (ROOT / "skills" / skill / "agents" / f"{agent}.md").read_text()
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["model"] == DEFAULT_AGENT_MODEL
    assert frontmatter["effort"] == DEFAULT_AGENT_EFFORT


def test_every_agent_role_shares_one_model_and_effort() -> None:
    from evo.oracle.subagent import DEFAULT_ORACLE_BACKEND, DEFAULT_ORACLE_MODEL

    config = EvoConfig.from_yaml(ROOT / "evo" / "configs" / "default.yaml")
    assert config.executor_model == DEFAULT_AGENT_MODEL
    assert config.generator_model == DEFAULT_AGENT_MODEL
    assert DEFAULT_ORACLE_MODEL[DEFAULT_ORACLE_BACKEND] == DEFAULT_AGENT_MODEL
    assert DEFAULT_AGENT_MODEL == "gpt-6.1-sol"
    assert DEFAULT_AGENT_EFFORT == "medium"


def test_fast_mode_is_hard_disabled() -> None:
    assert CLAUDE_FAST_MODE_DISABLED_ENV == {
        "CLAUDE_CODE_DISABLE_FAST_MODE": "1"
    }
    settings = json.loads(CLAUDE_FAST_MODE_DISABLED_SETTINGS)
    assert settings["fastMode"] is False
    assert settings["fastModePerSessionOptIn"] is True


def test_generator_sdk_snapshot_restores_mutations(
    tmp_path: Path, monkeypatch,
) -> None:
    sdk = tmp_path / "assembly_sdk"
    sdk.mkdir()
    runtime = sdk / "runtime.py"
    runtime.write_text("ORIGINAL\n")
    monkeypatch.setattr("evo.engine.agents.ASSEMBLY_SDK_DIR", sdk)
    snapshot = _snapshot_frozen_sdk()
    runtime.write_text("MUTATED\n")
    (sdk / "injected.py").write_text("BAD\n")
    changed = _restore_frozen_sdk(snapshot)
    assert changed == ["injected.py", "runtime.py"]
    assert runtime.read_text() == "ORIGINAL\n"
    assert not (sdk / "injected.py").exists()


def test_default_execution_is_full_pool_with_two_workers() -> None:
    config = EvoConfig.from_yaml(ROOT / "evo" / "configs" / "default.yaml")
    assert config.batch_size == 0
    assert config.parallel == 2
    assert config.split == "train"
    assert config.max_train_samples == 69
    assert config.splits_path.name in {
        "splits.json", "splits_experiment_50_30_50_seed42.json",
    }
    assert config.executor_backend == "responses"
    assert config.generator_backend == "responses"
    assert config.executor_model == DEFAULT_AGENT_MODEL
    assert config.generator_model == DEFAULT_AGENT_MODEL


def test_split_manifest_cannot_silently_target_another_dataset(
    tmp_path: Path, monkeypatch,
) -> None:
    tasks = tmp_path / "new" / "tasks" / "task_00" / "0" / "parts"
    tasks.mkdir(parents=True)
    (tasks / "part_0.obj").write_text("v 0 0 0\n")
    gt = tmp_path / "new" / "gt" / "task_00" / "0"
    gt.mkdir(parents=True)
    (gt / "scene.urdf").write_text('<robot name="x"><link name="x"/></robot>')
    stale = tmp_path / "old_splits.json"
    stale.write_text(json.dumps([
        {"task_id": "task_99", "instance": "9", "split": "train"},
    ]))
    monkeypatch.setenv("EVO_SPLITS_PATH", str(stale))
    config = EvoConfig(tasks_root=tmp_path / "new" / "tasks", gt_root=tmp_path / "new" / "gt")
    with pytest.raises(ValueError, match="cross-dataset split"):
        config.load_splits()


def test_responses_request_settings_disable_storage_and_fast_lane() -> None:
    settings = _settings("high")
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "high"
    assert settings.store is False
    assert settings.extra_args == {"service_tier": DEFAULT_SERVICE_TIER}
    assert DEFAULT_SERVICE_TIER == "default"


def test_anthropic_style_base_url_is_normalized_to_v1(monkeypatch) -> None:
    monkeypatch.delenv("EVO_RESPONSES_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.alayalab.ai")
    assert _responses_base_url() == "https://api.alayalab.ai/v1"


def test_openai_api_base_wins_over_other_provider_roots(monkeypatch) -> None:
    monkeypatch.delenv("EVO_RESPONSES_PROVIDER", raising=False)
    # A workstation ~/.env carries OPENAI_API_BASE for the Responses provider
    # alongside an OPENAI_BASE_URL for a different one. Preferring the latter
    # would send the OpenRouter key to the wrong host.
    monkeypatch.setenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.alayalab.ai/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")
    monkeypatch.delenv("OPENROUTER_EVO_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider-key")
    base_url = _responses_base_url()
    assert base_url == "https://openrouter.ai/api/v1"
    assert _responses_api_key(base_url) == "router-key"


def test_env_chain_reads_the_home_credential_file(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    (home / "work").mkdir(parents=True)
    (home / ".env").write_text(
        "OPENROUTER_API_KEY=from-home\nOPENAI_API_BASE=https://openrouter.ai/api/v1\n"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    assert (home / ".env") in env_chain()
    load_env_chain()
    assert os.environ["OPENROUTER_API_KEY"] == "from-home"
    assert os.environ["OPENAI_API_BASE"] == "https://openrouter.ai/api/v1"


def test_openrouter_uses_dedicated_key_and_openai_model_namespace(
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVO_RESPONSES_PROVIDER", raising=False)
    base_url = "https://openrouter.ai/api/v1"
    monkeypatch.delenv("OPENROUTER_EVO_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "different-provider-key")
    assert _responses_api_key(base_url) == "router-key"
    assert _provider_model_name("gpt-5.6-luna", base_url) == (
        "openai/gpt-5.6-luna"
    )
    assert _provider_model_name("openai/gpt-6.1-sol", base_url) == (
        "openai/gpt-6.1-sol"
    )


def test_openrouter_prefers_the_experiment_spend_key(monkeypatch) -> None:
    monkeypatch.delenv("EVO_RESPONSES_PROVIDER", raising=False)
    # ~/.env carries both a general OpenRouter key and OPENROUTER_EVO_KEY; the
    # latter is the one whose spend we track for experiments, so it wins.
    base_url = "https://openrouter.ai/api/v1"
    monkeypatch.setenv("OPENROUTER_EVO_KEY", "evo-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "different-provider-key")
    assert _responses_api_key(base_url) == "evo-key"
    monkeypatch.delenv("OPENROUTER_EVO_KEY")
    assert _responses_api_key(base_url) == "router-key"
    # A non-OpenRouter host never sees either OpenRouter key.
    monkeypatch.setenv("OPENROUTER_EVO_KEY", "evo-key")
    assert _responses_api_key("https://api.alayalab.ai/v1") == (
        "different-provider-key"
    )


def test_responses_backend_uses_explicit_item_state() -> None:
    source = inspect.getsource(run_responses_agent)
    assert "previous_response_id=" not in source
    assert "conversation_id=" not in source
    assert "result.to_input_list()" in source


def test_vminimal_executor_context_is_compact_and_runtime_owned() -> None:
    context, specialists = _load_skill_as_executor_context(
        ROOT / "skills" / "v_minimal",
        runtime_globals_injected=True,
    )
    assert len(context) < 30_000
    assert "Tool source is intentionally not inlined" in context
    assert "# tools/ source" not in context
    assert "def _export_one_link_mesh" not in context
    assert "ASSEMBLY_SDK_DIR`, `SKILL_DIR`, `INPUT_DIR`, and `OUTPUT_DIR` are injected" in context
    assert set(specialists) == {"validate-assembly", "validate-joints"}


def test_executor_context_appends_evolved_helper_contract(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# workflow\n")
    (skill / "ACTIONS.md").write_text("# frozen sdk\n")
    (skill / "TOOLS.md").write_text(
        "### `tools/geometry.py::axis(value)`\n\n"
        "- **Parameters:** value\n"
        "- **Returns:** vector\n"
        "- **Raises:** ValueError\n"
        "- **Example:** `axis('X')`\n"
    )
    context, specialists = _load_skill_as_executor_context(
        skill,
        runtime_globals_injected=True,
    )
    assert "# ACTIONS.md\n\n# frozen sdk" in context
    assert "# TOOLS.md\n\n### `tools/geometry.py::axis(value)`" in context
    assert "TOOLS.md is the evolved-helper contract" in context
    assert specialists == {}


def test_experiment_subagent_overrides_frontmatter(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    agents = skill / "agents"
    agents.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# workflow\n")
    (agents / "validator.md").write_text(
        "---\n"
        "name: validator\n"
        "description: validate\n"
        "model: gpt-5.6-luna\n"
        "effort: low\n"
        "---\n"
        "Return a verdict.\n"
    )
    _context, specialists = _load_skill_as_executor_context(
        skill,
        subagent_model="gpt-6.1-sol",
        subagent_effort="medium",
        runtime_globals_injected=True,
    )
    validator = specialists["validator"]
    assert validator.model == "gpt-6.1-sol"
    assert validator.effort == "medium"


def test_executor_prompt_has_no_undefined_phase_protocol() -> None:
    source = inspect.getsource(run_executor)
    assert "Begin Phase" not in source
    assert "Run Phases" not in source
    assert "Phase 7" not in source


def test_scoped_file_tools_reject_paths_outside_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    with pytest.raises(PermissionError):
        _resolve_scoped(
            str(tmp_path / "outside.txt"), cwd=allowed, roots=[allowed]
        )


def test_text_tool_output_is_bounded_and_preserves_tail() -> None:
    text = "HEAD" + ("x" * (MAX_TEXT_TOOL_OUTPUT_CHARS * 2)) + "TAIL"
    bounded = _bounded_text(text, label="test")
    assert len(bounded) <= MAX_TEXT_TOOL_OUTPUT_CHARS
    assert bounded.startswith("HEAD")
    assert bounded.endswith("TAIL")
    assert "truncated" in bounded


def test_raw_responses_artifact_read_is_blocked(tmp_path: Path) -> None:
    artifact = tmp_path / "executor.responses.json"
    artifact.write_text('{"items": ["' + ("x" * 100_000) + '"]}')
    read_tool = next(
        tool for tool in make_file_tools(
            cwd=tmp_path, read_roots=[tmp_path], write_root=tmp_path,
        )
        if tool.name == "Read"
    )
    result = asyncio.run(read_tool.on_invoke_tool(
        None,
        json.dumps({"file_path": str(artifact)}),
    ))
    assert "Raw Responses trajectory blocked" in str(result)
    assert "100000" not in str(result)


def test_sim_mcp_injects_runtime_owned_globals(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_send(code: str, port: int):
        captured["code"] = code
        captured["port"] = str(port)
        return {"status": "ok", "stdout": "ok", "result": None}

    monkeypatch.setattr(
        "evo.engine.responses_backend.send_to_sim_worker", fake_send
    )
    server = SimResponsesMCPServer(12000, {
        "SKILL_DIR": "/skill",
        "INPUT_DIR": "/input",
        "OUTPUT_DIR": "/output",
    })
    result = asyncio.run(server.call_tool(
        "mcp__sim__execute_python_code",
        {"code": "print(SKILL_DIR, INPUT_DIR, OUTPUT_DIR)"},
    ))
    assert not result.isError
    assert captured["port"] == "12000"
    assert "SKILL_DIR = '/skill'" in captured["code"]
    assert "INPUT_DIR = '/input'" in captured["code"]
    assert "OUTPUT_DIR = '/output'" in captured["code"]
    assert captured["code"].endswith(
        "print(SKILL_DIR, INPUT_DIR, OUTPUT_DIR)"
    )


def test_sim_mcp_prevents_runtime_global_overwrite(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_send(code: str, port: int):
        captured["code"] = code
        return {"status": "ok", "stdout": "ok", "result": None}

    monkeypatch.setattr(
        "evo.engine.responses_backend.send_to_sim_worker", fake_send
    )
    server = SimResponsesMCPServer(12000, {"SKILL_DIR": "/skill"})
    result = asyncio.run(server.call_tool(
        "mcp__sim__execute_python_code",
        {"code": "SKILL_DIR = '/wrong'\nprint(SKILL_DIR)"},
    ))
    assert not result.isError
    assert "SKILL_DIR = '/skill'" in captured["code"]
    assert "SKILL_DIR = '/wrong'" not in captured["code"].splitlines()
    assert "__evo_ignored_runtime_assignment_SKILL_DIR = '/wrong'" in captured["code"]
    assert captured["code"].endswith("print(SKILL_DIR)")


def test_runtime_values_are_mirrored_and_protected_in_environ(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SKILL_DIR", raising=False)
    namespace: dict = {}
    protected = _inject_runtime_globals(
        "import os\n"
        "SKILL_DIR = '/wrong-global'\n"
        "os.environ['SKILL_DIR'] = '/wrong-env'\n"
        "OBSERVED = (SKILL_DIR, os.environ['SKILL_DIR'])",
        {"SKILL_DIR": "/skill"},
    )
    exec(protected, namespace)
    assert namespace["OBSERVED"] == ("/skill", "/skill")
    assert "__EVO_IGNORED_RUNTIME_ASSIGNMENT_SKILL_DIR" in os.environ


def test_runtime_global_protection_covers_nontrivial_store_targets() -> None:
    protected = _inject_runtime_globals(
        "for SKILL_DIR in ['bad']:\n    print(SKILL_DIR)\n"
        "del OUTPUT_DIR",
        {"SKILL_DIR": "/skill", "OUTPUT_DIR": "/output"},
    )
    assert "for __evo_ignored_runtime_assignment_SKILL_DIR" in protected
    assert "print(SKILL_DIR)" in protected
    assert "del __evo_ignored_runtime_assignment_OUTPUT_DIR" in protected


def test_null_responses_output_gets_explicit_diagnostic(
    tmp_path: Path, monkeypatch,
) -> None:
    response = SimpleNamespace(
        output=None,
        id="resp_null",
        status="completed",
        error=None,
        incomplete_details=None,
        model_dump=lambda **_: {
            "id": "resp_null", "status": "completed", "output": None,
        },
    )

    async def fake_fetch(_self, *args, **kwargs):
        return response

    monkeypatch.setattr(OpenAIResponsesModel, "_fetch_response", fake_fetch)
    diagnostic = tmp_path / "invalid.jsonl"
    model = GuardedOpenAIResponsesModel(
        "gpt-6.1-sol", object(), invalid_response_path=diagnostic,
    )
    with pytest.raises(EmptyResponsesOutputError, match="output=null"):
        asyncio.run(model._fetch_response(stream=False))
    payload = json.loads(diagnostic.read_text())
    assert payload["response"]["id"] == "resp_null"
    assert payload["response"]["output"] is None


def _empty_usage() -> dict[str, int | float]:
    return {
        "n_api_calls": 2,
        "fresh_input": 10,
        "cache_read": 0,
        "cache_write_5m": 0,
        "cache_write_1h": 0,
        "output": 5,
        "cost_usd": 0.0,
        "cache_misses": 0,
    }


def test_executor_training_path_routes_to_the_shared_default(
    tmp_path: Path, monkeypatch,
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Minimal assembly skill\n")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    captured: dict = {}

    async def fake_run_responses_agent(**kwargs):
        captured.update(kwargs)
        output_dir.mkdir(exist_ok=True)
        (output_dir / "furniture.urdf").write_text(
            '<robot name="smoke"><link name="base"/></robot>'
        )
        kwargs["artifact_path"].write_text("{}")
        return ResponsesRunOutcome(
            final_output="done", response_id="resp_smoke",
            usage_summary=_empty_usage(), artifact_path=kwargs["artifact_path"],
        )

    monkeypatch.setattr(
        "evo.engine.agents.SimWorkerManager", lambda **_: nullcontext()
    )
    monkeypatch.setattr(
        "evo.engine.responses_backend.run_responses_agent",
        fake_run_responses_agent,
    )
    result = asyncio.run(run_executor(
        factory="task", instance="0", input_dir=input_dir,
        output_dir=output_dir, skill_dir=skill,
    ))
    assert result.success
    assert captured["model"] == DEFAULT_AGENT_MODEL
    assert captured["effort"] == DEFAULT_AGENT_EFFORT
    assert captured["subagent_default_effort"] == DEFAULT_AGENT_EFFORT
    assert captured["runtime_globals"] == {
        "ASSEMBLY_SDK_DIR": str(ROOT / "evo" / "assembly_sdk"),
        "SKILL_DIR": str(output_dir / "skill_snapshot"),
        "INPUT_DIR": str(input_dir),
        "OUTPUT_DIR": str(output_dir),
    }
    assert json.loads((output_dir / "result.json").read_text())["backend"] == "responses"


def test_generator_training_path_routes_to_the_shared_default(
    tmp_path: Path, monkeypatch,
) -> None:
    current = tmp_path / "current"
    current.mkdir()
    (current / "SKILL.md").write_text("# Before\n")
    output = tmp_path / "round" / "skill"
    log_file = tmp_path / "round" / "generator.log"
    log_file.parent.mkdir()
    captured: dict = {}

    async def fake_run_responses_agent(**kwargs):
        captured.update(kwargs)
        (output / "SKILL.md").write_text("# After\n")
        (output.parent / "generator_journal.md").write_text("Verified\n")
        kwargs["artifact_path"].write_text("{}")
        return ResponsesRunOutcome(
            final_output="updated", response_id="resp_generator",
            usage_summary=_empty_usage(), artifact_path=kwargs["artifact_path"],
        )

    monkeypatch.setattr(
        "evo.engine.responses_backend.run_responses_agent",
        fake_run_responses_agent,
    )
    result = asyncio.run(run_generator(
        current_skill_dir=current,
        current_skill_content={"SKILL.md": "# Before\n"},
        feedback_history=[], output_skill_dir=output, log_file=log_file,
        sim_port=None,
    ))
    assert "SKILL.md" in result.files_modified
    assert captured["model"] == DEFAULT_AGENT_MODEL
    assert captured["effort"] == DEFAULT_AGENT_EFFORT
    result_doc = json.loads((output.parent / "generator.result.json").read_text())
    assert result_doc["backend"] == "responses"


# --- cost accounting -------------------------------------------------------

def _usage(entries, requests=None):
    """A Responses Usage stand-in carrying per-request token details."""
    return SimpleNamespace(
        requests=requests if requests is not None else len(entries),
        input_tokens=sum(e["input_tokens"] for e in entries),
        input_tokens_details=SimpleNamespace(
            cached_tokens=sum(
                e["input_tokens_details"]["cached_tokens"] for e in entries
            )
        ),
        output_tokens=sum(e["output_tokens"] for e in entries),
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        request_usage_entries=entries,
    )


def test_cost_splits_fresh_input_from_cache_reads() -> None:
    # The aggregate input_tokens counts cached and uncached together, so
    # pricing it as fresh input overstates a cached run by ~10x.
    usage = _usage([{
        "input_tokens": 100_000,
        "output_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 90_000, "cache_write_tokens": 5_000},
    }])
    tokens = pricing.token_breakdown(usage)
    assert tokens == {
        "fresh_input": 10_000,
        "cache_read": 90_000,
        "cache_write": 5_000,
        "output": 1_000,
        "reasoning": 0,
    }
    cost, priced = pricing.cost_usd("gpt-6.1-sol", tokens)
    assert priced
    # 10k*$2 + 90k*$0.1 + 5k*$2.5 + 1k*$10 per 1M (gpt-6.1-sol list price)
    assert cost == pytest.approx((20_000 + 9_000 + 12_500 + 10_000) / 1e6)


def test_a_responses_run_is_no_longer_priced_at_zero() -> None:
    summary = pricing.summarize(
        _usage([{
            "input_tokens": 4_000,
            "output_tokens": 500,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        }]),
        DEFAULT_AGENT_MODEL,
    )
    assert summary["cost_usd"] > 0
    assert summary["cost_priced"] is True
    # Key names must match _summarize_session so both cost paths are
    # interchangeable for result.json and the round tables.
    assert {
        "n_api_calls", "fresh_input", "cache_read",
        "cache_write_5m", "cache_write_1h", "output", "cost_usd", "cache_misses",
    } <= set(summary)


def test_unknown_model_reports_unpriced_rather_than_free() -> None:
    summary = pricing.summarize(
        _usage([{
            "input_tokens": 1_000,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        }]),
        "some-unlisted-model",
    )
    assert summary["cost_priced"] is False
    assert summary["cost_usd"] == 0.0


def test_provider_namespaced_model_resolves_to_the_same_rates() -> None:
    assert pricing.rates_for("openai/gpt-6.1-sol") is pricing.rates_for("gpt-6.1-sol")


def test_oracle_cost_reaches_the_run_artifact() -> None:
    # The oracle outspends the executor per task; a cost summary that
    # counted only the executor understated the run several-fold.
    from evo.oracle.check_urdf import OracleResult

    result = OracleResult(passed=True, checks=[], cost_usd=0.36)
    assert result.to_dict()["cost_usd"] == 0.36
    assert OracleResult(passed=True, checks=[]).cost_usd is None


# --- the generator must learn what happened to its last candidate ----------

_AB_REJECTED = {
    "round": 1,
    "pass_rate": 0.3,
    "per_check_pass_rate": {"subagent_verdict": "3/10"},
    "generator_output_rejected": "critical_metric_regression",
    "candidate_ab_test": {
        "accepted": False,
        "action": "REJECT_CANDIDATE",
        "reason": "critical_metric_regression",
        "baseline_pass_rate": 0.3,
        "candidate_pass_rate": 0.0,
        "baseline_mean_score": 0.525,
        "candidate_mean_score": 0.35,
        "sample_count": 10,
        "critical_regressions": ["task_01/27:subagent_verdict"],
    },
}


def test_rejected_candidate_is_reported_to_the_generator() -> None:
    from evo.engine.agents import _format_previous_candidate_outcome

    note = _format_previous_candidate_outcome(_AB_REJECTED)
    assert "REJECTED" in note and "DISCARDED" in note
    # The controlled numbers, not just the verdict.
    assert "30% -> 0%" in note
    assert "0.525 -> 0.350" in note
    assert "task_01/27:subagent_verdict" in note
    # And the warning that stops it re-reading the trajectory as proof.
    assert "NOT a controlled comparison" in note


def test_accepted_candidate_is_reported_as_in_effect() -> None:
    from evo.engine.agents import _format_previous_candidate_outcome

    accepted = dict(_AB_REJECTED)
    accepted.pop("generator_output_rejected")
    accepted["candidate_ab_test"] = {
        **_AB_REJECTED["candidate_ab_test"],
        "accepted": True,
        "action": "ACCEPT_CANDIDATE",
        "reason": None,
        "candidate_pass_rate": 0.5,
        "candidate_mean_score": 0.6,
        "critical_regressions": [],
    }
    note = _format_previous_candidate_outcome(accepted)
    assert "ACCEPTED" in note and "in effect" in note
    assert "REJECTED" not in note


def test_no_note_before_any_candidate_has_been_judged() -> None:
    from evo.engine.agents import _format_previous_candidate_outcome

    assert _format_previous_candidate_outcome({}) == ""
    assert _format_previous_candidate_outcome({"round": 0, "pass_rate": 0.1}) == ""


def test_generator_prompt_carries_the_previous_ab_verdict(
    tmp_path: Path, monkeypatch,
) -> None:
    """End to end: the verdict must reach the prompt, not just a log file.

    It previously reached round_feedback.json and the experiment report but
    was never read back into the prompt, so the Generator kept optimizing
    against a result it could not see.
    """
    current = tmp_path / "current"
    current.mkdir()
    (current / "SKILL.md").write_text("# Before\n")
    output = tmp_path / "round" / "skill"
    log_file = tmp_path / "round" / "generator.log"
    log_file.parent.mkdir()
    captured: dict = {}

    async def fake_run_responses_agent(**kwargs):
        captured.update(kwargs)
        (output / "SKILL.md").write_text("# After\n")
        (output.parent / "generator_journal.md").write_text("Verified\n")
        kwargs["artifact_path"].write_text("{}")
        return ResponsesRunOutcome(
            final_output="updated", response_id="resp_generator",
            usage_summary=_empty_usage(), artifact_path=kwargs["artifact_path"],
        )

    monkeypatch.setattr(
        "evo.engine.responses_backend.run_responses_agent",
        fake_run_responses_agent,
    )
    # [-1] is the current round (its A/B has not run); [-2] carries the verdict.
    history = [_AB_REJECTED, {"round": 2, "pass_rate": 0.2, "runs": []}]
    asyncio.run(run_generator(
        current_skill_dir=current,
        current_skill_content={"SKILL.md": "# Before\n"},
        feedback_history=history, output_skill_dir=output, log_file=log_file,
        sim_port=None,
    ))
    prompt = captured["prompt"]
    assert "REJECTED" in prompt
    assert "task_01/27:subagent_verdict" in prompt
    assert "30% -> 0%" in prompt
    assert "NOT a controlled comparison" in prompt


def test_read_tool_attaches_only_vision_safe_valid_images(tmp_path: Path) -> None:
    # Run 8 round 5 aborted with a provider 400: the executor wrote an SVG, Read
    # saw image/svg+xml and attached it as an image. Only PNG/JPEG/GIF/WebP with
    # a valid header may become vision input; everything else reads as text.
    read_tool = next(
        tool for tool in make_file_tools(
            cwd=tmp_path, read_roots=[tmp_path], write_root=tmp_path,
        )
        if tool.name == "Read"
    )

    def read(name: str):
        return asyncio.run(read_tool.on_invoke_tool(
            None, json.dumps({"file_path": str(tmp_path / name)}),
        ))

    (tmp_path / "ok.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    out = read("ok.png")
    assert getattr(out[0], "image_url", "").startswith("data:image/png;base64,")

    (tmp_path / "plot.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'><rect/></svg>")
    out = read("plot.svg")
    assert not hasattr(out[0], "image_url") and "<svg" in out[0].text

    (tmp_path / "truncated.png").write_bytes(b"\x89PN")
    out = read("truncated.png")
    assert not hasattr(out[0], "image_url") and "not a valid image/png" in out[0].text


def test_local_cpa_provider_routes_url_key_and_bare_model_ids(monkeypatch) -> None:
    from evo.engine.responses_backend import _provider_model_name
    monkeypatch.setenv("EVO_RESPONSES_PROVIDER", "local_cpa")
    monkeypatch.setenv("LOCAL_CPA_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("LOCAL_CPA_KEY", "cpa-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_EVO_KEY", "evo-key")
    base_url = _responses_base_url()
    assert base_url == "http://127.0.0.1:8080/v1"
    assert _responses_api_key(base_url) == "cpa-key"
    assert _provider_model_name("gpt-6.1-sol", base_url) == "gpt-6.1-sol"
    monkeypatch.delenv("EVO_RESPONSES_PROVIDER")
    assert _responses_base_url() == "https://openrouter.ai/api/v1"
    assert _responses_api_key("https://openrouter.ai/api/v1") == "evo-key"


def test_responses_request_timeout_follows_effort(monkeypatch) -> None:
    from evo.engine.responses_backend import TIMEOUT_ENV, _request_timeout
    monkeypatch.delenv(TIMEOUT_ENV, raising=False)
    assert _request_timeout("low") == 300.0
    assert _request_timeout("medium") == 300.0
    assert _request_timeout("high") == 600.0
    assert _request_timeout(None) in (300.0, 600.0)
    monkeypatch.setenv(TIMEOUT_ENV, "120")
    assert _request_timeout("high") == 120.0


def test_responses_client_retries_requests_in_place(monkeypatch) -> None:
    from evo.engine.responses_backend import MAX_RETRIES_ENV, _client
    monkeypatch.setenv("EVO_RESPONSES_PROVIDER", "local_cpa")
    monkeypatch.setenv("LOCAL_CPA_URL", "http://127.0.0.1:8317/v1")
    monkeypatch.setenv("LOCAL_CPA_KEY", "test-key")
    monkeypatch.delenv(MAX_RETRIES_ENV, raising=False)
    client = _client("low")
    assert client.max_retries == 2 and client.timeout == 300.0
    monkeypatch.setenv(MAX_RETRIES_ENV, "0")
    assert _client("high").max_retries == 0


def test_first_request_gets_the_short_timeout_only_at_low_effort(monkeypatch) -> None:
    import asyncio
    from openai import NOT_GIVEN
    from openai._models import FinalRequestOptions
    from evo.engine.responses_backend import FIRST_TIMEOUT_ENV, SessionClient, _first_request_timeout
    monkeypatch.delenv(FIRST_TIMEOUT_ENV, raising=False)
    assert _first_request_timeout("low") == 90.0 and _first_request_timeout("high") is None
    monkeypatch.setenv(FIRST_TIMEOUT_ENV, "0")
    assert _first_request_timeout("low") is None
    monkeypatch.delenv(FIRST_TIMEOUT_ENV, raising=False)

    seen = []

    async def fake_request(self, cast_to, options, *, stream=False, stream_cls=None):
        seen.append(options.timeout)
        return "ok"

    monkeypatch.setattr("openai._base_client.AsyncAPIClient.request", fake_request)
    client = SessionClient(base_url="http://127.0.0.1:1/v1", api_key="k", timeout=300.0,
                           max_retries=2, first_request_timeout=90.0)
    opts = FinalRequestOptions(method="post", url="/responses", timeout=NOT_GIVEN)

    async def go():
        for _ in range(3):
            await client.request(object, opts)
    asyncio.run(go())
    assert seen[0] == 90.0 and seen[1] is NOT_GIVEN and seen[2] is NOT_GIVEN
    assert client.timeout == 300.0 and client.max_retries == 2



def test_acceptance_check_rejection_is_reported_without_ab_wording() -> None:
    from evo.engine.agents import _format_previous_candidate_outcome

    prev = {"round": 3, "pass_rate": 0.3, "generator_output_rejected": "candidate skill preflight failed: x"}
    note = _format_previous_candidate_outcome(prev)
    assert "REJECTED by the acceptance check" in note and "preflight failed: x" in note
    assert "A/B" not in note and "replayed" not in note
    adopted = _format_previous_candidate_outcome({"round": 3, "pass_rate": 0.3, "generator_changed": True}, same_tasks=True)
    assert "ADOPTED" in note or "ADOPTED" in adopted
    assert "same tasks" in adopted and "A/B" not in adopted
