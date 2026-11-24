"""Single source of truth for the model and reasoning effort every agent uses.

The loop drives five model-backed roles, and they all run on the same
Responses provider (see `evo.engine.responses_backend`):

    executor            turns a task's OBJ parts into a URDF (what it must
                        author depends on the dataset view)
    generator           rewrites the skill
    subagent oracle     visual judge that produces the training signal
    validate-*          in-executor visual checks  (skill frontmatter; the
                        v_beginner seed ships validate-joints only)

Keeping one model and one effort across all five makes a round's results
attributable to the skill rather than to a model mix, and stops the defaults
from drifting apart across `default.yaml`, the agent entrypoints, the CLI
prompts, and the skill frontmatter. Every one stays overridable per run
(`--executor-model`, `--oracle-effort`, `EVO_ORACLE_MODEL`, and friends).

The two validators are declared in `skills/*/agents/*.md` frontmatter, which
cannot import this module; `tests/test_agent_runtime_config.py` asserts that
frontmatter matches the constants here so the two cannot diverge silently.
"""
from __future__ import annotations

# Model alias as the repo names it. `_provider_model_name` maps this to the
# provider's id (on OpenRouter: `openai/gpt-6.1-sol`).
DEFAULT_AGENT_MODEL = "gpt-6.1-sol"

# Reasoning effort passed as ModelSettings.reasoning.effort.
DEFAULT_AGENT_EFFORT = "medium"

__all__ = ["DEFAULT_AGENT_MODEL", "DEFAULT_AGENT_EFFORT"]
