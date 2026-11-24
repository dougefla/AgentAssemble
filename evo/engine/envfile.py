"""Minimal `.env` loader — flexible config without sourcing a shell script.

Put host-specific config in a `.env` at the repo root and it's picked up
automatically by the `evo` CLI and the loop. Provider credentials usually live
outside the repo instead (`~/.env`, or `~/work/.env` on DSW); `load_env_chain`
reads all three so a checkout never has to carry secrets. Recognised keys (all
optional):

    LIGHTPHYX_CLI=/path/to/light-3js-phyx/packages/server/dist/cli.js
    EVO_RENDER_BACKEND=lightphyx      # oracle renderer: lightphyx | blender
    BLENDER_BIN=/path/to/blender      # legacy oracle A/B fallback only
                                      # (EVO_RENDER_BACKEND=blender)
    EVO_DATASET_ROOT=/path/to/dataset/artvip_sanitized
    EVO_DATA_ROOT=...                # overrides <dataset>/tasks
    EVO_GT_ROOT=...                  # overrides <dataset>/gt
    EVO_RAW_DATA_ROOT=...            # raw ArtVIP root, for re-sanitizing
    ANTHROPIC_BASE_URL=...           # Claude API/proxy auth (--auth proxy)
    ANTHROPIC_AUTH_TOKEN=...
    ANTHROPIC_MODEL=...
    OPENAI_API_BASE=https://.../v1   # Responses provider root; wins over the
                                     # OPENAI_BASE_URL/ANTHROPIC_BASE_URL pair
    OPENAI_BASE_URL=https://.../v1   # Responses-compatible provider root
    OPENAI_API_KEY=...               # OpenAI-compatible provider auth
    OPENROUTER_EVO_KEY=...           # OpenRouter experiment-spend key (preferred)
    OPENROUTER_API_KEY=...           # OpenRouter Responses auth (fallback)
    EVO_RESPONSES_PROVIDER=local_cpa # route every agent to LOCAL_CPA_URL/KEY
    LOCAL_CPA_URL=http://.../v1      # local Responses-compatible gateway
    LOCAL_CPA_KEY=...                # its auth (bare model ids, e.g. gpt-6.1-sol)
    EVO_ORACLE_BACKEND=responses     # subagent oracle judge: responses | claude
    EVO_ORACLE_MODEL=...             # override the subagent oracle judge model

Already-exported environment variables win unless override=True, so an explicit
`export FOO=...` in your shell still takes precedence over the file.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Load KEY=VALUE lines from `path` (default: <repo>/.env) into os.environ.

    Returns the list of keys set. Lines that are blank, comments (`#`), or have
    no `=` are ignored; a leading `export ` and surrounding quotes are stripped.
    """
    path = path or (PROJECT_ROOT / ".env")
    loaded: list[str] = []
    if not path.is_file():
        return loaded
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


def env_chain() -> list[Path]:
    """The `.env` files consulted, highest precedence first.

    `~/work/.env` is the DSW provider file, `<repo>/.env` holds the
    dataset/runtime settings that belong to this checkout, and `~/.env` is the
    workstation-wide credential file. Because `load_dotenv` never overwrites a
    key that is already set, the first file to define a key wins.
    """
    home = Path.home()
    return [home / "work/.env", PROJECT_ROOT / ".env", home / ".env"]


def load_env_chain(*, override: bool = False) -> list[str]:
    """Load every file in `env_chain()` into os.environ; return the keys set.

    Single entry point for anything that needs provider credentials — the
    Responses backend, the subagent oracle, the loop, the CLI, and the offline
    report tooling all call this so they resolve the same configuration.
    """
    loaded: list[str] = []
    for path in env_chain():
        loaded.extend(load_dotenv(path, override=override))
    return loaded
