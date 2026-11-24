"""Portable path defaults — single source of truth for cross-platform paths.

Every host-specific path that used to be hardcoded somewhere lives here, driven
by an env var with a sensible fallback. Mac dev machines keep the old behavior
by exporting the right env vars (e.g. via a `.env` file at the repo root, or a
`.env.local`); Linux servers get reasonable defaults out of the box.

There are two distinct dataset shapes in the project:

  1. **Sanitized layout** (what loop.py / the executor / the oracle consume):

         <root>/tasks/<task_id>/<instance>/parts/*.obj + thumbnail.png
         <root>/gt/<task_id>/<instance>/scene.urdf

     Default root: `./dataset/artvip_full` (gitignored — populate it or set EVO_DATASET_ROOT).
     Override the *whole* root with EVO_DATASET_ROOT, or the two halves
     individually with EVO_DATA_ROOT (tasks) / EVO_GT_ROOT (gt).

  2. **Raw InfiniPart layout** (what `evo/data/loader.py` consumes when
     generating splits.json):

         <root>/<FactoryName>/<Instance>/scene.urdf
         <root>/<FactoryName>/<Instance>/outputs/.../objs/*.obj

     Default root: empty (caller must export EVO_RAW_DATA_ROOT).

Env vars
--------
BLENDER_BIN          Path or command for the Blender executable. Used ONLY
                     by the legacy oracle render fallback
                     (EVO_RENDER_BACKEND=blender, kept for A/B comparisons
                     until the visual judges are recalibrated).
                     Default: `blender` (assumed on PATH).
EVO_DATASET_ROOT     Root of a sanitized dataset; sets both task input and
                     GT roots to `<root>/tasks` and `<root>/gt`.
                     Default: ./dataset/artvip_full
EVO_DATA_ROOT        Sanitized task-input root; overrides
                     `${EVO_DATASET_ROOT}/tasks` if set explicitly.
EVO_GT_ROOT          Sanitized GT root; overrides
                     `${EVO_DATASET_ROOT}/gt` if set explicitly.
EVO_RAW_DATA_ROOT    Raw InfiniPart root for loader.py / sanitize scripts.
                     No default — set it explicitly when re-generating splits.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default dataset root (the active ArtVIP-derived dataset). Its content is
# gitignored; populate it or set EVO_DATASET_ROOT / EVO_DATA_ROOT / EVO_GT_ROOT.
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "artvip_full"


def _env_path(name: str, default: str | Path) -> Path:
    """Read an env-var-supplied path, expanding `~` and `$VAR` references."""
    raw = os.environ.get(name) or str(default)
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def blender_bin() -> Path:
    """Resolve the Blender executable (`BLENDER_BIN` env var wins).

    Legacy oracle fallback only: consumed by the Blender render path behind
    EVO_RENDER_BACKEND=blender, kept for A/B comparisons until the visual
    judges are recalibrated on lightphyx renders.
    """
    return _env_path("BLENDER_BIN", "blender")


def dataset_root() -> Path:
    """Root of a sanitized dataset (parent of `tasks/` and `gt/`)."""
    return _env_path("EVO_DATASET_ROOT", DEFAULT_DATASET_ROOT)


def data_root() -> Path:
    """Root containing per-task input parts (sanitized layout).

    Resolution order: EVO_DATA_ROOT → ${EVO_DATASET_ROOT}/tasks → ./dataset/artvip_full/tasks
    """
    if "EVO_DATA_ROOT" in os.environ:
        return _env_path("EVO_DATA_ROOT", "")
    return dataset_root() / "tasks"


def gt_root() -> Path:
    """Root containing ground-truth URDFs (sanitized layout).

    Resolution order: EVO_GT_ROOT → ${EVO_DATASET_ROOT}/gt → ./dataset/artvip_full/gt
    """
    if "EVO_GT_ROOT" in os.environ:
        return _env_path("EVO_GT_ROOT", "")
    return dataset_root() / "gt"


def raw_data_root() -> Path | None:
    """Root of raw InfiniPart data (for loader.py only).

    Returns None if EVO_RAW_DATA_ROOT is unset — callers should fail loud
    rather than silently scan the wrong directory.
    """
    raw = os.environ.get("EVO_RAW_DATA_ROOT")
    if not raw:
        return None
    return Path(os.path.expandvars(os.path.expanduser(raw)))


# Convenience module-level constants for callers that just want a value at
# import time. These are snapshots — re-export the functions above if you
# need late-binding behavior (e.g. tests that mutate os.environ).
BLENDER_BIN = blender_bin()
DATASET_ROOT = dataset_root()
DATA_ROOT = data_root()
GT_ROOT = gt_root()
