"""Sim-worker entry point for Assembly SDK v1.

Load this file at the start of a session (the sim worker preloads it; a
skill may also re-exec it defensively — the operation is idempotent):

    exec(open(f"{ASSEMBLY_SDK_DIR}/runtime.py").read())

It binds an ``assembly`` object with explicit, typed method arguments. All
durable state lives in a state file under OUTPUT_DIR; the in-process mesh
cache survives re-execution. The SDK directory is frozen by the training
runtime and is not part of the evolvable skill.
"""
from __future__ import annotations

import sys
from pathlib import Path

if "ASSEMBLY_SDK_DIR" not in globals():
    raise RuntimeError(
        "ASSEMBLY_SDK_DIR is required; runtime.py must be loaded through the "
        "training runtime's injected immutable global"
    )
_SDK_DIR = Path(ASSEMBLY_SDK_DIR).resolve()
if not (_SDK_DIR / "core.py").is_file():
    raise RuntimeError(f"ASSEMBLY_SDK_DIR does not contain core.py: {_SDK_DIR}")
_PROJECT_ROOT = _SDK_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evo.assembly_sdk.host_runtime import get_assembly  # noqa: E402
from evo.assembly_sdk.core import AssemblyError, SDK_VERSION  # noqa: E402

assembly = get_assembly(globals())
# Sim calls share durable assembly state but an agent may defensively re-run
# this loader. Refresh the common read-only handle snapshot every time so a
# stale local cannot abort a training sample. This never mutates SDK state.
try:
    parts = assembly.inspect_parts()
except AssemblyError:
    parts = []
print(f"[Assembly SDK {SDK_VERSION}] ready; use assembly.<method>(explicit_args)")
