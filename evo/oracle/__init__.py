"""Pluggable oracle layer.

An *oracle* scores a predicted URDF against ground truth and returns an
``OracleResult`` — a 1-bit verdict (``passed``) plus per-check detail
(``checks``). ``OracleResult`` is the common contract every oracle returns,
so the loop and offline tooling treat all oracles uniformly.

Two oracles ship today:

* ``script``   — :func:`evo.oracle.check_urdf.run_oracle`. Hand-coded URDF
  text/geometry comparison (free, fast, deterministic). The default entry.
* ``subagent`` — :func:`evo.oracle.subagent.run_subagent_oracle_async`. A
  visual LLM judge over rendered rest + motion frames.

Adding a new oracle method
--------------------------
1. Write ``evo/oracle/<name>.py`` exposing a callable that returns an
   ``OracleResult``.
2. Register it in :data:`ORACLES` below with a *lazy* loader (so selecting
   the ``script`` oracle never imports heavy deps like the Agent SDK or
   Blender that another oracle needs).
3. Wire it into ``evo.engine.loop._run_oracle_dispatch`` — that function is
   the single switchboard that adapts each oracle's call signature (some are
   synchronous, some async with extra render options).

The registry keeps oracle discovery in one place; the dispatcher keeps the
per-oracle calling convention explicit. New scoring methods (the planned
non-``check_urdf`` oracles) slot in at both points without touching callers.
"""
from __future__ import annotations

from typing import Callable

from evo.oracle.check_urdf import (
    CheckResult,
    OracleResult,
    Thresholds,
    find_urdf,
    run_oracle,
)


def _load_script() -> Callable:
    from evo.oracle.check_urdf import run_oracle as _f
    return _f


def _load_subagent() -> Callable:
    from evo.oracle.subagent import run_subagent_oracle_async as _f
    return _f


# name -> lazy loader returning that oracle's entrypoint callable.
ORACLES: dict[str, Callable[[], Callable]] = {
    "script": _load_script,
    "objective": _load_script,
    "subagent": _load_subagent,
}


def get_oracle(name: str) -> Callable:
    """Return the oracle entrypoint registered under ``name`` (imported lazily).

    Raises ``ValueError`` for an unknown mode. The returned callables differ
    in signature (script is sync, subagent is async) — see
    ``evo.engine.loop._run_oracle_dispatch`` for how each is invoked.
    """
    try:
        loader = ORACLES[name]
    except KeyError:
        raise ValueError(
            f"unknown oracle mode {name!r}; known: {sorted(ORACLES)}"
        ) from None
    return loader()


__all__ = [
    "OracleResult",
    "CheckResult",
    "Thresholds",
    "find_urdf",
    "run_oracle",
    "ORACLES",
    "get_oracle",
]
