#!/usr/bin/env python3
"""evo — the user-facing entry point for evo-assemble.

Three actions:

    evo train     # co-evolve the /assemble skill (the loop)
    evo infer     # run the skill on task(s) to produce URDFs (no evolution)
    evo test      # run the test suite (pytest)

Run bare (`evo`) for an interactive, guided menu that prompts for the key
options, shows the exact command it will run, and confirms before launching.
Pass a subcommand + flags for non-interactive / CI use, e.g.:

    evo train --split train,validation --executor-backend responses \\
              --generator-backend responses --oracle objective \\
              --parallel 2 --max-rounds 3 --headless --yes
    evo infer --task task_01/0 --oracle script
    evo test -k canonical

`train` and `infer` delegate to `python -m evo.engine.loop`; `infer` just adds
`--no-generator --max-rounds 1`. Anything after a literal `--` is forwarded to
the loop verbatim, so every loop flag remains reachable.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from evo.engine.envfile import load_env_chain  # noqa: E402
from evo.engine.model_defaults import (  # noqa: E402
    DEFAULT_AGENT_EFFORT,
    DEFAULT_AGENT_MODEL,
)
from evo.oracle.subagent import (  # noqa: E402
    DEFAULT_ORACLE_BACKEND,
    DEFAULT_ORACLE_MODEL,
)

# Defaults offered by the interactive prompts; the loop applies the same
# fallbacks when the corresponding flag is omitted.
DEFAULT_ORACLE_JUDGE_MODEL = DEFAULT_ORACLE_MODEL[DEFAULT_ORACLE_BACKEND]
_EFFORTS = ["low", "medium", "high", "max"]

# ---------------------------------------------------------------------------
# Prompt helpers (questionary if available, else stdlib input())
# ---------------------------------------------------------------------------

try:
    import questionary  # type: ignore
    _HAVE_Q = True
except Exception:  # noqa: BLE001
    _HAVE_Q = False


def _select(msg: str, choices: list[str], default: str | None = None) -> str:
    default = default if default in choices else choices[0]
    if _HAVE_Q:
        return questionary.select(msg, choices=choices, default=default).ask() or default
    while True:
        print(f"\n{msg}")
        for i, c in enumerate(choices, 1):
            print(f"  {i}) {c}" + ("  [default]" if c == default else ""))
        raw = input("> ").strip()
        if not raw:
            return default
        if raw in choices:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]


def _text(msg: str, default: str = "") -> str:
    if _HAVE_Q:
        return (questionary.text(msg, default=default).ask() or default).strip()
    raw = input(f"{msg} [{default}]: ").strip()
    return raw or default


def _confirm(msg: str, default: bool = True) -> bool:
    if _HAVE_Q:
        return bool(questionary.confirm(msg, default=default).ask())
    raw = input(f"{msg} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def _loop_base() -> list[str]:
    return [sys.executable, "-m", "evo.engine.loop"]


def _add(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd += [flag, value]


def build_train_cmd(opts: dict) -> list[str]:
    cmd = _loop_base()
    _add(cmd, "--split", opts.get("split"))
    _add(cmd, "--skill-dir", opts.get("skill_dir"))
    _add(cmd, "--executor-backend", opts.get("executor_backend"))
    _add(cmd, "--generator-backend", opts.get("generator_backend"))
    _add(cmd, "--executor-model", opts.get("executor_model"))
    _add(cmd, "--generator-model", opts.get("generator_model"))
    _add(cmd, "--auth", opts.get("auth"))
    _add(cmd, "--generator-auth", opts.get("generator_auth"))
    _add(cmd, "--oracle-mode", opts.get("oracle"))
    _add(cmd, "--oracle-model", opts.get("oracle_model"))
    _add(cmd, "--oracle-effort", opts.get("oracle_effort"))
    _add(cmd, "--oracle-votes", opts.get("oracle_votes"))
    _add(cmd, "--regression-tol", opts.get("regression_tol"))
    _add(cmd, "--executor-effort", opts.get("executor_effort"))
    _add(cmd, "--generator-effort", opts.get("generator_effort"))
    _add(cmd, "--max-rounds", opts.get("max_rounds"))
    _add(cmd, "--batch-size", opts.get("batch_size"))
    _add(cmd, "--parallel", opts.get("parallel"))
    if opts.get("headless"):
        cmd.append("--headless")
    if opts.get("verbose"):
        cmd.append("-v")
    cmd += opts.get("passthrough", [])
    return cmd


def build_infer_cmd(opts: dict) -> list[str]:
    cmd = _loop_base()
    # inference = one round, executors only (no skill edits)
    cmd += ["--no-generator", "--max-rounds", "1"]
    task = (opts.get("task") or "").strip()
    if task:
        cmd += ["--include", task]
    _add(cmd, "--skill-dir", opts.get("skill_dir"))
    _add(cmd, "--executor-backend", opts.get("executor_backend"))
    _add(cmd, "--executor-model", opts.get("executor_model"))
    _add(cmd, "--auth", opts.get("auth"))
    _add(cmd, "--oracle-mode", opts.get("oracle"))
    _add(cmd, "--oracle-model", opts.get("oracle_model"))
    _add(cmd, "--parallel", opts.get("parallel"))
    if opts.get("headless"):
        cmd.append("--headless")
    if opts.get("verbose"):
        cmd.append("-v")
    cmd += opts.get("passthrough", [])
    return cmd


def build_test_cmd(opts: dict) -> list[str]:
    cmd = [sys.executable, "-m", "pytest"]
    if opts.get("k"):
        cmd += ["-k", opts["k"]]
    cmd += opts.get("passthrough", [])
    return cmd


# ---------------------------------------------------------------------------
# Interactive flows
# ---------------------------------------------------------------------------

def interactive_train() -> dict:
    print("\n— Train: co-evolve the /assemble skill —")
    o: dict = {}
    o["split"] = _text("Dataset split(s) (comma-sep)", "train,validation")
    o["executor_backend"] = _select("Executor backend", ["responses", "claude", "codex"], "responses")
    o["generator_backend"] = _select("Generator backend", ["responses", "claude", "codex"], "responses")
    o["executor_model"] = _text("Executor model", DEFAULT_AGENT_MODEL)
    o["generator_model"] = _text("Generator model", DEFAULT_AGENT_MODEL)
    o["auth"] = _select("API auth", ["proxy", "local"], "proxy")
    o["generator_auth"] = _select("Generator auth", ["local", "proxy"], o["auth"])
    o["oracle"] = _select("Oracle", ["subagent", "objective", "script"], "subagent")
    if o["oracle"] == "subagent":
        o["oracle_model"] = _text("Oracle judge model", DEFAULT_ORACLE_JUDGE_MODEL)
        o["oracle_effort"] = _select("Oracle effort", _EFFORTS, DEFAULT_AGENT_EFFORT)
    o["max_rounds"] = _text("Max rounds", "3")
    o["batch_size"] = _text("Batch size (0 = all filtered tasks)", "0")
    o["executor_effort"] = _select("Executor effort", _EFFORTS, DEFAULT_AGENT_EFFORT)
    o["generator_effort"] = _select("Generator effort", _EFFORTS, DEFAULT_AGENT_EFFORT)
    o["parallel"] = _text("Parallel sim workers", "2")
    o["headless"] = _confirm("Headless sim worker (compatibility flag)?", True)
    o["verbose"] = _confirm("Verbose logs?", True)
    return o


def interactive_infer() -> dict:
    print("\n— Infer: run the skill on task(s) to produce URDFs (no evolution) —")
    o: dict = {}
    o["task"] = _text("Task(s) to run (e.g. task_01/0  or  task_01/0,task_03/2)", "task_01/0")
    o["executor_backend"] = _select("Executor backend", ["responses", "claude", "codex"], "responses")
    o["executor_model"] = _text("Executor model", DEFAULT_AGENT_MODEL)
    o["auth"] = _select("Auth", ["proxy", "local"], "proxy")
    o["oracle"] = _select("Oracle (always scored)", ["subagent", "objective", "script"], "subagent")
    if o["oracle"] == "subagent":
        o["oracle_model"] = _text("Oracle judge model", DEFAULT_ORACLE_JUDGE_MODEL)
    o["parallel"] = _text("Parallel sim workers", "1")
    o["headless"] = _confirm("Headless sim worker (compatibility flag)?", True)
    o["verbose"] = _confirm("Verbose logs?", True)
    return o


def interactive_test() -> dict:
    print("\n— Test: run the pytest suite —")
    return {"k": _text("Filter (-k expression, blank = all)", "")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    load_env_chain()  # <repo>/.env + ~/.env so subprocesses inherit them
    raw = list(sys.argv[1:] if argv is None else argv)
    raw, passthrough = _split_passthrough(raw)

    p = argparse.ArgumentParser(
        prog="evo",
        description="evo-assemble — train / infer / test the /assemble skill.",
    )
    sub = p.add_subparsers(dest="action")
    common_loop = []
    for name in ("train", "infer"):
        sp = sub.add_parser(name, help=f"{name} (delegates to the evo loop)")
        sp.add_argument("--skill-dir")
        sp.add_argument("--executor-backend", choices=("responses", "claude", "codex"))
        sp.add_argument("--executor-model")
        sp.add_argument("--auth", choices=("local", "proxy"))
        sp.add_argument("--oracle", dest="oracle", choices=("objective", "script", "subagent"))
        sp.add_argument("--parallel")
        sp.add_argument("--headless", action="store_true")
        sp.add_argument("-v", "--verbose", action="store_true")
        sp.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
        sp.add_argument("--print-only", action="store_true", help="print the command and exit")
        if name == "train":
            sp.add_argument("--split")
            sp.add_argument("--generator-backend", choices=("responses", "claude", "codex"))
            sp.add_argument("--generator-model")
            sp.add_argument("--generator-auth", choices=("local", "proxy"))
            sp.add_argument("--oracle-effort", choices=("low", "medium", "high", "max"))
            sp.add_argument("--oracle-votes")
            sp.add_argument("--regression-tol")
            sp.add_argument("--executor-effort", choices=("low", "medium", "high", "max"))
            sp.add_argument("--generator-effort", choices=("low", "medium", "high", "max"))
            sp.add_argument("--max-rounds")
            sp.add_argument("--batch-size")
        else:
            sp.add_argument("--task")
    spt = sub.add_parser("test", help="run the pytest suite")
    spt.add_argument("-k", help="pytest -k filter")
    spt.add_argument("-y", "--yes", action="store_true")
    spt.add_argument("--print-only", action="store_true")

    args = p.parse_args(raw)
    action = args.action
    interactive = action is None

    if interactive:
        action = _select("What do you want to do?", ["train", "infer", "test"], "train")
        opts = {"train": interactive_train, "infer": interactive_infer, "test": interactive_test}[action]()
        opts["passthrough"] = passthrough
        skip_confirm = False
        print_only = False
    else:
        # argparse already maps --foo-bar → foo_bar and stores strings/bools/None,
        # which is exactly what the builders expect.
        opts = {k: v for k, v in vars(args).items() if k not in ("action", "yes", "print_only")}
        opts["passthrough"] = passthrough
        skip_confirm = getattr(args, "yes", False)
        print_only = getattr(args, "print_only", False)

    builder = {"train": build_train_cmd, "infer": build_infer_cmd, "test": build_test_cmd}[action]
    cmd = builder(opts)

    if action == "test":
        import importlib.util
        if importlib.util.find_spec("pytest") is None:
            print(
                "pytest is not installed in this Python environment "
                f"({sys.executable}).\n"
                'Install the dev extras first:  pip install -e ".[dev]"'
            )
            return 1

    print("\nCommand:\n  " + " ".join(cmd) + "\n")
    if print_only:
        return 0
    if not skip_confirm and not _confirm("Run this now?", True):
        print("Aborted.")
        return 1

    try:
        return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
