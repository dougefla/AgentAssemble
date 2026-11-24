#!/usr/bin/env python3
"""Batch evaluation: run the oracle on all predicted URDFs and report pass rate.

Expects directory layout:
    gt_root/
        BarChairFactory/0/scene.urdf
        MicrowaveSapienFactory/0/scene.urdf
        ...
    pred_root/
        BarChairFactory/0/furniture.urdf  (or any .urdf)
        MicrowaveSapienFactory/0/furniture.urdf
        ...

Usage:
    python batch_eval.py <gt_root> <pred_root> [--verbose] [--json] [--limit N]

Output: per-task verdicts + aggregate pass rate + failure breakdown.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from evo.oracle.check_urdf import OracleResult, Thresholds, find_urdf, run_oracle


@dataclass
class TaskResult:
    factory: str
    instance: str
    result: OracleResult | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.result is not None and self.result.passed

    @property
    def key(self) -> str:
        return f"{self.factory}/{self.instance}"


def discover_tasks(gt_root: Path, pred_root: Path) -> list[tuple[str, str, Path, Path]]:
    """Find (factory, instance) pairs that exist in both GT and pred roots."""
    tasks = []
    for factory_dir in sorted(gt_root.iterdir()):
        if not factory_dir.is_dir():
            continue
        factory_name = factory_dir.name
        for instance_dir in sorted(factory_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
            instance_name = instance_dir.name
            gt_dir = instance_dir
            pred_dir = pred_root / factory_name / instance_name

            # Check both have URDFs
            gt_has_urdf = any(gt_dir.glob("*.urdf"))
            pred_has_urdf = pred_dir.exists() and any(pred_dir.glob("*.urdf"))

            if gt_has_urdf and pred_has_urdf:
                tasks.append((factory_name, instance_name, gt_dir, pred_dir))

    return tasks


def run_batch(
    gt_root: Path,
    pred_root: Path,
    thresholds: Thresholds,
    verbose: bool = False,
    limit: int | None = None,
) -> list[TaskResult]:
    """Run oracle on all discovered task pairs."""
    tasks = discover_tasks(gt_root, pred_root)
    if limit:
        tasks = tasks[:limit]

    results: list[TaskResult] = []
    for i, (factory, instance, gt_dir, pred_dir) in enumerate(tasks):
        key = f"{factory}/{instance}"
        try:
            result = run_oracle(gt_dir, pred_dir, thresholds=thresholds, verbose=False)
            tr = TaskResult(factory, instance, result)
        except Exception as e:
            tr = TaskResult(factory, instance, None, error=str(e))

        results.append(tr)
        status = "PASS" if tr.passed else "FAIL"
        score = f"{tr.result.score:.2f}" if tr.result else "ERR"
        if verbose or not tr.passed:
            print(f"[{i+1}/{len(tasks)}] {status} {key} (score={score})")

    return results


def print_summary(results: list[TaskResult], as_json: bool = False):
    """Print aggregate statistics."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = passed / total * 100 if total > 0 else 0

    # Failure breakdown by check type
    check_failures: dict[str, int] = {}
    for r in results:
        if r.result and not r.result.passed:
            for c in r.result.checks:
                if not c.passed:
                    check_failures[c.name] = check_failures.get(c.name, 0) + 1

    # Failure breakdown by factory
    factory_stats: dict[str, dict[str, int]] = {}
    for r in results:
        f = r.factory
        if f not in factory_stats:
            factory_stats[f] = {"total": 0, "passed": 0}
        factory_stats[f]["total"] += 1
        if r.passed:
            factory_stats[f]["passed"] += 1

    if as_json:
        output = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 1),
            "check_failures": dict(sorted(check_failures.items(), key=lambda x: -x[1])),
            "per_factory": {
                k: {**v, "pass_rate": round(v["passed"] / v["total"] * 100, 1)}
                for k, v in sorted(factory_stats.items())
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total:     {total}")
        print(f"Passed:    {passed}")
        print(f"Failed:    {failed}")
        print(f"Pass rate: {pass_rate:.1f}%")

        if check_failures:
            print(f"\nFailure breakdown by check type:")
            for name, count in sorted(check_failures.items(), key=lambda x: -x[1]):
                print(f"  {name}: {count} failures ({count/failed*100:.0f}% of failures)")

        if factory_stats:
            print(f"\nPer-factory pass rate:")
            for name, stats in sorted(factory_stats.items()):
                pr = stats["passed"] / stats["total"] * 100
                bar = "█" * int(pr / 5) + "░" * (20 - int(pr / 5))
                print(f"  {name:40s} {bar} {pr:5.1f}% ({stats['passed']}/{stats['total']})")


def main():
    parser = argparse.ArgumentParser(
        description="Batch URDF comparison evaluation",
    )
    parser.add_argument("gt_root", type=Path, help="Root of ground-truth data")
    parser.add_argument("pred_root", type=Path, help="Root of predicted outputs")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--limit", "-n", type=int, help="Max tasks to evaluate")
    args = parser.parse_args()

    th = Thresholds()
    results = run_batch(args.gt_root, args.pred_root, th, verbose=args.verbose, limit=args.limit)

    if not results:
        print("No task pairs found. Check directory structure:")
        print(f"  GT:   {args.gt_root}/<Factory>/<Instance>/scene.urdf")
        print(f"  Pred: {args.pred_root}/<Factory>/<Instance>/*.urdf")
        sys.exit(2)

    print_summary(results, as_json=args.json)

    pass_rate = sum(1 for r in results if r.passed) / len(results) * 100
    sys.exit(0 if pass_rate >= 50 else 1)


if __name__ == "__main__":
    main()
