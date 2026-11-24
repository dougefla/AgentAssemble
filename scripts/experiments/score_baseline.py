#!/usr/bin/env python
"""Score a baseline on the test split of the evaluation subset with the same oracle and fitness as the main experiment.

Reads <pred-root>/<task>/<inst>/<urdf-name> (one prediction per task), configures the scored checks / joint dims
from the dataset variant exactly like evo/engine/loop.py, runs evo.oracle.check_urdf.run_oracle against gt/, and
reports mean session fitness (evo.engine.paired_gate.session_fitness) and pass rate over ALL test tasks; a task
with no URDF scores 0.

The subset folder is read from EVO_DATASET_ROOT.

usage: python scripts/experiments/score_baseline.py --pred-root DIR --out FILE.json [--urdf-name furniture.urdf]
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
DATA = Path(os.environ["EVO_DATASET_ROOT"])

def _configure():
    from evo.oracle.check_urdf import JointReport, configure_scored_checks
    v = json.loads((DATA / "dataset_variant.json").read_text())
    if v.get("scored_joint_dims"):
        JointReport.configure_dims(v["scored_joint_dims"])
    if v.get("scored_checks"):
        configure_scored_checks(v["scored_checks"])

def score_one(job):
    task, pred_dir = job
    _configure()
    from evo.oracle.check_urdf import run_oracle
    from evo.engine.paired_gate import session_fitness
    try:
        r = run_oracle(DATA / "gt" / task / "0", Path(pred_dir)).to_dict()
        err = None
    except Exception as e:
        r, err = None, f"{type(e).__name__}: {e}"
    return {"task_id": task, "fitness": session_fitness(r), "passed": bool(r and r.get("passed")), "error": err,
            "failed_checks": [k for k, c in ((r or {}).get("checks") or {}).items() if c.get("scored", True) and not c.get("passed")]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--urdf-name", default="furniture.urdf")
    a = ap.parse_args()
    test = [r["task_id"] for r in json.loads((DATA / "splits.json").read_text()) if r["split"] == "test"]
    jobs, res = [], []
    for t in test:
        if (a.pred_root / t / "0" / a.urdf_name).is_file():
            jobs.append((t, str(a.pred_root / t / "0")))
        else:
            res.append({"task_id": t, "fitness": 0.0, "passed": False, "error": "no URDF", "failed_checks": []})
    with ProcessPoolExecutor(4) as ex:
        res += list(ex.map(score_one, jobs))
    res.sort(key=lambda r: r["task_id"])
    n = len(res)
    summ = {"pred_root": str(a.pred_root), "tasks": n, "with_urdf": len(jobs),
            "fitness": round(sum(r["fitness"] for r in res) / n, 4), "pass_rate": round(sum(r["passed"] for r in res) / n, 4)}
    a.out.write_text(json.dumps({"summary": summ, "per_task": res}, indent=1))
    print(json.dumps(summ))
main()
