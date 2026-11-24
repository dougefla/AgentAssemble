"""Test-split comparison (40 tasks x 3 sessions) of skill versions per executor model: mean fitness, pass rate,
paired difference vs that model seed skill.

usage: python scripts/experiments/compare_test.py <runs.json> <runs dir> [out.json]
<runs.json> maps {executor model: {skill name: run folder name}} and must contain a "seed" entry per model."""
import json, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from evo.engine.paired_gate import read_task_sessions, session_fitness, session_passed, paired_stats
R = Path(sys.argv[2])
RUNS = json.loads(Path(sys.argv[1]).read_text())
out = {}
for model, vers in RUNS.items():
    per = {}
    for v, run in vers.items():
        fit, pas = {}, {}
        for d in sorted((R / run / "round_0" / "outputs").glob("task_*/*")):
            ss = read_task_sessions(d)
            fit[d.parent.name] = [session_fitness(s) for s in ss]; pas[d.parent.name] = [float(session_passed(s)) for s in ss]
        per[v] = (fit, pas)
    out[model] = {}
    for v, (fit, pas) in per.items():
        n = sum(len(x) for x in fit.values())
        row = {"run": vers[v], "tasks": len(fit), "sessions": n,
               "fitness": round(sum(sum(x) / len(x) for x in fit.values()) / len(fit), 4),
               "pass_rate": round(sum(sum(x) / len(x) for x in pas.values()) / len(pas), 4)}
        if v != "seed" and "seed" in per:
            s = paired_stats(fit, per["seed"][0]); row["vs_seed"] = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in s.items()}
        out[model][v] = row
        print(model, v, row)
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
