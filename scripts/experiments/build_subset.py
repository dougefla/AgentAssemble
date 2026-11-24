"""Draw the 120-object evaluation subset from the screening of the seed Skill (213 eligible objects x 3 sessions).

Per split the subset takes 22 never_low, 10 never_high and 8 always objects (40 per split) and no sometimes-pass
object. never = 0/3 seed passes (low: mean seed fitness < 0.35, high: >= 0.35), always = 3/3 passes.
The result of the paper is stored in benchmark/splits_main.json and benchmark/seed_screening.json.

usage: python scripts/experiments/build_subset.py <screen_run_dir> <full_view_dir> <out_dir> [per_split=40] [--write]
  <screen_run_dir>  run folder of the seed Skill on the full view (--no-generator --executor-reps 3)
  <full_view_dir>   the rotated-and-scaled view of all ArtiAssembly objects (contains tiers.json)
  <out_dir>         where the subset view is written (use it as EVO_DATASET_ROOT)"""
import json, random, subprocess, sys, statistics as st, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from evo.engine.paired_gate import read_task_sessions, session_fitness, session_passed
args = [a for a in sys.argv[1:] if a != "--write"]; write = "--write" in sys.argv
run, SRC, out = Path(args[0]), args[1].rstrip("/"), args[2].rstrip("/")
N = int(args[3]) if len(args) > 3 else 40
R = str(ROOT / "benchmark") + "/"
tiers = json.load(open(SRC + "/tiers.json"))["tasks"]
scr = {}
for o in sorted(run.glob("round_0/outputs/*/0")):
    ss = read_task_sessions(o)
    scr[o.parent.name] = {"n": len(ss), "fit": [session_fitness(s) for s in ss], "pass": [int(session_passed(s)) for s in ss]}
json.dump(scr, open(R + "seed_screening.json", "w"), indent=1)
pools = {"never_low": [], "always": [], "sometimes": [], "never_high": []}
for t, s in sorted(scr.items()):
    if s["n"] < 3: continue
    p = sum(s["pass"]); m = st.mean(s["fit"])
    if p == s["n"]: pools["always"].append(t)
    elif p == 0 and m < 0.35: pools["never_low"].append(t)
    elif p == 0: pools["never_high"].append(t)
    else: pools["sometimes"].append(t)
print("screened", len(scr), "sessions", sum(s["n"] for s in scr.values()), "mean fitness %.3f" % st.mean(st.mean(s["fit"]) for s in scr.values()),
      "session pass %.1f%%" % (100 * st.mean(st.mean(s["pass"]) for s in scr.values())))
print("pools", {k: len(v) for k, v in pools.items()})
quota = {"never_low": round(N * 0.55), "never_high": round(N * 0.25), "always": N - round(N * 0.55) - round(N * 0.25)}
print("quota per split", quota, "max feasible per split", {k: len(pools[k]) // 3 for k in quota})
if not write: sys.exit()
rng = random.Random(44)
for k in quota: rng.shuffle(pools[k])
split, kind = {}, {}
for i, sp in enumerate(("train", "val", "test")):
    for k, q in quota.items():
        got = pools[k][i * q:(i + 1) * q]; assert len(got) == q, (sp, k, len(got))
        for t in got: split[t] = sp; kind[t] = k
tasks = sorted(split); name = Path(out).name
open(R + "tasks_main.txt", "w").write("\n".join(tasks) + "\n")
json.dump({"split": split, "kind": kind}, open(R + "splits_main.kinds.json", "w"), indent=1)
subprocess.run([sys.executable, str(ROOT / "scripts/dataset/subset_view.py"), "--source", SRC,
                "--tasks-file", R + "tasks_main.txt", "--name", name, "--out", out, "--force", "--note",
                f"main experiment {N}/{N}/{N}, pose-scored; per split {quota} by 3 seed-skill sessions per task on the 213 eligible tasks "
                "(never = 0/3 passes, low/high by mean fitness 0.35; always = 3/3); no sometimes-pass tasks; splits reassigned, seed 44"], check=True)
for fn in ("splits.json", "index.json"):
    d = json.load(open(f"{out}/{fn}")); n = 0
    for r in d:
        if r.get("task_id") in split: r["split"] = split[r["task_id"]]; n += 1
    json.dump(d, open(f"{out}/{fn}", "w"), indent=1); print(fn, n)
v = json.load(open(out + "/dataset_variant.json")); print("scored_checks has link_scale:", "link_scale" in v["scored_checks"])
for sp in ("train", "val", "test"):
    ts = [t for t in tasks if split[t] == sp]
    print(sp, len(ts), dict(collections.Counter(kind[t] for t in ts)), "seed fitness %.3f" % st.mean(st.mean(scr[t]["fit"]) for t in ts),
          "within sd %.3f" % st.mean(st.pstdev(scr[t]["fit"]) for t in ts), "pass %.1f%%" % (100 * st.mean(st.mean(scr[t]["pass"]) for t in ts)),
          dict(collections.Counter(tiers[t]["joint_bucket"] for t in ts)))
print("DATASET", out)
