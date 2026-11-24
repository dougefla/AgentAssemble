#!/usr/bin/env python3
"""Tag every artic300 task instead of filtering it: tiers.json.

Inputs are re-scored seed-Skill rollouts (scripts/rescore_oracle.py output) on
the three views plus the per-link thumbnail visibility. Per task:

  exclude            reasons the task itself is defective (Isaac core validation
                     failed, known wrong object scale); null for a usable task
  hidden_links       links with no visible pixels in the reference image: their
                     size / orientation cannot be read off the image
  simple / rot / rotscale   seed Skill: sessions, passes, mean continuous fitness
  base_tier          A: seed solves the simple view in >= 2 of 3 sessions
                     B: 1 of 3      C: 0 of 3
  group (rotscale)   sentinel: seed fitness >= 0.9 in the rotscale view
                     frontier: 0.15 <= fitness < 0.9
                     far:      fitness < 0.15, or no rotscale sessions and base tier C

Nothing here depends on an evolved Skill: the tags are fixed before evolution.

    .venv/bin/python scripts/dataset/build_tiers.py --data ~/Dataset/artic300_rotscale \
        --simple report/oracle3-rescore/simple_seed.json --rot report/oracle3-rescore/rot_seed.json \
        --rotscale report/oracle3-rescore/rotscale_R0_230x3.json [more rotscale files, round_0 only]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evo.engine.paired_gate import session_fitness  # noqa: E402

BAD_SCALE = {"t2_057", "t4_089"}     # README: ~10x too small


def _sessions(path: Path, *, only_round0: bool = False) -> dict[str, list[dict | None]]:
    out: dict[str, list] = collections.defaultdict(list)
    for r in json.load(open(path))["rows"]:
        if only_round0 and r["round"] != "round_0":
            continue
        out[r["task"]].append(r["new"] if r["new"] and "passed" in r["new"] else None)
    return out


def _summ(sessions: list[dict | None]) -> dict | None:
    if not sessions:
        return None
    fit = [session_fitness(s) for s in sessions]
    return {"sessions": len(sessions), "passes": sum(1 for s in sessions if s and s["passed"]),
            "fitness": round(sum(fit) / len(fit), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--simple", type=Path, required=True)
    ap.add_argument("--rot", type=Path)
    ap.add_argument("--rotscale", type=Path, nargs="*", default=[])
    args = ap.parse_args()
    data = args.data.expanduser().resolve()
    index = {r["task_id"]: r for r in json.load(open(data / "index.json"))}
    splits = {r["task_id"]: r for r in json.load(open(data / "splits.json"))}
    simple = _sessions(args.simple)
    rot = _sessions(args.rot) if args.rot else {}
    rs: dict[str, list] = collections.defaultdict(list)
    for f in args.rotscale:
        for t, ss in _sessions(f, only_round0=True).items():
            rs[t].extend(ss)
    tiers = {}
    for t, row in sorted(index.items()):
        vis_p = data / "gt" / t / row.get("instance", "0") / "link_visibility.json"
        vis = json.load(open(vis_p)) if vis_p.is_file() else {}
        exclude = []
        if not row.get("isaac_core_pass", True):
            exclude.append("isaac_core_fail")
        if any(row.get("source_id", "").startswith(b) for b in BAD_SCALE):
            exclude.append("object_scale_wrong")
        s, r, q = _summ(simple.get(t, [])), _summ(rot.get(t, [])), _summ(rs.get(t, []))
        sp = s["passes"] if s else 0
        base = "A" if sp >= 2 else ("B" if sp == 1 else "C")
        if q is not None:
            group = "sentinel" if q["fitness"] >= 0.9 else ("frontier" if q["fitness"] >= 0.15 else "far")
        else:
            group = "far" if base == "C" else "unknown"
        nj = row["n_joints"]
        tiers[t] = {
            "source_id": row.get("source_id"), "split": splits.get(t, {}).get("split"),
            "difficulty": splits.get(t, {}).get("difficulty"), "n_joints": nj,
            "joint_bucket": "1" if nj <= 1 else ("2-3" if nj <= 3 else ("4-6" if nj <= 6 else "7+")),
            "n_loop_closure_usd_only": row.get("n_loop_closure_joints_usd_only", 0),
            "exclude": exclude or None,
            "hidden_links": vis.get("hidden_links", []), "small_links": vis.get("small_links", []),
            "simple": s, "rot": r, "rotscale": q, "base_tier": base, "group": group,
        }
    (data / "tiers.json").write_text(json.dumps({
        "about": "seed-Skill based tags, fixed before evolution; see scripts/dataset/build_tiers.py",
        "tasks": tiers}, indent=1))
    c = collections.Counter
    usable = {t: v for t, v in tiers.items() if not v["exclude"]}
    print(f"{data.name}: {len(tiers)} tasks, excluded {len(tiers) - len(usable)} "
          f"{dict(c(r for v in tiers.values() for r in (v['exclude'] or [])))}")
    print("base tier:", dict(sorted(c(v["base_tier"] for v in usable.values()).items())))
    print("group:", dict(sorted(c(v["group"] for v in usable.values()).items())))
    print("hidden-link tasks:", sum(1 for v in usable.values() if v["hidden_links"]))
    for split in sorted({v["split"] for v in usable.values() if v["split"]}):
        sub = [v for v in usable.values() if v["split"] == split]
        print(f"  {split}: n={len(sub)} tier {dict(sorted(c(v['base_tier'] for v in sub).items()))} "
              f"group {dict(sorted(c(v['group'] for v in sub).items()))} "
              f"hidden {sum(1 for v in sub if v['hidden_links'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
