#!/usr/bin/env python3
"""Cut a task subset out of an artic300 view into a new dataset root.

Used for the 2026-11-04 dataset filter: tasks that fail >= 2 of 3 seeds on the
simple view are dropped, the survivors are re-run on the rot view, the survivors
of that become the final dataset, materialised as its own rotscale view.

  tasks/<t>, gt/<t>, gt_renders/<t>   copied (hard links when possible) for the listed tasks
  splits.json, index.json             filtered to the listed tasks (partition inherited)
  mapping.json                        task-keyed dicts filtered, other keys kept
  dataset_variant.json                dataset = <name>, selection = provenance, n_instances updated
  README.md                           provenance note + per-split table

Usage:
  python scripts/dataset/subset_view.py --source ~/Dataset/artic300_rotscale \
      --tasks-file keep.txt --name artic180_rotscale --out ~/Dataset/artic180_rotscale \
      --note "tasks passing >= 2 of 3 seeds on artic300_simple and on artic300_rot"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tasks-file", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="", help="one-line provenance for dataset_variant.json / README")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    args = ap.parse_args()
    src = args.source.expanduser().resolve()
    out = args.out.expanduser()
    keep = [l.split()[0] for l in args.tasks_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    keep_set = set(keep)
    rows = json.loads((src / "splits.json").read_text())
    have = {r["task_id"] for r in rows}
    missing = sorted(keep_set - have)
    if missing:
        raise SystemExit(f"{len(missing)} listed tasks not in {src}: {missing[:5]}")
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists (use --force to replace)")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    n_files = 0
    for sub in ("tasks", "gt", "gt_renders"):
        if not (src / sub).is_dir():
            continue
        for t in keep:
            s = src / sub / t
            if not s.is_dir():
                continue
            shutil.copytree(s, out / sub / t, copy_function=_link_or_copy)
            n_files += sum(1 for _ in (out / sub / t).rglob("*") if _.is_file())

    new_rows = [r for r in rows if r["task_id"] in keep_set]
    (out / "splits.json").write_text(json.dumps(new_rows, indent=1) + "\n")
    if (src / "index.json").is_file():
        idx = json.loads((src / "index.json").read_text())
        if isinstance(idx, list):
            idx = [r for r in idx if r.get("task_id") in keep_set]
        (out / "index.json").write_text(json.dumps(idx, indent=1) + "\n")
    if (src / "mapping.json").is_file():
        m = json.loads((src / "mapping.json").read_text())
        if isinstance(m, dict):
            for k, v in list(m.items()):
                if isinstance(v, dict) and v and all(str(kk).startswith("task_") for kk in v):
                    m[k] = {kk: vv for kk, vv in v.items() if kk in keep_set}
        (out / "mapping.json").write_text(json.dumps(m, indent=1) + "\n")

    today = dt.date.today().isoformat()
    variant = json.loads((src / "dataset_variant.json").read_text())
    variant["parent_dataset"] = variant.get("dataset")
    variant["dataset"] = args.name
    variant["n_instances"] = len(new_rows)
    variant["selection"] = {
        "date": today, "source_view": str(src), "tasks_file": str(args.tasks_file.resolve()),
        "n_tasks": len(new_rows), "criterion": args.note,
    }
    (out / "dataset_variant.json").write_text(json.dumps(variant, indent=1) + "\n")

    by = {s: [r for r in new_rows if r["split"] == s] for s in sorted({r["split"] for r in new_rows})}
    lines = [f"| split | n | easy | medium | hard | mean joints |", "|---|---|---|---|---|---|"]
    for s, rs in by.items():
        d = Counter(r["difficulty"] for r in rs)
        lines.append(f"| {s} | {len(rs)} | {d['easy']} | {d['medium']} | {d['hard']} | {sum(r['n_joints'] for r in rs) / max(1, len(rs)):.2f} |")
    head = (f"# {args.name} — {len(new_rows)}-task subset of {src.name}\n\n"
            f"Built {today} by `scripts/dataset/subset_view.py` from `{src}` with the task list `{args.tasks_file.resolve()}`.\n"
            f"Selection: {args.note or '(no note)'}\n\n"
            f"Task files, GT and GT renders are the parent view's files (hard links where possible); the train/test "
            f"partition is inherited from the parent's splits.json.\n\n" + "\n".join(lines) + "\n\n---\n\n"
            f"Parent README follows.\n\n")
    parent_readme = (src / "README.md").read_text() if (src / "README.md").is_file() else ""
    (out / "README.md").write_text(head + parent_readme)
    print(f"{out}: {args.name}, {len(new_rows)} tasks ({', '.join(f'{s} {len(rs)}' for s, rs in by.items())}), {n_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
