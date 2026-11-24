#!/usr/bin/env python3
"""Write ``gt/<task>/<inst>/part_equivalence.json`` for every instance of a dataset.

The classes name the interchangeable (congruent) parts of each object, derived
from the GT geometry by :func:`evo.oracle.part_equivalence.equivalence_classes`.
The oracle reads the file when present and scores the closest assignment among
the permutations inside each class, so swapping two identical drawers is never
penalised. Re-run after changing the criteria; the file is small and
human-editable.

Usage:
    .venv/bin/python scripts/dataset/write_part_equivalence.py ~/Dataset/artvip_posefill [...]
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evo.oracle.part_equivalence import (  # noqa: E402
    DEFAULT_MAX_DISTANCE, DEFAULT_MAX_SCALE_DIFF, write_part_equivalence,
)


def write_dataset(root: Path, *, max_distance: float, max_scale_diff: float) -> dict:
    sizes: collections.Counter = collections.Counter()
    n_inst = 0
    with_classes = 0
    for gt_dir in sorted(p.parent for p in (root / "gt").glob("task_*/*/scene.urdf")):
        n_inst += 1
        classes = write_part_equivalence(gt_dir, max_distance=max_distance, max_scale_diff=max_scale_diff, with_transforms=True)
        if classes:
            with_classes += 1
            for c in classes:
                sizes[len(c)] += 1
    return {"dataset": root.name, "instances": n_inst, "with_classes": with_classes, "class_sizes": dict(sorted(sizes.items()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    ap.add_argument("--max-scale-diff", type=float, default=DEFAULT_MAX_SCALE_DIFF)
    args = ap.parse_args(argv)
    for root in args.roots:
        root = root.expanduser().resolve()
        if not (root / "gt").is_dir():
            print(f"skip {root}: no gt/")
            continue
        print(write_dataset(root, max_distance=args.max_distance, max_scale_diff=args.max_scale_diff))
    return 0


if __name__ == "__main__":
    sys.exit(main())
