#!/usr/bin/env python3
"""Build ArtVIP train/val/test splits with difficulty + challenge tags.

Two outputs:
  1. dataset/artvip_sanitized/splits.json   — initial automatic split
  2. reports/artvip_splits.html                — review/edit page (loads splits.json,
                                              user edits in browser, exports back)

Initial split logic:
  - features per instance:  n_parts, n_links, n_joints, joint-type counts
  - factory_group:          coarse semantic bucket (container / appliance /
                            tool / dispenser / stationary / other)
  - difficulty:              easy (≤5) / medium (≤9) / hard (>9)
                            score = n_parts + 2 × n_movable + 3 × (type_variety - 1)
  - challenge tags:          multi-revolute / multi-prismatic / multi-joint /
                            mixed-types / continuous / many-parts / simple-binary
  - holdout set:            entire factories held out for cross-category
                            zero-shot evaluation. One per difficulty level,
                            picked from factories whose instances are all the
                            same difficulty and from distinct groups, so the
                            holdout covers easy×stationary, medium×tool,
                            hard×container.
  - split (60/20/20):       remaining instances stratified by (difficulty,
                            group) so each train/val/test bucket keeps the
                            same ratio. Seeded.

Re-running the script regenerates splits.json + the HTML page.  The HTML page
saves edits to localStorage; user clicks Export JSON to get the corrected
splits.json back.
"""
from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "artvip_sanitized"
SPLIT_OUT = DATA / "splits.json"
HTML_OUT = ROOT / "reports" / "artvip_splits.html"


# Coarse semantic grouping. Keep this short — user can override per-instance
# in the review page.
FACTORY_GROUP = {
    # containers (cabinets / drawers / boxes — the "fridge" archetype)
    "ikea_furniture": "container",
    "wardrobe": "container",
    "cabinet": "container",
    "shoe_cabinet": "container",
    "drawers": "container",
    "cardboard_box": "container",
    "basket": "container",
    "trash_can": "container",
    "toolbox": "container",
    "sliding_door": "container",
    "table": "container",
    # major appliances (typically box body + door / lid + interior parts)
    "dishwasher": "appliance",
    "washing_machine": "appliance",
    "refrigerator": "appliance",
    "microwave": "appliance",
    "oven": "appliance",
    "coffee_machine": "appliance",
    "rice_cooker": "appliance",
    "kettle": "appliance",
    "toaster": "appliance",
    "blender": "appliance",
    "table_fan": "appliance",
    "fan": "appliance",
    "toilet": "appliance",
    "faucet": "appliance",
    "laptop": "appliance",
    # hand tools — articulated grippers / cutters
    "scissors": "tool",
    "tongs": "tool",
    "pliers": "tool",
    "stapler": "tool",
    "tweezers": "tool",
    "knife": "tool",
    # dispensers (pump / lever)
    "pump_bottle": "dispenser",
    "soap_dispenser": "dispenser",
    "spray_bottle": "dispenser",
    # stationary articulated objects (folders / books)
    "folder": "stationary",
    "binder": "stationary",
    "book": "stationary",
}


# Factories held out entirely from train/val/test — they form the
# zero-shot generalization probe. One factory per difficulty, each from
# a distinct group, each pure in difficulty (all instances at the same
# level). User can override on the review page (move individual rows
# into/out of the holdout split).
HOLDOUT_FACTORIES = {
    "folder",          # easy × stationary (9 instances)
    "scissors",        # medium × tool (8 instances)
    "cardboard_box",   # hard × container (10 instances)
}


def _parse_axis(axis_str: str | None) -> tuple[float, float, float]:
    if not axis_str:
        return (1.0, 0.0, 0.0)
    parts = axis_str.split()
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _movable_joint_summaries(joints) -> list[dict]:
    """Lightweight per-joint extraction for dependency analysis."""
    out = []
    for j in joints:
        jtype = (j.get("type") or "").upper()
        if jtype not in ("REVOLUTE", "PRISMATIC", "CONTINUOUS"):
            continue
        p = j.find("parent"); c = j.find("child")
        axis = j.find("axis")
        limit = j.find("limit")
        out.append({
            "type": jtype,
            "parent": p.get("link") if p is not None else None,
            "child": c.get("link") if c is not None else None,
            "axis": _parse_axis(axis.get("xyz") if axis is not None else None),
            "lower": float(limit.get("lower")) if limit is not None and limit.get("lower") else 0.0,
            "upper": float(limit.get("upper")) if limit is not None and limit.get("upper") else 0.0,
        })
    return out


def detect_motion_dependency_tags(joints) -> list[str]:
    """Geometry-free heuristics for joint motion sequencing.

    coaxial-joints:
      ≥2 revolute joints share the same parent link AND have parallel
      axes (|cos(angle)| ≥ 0.95).  Typical: toilet seat + lid hinging on
      the same pivot, watch case lids, double-leaf doors.

    enclosed-motion:
      ≥1 revolute "door" (lower≤0 and upper>0 with range ≥ π/2) AND
      ≥1 other movable joint sharing the same parent link.  Typical:
      dishwasher (door + interior racks), washing-machine (lid +
      drum), oven (door + interior tray), microwave (door + turntable).
      The "door" hint comes from the limit range — a hinge with at
      least 90° travel, opening from a closed position.
    """
    import math

    movables = _movable_joint_summaries(joints)
    tags: list[str] = []

    # coaxial-joints
    revolutes_by_parent: dict[str, list[dict]] = defaultdict(list)
    for m in movables:
        if m["type"] == "REVOLUTE" and m["parent"]:
            revolutes_by_parent[m["parent"]].append(m)

    for parent, group in revolutes_by_parent.items():
        if len(group) < 2:
            continue
        # Any pair with parallel axes?
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i]["axis"], group[j]["axis"]
                na = math.sqrt(sum(x*x for x in a)) or 1
                nb = math.sqrt(sum(x*x for x in b)) or 1
                cos_ab = abs(sum(ax*bx for ax, bx in zip(a, b))) / (na * nb)
                if cos_ab >= 0.95:
                    tags.append("coaxial-joints")
                    break
            else:
                continue
            break
        if "coaxial-joints" in tags:
            break

    # enclosed-motion: a door-shaped revolute + at least one sibling joint
    # that's NOT another door.
    #   door  := REVOLUTE with range ∈ [80°, 200°]    (typical fold-open)
    #   other := PRISMATIC, CONTINUOUS, REVOLUTE with range < 80° (button),
    #            or REVOLUTE with range > 200° (drum / 360° spin)
    # Wardrobes with two parallel doors (both ≈90°) won't fire — both are
    # doors, no "other".  Washing-machines (door + drum), dishwashers
    # (door + prismatic trays), toolboxes (lid + interior trays) fire.
    DOOR_MIN, DOOR_MAX = math.pi / 2 - 0.1, math.pi * 200 / 180  # ~80°..200°
    def is_door(m):
        if m["type"] != "REVOLUTE":
            return False
        rng = m["upper"] - m["lower"]
        return DOOR_MIN <= rng <= DOOR_MAX

    by_parent: dict[str, list[dict]] = defaultdict(list)
    for m in movables:
        if m["parent"]:
            by_parent[m["parent"]].append(m)
    for parent, group in by_parent.items():
        if len(group) < 2:
            continue
        has_door = any(is_door(m) for m in group)
        has_non_door = any(not is_door(m) for m in group)
        if has_door and has_non_door:
            tags.append("enclosed-motion")
            break

    return tags


def features_for_instance(task_id: str, instance: str, factory: str) -> dict | None:
    """Read parts/ + GT URDF, return feature dict or None if missing."""
    parts_dir = DATA / "tasks" / task_id / instance / "parts"
    gt_urdf = DATA / "gt" / task_id / instance / "scene.urdf"
    if not gt_urdf.exists():
        return None
    parts = list(parts_dir.glob("*.obj"))
    try:
        tree = ET.parse(gt_urdf)
    except Exception:
        return None
    joints = tree.findall(".//joint")
    links = tree.findall(".//link")
    jtypes = [j.get("type", "?").upper() for j in joints]
    n_revolute = sum(1 for t in jtypes if t == "REVOLUTE")
    n_prismatic = sum(1 for t in jtypes if t == "PRISMATIC")
    n_continuous = sum(1 for t in jtypes if t == "CONTINUOUS")
    n_fixed = sum(1 for t in jtypes if t == "FIXED")
    n_movable = n_revolute + n_prismatic + n_continuous
    n_parts = len(parts)

    type_variety = sum(1 for c in (n_revolute, n_prismatic, n_continuous) if c > 0)
    score = n_parts + 2 * n_movable + 3 * max(0, type_variety - 1)
    if score <= 5:
        difficulty = "easy"
    elif score <= 9:
        difficulty = "medium"
    else:
        difficulty = "hard"

    tags: list[str] = []
    if n_movable >= 3:
        tags.append("multi-joint")
    if n_revolute >= 2:
        tags.append("multi-revolute")
    if n_prismatic >= 2:
        tags.append("multi-prismatic")
    if n_revolute > 0 and n_prismatic > 0:
        tags.append("mixed-types")
    if n_continuous > 0:
        tags.append("continuous")
    if n_parts >= 6:
        tags.append("many-parts")
    if n_parts == 2 and n_movable == 1:
        tags.append("simple-binary")
    # joint-dependency tags (sequencing / occlusion) — see detect_motion_dependency_tags
    tags.extend(detect_motion_dependency_tags(joints))

    group = FACTORY_GROUP.get(factory, "other")

    return {
        "task_id": task_id,
        "instance": instance,
        "factory": factory,
        "group": group,
        "n_parts": n_parts,
        "n_links": len(links),
        "n_joints": len(joints),
        "n_revolute": n_revolute,
        "n_prismatic": n_prismatic,
        "n_continuous": n_continuous,
        "n_fixed": n_fixed,
        "n_movable": n_movable,
        "complexity_score": score,
        "difficulty": difficulty,
        "challenges": tags,
    }


def _factory_majority_difficulty(rows: list[dict]) -> str:
    """Mode of `difficulty` over all instances of one factory.

    Ties broken by hard > medium > easy (lean toward harder bucket so a
    mixed-difficulty factory is treated conservatively in stratification).
    """
    c = Counter(r["difficulty"] for r in rows)
    if not c:
        return "easy"
    top_count = max(c.values())
    candidates = {d for d, n in c.items() if n == top_count}
    for d in ("hard", "medium", "easy"):
        if d in candidates:
            return d
    return "easy"


def stratified_split(
    rows: list[dict],
    train: float = 0.6,
    val: float = 0.2,
    seed: int = 42,
    by: str = "factory",
) -> None:
    """Mutates rows in place — sets row['split'] ∈ {train, val, test, holdout}.

    Step 1: any instance whose factory is in HOLDOUT_FACTORIES gets
    `split = "holdout"` and is removed from the stratified pool. The
    holdout factories are chosen for cross-category zero-shot evaluation
    — the agent never sees a `folder` / `scissors` / `cardboard_box`
    during training.

    Step 2: remaining instances are split. Two modes:

    `by="factory"` (paper policy, default):
        Every instance of a factory lands in one split (train, val, or
        test). Factories are bucketed by (majority_difficulty, group),
        sorted by instance count desc within each bucket, and assigned
        greedily to whichever split is currently most below its target.
        This blocks the leak where the executor sees factory F's
        instances during evolution and then is evaluated on F.

    `by="instance"` (legacy):
        Instances within each (difficulty, group) bucket are split
        60/20/20.  Same factory can appear in all three splits.
    """
    rng = random.Random(seed)

    # Reserve holdout factories first.
    holdout_rows = []
    pool: list[dict] = []
    for r in rows:
        if r["factory"] in HOLDOUT_FACTORIES:
            r["split"] = "holdout"
            holdout_rows.append(r)
        else:
            pool.append(r)

    if by == "instance":
        # Legacy: per-instance stratified split.
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in pool:
            buckets[(r["difficulty"], r["group"])].append(r)
        for bucket_rows in buckets.values():
            rng.shuffle(bucket_rows)
            n = len(bucket_rows)
            n_train = round(n * train)
            n_val = round(n * val)
            for i, r in enumerate(bucket_rows):
                if i < n_train:
                    r["split"] = "train"
                elif i < n_train + n_val:
                    r["split"] = "val"
                else:
                    r["split"] = "test"
        return

    if by != "factory":
        raise ValueError(f"by must be 'factory' or 'instance', got {by!r}")

    # by="factory": cluster instances by factory, assign whole factories.
    factory_rows: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        factory_rows[r["factory"]].append(r)

    # Stratify by difficulty only, NOT (difficulty × group). With ~22
    # non-holdout factories, a 2-D bucketing yields cells of size 1–3
    # that the bin-packer can't split across train/val/test without
    # leaving entire (diff×group) cells in just one split. Difficulty
    # is the dimension that matters most for the experiment — group
    # is just a coarse semantic axis carried through for the figures.
    fac_buckets: dict[str, list[str]] = defaultdict(list)
    for fname, frs in factory_rows.items():
        diff = _factory_majority_difficulty(frs)
        fac_buckets[diff].append(fname)

    # Greedy bin-pack within each difficulty bucket, sorted by size desc.
    target_ratio = {"train": train, "val": val,
                    "test": max(0.0, 1.0 - train - val)}

    for diff_key, factories in fac_buckets.items():
        bucket_total = sum(len(factory_rows[f]) for f in factories)
        bucket_target = {s: target_ratio[s] * bucket_total for s in target_ratio}
        bucket_counts = {"train": 0, "val": 0, "test": 0}
        # Largest factories first; deterministic tiebreak by name.
        factories.sort(key=lambda f: (-len(factory_rows[f]), f))
        for fname in factories:
            n = len(factory_rows[fname])
            deficits = {s: bucket_target[s] - bucket_counts[s] for s in bucket_counts}
            chosen = max(deficits, key=deficits.get)
            bucket_counts[chosen] += n
            for r in factory_rows[fname]:
                r["split"] = chosen


def existing_thumbs(task_id: str, instance: str) -> dict[str, str]:
    """Resolve the asset paths to embed into the HTML.

    Returns relative URLs from reports/artvip_splits.html to dataset assets.
    """
    base = f"../dataset/artvip_sanitized"
    parts_thumb = f"{base}/tasks/{task_id}/{instance}/thumbnail.png"
    rendered = f"{base}/gt_renders/{task_id}/{instance}/render_assembled.png"
    # joint anim GIFs — pattern anim_<joint>.gif under gt_renders/<task>/<inst>/
    anim_dir = DATA / "gt_renders" / task_id / instance
    anims: list[str] = []
    if anim_dir.is_dir():
        for gif in sorted(anim_dir.glob("anim_*.gif")):
            anims.append(f"{base}/gt_renders/{task_id}/{instance}/{gif.name}")
    return {
        "thumbnail": parts_thumb,
        "rendered": rendered,
        "anims": anims,
    }


def build_html(rows: list[dict]) -> str:
    """Self-contained HTML page with embedded data + edit/export logic.

    Loosely modeled on reports/artvip_axis_correction.html — same layout language,
    localStorage persistence, Export/Import JSON buttons.
    """
    # Add asset paths to each row for the page to consume directly.
    for r in rows:
        r["assets"] = existing_thumbs(r["task_id"], r["instance"])

    challenge_vocab = sorted({t for r in rows for t in r["challenges"]} | {
        "multi-joint", "multi-revolute", "multi-prismatic", "mixed-types",
        "continuous", "many-parts", "simple-binary",
    })

    group_vocab = sorted({r["group"] for r in rows} | {
        "container", "appliance", "tool", "dispenser", "stationary", "other",
    })

    header_stats = {
        "n_total": len(rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "by_group": dict(Counter(r["group"] for r in rows)),
    }

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ArtVIP train/val/test splits review</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 20px; background: #f5f5f7; color: #1f2937; }}
  h1 {{ margin: 0 0 6px 0; font-size: 1.4em; }}
  .meta {{ color: #6b7280; margin-bottom: 12px; font-size: 0.9em; }}
  .help {{ background: #fef3c7; border-left: 3px solid #f59e0b; padding: 10px 14px; margin-bottom: 16px; border-radius: 4px; font-size: 0.9em; }}
  .help code {{ background: #fde68a; padding: 1px 4px; border-radius: 2px; }}
  .toolbar {{
    position: sticky; top: 0; background: #fff; padding: 10px 14px;
    margin-bottom: 14px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    z-index: 100; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }}
  .toolbar input[type="search"] {{ padding: 5px 8px; border: 1px solid #d1d5db; border-radius: 4px; min-width: 220px; }}
  .toolbar select {{ padding: 5px 8px; border: 1px solid #d1d5db; border-radius: 4px; }}
  .toolbar button {{ padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9em; }}
  .toolbar .primary {{ background: #2563eb; color: #fff; }}
  .toolbar .secondary {{ background: #e5e7eb; color: #1f2937; }}
  .toolbar .danger {{ background: #fee2e2; color: #b91c1c; }}
  .stats {{ font-size: 0.85em; color: #6b7280; margin-left: auto; }}
  .stats b {{ color: #1f2937; }}
  .card {{
    background: #fff; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05); border-left: 4px solid #d1d5db;
  }}
  .card.touched {{ border-left-color: #f59e0b; }}
  .card.train  {{ background: #f0fdf4; }}
  .card.val    {{ background: #fff7ed; }}
  .card.test   {{ background: #faf5ff; }}
  .card.holdout{{ background: #fef2f2; border-left-color: #dc2626; }}
  .card-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; gap: 12px; }}
  .card-head h3 {{ margin: 0; font-size: 1em; }}
  .card-head .info {{ font-size: 0.82em; color: #6b7280; font-family: monospace; }}
  .grid {{ display: grid; grid-template-columns: 180px 180px 1fr; gap: 10px; align-items: start; }}
  .col-thumb img, .col-render img {{ width: 100%; max-height: 180px; object-fit: contain; display: block; border: 1px solid #e5e7eb; border-radius: 4px; background: #fafafa; }}
  .col-thumb .label, .col-render .label {{ font-size: 0.78em; color: #6b7280; margin-top: 3px; text-align: center; }}
  .anims {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
  .anims img {{ width: 100px; height: 100px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 4px; background: #fff; }}
  .anims .label {{ font-size: 0.72em; color: #6b7280; text-align: center; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 16px; padding: 8px 10px; background: #f3f4f6; border-radius: 4px; align-items: center; }}
  .controls label {{ font-weight: 500; font-size: 0.9em; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .chip {{ padding: 3px 9px; border-radius: 12px; font-size: 0.78em; cursor: pointer; user-select: none;
          border: 1px solid #d1d5db; color: #6b7280; background: #fff; }}
  .chip.on {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
  .split-radio {{ display: flex; gap: 8px; }}
  .split-radio label {{ font-weight: normal; padding: 3px 10px; border-radius: 4px; cursor: pointer; border: 1px solid #d1d5db; }}
  .split-radio input {{ display: none; }}
  .split-radio input:checked + span {{ font-weight: 600; }}
  .split-radio label:has(input[value=train]:checked) {{ background: #d1fae5; border-color: #10b981; }}
  .split-radio label:has(input[value=val]:checked)   {{ background: #fed7aa; border-color: #f97316; }}
  .split-radio label:has(input[value=test]:checked)  {{ background: #e9d5ff; border-color: #a855f7; }}
  .split-radio label:has(input[value=holdout]:checked) {{ background: #fee2e2; border-color: #dc2626; }}
</style>
</head>
<body>
<h1>ArtVIP splits review</h1>
<div class="meta">{len(rows)} instances · auto-stratified 60/20/20 by (difficulty × group). Edit per-instance, then Export.</div>

<div class="help">
  <b>怎么用</b>：
  <ol style="margin: 4px 0 0 18px;">
    <li>每张卡片显示 thumbnail + GT 渲染 + 关节 GIF</li>
    <li>Difficulty / Group / Challenges / Split 可改</li>
    <li>所有改动自动存到 <code>localStorage</code></li>
    <li>做完点 <b>Export JSON</b> 下载 <code>splits.json</code>，覆盖到 <code>dataset/artvip_sanitized/splits.json</code></li>
  </ol>
  <b>4 个 split</b>：<span style="color:#10b981">train</span> / <span style="color:#f97316">val</span> /
  <span style="color:#a855f7">test</span> / <span style="color:#dc2626">holdout</span> —
  holdout 是按 factory 整体留出的零样本测试集（folder × easy, scissors × medium, cardboard_box × hard），
  train/val/test 在剩余实例里 60/20/20 分层抽样。
</div>

<div class="toolbar">
  <input type="search" id="filter" placeholder="filter by task / factory / group …">
  <select id="filter-split">
    <option value="all">all splits</option>
    <option value="train">train only</option>
    <option value="val">val only</option>
    <option value="test">test only</option>
    <option value="holdout">holdout only</option>
  </select>
  <select id="filter-difficulty">
    <option value="all">all difficulty</option>
    <option value="easy">easy</option>
    <option value="medium">medium</option>
    <option value="hard">hard</option>
  </select>
  <button class="primary" onclick="exportJSON()">Export JSON</button>
  <button class="secondary" onclick="importJSON()">Import JSON</button>
  <button class="danger" onclick="clearLocal()">清空 localStorage</button>
  <span class="stats" id="stats"></span>
</div>

<div id="cards"></div>

<script>
const INITIAL = {json.dumps(rows, ensure_ascii=False)};
const CHALLENGE_VOCAB = {json.dumps(challenge_vocab)};
const GROUP_VOCAB = {json.dumps(group_vocab)};
const STORAGE_KEY = "artvip_splits_v1";

function loadState() {{
  try {{
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (s && s.rows) return s;
  }} catch (e) {{}}
  return null;
}}
function saveState(state) {{
  state.savedAt = new Date().toISOString();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}}

let STATE = loadState() || {{ rows: INITIAL.map(r => ({{...r}})) }};

function rowKey(r) {{ return `${{r.task_id}}/${{r.instance}}`; }}

function renderCards() {{
  const container = document.getElementById("cards");
  const filterText = document.getElementById("filter").value.toLowerCase();
  const filterSplit = document.getElementById("filter-split").value;
  const filterDiff = document.getElementById("filter-difficulty").value;

  const visible = STATE.rows.filter(r => {{
    if (filterSplit !== "all" && r.split !== filterSplit) return false;
    if (filterDiff !== "all" && r.difficulty !== filterDiff) return false;
    if (filterText) {{
      const blob = `${{r.task_id}} ${{r.instance}} ${{r.factory}} ${{r.group}} ${{r.challenges.join(" ")}}`.toLowerCase();
      if (!blob.includes(filterText)) return false;
    }}
    return true;
  }});

  container.innerHTML = visible.map(r => cardHTML(r)).join("");
  updateStats();

  // Wire up event handlers
  visible.forEach(r => {{
    const k = rowKey(r);
    const card = document.querySelector(`[data-key="${{k}}"]`);
    if (!card) return;
    card.querySelectorAll('.chip').forEach(chip => {{
      chip.onclick = () => toggleChallenge(r, chip.dataset.tag);
    }});
    card.querySelectorAll('.split-radio input').forEach(rb => {{
      rb.onchange = () => updateField(r, "split", rb.value);
    }});
    const diff = card.querySelector('select.difficulty');
    if (diff) diff.onchange = () => updateField(r, "difficulty", diff.value);
    const grp = card.querySelector('select.group');
    if (grp) grp.onchange = () => updateField(r, "group", grp.value);
  }});
}}

function cardHTML(r) {{
  const k = rowKey(r);
  const a = r.assets || {{}};
  const animsHTML = (a.anims || []).map(url => {{
    const name = url.split("/").pop().replace(/^anim_|\\.gif$/g, "");
    return `<div><img src="${{url}}" loading="lazy" title="${{name}}"><div class="label">${{name}}</div></div>`;
  }}).join("");
  const chipsHTML = CHALLENGE_VOCAB.map(t => {{
    const on = (r.challenges || []).includes(t) ? " on" : "";
    return `<span class="chip${{on}}" data-tag="${{t}}">${{t}}</span>`;
  }}).join("");
  const groupOpts = GROUP_VOCAB.map(g => `<option ${{g===r.group?'selected':''}}>${{g}}</option>`).join("");
  const splitClass = r.split || "train";
  return `<div class="card ${{splitClass}}" data-key="${{k}}">
    <div class="card-head">
      <h3>${{r.task_id}}/${{r.instance}}</h3>
      <span class="info">factory=<b>${{r.factory}}</b> · parts=${{r.n_parts}} · links=${{r.n_links}} · joints=${{r.n_joints}}
        (REV=${{r.n_revolute}}, PRIS=${{r.n_prismatic}}, CONT=${{r.n_continuous}}, FIX=${{r.n_fixed}})
        · score=${{r.complexity_score}}</span>
    </div>
    <div class="grid">
      <div class="col-thumb">
        <img src="${{a.thumbnail || ''}}" loading="lazy" onerror="this.style.opacity=0.2">
        <div class="label">thumbnail</div>
      </div>
      <div class="col-render">
        <img src="${{a.rendered || ''}}" loading="lazy" onerror="this.style.opacity=0.2">
        <div class="label">render_assembled</div>
      </div>
      <div>
        <div class="anims">${{animsHTML || '<i style="color:#999;font-size:.8em">no anim gifs</i>'}}</div>
        <div class="controls">
          <label>Difficulty:
            <select class="difficulty">
              <option ${{r.difficulty==='easy'?'selected':''}}>easy</option>
              <option ${{r.difficulty==='medium'?'selected':''}}>medium</option>
              <option ${{r.difficulty==='hard'?'selected':''}}>hard</option>
            </select>
          </label>
          <label>Group:
            <select class="group">${{groupOpts}}</select>
          </label>
          <label>Split:
            <span class="split-radio">
              <label><input type="radio" name="split-${{k}}" value="train"   ${{r.split==='train'?'checked':''}}><span>train</span></label>
              <label><input type="radio" name="split-${{k}}" value="val"     ${{r.split==='val'?'checked':''}}><span>val</span></label>
              <label><input type="radio" name="split-${{k}}" value="test"    ${{r.split==='test'?'checked':''}}><span>test</span></label>
              <label><input type="radio" name="split-${{k}}" value="holdout" ${{r.split==='holdout'?'checked':''}}><span>holdout</span></label>
            </span>
          </label>
        </div>
        <div style="margin-top:6px"><label style="font-size:.85em;color:#6b7280">Challenges:</label>
          <div class="chips">${{chipsHTML}}</div>
        </div>
      </div>
    </div>
  </div>`;
}}

function toggleChallenge(row, tag) {{
  row.challenges = row.challenges || [];
  const i = row.challenges.indexOf(tag);
  if (i >= 0) row.challenges.splice(i, 1);
  else row.challenges.push(tag);
  row.touched = true;
  saveState(STATE);
  renderCards();
}}
function updateField(row, field, value) {{
  row[field] = value;
  row.touched = true;
  saveState(STATE);
  renderCards();
}}

function updateStats() {{
  const c = {{}};
  for (const r of STATE.rows) {{
    const s = r.split || "?";
    c[s] = (c[s] || 0) + 1;
  }}
  const dByS = {{}};
  for (const r of STATE.rows) {{
    const k = `${{r.split}}/${{r.difficulty}}`;
    dByS[k] = (dByS[k] || 0) + 1;
  }}
  document.getElementById("stats").innerHTML =
    `<b>${{c.train||0}}</b> train / <b>${{c.val||0}}</b> val / <b>${{c.test||0}}</b> test / <b>${{c.holdout||0}}</b> holdout
     · easy=${{(dByS['train/easy']||0)+(dByS['val/easy']||0)+(dByS['test/easy']||0)+(dByS['holdout/easy']||0)}}
     · med=${{(dByS['train/medium']||0)+(dByS['val/medium']||0)+(dByS['test/medium']||0)+(dByS['holdout/medium']||0)}}
     · hard=${{(dByS['train/hard']||0)+(dByS['val/hard']||0)+(dByS['test/hard']||0)+(dByS['holdout/hard']||0)}}`;
}}

function exportJSON() {{
  const out = STATE.rows.map(r => {{
    const {{ assets, touched, ...keep }} = r;
    return keep;
  }});
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "splits.json"; a.click();
  URL.revokeObjectURL(url);
}}

function importJSON() {{
  const inp = document.createElement("input");
  inp.type = "file"; inp.accept = "application/json";
  inp.onchange = e => {{
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = () => {{
      try {{
        const arr = JSON.parse(reader.result);
        if (!Array.isArray(arr)) throw new Error("expected array");
        const lookup = {{}};
        for (const r of arr) lookup[`${{r.task_id}}/${{r.instance}}`] = r;
        for (const r of STATE.rows) {{
          const u = lookup[`${{r.task_id}}/${{r.instance}}`];
          if (u) Object.assign(r, u, {{assets: r.assets, touched: true}});
        }}
        saveState(STATE);
        renderCards();
      }} catch (err) {{ alert("Import failed: " + err); }}
    }};
    reader.readAsText(file);
  }};
  inp.click();
}}

function clearLocal() {{
  if (!confirm("Reset all in-browser edits?")) return;
  localStorage.removeItem(STORAGE_KEY);
  STATE = {{ rows: INITIAL.map(r => ({{...r}})) }};
  renderCards();
}}

document.getElementById("filter").addEventListener("input", renderCards);
document.getElementById("filter-split").addEventListener("change", renderCards);
document.getElementById("filter-difficulty").addEventListener("change", renderCards);
renderCards();
</script>
</body></html>
"""


def _merge_human_edits(auto_rows: list[dict], existing_path: Path) -> int:
    """Pull human-reviewed `difficulty / group / challenges` out of
    `existing_path` and overlay them onto the freshly-computed `auto_rows`.

    Mutates `auto_rows` in place. Returns the number of instances that
    received an overlay.

    `split` is deliberately NOT preserved — it must be re-derived after
    every script run because the by-factory algorithm can't honor a
    previous instance-level split (different policy entirely).  Auto-
    derived numeric fields (n_parts, joint counts, scores) always come
    from the fresh URDF/parts read.
    """
    if not existing_path.exists():
        return 0
    try:
        existing = json.loads(existing_path.read_text())
    except Exception:
        return 0
    by_key = {(r["task_id"], r["instance"]): r for r in existing if "task_id" in r}
    PRESERVED = ("difficulty", "group", "challenges")
    n = 0
    for r in auto_rows:
        old = by_key.get((r["task_id"], r["instance"]))
        if not old:
            continue
        for k in PRESERVED:
            if k in old:
                r[k] = old[k]
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.6)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Refuse to overwrite splits.json if it already exists.")
    parser.add_argument(
        "--preserve-edits", action="store_true",
        help="Read the existing splits.json (if any) and keep its human-edited "
             "labels (`difficulty`, `group`, `challenges`). `split` is always "
             "recomputed.  Auto-derived numeric features (n_parts, joint "
             "counts, scores) also get refreshed.",
    )
    parser.add_argument(
        "--split-by", choices=("factory", "instance"), default="factory",
        help="Splitting policy.  factory (default, paper): every instance of "
             "a factory lands in one split.  instance: legacy random per-"
             "instance split (same factory can appear in train+val+test).",
    )
    args = parser.parse_args()

    mapping = json.loads((DATA / "mapping.json").read_text())
    t2f = mapping.get("task_to_factory", {})
    rows: list[dict] = []
    for task_dir in sorted((DATA / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        factory = t2f.get(task_dir.name, "unknown")
        for inst_dir in sorted(task_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            r = features_for_instance(task_dir.name, inst_dir.name, factory)
            if r:
                rows.append(r)

    if args.preserve_edits:
        # Apply human label edits BEFORE the split so factory-level
        # bucket assignment uses corrected difficulty/group.
        n_overlay = _merge_human_edits(rows, SPLIT_OUT)
        print(f"--preserve-edits: overlaid human labels onto {n_overlay}/{len(rows)} rows")

    stratified_split(rows, train=args.train, val=args.val, seed=args.seed,
                     by=args.split_by)

    print(f"\n=== {len(rows)} instances ===")
    for k, label in (("split", "split"), ("difficulty", "difficulty"), ("group", "group")):
        c = Counter(r[k] for r in rows)
        print(f"  {label}: {dict(sorted(c.items()))}")

    if SPLIT_OUT.exists() and args.no_overwrite:
        print(f"\nrefusing to overwrite existing {SPLIT_OUT} (--no-overwrite)")
    else:
        clean = [{k: v for k, v in r.items() if k != "assets"} for r in rows]
        SPLIT_OUT.write_text(json.dumps(clean, indent=2))
        print(f"\nwrote {SPLIT_OUT}")

    HTML_OUT.write_text(build_html(rows))
    print(f"wrote {HTML_OUT}")
    print(f"\nopen the page in your browser, edit, click Export JSON, "
          f"replace {SPLIT_OUT} with the download.")


if __name__ == "__main__":
    main()
