# ArtiAssembly

ArtiAssembly has 300 articulated everyday objects in four tiers of complexity (T1 to T4 with 90, 90, 75, and 45
objects). Every object has a ground-truth URDF, per-link meshes, a reference image, and the parametric
OpenSCAD source of its geometry.

## Files in this folder

| File | Content |
|---|---|
| `objects.jsonl` | the 300 objects with id, name, tier, category, number of joints, and the text prompt |
| `splits_main.json` | the 120 objects of the evaluation subset with their split (train, val, test), tier, and joint counts |
| `seed_screening.json` | fitness and pass of the seed Skill S0 on the 213 eligible objects, three sessions each |

The evaluation subset is drawn from the screening. Objects that S0 never solves (`never_low`, `never_high`
by mean fitness below or above 0.35) and objects that it always solves (`always`) are sampled at 22, 10,
and 8 per split. `scripts/experiments/build_subset.py` reproduces the draw.

## Task view

The agent works on a view of the dataset in which the answers are removed.

```
<EVO_DATASET_ROOT>/
  tasks/<task_id>/0/parts/      part_<i>.obj, template.urdf, render_thumbnail.png   (what the Executor sees)
  gt/<task_id>/0/               scene.urdf and assets                               (Evaluator only)
  splits.json, index.json, mapping.json, dataset_variant.json
```

Every non-root mesh is rotated by a random rotation about its link origin and scaled so that its longest
bounding-box side is 1. The template keeps link names, the kinematic tree, joint types, and joint limits,
and it blanks the joint origins, the joint axes, the link orientations, and the link scales.

The scripts in `scripts/dataset` build the view. `build_artic100_rot_view.py` applies the random rotation and
the scaling and writes the template, and `subset_view.py` cuts the evaluation subset out of the full view.

## Download

The download link of the objects and of the prepared task view will be added here.
