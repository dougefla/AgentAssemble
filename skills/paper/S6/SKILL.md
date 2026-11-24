---
name: assemble
description: "Turn the OBJ parts and the blanked URDF template in INPUT_DIR into the complete URDF of the assembled, articulated object at OUTPUT_DIR/furniture.urdf: recover each part's orientation and size, then its joints (input/output only)."
---

# Task

## Input

`INPUT_DIR` (read-only) contains:

- `part_<i>.obj` (with `.mtl`): the mesh of each link of one articulated
  object, expressed in that link's own frame. The frame origin is the link's
  joint pivot, so every mesh sits at the origin. Except for the root link's
  part, each mesh has been **rotated by an unknown rotation** about that
  origin and **uniformly scaled** so that its longest bounding-box side is 1.
  The root link's part is given in place, upright and at its true size.
- `template.urdf`: the object's URDF with the answers removed. Every
  `? ? ?` is a blank:
  - a link's visual `<origin rpy>` — the rotation that brings its mesh back
    to the assembled orientation (xyz stays `0 0 0`);
  - a link's visual `<mesh scale>` — three equal numbers, the factor that
    restores the part's true size;
  - a joint's `<origin xyz>` (the child frame's position in the parent
    frame, i.e. the pivot) and `<axis xyz>` (in world coordinates).
  Everything else in the file — link names, tree, joint types, limits,
  mimic couplings, colours — is given and correct. The comment at the top
  of the file states the frame conventions. A `<mimic>` tag only couples a
  joint's value to another's in the finished URDF: leave it as it is; the
  SDK has no notion of mimic and you do not need to reproduce it when
  building or exporting.

A reference image of the assembled object is named in the prompt: each link
has the same flat colour there as in its `.mtl`.

## Output

`OUTPUT_DIR/furniture.urdf`: `template.urdf` with every `? ? ?` replaced
by three numbers, so that the file describes the assembled object at rest
and how each joint moves it. Change nothing else in the file.

## Tools

The persistent Python sim session has the frozen Assembly SDK preloaded as
`assembly` (see ACTIONS.md): it imports the parts, sets each part's scale,
builds links (with their visual rotation) and joints, renders the assembly
at rest and in motion, and can write the URDF with the scales and
rotations baked in.

## Constrain depth before screen-space placement corrections

A good thumbnail overlay does not establish a correct 3D attachment. Before
moving a revolute pivot to reduce a colour-region offset, constrain its
unseen coordinate from the **parent's actual attachment geometry**: the
bearing/axle centre, hinge edge, bracket plane, or support socket. Inspect
that neighborhood in the parent's restored mesh, not just its global AABB.
For a non-root parent use its current scale, visual rotation, and world
pivot to transform the measured attachment into world coordinates.

Hold this measured attachment constraint while correcting the remaining
projected coordinates. In particular, do not move a wheel away from its
axle, a handle out of its socket, or a hinge off its bracket just to match
screen centres. Colour-region centres are not mechanical landmarks. If an
attachment-constrained placement still has a silhouette error, reconsider
orientation/scale or visibility instead of adding an unconstrained 3D nudge.
After changing a parent's rotation or scale, remeasure its child attachment
locations before independently adjusting descendants. Prefer an actual
small attachment feature over the part centroid, floor level, or a rounded
fraction of the object's overall dimensions. If no feature is identifiable,
keep the depth uncertainty explicit rather than claiming the overlay solves it.

## One-sided slider travel: check remaining guide engagement

For each prismatic joint with zero at one limit, check the **nonzero legal
endpoint**, not just q=0. The thumbnail may already show the slider extended;
"outward" then need not be its remaining legal motion. Measure the moving
member's trailing edge and the stationary guide/mouth along the restored
travel direction. For each candidate axis sign compute endpoint positions
as `x(q) = x(0) + q * axis`, using the template's actual signed limit.

Reject a sign that sends an already-extended member farther out until its
trailing edge leaves the guide, or pushes its front/handle through a fixed
back wall, when the opposite sign retracts it into the available cavity.
Use actual rail engagement and stopping surfaces, not an assumption that
increasing q always opens or moves upward. Check both signs without changing
the rest pose, scale, rotation, or given limits to make one work. Render the
chosen nonzero endpoint to verify the numerical engagement check, then reset.
If both signs remain mechanically plausible (especially for two-sided
adjustments), record the ambiguity; a rest overlay cannot resolve it.

## Root-registered rest-pose check

Before finalizing, render the **complete, final-parameter** assembly and read
its image. Compare it with the thumbnail using
`tools.image_compare.compare_views`, supplying the template's link RGB colours,
the fixed root name, and an overlay path inside OUTPUT_DIR. Read the panel: it
shows the target, a preview registered on the fixed root, and their blend.
This removes misleading differences from automatic camera framing.

Use comparable camera angles (normally azimuth 45, elevation 25); if the
root's span residual exceeds 0.1, first adjust the view or inspect root
occlusion rather than treating the measurements as placement corrections.
For a child whose orientation and size already agree, a displaced colour
region is evidence of a misplaced pivot: inspect its `center_delta_px` and
move the joint, not the visual xyz. Use known root dimensions and the
projected world directions to estimate a correction; image +Y points down.
For similar X/Y span ratios, check scale before translation. Do not force a
3D correction from an occluded region or a single-view depth ambiguity.
Re-render and inspect the registered panel after any significant correction.
Keep mesh origins intact and all visual xyz at zero.
If the simulator/API is unavailable, stop and write FAILED.txt, not retries.
