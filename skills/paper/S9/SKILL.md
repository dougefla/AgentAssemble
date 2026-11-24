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

## Recover revolute bearings from circular rims

When a revolute direction is uncertain, use
`tools.round_features.find_round_features` on the **supplied mesh vertices
and triangular faces**, before relying on whole-part PCA or the largest
plate normal. It finds closed circular sharp-edge loops without requiring
a guessed cutting plane, pivot neighborhood radius, or restored orientation.
A part's long axis or broad face need not identify its proximal bearing.

Inspect the child features whose `origin_line_distance` is small compared
with the part span: their circle normals are candidate **local** bearing
axes through the preserved mesh origin. Match these to actual parent
bearing/pin rims near the attachment, not to unrelated screws. Transform
parent features with its current scale, visual rotation and world pivot;
transform child axes with the candidate visual rotation. Matched axle lines
must be coaxial. This constrains the restoration rotation as well as the
world joint axis; do not simply paste a raw-mesh axis into the URDF.

Several concentric rims corroborate a bearing; radius agreement may support
a scale estimate only for genuinely mating surfaces. The helper does not
choose the joint, rotation sign, scale or rest facing. Inspect remaining
in-plane orientation against the thumbnail. For a non-root parent, recheck
its distal bearing after restoring it, rather than assuming all joints of
an arm share one axis. If no trustworthy rim is found, report that absence
and use other geometry; an empty result is not an execution failure.

## Seat sliders transversely before fitting extension

A prismatic rest origin has one coordinate **along** travel and two
coordinates **across** it. Do not fit all three freely to a colour-region
centre. Before adjusting extension, measure the parent guide's cross-section
and the child's corresponding runner, shaft, or seating surfaces in their
restored geometry. Constrain the two transverse coordinates by the actual
rail/channel, concentric shaft bore, or supporting shelf; preserve the
mesh-origin offset from these surfaces instead of placing the origin itself
on the shelf or at the guide's centre.

For a planar mating pair, let `c` be a point on the child surface after
rotation/scale but before translation, and `n·x=d` the matching parent plane
in world coordinates. The rest pivot must satisfy `n·p = d - n·c` (including
any measured clearance). Use independent guide normals perpendicular to
the travel axis to determine the transverse placement. Do not use a front
stop plane to constrain extension as though it were a guide sidewall.
For cylindrical guides, align the shaft and bore centre lines instead.

With these constraints held, fit only extension along the guide from a
visible corresponding front/handle landmark, not the centre of an occluded
colour region. Check the seated cross-section at q=0 and at the legal
endpoint. An image discrepancy does not justify moving a drawer through
its shelf, a footrest off its rail, or a shaft off its bore. If the mating
geometry is absent, retain uncertainty rather than inventing a guide height.

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
