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
