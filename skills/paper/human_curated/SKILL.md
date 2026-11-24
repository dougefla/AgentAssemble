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

# Procedure

## What has to be solved

Every non-root link `i` has three unknowns, namely a rotation `R_i`, a scale `s_i` and a world pivot `p_i`. A mesh vertex `v` rests at `p_i + s_i * R_i @ v`. Every joint also needs a unit axis `a` in world coordinates. Link frames are never rotated, so the joint `<origin xyz>` equals `p_child - p_parent` (with `p_root = 0`). The result is scored numerically with tight tolerances on axis direction, pivot, scale, rest surface and swept motion. Take every number from mesh geometry. Use the images to choose between discrete alternatives and to catch gross errors.

## Step 0. Read the givens (2 to 3 calls)

1. Read `template.urdf`. Write down the tree (parent, child, type, lower, upper for each joint) and the colour of each link. Order the links breadth-first from the root, because a child can only be fitted after its parent is placed.
2. View the reference image. For each link note which parent surface it touches, where its bulk extends from that contact (up, front, left), and any asymmetric detail (handle, knob, bend, hole pattern).
3. Remember the world convention, in which Z is up, -Y is the front, and units are metres.

## Step 1. One helper cell, then measure every part (2 to 3 calls)

Define small numpy helpers once in the persistent session and reuse them.

- An OBJ loader that returns vertices and triangles.
- `rpy -> R` with the URDF convention `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)` and its inverse. Assert `det(R) = +1` every time.
- Dominant flat-face normals, obtained by summing triangle areas over clustered unit normals.
- Principal axes from the PCA of the vertices, with the extents along them.
- Axis of revolution of a set of faces, given by the eigenvector with the smallest eigenvalue of the area-weighted sum of `n n^T`. Side faces of a cylinder all have normals perpendicular to its axis.
- Circle fit, which projects selected vertices on the plane perpendicular to an axis and returns centre and radius by least squares.

Then call `import_parts` and print, for all parts in one cell, the AABB, the PCA extents, the top dominant normals, and where the origin lies inside the AABB (as a fraction per axis). The origin is the pivot, so its location in the part is strong evidence. An origin on one edge of a flat panel marks the hinge edge. An origin at the centre of a round boss marks the hinge pin.

## Step 2. Fix the camera once (2 calls)

Create the root link (rotation zero, scale 1) and render it alone with `render_preview` at the default camera. The reference image is normally an oblique front view close to that default. If the root looks different, change azimuth and elevation until it matches, then keep that camera for every later comparison.

## Step 3. Solve one joint at a time, parents first

Work in the order axis, rotation, scale, pivot. All parent geometry is in world coordinates (the root as given, a solved link through `p + s * R @ v`).

**3a. Axis in the child mesh (`a_c`) and in the world (`a_w`).**

- Revolute. In the child, the hinge line passes through the mesh origin. Select the child faces within a small radius of the origin and compute their axis of revolution. Confirm that many nearby vertices share one distance from that line (a pin, barrel or bore). If the part has no round feature there, take the dominant normal or PCA axis that agrees with the image (a door hinge runs along the hinge edge). In the parent, find the mating bore, pin, knuckle or opening edge in the region where the image shows the child attached, and compute its axis and centre line the same way.
- Prismatic. The axis is the direction in which the child slides freely in the parent, which is the long direction of the rail, slot or cavity, or the normal of the cavity opening. In the child it is the matching rail direction or the PCA axis along which the cross-section stays constant.
- If `a_w` is within about 2 degrees of a world axis and the parent is an axis-aligned manufactured object, snap it to the exact world axis. Otherwise keep the fitted direction. Normalise it.
- Sign. The positive direction moves the child from `lower` towards `upper`. Choose the sign so that the larger side of the range moves the child into free space (a drawer comes out of the cabinet, a door or lid swings away from the body, by the right-hand rule about `a_w`). With a symmetric range and no other cue, point the axis along the nearest positive world direction. Step 4 checks this choice.

**3b. Rotation `R`.** It must send `a_c` to `a_w`. That leaves one angle about the axis, fixed by a second direction pair `(b_c, b_w)`. Take `b_c` from a dominant face normal or PCA axis of the child that is not parallel to `a_c`, and `b_w` from the direction that feature has at rest (a panel normal equal to the normal of the parent face it closes, an arm pointing towards the next joint as seen in the image, a flat base parallel to the parent surface). Orthogonalise `b` against `a` on both sides and set `R = [a_w, b_w, a_w x b_w] @ [a_c, b_c, a_c x b_c]^T`. Sign choices of `a_c` and `b_c` give up to four discrete candidates, and symmetric parts give more. Decide between them with asymmetric evidence in this order, namely the side of the pivot on which the bulk of the part lies, the position of handles and knobs (they face outward, usually towards -Y), and the reference image. When the assembled object is clearly axis-aligned, `R` should send the child's dominant normals to world axes, so reject candidates that leave the part tilted.

**3c. Scale `s`.** Compute it after `R`, as a length that the parent fixes divided by the same length measured on `R @ v`. Good references, best first, are the radius of a pin against the radius of its bore, the width of a part against the gap between the parent cheeks or the opening it fills, the height or width of a door against its opening, and the link's size relative to the root in the reference image as a last resort. Take two independent references when available. They should agree within a few percent, and a larger disagreement means `R` is a wrong discrete candidate. Parts that are copies of each other (equal vertex counts and equal PCA extents, such as a stack of drawers) share one scale and the same rule for `R`.

**3d. Pivot `p`.** Pick a reference point `x` whose rest location on the parent is known, and solve `p = x_world - s * R @ x_child`.

- Revolute. `p` must lie on the parent hinge line found in 3a. Start from the foot of the perpendicular and slide along the axis until the child's hinge feature is centred on the parent's (equal overhang at both ends, or shoulder against shoulder). Any point on the line gives the same motion, yet the rest surface is scored, so the slide still matters.
- Prismatic. Fix the two coordinates across the axis by centring the child's cross-section in the parent's channel or opening. Fix the coordinate along the axis from the rest contact, such as a front panel flush with the parent's front face or a stop touching.

**3e. Build and look (2 to 3 calls per link).** Call `set_part_scale`, `create_link` with the rpy of `R`, `create_joint` with the template's name, type and limits, then `set_joint_world_frame` with `p` and `a_w`. Keep the visual offset at zero. Compare the link's world AABB from `inspect_parts` with the AABB of `p + s * R @ v` computed in numpy. A mismatch means the rpy convention is wrong and must be fixed before continuing. Render with the fixed camera and compare with the reference image for position, size relative to the parent, facing direction and silhouette. If the link is wrong, name the fault first (flipped about an axis, wrong scale, shifted along the axis) and recompute that one quantity from geometry. Blind nudging of numbers wastes the budget.

## Step 4. Motion check per joint (1 to 2 calls each)

Run `render_joint_sweep` and look at the first and last frame.

- A revolute child must stay attached at the hinge. A hinge feature that orbits means the pivot is off the hinge line.
- A wobbling or tilting sweep means the axis direction is off.
- The child must move into free space and never through its parent over the whole range. If it dives into the parent, flip the axis sign.
- A prismatic child must stay inside its guide and keep its clearance.
- Descendants must move rigidly with the link.

## Step 5. Whole-object check and output (3 to 4 calls)

1. Render the full assembly at rest with the fixed camera. If an image library is importable, load this render and the reference image, build one mask per link from the pixels nearest its colour, normalise by the bounding box of the whole foreground, and compare each link's mask box. A link whose box differs by more than a few percent of the object size needs another look at its scale or pivot.
2. Check the numbers. Every scale is positive with three equal values, every axis has unit length, every revolute pivot lies on its parent hinge line, each joint origin equals `p_child - p_parent`, and no link interpenetrates its parent deeply at rest.
3. Run `audit` and fix every issue it reports.
4. Call `export_urdf(OUTPUT_DIR)`. Confirm that `furniture.urdf` exists, contains no `?`, and keeps the template's link names, joint names, types and limits.

## Budget (about 60 tool calls)

- Spend about 8 calls on Steps 0 to 2, about 5 to 8 per joint on Steps 3 and 4, and keep 6 in reserve for Step 5.
- Batch work. Measure all parts in one cell. Solve axis, rotation, scale and pivot of a link in one cell and render once afterwards.
- With many links, solve one member of each group of identical parts fully and place the others by the offset between their mating features on the parent. Sweep one joint per group plus any joint that looked doubtful.
- Allow at most two correction rounds per link, then keep the best candidate and move on, since an unfinished tree scores nothing.
- When about 10 calls remain, stop refining, place every remaining link with its best estimate, run `audit`, and export.
