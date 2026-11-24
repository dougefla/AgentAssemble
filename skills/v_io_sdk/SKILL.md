---
name: assemble
description: "Turn the OBJ parts and the blanked URDF template in INPUT_DIR into the complete URDF of the assembled, articulated object at OUTPUT_DIR/furniture.urdf (input/output only)."
---

# Task

## Input

`INPUT_DIR` (read-only) contains:

- `part_<i>.obj` (with `.mtl` / textures): the mesh of each part of one
  articulated object;
- `template.urdf`: the object's URDF with the answers removed. Every
  `? ? ?` is a blank: the `<origin xyz>` and `<origin rpy>` of a link's
  visual, or the `<origin xyz>` and `<axis xyz>` of a joint. Everything
  else in the file is given and correct.

A reference image of the assembled object is named in the prompt.

## Output

`OUTPUT_DIR/furniture.urdf`: `template.urdf` with every `? ? ?` replaced
by three numbers, so that the file describes the assembled object at rest
and how each joint moves it. Change nothing else in the file.

## Tools

The persistent Python sim session has the frozen Assembly SDK preloaded as
`assembly` (see ACTIONS.md): it imports the parts, builds links and joints,
renders the assembly at rest and in motion, and can write the URDF.
