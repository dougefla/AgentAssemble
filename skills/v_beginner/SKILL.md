---
name: assemble
description: "Annotate the joints of pre-placed OBJ parts into a URDF with the frozen Assembly SDK."
user-invocable: true
argument-hint: "[obj_directory] [output_directory]"
---

# Annotate articulated joints on pre-placed parts

Produce `OUTPUT_DIR/furniture.urdf` and its meshes from the read-only OBJ files
in `INPUT_DIR`. The simulation backend is available only through
`mcp__sim__execute_python_code` — a persistent Python session.

## Task contract

Every part already sits in its correct rest pose the moment it is imported.
Nothing needs to be placed, aligned, scaled, or grounded, and nothing may be
moved. Your only job is to say how the parts articulate:

- which part is the fixed base (root) and which parts move;
- the tree: each moving part's parent;
- for each moving part: joint type (`REVOLUTE` or `PRISMATIC`; `FIXED` for a
  part that is rigidly attached), the pivot (joint origin), the axis direction,
  and the travel limits.

A hinge's pivot is where its axis line passes — an edge of the panel, not the
panel's center. A slide's pivot can be any point on the moving part; only its
direction matters. Limits are one-sided from the rest pose: the part is closed
or retracted at q=0 and opens or extends toward the nonzero end.

## Runtime contract

Every sim call receives four immutable Python globals:

- `ASSEMBLY_SDK_DIR`: frozen, versioned runtime owned by the training system.
- `SKILL_DIR`: this evolvable policy directory.
- `INPUT_DIR`: read-only input OBJ directory.
- `OUTPUT_DIR`: writable candidate output directory.

The session is persistent: your Python variables, imports, and helpers
survive across calls, and `assembly` is already preloaded when the session
starts. Re-executing the loader is optional and harmless:

```python
exec(open(f"{ASSEMBLY_SDK_DIR}/runtime.py").read())
```

This rebinds `assembly` and refreshes a read-only `parts` snapshot from the
persistent assembly state. Rebuild dictionaries such as
`by_label = {p["label"]: p["handle"] for p in parts}` after every runtime load;
never reuse any other Python local from a previous sim call.
The session is a plain Python interpreter, not a Jupyter kernel: `display()`,
IPython helpers, and `%` magics do not exist. Use `print()` for data and
write images to files; the `Read` tool shows you rendered PNGs.

Use only `assembly`'s explicit methods. Never mutate assembly state
dictionaries, cached mesh vertices, part transforms, runtime globals, or
parts looked up by guessed loader-internal names such as `part_2`. Stable
part handles returned by `assembly.import_parts()` are the only part
identities. `clear_scene()` invalidates all of those handles. Do not clear
after importing; `import_parts(..., replace=True)` already resets the scene.
If a correction truly requires clearing, immediately re-import and rebuild
the handle map from the new return value before calling any part operation.

The SDK owns import coordinates, persistent state, topology invariants, FK,
the visual transform of every link, rendering, collision AABBs, and
deterministic URDF/OBJ export. Do not reproduce those mechanisms in skill
code. `ACTIONS.md` is the compact API. Its signatures are exact and frozen:
do not edit that file or restate an SDK signature differently. Arguments
shown after `*` are keyword-only.

## Minimal workflow

1. Import and inspect geometry (`replace=True` resets any earlier scene):

   ```python
   parts = assembly.import_parts(INPUT_DIR, replace=True)
   by_label = {p["label"]: p["handle"] for p in parts}
   print(parts)
   print(assembly.inspect_parts())
   ```

   `inspect_parts()` world AABBs are the rest-pose positions. Use them to tell
   the base from the movers and to read off pivots: a door's hinge is along
   one vertical edge of its AABB where it meets the body; a drawer slides
   along the axis in which it is inset into the body. The thumbnail, when
   provided, only helps with semantics (which way a lid opens); never use it
   to reposition anything.

2. Register links and joints incrementally with `create_link()` and
   `create_joint()`. One part per link unless two parts are clearly one rigid
   piece (a handle on its door). Give every independently movable part its
   own joint; never merge symmetric arms or blades merely because their
   motions are coupled. `create_joint()` takes the pivot in the parent's
   frame; when you know it in world coordinates, create the joint with any
   origin and then call `set_joint_world_frame()`, which converts a world
   pivot and axis for you. Both work in a valid partial forest. FIXED joints
   ignore `axis` and take no limits.

3. Call `audit()`. Do not proceed while `audit()["ok"]` is false. The audit
   also confirms `mesh_frame == "world"` and that no mesh drifted.

4. Render and check:

   ```python
   assembly.render_preview(f"{OUTPUT_DIR}/render_assembled.png")
   for joint in movable_joints:
       assembly.render_joint_sweep(joint, OUTPUT_DIR)   # <joint>_<index>_<pct>pct.png
   ```

   Pass each joint's sweep frames to `validate-joints`. Its verdict is a
   diagnostic opinion; the objective oracle remains the evaluator.

5. Correct parameters with `update_joint()`, `reparent_joint()`, or
   `set_joint_world_frame()`; fix a wrong parent by reparenting instead of
   deleting and re-creating the joint. Re-audit, reset to q=0, then call
   `export_urdf(OUTPUT_DIR)`.

A deterministic SDK rejection comes back from the sim tool as a structured
result `{"ok": false, "error": {"code": ..., "operation": ..., "message": ...}}`;
it is not an engine failure. Fix the offending arguments and retry. A
rejection with `mesh_frame: "world"` means you tried to move a mesh: the fix
is always to change a joint, never a pose.
Infrastructure failures (API/context, sim worker startup/socket, or missing
injected globals) are training-system failures: write the reason to
`OUTPUT_DIR/FAILED.txt`, stop, and report; do not retry or work around them.

## Coordinate contract

The source OBJ vertex frame is the world frame at rest. At all joint values:

```text
T_world_mesh(q) = T_world_link(q) @ T_link_visual
T_world_child(q) = T_world_parent(q) @ T_joint_origin @ T_joint_motion(q)
```

The root link frame is world identity. A joint origin locates the child link
frame at q=0; for a revolute joint this is the hinge frame, not the mesh
center. The SDK solves `T_link_visual` itself so that at q=0
`T_world_mesh` is the identity for every link: meshes stay exactly where they
were imported whatever you do to the joints. `set_link_visual_pose()`,
`ground()`, `bake_current_pose_as_rest()`, and `set_part_scale()` are refused
in this mode; a valid partial forest is supported during incremental
authoring.

q=0 must reproduce the reference pose. Closed doors, lids, and drawers use a
one-sided range touching zero. Joint sweep `000pct` is the sorted lower
limit, not necessarily q=0: with a canonical positive world axis, opening in
the positive direction yields closed-to-open, while opening in the negative
direction yields open-to-closed.

## Hard rules

1. Never write inside `INPUT_DIR` or use external datasets.
2. Never invoke the simulation backend outside its MCP tool.
3. Never edit or copy code under `ASSEMBLY_SDK_DIR`.
4. Never access a part by loader-internal name; use returned stable handles.
5. Never reuse a part handle after `clear_scene()`; re-import and rebuild maps.
6. Never hand-write FK, URDF XML, OBJ export, or render setup.
7. Never move, scale, or ground a part; only joints are yours to author.
8. Never export unless `assembly.audit(require_complete_tree=True)["ok"]`.
9. Keep the evolvable skill focused on inference policy, geometry heuristics,
   and validators—not deterministic mechanics.
