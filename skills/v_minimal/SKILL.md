---
name: assemble
description: "Assemble articulated OBJ parts into a URDF with the frozen Assembly SDK."
user-invocable: true
argument-hint: "[obj_directory] [output_directory]"
---

# Assemble articulated furniture

Produce `OUTPUT_DIR/furniture.urdf` and its meshes from the read-only OBJ files
in `INPUT_DIR`. The simulation backend is available only through
`mcp__sim__execute_python_code` — a persistent Python session.

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

Use only `assembly`'s explicit methods. Never mutate assembly state
dictionaries, cached mesh vertices, part transforms, runtime globals, or
parts looked up by guessed loader-internal names such as `part_2`. Stable
part handles returned by `assembly.import_parts()` are the only part
identities. `clear_scene()` invalidates all of those handles. Do not clear
after importing; `import_parts(..., replace=True)` already resets the scene.
If a correction truly requires clearing, immediately re-import and rebuild
the handle map from the new return value before calling any part operation.

The SDK owns import coordinates, persistent state, topology invariants, FK,
grounding, rendering, collision AABBs, and deterministic URDF/OBJ export. Do
not reproduce those mechanisms in skill code. `ACTIONS.md` is the compact API.
Its signatures are exact and frozen: do not edit that file or restate an SDK
signature differently. Arguments shown after `*` are keyword-only.

## Minimal workflow

1. Inspect the reference thumbnail once when provided.
2. Import and inspect geometry (`replace=True` resets any earlier scene):

   ```python
   parts = assembly.import_parts(INPUT_DIR, replace=True)
   by_label = {p["label"]: p["handle"] for p in parts}
   print(parts)
   print(assembly.inspect_parts())
   ```

   Input parts are metric meshes in their original relative scale: do not
   rescale them. Use `inspect_parts()` AABBs and the thumbnail to decide
   which part is which. Call `set_part_scale()` only when a part's extent is
   clearly inconsistent with the assembled object (for example a handle
   larger than its door). Never fabricate a handle string.

3. Infer rigid groups, root, topology, joint types, origins, axes, limits, and
   q=0 mesh poses. Register them incrementally with `create_link()` and
   `create_joint()`. It is safe to call `set_link_visual_pose()` on a link in a
   valid partial forest and `set_joint_world_frame()` as soon as that joint
   exists; the SDK deliberately supports this common authoring order. An
   isolated temporary link can be given its final semantic name by calling
   `create_link(final_name, same_parts)` or `rename_link()`; both keep the
   link's visual pose. Use `update_link_parts()` to atomically swap the
   complete part groups of two isolated links. FIXED joints ignore `axis`
   and take no limits.
4. Finish one connected tree and place every mesh frame with
   `set_link_visual_pose()`. The SDK solves the link-relative visual transform.
5. Call `ground()`, then `audit()`. Do not proceed while `audit()["ok"]` is
   false.
6. Render and check:

   ```python
   assembly.render_preview(f"{OUTPUT_DIR}/render_assembled.png")
   for joint in movable_joints:
       assembly.render_joint_sweep(joint, OUTPUT_DIR)   # <joint>_<index>_<pct>pct.png
   ```

   Pass the rest render (plus the thumbnail path) to `validate-assembly`, and
   each joint's sweep frames to `validate-joints`. Treat their verdicts as
   diagnostic opinions; the objective oracle remains the evaluator.
7. Correct parameters with `update_joint()`, `reparent_joint()`,
   `set_joint_world_frame()`, `set_link_visual_pose()`, `update_link_parts()`,
   or the rename/delete APIs; all of them work in a valid partial forest,
   so fix a wrong parent by reparenting instead of deleting and re-creating
   the joint. Re-audit, reset to q=0, then call `export_urdf(OUTPUT_DIR)`.

A deterministic SDK rejection comes back from the sim tool as a structured
result `{"ok": false, "error": {"code": ..., "operation": ..., "message": ...}}`;
it is not an engine failure. Fix the offending arguments and retry.
Infrastructure failures (API/context, sim worker startup/socket, or missing
injected globals) are training-system failures: write the reason to
`OUTPUT_DIR/FAILED.txt`, stop, and report; do not retry or work around them.

## Coordinate contract

The source OBJ vertex frame is preserved exactly. At all joint values:

```text
T_world_mesh(q) = T_world_link(q) @ T_link_visual
T_world_child(q) = T_world_parent(q) @ T_joint_origin @ T_joint_motion(q)
```

The root link frame is world identity. A joint origin locates the child link
frame at q=0; for a revolute joint this is normally the hinge frame, not the
mesh center. `set_link_visual_pose()` accepts the desired mesh WORLD pose and
computes `T_link_visual` safely. It moves only that visual mesh; it does not
move child link frames. To reposition a subtree, update its incoming joint
with `set_joint_world_frame()`.

q=0 must reproduce the reference pose. Closed doors, lids, and drawers
normally use a one-sided range whose endpoint is zero. Normalize the complete
assembly with `ground()` rather than applying per-part offsets.

## Hard rules

1. Never write inside `INPUT_DIR` or use external datasets.
2. Never invoke the simulation backend outside its MCP tool.
3. Never edit or copy code under `ASSEMBLY_SDK_DIR`.
4. Never access a part by loader-internal name; use returned stable handles.
5. Never reuse a part handle after `clear_scene()`; re-import and rebuild maps.
6. Never hand-write FK, URDF XML, OBJ export, or render setup.
7. Never export unless `assembly.audit(require_complete_tree=True)["ok"]`.
8. Keep the evolvable skill focused on inference policy, geometry heuristics,
   and validators—not deterministic mechanics.
