# Assembly SDK action reference

The sim worker preloads the frozen SDK: `assembly` is already bound in the
persistent session. Re-executing the loader is optional and idempotent:

```python
exec(open(f"{ASSEMBLY_SDK_DIR}/runtime.py").read())
```

All arguments are explicit. Calls either return JSON-safe values or raise a
structured `AssemblyError` whose code is one of: `invalid_argument`,
`unknown_handle`, `ambiguous_geometry`, `invariant_violation`,
`sim_runtime`, or `io_failure`. Wrong argument types (for example a list
where a link name is expected) are reported as `invalid_argument`, never as
a raw Python traceback. For exploratory probes,
`assembly.safe_call("method_name", keyword=value)` always returns a structured
`{"ok": ...}` dictionary.

The signatures below preserve Python's argument order and keyword-only
markers exactly (default literals are compacted for readability). Every
argument after `*` must be passed by name. This file is the frozen SDK
contract: the Generator may improve policy and tools, but must not edit
`ACTIONS.md` or restate an SDK signature differently elsewhere.

## Core API

| Method | Contract |
|---|---|
| `clear_scene()` | Remove loaded parts and reset SDK state. This invalidates every previously returned part handle; call `import_parts()` again and rebuild handle maps before any further part operation. |
| `import_parts(INPUT_DIR, *, replace=True)` | Import all OBJs, preserve source vertices exactly, and return stable handles such as `part:part_0.obj`. Repeating the identical directory with `replace=False` is a read-only idempotent handle lookup. |
| `inspect_parts()` | Return handle, label, vertex count, and local/world AABBs. |
| `set_part_scale(handle, scale)` | Set a finite positive uniform semantic scale for one anonymized part. The value persists and is used by every geometry, render, and export path. |
| `state()` | Detached JSON-safe state snapshot for inspection only. |
| `create_link(name, parts, *, visual_xyz=(0,0,0), visual_rpy=(0,0,0))` | Register one or more part handles as a rigid link. Re-registering the exact group owned by one isolated temporary link atomically renames it and keeps that link's visual pose; unsafe duplicate ownership is rejected. |
| `update_link_parts(name, parts)` | Correct a rigid grouping atomically without rebuilding the tree. If another isolated link owns exactly the requested group, the two links exchange groups atomically. |
| `rename_link(name, new_name)` / `delete_link(name)` | Rename safely, or delete an isolated link after its incident joints are removed. |
| `create_joint(name, joint_type, parent, child, origin_xyz, *, origin_rpy=(0,0,0), axis="Z", limits=None)` | Add FIXED/REVOLUTE/CONTINUOUS/PRISMATIC joint; prevents multiple parents, cycles, zero axes, and invalid limits. FIXED joints validate `axis` but always store Z and accept no limits. |
| `update_joint(name, *, joint_type=..., origin_xyz=..., origin_rpy=..., axis=..., limits=...)` | Update joint parameters atomically while preserving the current q; meaningless limits/axis arguments are rejected. |
| `rename_joint(name, new_name)` / `delete_joint(name)` | Rename or remove a joint while keeping joint-position state consistent. |
| `reparent_joint(name, new_parent)` | Change topology while preserving the child's complete q=0 world frame; valid partial forests are supported. Reparenting onto a descendant is rejected as a cycle. |
| `set_joint_world_frame(name, world_origin, world_axis=None)` | Convert an intuitive q=0 world pivot/axis to the correct parent/joint-local values. Axis may be positional or keyword (rejected for FIXED joints); valid partial forests are supported during incremental authoring. |
| `set_link_visual_pose(name, world_xyz, world_rpy=(0,0,0))` | Solve the link-relative visual transform from a desired q=0 WORLD mesh pose. Rotation may be positional or keyword, for example `world_rpy=[rx, ry, rz]`. Child link frames do not move; valid partial forests are supported. |
| `bake_current_pose_as_rest(name, *, allow_nonzero_joint_positions=False)` | Bake a displayed visual pose; refuses nonzero q by default to prevent accidental motion-in-rest. |
| `set_joint_positions({name: q})` | Validate limits, run FK, and update the scene. |
| `reset_joints()` | Set every joint to q=0 and reapply FK. |
| `ground(z=0.0)` | Translate the complete q=0 assembly to the ground without changing articulation or the current q values. Idempotent; measurements report `measured_at="q=0"`. |
| `audit(*, require_complete_tree=True, verify_source_vertices=True)` | Objective structural/runtime audit; returns `ok`, counts, and issues. |
| `collision_aabbs(*, epsilon=1e-5)` | Fast broad-phase link-pair AABB overlaps at the current pose. |
| `render_preview(path, *, azimuth_deg=45, elevation_deg=25, resolution=512, views=2)` | Deterministic framed PNG, including valid partial forests; appends `.png` when needed and reports every forest root. Multiple roots mean the assembly is not connected yet. `azimuth_deg` 0 views the front (camera on −Y), increasing counter-clockwise from above; 45 = front-right, 225 = back-left. **`views=2` (default) also renders the opposite side as `<stem>_az<NNN>.png`; every path is listed in the returned `views`.** The dataset has no shared front convention, so one fixed camera shows the featureless back of some objects — judge from all returned views, not just `path`. |
| `render_joint_sweep(joint, output_dir, *, steps=4, azimuth_deg=45, elevation_deg=25, resolution=512, views=2)` | Render `steps` evenly spaced legal poses as `<output_dir>/<joint>_<index>_<pct>pct.png` (for `steps=4`: `000`, `033`, `067`, `100`) with one fixed camera fitted to the union motion envelope; restores q=0 even if rendering fails. **`views=2` (default) renders every pose from the opposite side too, as `<joint>_<index>_<pct>pct_az<NNN>.png` — a single fixed azimuth leaves some joints occluded behind the body, where the frames are identical and motion direction cannot be judged at all.** The returned list is grouped by camera — all poses for the primary camera first, then the same poses for each extra camera — and every entry carries its `azimuth_deg`; each camera is fixed across its own frames. Like `render_preview`, `output_dir` must stay inside `OUTPUT_DIR` (relative paths resolve against it). |
| `export_urdf(OUTPUT_DIR, *, history=False)` | Reset q=0, re-audit, then write deterministic URDF and namespaced material/texture assets; `meshes/` is staged and swapped in before the URDF is replaced, so a failed export never leaves a URDF pointing at missing meshes. Returns actionable warnings. |

## Example complete call

```python
exec(open(f"{ASSEMBLY_SDK_DIR}/runtime.py").read())
parts = assembly.import_parts(INPUT_DIR)
by_label = {p["label"]: p["handle"] for p in parts}

assembly.create_link("body", by_label["part_0"])
assembly.create_link("door", by_label["part_1"])
assembly.create_joint(
    "door_joint", "REVOLUTE", "body", "door",
    origin_xyz=[0.42, 0.0, 0.55], axis="Z", limits=[0.0, 1.57],
)
assembly.set_link_visual_pose("body", [0.0, 0.0, 0.0])
assembly.set_link_visual_pose("door", [0.0, 0.0, 0.0])
assembly.ground()
print(assembly.audit())
assembly.render_preview(f"{OUTPUT_DIR}/render_assembled.png")
assembly.export_urdf(OUTPUT_DIR)
```
