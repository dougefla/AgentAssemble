# Frozen Assembly SDK

`evo/assembly_sdk` is the versioned deterministic runtime beneath the
evolvable assembly skill. It runs pure-Python in the host sim-worker process
(`host_runtime.py`): meshes are parsed from the source OBJs, durable state
lives in `$OUTPUT_DIR/.assembly_state.json`, and rendering goes through the
lightphyx renderer. The training loop injects the SDK's absolute path as
`ASSEMBLY_SDK_DIR`; the sim worker preloads it at session start, and the
explicit loader line remains supported (re-executing it is idempotent):

```python
exec(open(f"{ASSEMBLY_SDK_DIR}/runtime.py").read())
```

The resulting `assembly` object owns coordinate conventions, stable part
identity, persistent assembly state, graph invariants, FK, pose conversions,
grounding, rendering, audits, material-preserving OBJ/URDF export, and
structured errors. Generator file-tool write scope is limited to the per-round
skill, and the training runtime snapshots/restores these SDK sources around a
Generator sim session; an attempted mutation fails that Generator round.

Compatibility follows semantic versioning through `assembly.version`. Scene
state has its own `schema_version` and is rejected explicitly when unsupported.
SDK 1.1 keeps scene schema v1 and strengthens transactional rollback, error
classification, identity caching, and material namespaces. SDK 1.1.1 also
validates partial forests before FK and defensively rejects graph revisits, so
corrupt cyclic scene state fails fast instead of hanging preview operations.
`render_preview()` now reports forest `roots` as the authoritative partial-
assembly diagnostic; the SDK 1.1.0 `unposed_links` key remains for compatibility.
SDK 1.1.2 fixes motion visualization: `render_joint_sweep()` fits one camera to
the union envelope of every q pose, then holds camera location, rotation, and
orthographic scale fixed across all frames. Returned frame metadata includes
the exact camera contract for regression tests and downstream reports.
SDK 1.2 adds a persistent positive uniform scale per imported part for the
ArtVIP Assembly 115 anonymous-part benchmark. `set_part_scale()` changes this
semantic scale without mutating the source mesh datablock; inspect, ground,
collision, preview and sweep consume the scaled object transform, while export
bakes the factor into deterministic link meshes. Export also writes
`parts_map.json` contract v2 with each source path and scale so downstream
rendering exactly reproduces the candidate instead of silently reloading unit
cube source geometry.
SDK 1.2.1 makes incremental authoring match the agent workflow:
`set_link_visual_pose()` accepts a valid partial forest, and
`set_joint_world_frame()` accepts one as soon as its joint exists. Cycles and
other structural corruption still fail immediately; `ground()`, complete
`audit()`, motion sweeps, and export continue to require one connected tree.
SDK 1.2.2 hardens fresh-call ergonomics for low-reasoning executors. Every
runtime load exposes a refreshed read-only `parts` snapshot, identical
`import_parts(..., replace=False)` calls are idempotent handle lookups, and
`set_link_visual_pose()` accepts `world_rpy` either positionally or by keyword.
SDK 1.2.3 makes common semantic-correction operations atomic: registering the
same isolated rigid group under a semantic name renames its temporary link,
`update_link_parts()` can exchange the complete groups of two isolated links,
and `set_joint_world_frame()` accepts `world_axis` positionally or by keyword.
SDK 1.2.4 moves the same contract to the pure-Python sim-worker runtime and
completes the agent-facing error boundary: deterministic
`AssemblyError` contract rejections are returned by the sim worker as
structured, recoverable tool results instead of being mislabeled as engine
tracebacks. Actual simulation, transport, renderer, and filesystem failures
remain blocking training-system errors.

## Scope boundary

The exported URDF is an assembly/evaluation artifact: it contains links,
visual meshes, and articulated joints. It intentionally does **not** synthesize
`<inertial>` or `<collision>` elements. Add physically calibrated mass,
inertia, and collision geometry before using an output in a dynamics simulator
such as PyBullet, Isaac, or MuJoCo.

Material names and texture destinations are namespaced by stable part handle,
so identically named assets from different source directories cannot overwrite
one another. Missing MTL/texture references are returned as export warnings.

The readiness gates are:

```bash
python3 -m pytest -q
python3 scripts/preflight_training_system.py
python3 scripts/preflight_assembly_sdk_contracts.py
python3 scripts/preflight_assembly_sdk.py --dataset "$EVO_DATASET_ROOT" --samples 3
```

The contract preflight is model-free and exercises rollback, structured errors,
partial rendering, revision behavior, correction APIs, resource cleanup, and
material namespaces in a real sim worker. The last command checks representative
ArtVIP tasks for source-coordinate preservation, stable handle recovery,
grounding idempotence, deterministic export both within and across worker
processes, independent q=0 FK replay, and headless rendering.
