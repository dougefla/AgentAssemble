"""lightphyx backend for the oracle renderer — drop-in for render_pose().

Replaces the one-Blender-process-per-PNG path in render_motion.py with a
persistent light-3js-phyx server (three.js + Rapier, headless Chromium).
The camera framing mirrors BLENDER_SCRIPT exactly (same azimuth/elevation
formula, same distance = AABB-diagonal * 1.8, same 50mm-lens FOV), so frames
are composition-compatible with the Blender backend. Lighting differs by
design: lightphyx renders uniform-ambient (albedo), Blender used a sun lamp —
GT and pred are always rendered by the same backend so comparisons stay fair.

Mechanism: lightphyx cannot place bare meshes, so each pose is flattened into
a temporary URDF — one base link plus one fixed-jointed child link per mesh,
with the world transform (M_link @ M_vis, FK done host-side by build_pose)
baked into the joint origin. Loaded, rendered via server-side PNG save,
unloaded. Mesh paths are written relative to the temp URDF so they resolve
under the server's /files/<abspath> static route; OBJ/MTL/textures load from
their original locations (mesh templates are cached by URL server-side).

Sims (one env each; LIGHTPHYX_CLI locates the server, see lightphyx docs)
are leased from a small per-process pool: a lease is exclusive, so scene
mutations on one env never interleave. Sim workers render one frame at a
time and keep the default of one Sim; the training loop calls
configure_render_pool(ORACLE_RENDER_SIMS) so the oracle's asyncio.to_thread
renders of different tasks proceed in parallel instead of queueing behind a
single env.
"""
from __future__ import annotations

import atexit
import inspect
import math
import os
import threading
import uuid
from pathlib import Path
from xml.sax.saxutils import quoteattr

# Blender camera: 50mm lens on a 36mm sensor, square render → vertical FOV.
_FOV_DEG = math.degrees(2 * math.atan(18.0 / 50.0))  # 39.5978°
_RES = 512
_ELEV_DEG = 25.0
_DIST_FACTOR = 1.8

# Leased Sims. _IDLE holds connected Sims nobody is using; _LIVE counts every
# Sim that exists (idle or leased) so the pool never exceeds _MAX_SIMS.
_POOL = threading.Condition()
_IDLE: list = []
_LIVE = 0
_MAX_SIMS = 1
# Bumped by _shutdown(): Sims leased before a shutdown are discarded on return
# (their server is being recycled), never handed out again.
_GENERATION = 0
_ATEXIT_REGISTERED = False

# Concurrent oracle renders in the training-loop process. Every one holds an
# env on the shared pool, so the loop adds this to the pool capacity.
ORACLE_RENDER_SIMS = max(1, int(os.environ.get("EVO_ORACLE_RENDER_SIMS", "4")))
# Per-call deadline. The SDK default (120 s) let one hung page block a
# render — and, on the old close path, its cleanup — for minutes.
_RPC_TIMEOUT = 45.0
_AABB_CACHE: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}


class LightphyxUnavailable(RuntimeError):
    """lightphyx SDK missing or its server failed to launch."""


# evo only renders flattened, fixed-joint scenes: a kinematic env skips the
# per-load Rapier collider build and makes joint writes immediate.
_ENV_KWARGS = {"physics": False}


def _supported(callable_, **kwargs):
    """Keep only keyword arguments the installed SDK function accepts."""
    try:
        parameters = inspect.signature(callable_).parameters.values()
    except (TypeError, ValueError):
        return {}
    names = {parameter.name for parameter in parameters}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in names}


def _launch_local(lp):
    """Launch a private server with recovery controls when the SDK has them.

    Older lightphyx releases expose only ``launch(headless, out_dir, pool)``;
    passing unknown keywords would raise before Chromium even starts, so the
    extras are filtered against the live signature.
    """
    return lp.launch(**_supported(
        lp.launch, rpc_timeout=_RPC_TIMEOUT, startup_attempts=3, **_ENV_KWARGS,
    ))


def configure_render_pool(sims: int) -> None:
    """Allow up to `sims` concurrent renders (one env each) in this process."""
    global _MAX_SIMS
    with _POOL:
        _MAX_SIMS = max(1, int(sims))
        _POOL.notify_all()


def _new_sim():
    """Connect one new Sim: to the shared pool if there is one, else launch."""
    try:
        import lightphyx as lp
    except ImportError as e:
        raise LightphyxUnavailable(
            "EVO_RENDER_BACKEND=lightphyx but the `lightphyx` package is not "
            "installed. `pip install -e <light-3js-phyx>/sdk/python` (and set "
            "LIGHTPHYX_CLI if the repo isn't discoverable), or set "
            "EVO_RENDER_BACKEND=blender."
        ) from e
    shared_url = os.environ.get("EVO_LIGHTPHYX_URL")
    if shared_url:
        return lp.connect(shared_url, **_supported(
            lp.connect, rpc_timeout=_RPC_TIMEOUT, **_ENV_KWARGS,
        ))
    # Standalone SDK/preflight calls have no experiment-owned pool.
    # Close and retry a transient local Chromium startup rather than
    # surfacing the infrastructure burst to the assembly agent. Older
    # installed SDKs lack these keywords; _launch_local keeps their
    # one-shot API usable instead of failing on signature mismatch.
    return _launch_local(lp)


def _lease():
    """Take an idle Sim, or connect a new one while under the cap, else wait."""
    global _LIVE, _ATEXIT_REGISTERED
    with _POOL:
        while not _IDLE and _LIVE >= _MAX_SIMS:
            _POOL.wait()
        if _IDLE:
            return _IDLE.pop(), _GENERATION
        _LIVE += 1
        generation = _GENERATION
        if not _ATEXIT_REGISTERED:
            atexit.register(_shutdown)
            _ATEXIT_REGISTERED = True
    try:
        return _new_sim(), generation
    except BaseException:
        with _POOL:
            _LIVE -= 1
            _POOL.notify()
        raise


def _give_back(sim, generation: int, healthy: bool) -> None:
    global _LIVE
    with _POOL:
        keep = healthy and generation == _GENERATION
        if keep:
            _IDLE.append(sim)
        else:
            _LIVE -= 1
        _POOL.notify()
    if not keep:
        _discard(sim, graceful=healthy)


def _discard(sim, *, graceful: bool) -> None:
    """Drop a Sim. A broken one is NOT asked to destroy its env first.

    sim.close() sends destroy_envs and waits for the reply, which after a
    transport failure is exactly what can hang. Closing the socket is enough:
    the server reclaims every env of a dropped connection (light-3js-phyx
    d02d91a). close() still runs afterwards — it fails fast on the closed
    socket — so a privately launched server is shut down too.
    """
    if not graceful:
        client = getattr(sim, "_client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    try:
        sim.close()
    except Exception:
        pass


def _shutdown() -> None:
    """Close every idle Sim; Sims currently leased are dropped on return."""
    global _GENERATION, _LIVE
    with _POOL:
        _GENERATION += 1
        idle = list(_IDLE)
        _IDLE.clear()
        _LIVE -= len(idle)
        _POOL.notify_all()
    for sim in idle:
        _discard(sim, graceful=True)


def _local_aabb(obj_path: str):
    """Min/max corners of the OBJ's own vertices (cached by path)."""
    cached = _AABB_CACHE.get(obj_path)
    if cached is not None:
        return cached
    mn = [math.inf] * 3
    mx = [-math.inf] * 3
    with open(obj_path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                for i in range(3):
                    v = float(parts[i + 1])
                    if v < mn[i]:
                        mn[i] = v
                    if v > mx[i]:
                        mx[i] = v
    if mn[0] is math.inf:
        result = None  # no vertices — degenerate OBJ
    else:
        result = (tuple(mn), tuple(mx))
    _AABB_CACHE[obj_path] = result
    return result


def _mat_to_xyz_rpy(M):
    """Decompose a rigid 4x4 (row-major lists) into URDF xyz + rpy (ZYX)."""
    xyz = (M[0][3], M[1][3], M[2][3])
    # rpy_to_mat in render_motion.py builds R = Rz(y) @ Ry(p) @ Rx(r):
    #   R[2][0] = -sin(p)
    sp = -M[2][0]
    sp = max(-1.0, min(1.0, sp))
    p = math.asin(sp)
    if abs(sp) < 1.0 - 1e-9:
        r = math.atan2(M[2][1], M[2][2])
        y = math.atan2(M[1][0], M[0][0])
    else:  # gimbal lock: pitch = ±90°, roll/yaw degenerate — fold into roll
        r = math.atan2(-M[1][2], M[1][1])
        y = 0.0
    return xyz, (r, p, y)


def _split_uniform_scale(M):
    """Return (rigid_matrix, positive_uniform_scale) for ``M = rigid @ S``."""
    scales = [
        math.sqrt(sum(float(M[row][column]) ** 2 for row in range(3)))
        for column in range(3)
    ]
    if min(scales) <= 1e-12:
        raise ValueError(f"mesh transform has degenerate scale: {scales}")
    scale = sum(scales) / 3.0
    if max(abs(value - scale) for value in scales) > max(1e-6, scale * 1e-6):
        raise ValueError(f"mesh transform has non-uniform scale: {scales}")
    rigid = [list(row) for row in M]
    for row in range(3):
        for column in range(3):
            rigid[row][column] = float(M[row][column]) / scale
    return rigid, scale


def _with_uniform_scale(M, scale: float):
    """Compose a positive uniform scale on the right: ``M @ diag(s,s,s,1)``."""
    value = float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"mesh scale must be positive and finite, got {scale!r}")
    scaled = [list(row) for row in M]
    for row in range(3):
        for column in range(3):
            scaled[row][column] = float(M[row][column]) * value
    return scaled


def _mesh_entries(links, poses, base_dir: Path, parts_map: dict | None):
    """Yield (world_matrix, mesh_abs_path) per renderable mesh, mirroring
    BLENDER_SCRIPT's source policy: parts_map first, mesh_path fallback.

    ``parts_map`` v2 entries carry per-part uniform semantic scale; legacy
    link-keyed string lists remain readable with implicit scale 1.
    """
    parts_map = parts_map or {}
    parts_by_link = (
        parts_map.get("links", {})
        if parts_map.get("version") == 2
        else parts_map
    )
    entries = []
    for link_name, info in links.items():
        pose = poses.get(link_name)
        if pose is None:
            continue
        vx, vy, vz = info["visual_xyz"]
        rr, pp, yy = info["visual_rpy"]
        cr, sr = math.cos(rr), math.sin(rr)
        cp, sp = math.cos(pp), math.sin(pp)
        cy, sy = math.cos(yy), math.sin(yy)
        M_vis = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, vx],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, vy],
            [-sp, cp * sr, cp * cr, vz],
            [0.0, 0.0, 0.0, 1.0],
        ]
        M = [
            [sum(pose[i][k] * M_vis[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)
        ]
        src_entries = list(parts_by_link.get(link_name) or [])
        if not src_entries:
            mp = info.get("mesh_path")
            if mp:
                src_entries = [{
                    "path": os.path.join(str(base_dir), mp),
                    "scale": 1.0,
                }]
        for source in src_entries:
            if isinstance(source, dict):
                raw_path = source.get("path")
                scale = source.get("scale", 1.0)
            else:
                raw_path = source
                scale = 1.0
            if not isinstance(raw_path, (str, os.PathLike)):
                continue
            full = Path(raw_path).expanduser()
            if not full.is_absolute():
                full = base_dir / full
            full = full.resolve()
            if full.exists():
                entries.append((_with_uniform_scale(M, scale), str(full)))
    return entries


def _world_aabb(entries):
    """AABB over the transformed local-bbox corners of every mesh — the same
    conservative bound Blender computes from `obj.bound_box`, so the camera
    distance matches the Blender backend frame-for-frame."""
    mn = [math.inf] * 3
    mx = [-math.inf] * 3
    for M, path in entries:
        local = _local_aabb(path)
        if local is None:
            continue
        (lx, ly, lz), (hx, hy, hz) = local
        for cx in (lx, hx):
            for cy in (ly, hy):
                for cz in (lz, hz):
                    for i in range(3):
                        w = M[i][0] * cx + M[i][1] * cy + M[i][2] * cz + M[i][3]
                        if w < mn[i]:
                            mn[i] = w
                        if w > mx[i]:
                            mx[i] = w
    if mn[0] is math.inf:
        return None
    return tuple(mn), tuple(mx)


def _write_flat_urdf(path: Path, entries) -> None:
    """One base link + one fixed-jointed child per mesh, world pose baked in.
    Mesh filenames are relative to the URDF so the lightphyx static server
    resolves them back to their absolute /files/<path> locations."""
    urdf_dir = path.parent
    lines = ['<?xml version="1.0"?>', '<robot name="oracle_pose">',
             '  <link name="base"/>']
    for i, (M, mesh_path) in enumerate(entries):
        rigid, scale = _split_uniform_scale(M)
        (x, y, z), (r, p, w) = _mat_to_xyz_rpy(rigid)
        rel = os.path.relpath(mesh_path, urdf_dir)
        lines += [
            f'  <link name="m{i}">',
            '    <visual>',
            '      <geometry>',
            f'        <mesh filename={quoteattr(rel)} '
            f'scale="{scale:.9g} {scale:.9g} {scale:.9g}"/>',
            '      </geometry>',
            '    </visual>',
            '  </link>',
            f'  <joint name="j{i}" type="fixed">',
            f'    <origin xyz="{x:.9g} {y:.9g} {z:.9g}" rpy="{r:.9g} {p:.9g} {w:.9g}"/>',
            '    <parent link="base"/>',
            f'    <child link="m{i}"/>',
            '  </joint>',
        ]
    lines.append('</robot>')
    path.write_text("\n".join(lines))


def _transport_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError, OSError]
    try:
        import lightphyx as lp
        errors.append(lp.RpcError)
    except Exception:  # pragma: no cover - SDK missing is reported elsewhere
        pass
    return tuple(errors)


_TRANSPORT_ERRORS = _transport_errors()


# CAD-style feature edges on the RGB pass. Every consumer of these frames is a
# vision model (the executor's validators and the oracle judges), and the
# failure they make most is misreading a flat face: a drawer front taken for a
# panel, a seam missed, a hinge side guessed. Edges make those readable at
# 512px. Segmentation and depth are unaffected. EVO_RENDER_OUTLINE=0 restores
# the flat shading that runs 1-3 were judged under.
_OUTLINE = os.environ.get("EVO_RENDER_OUTLINE", "1").strip().lower() not in ("0", "false", "no", "off")


def _capture_with(sim, tmp_urdf: Path, tag: str, cam_pos, lookat,
                  fov_deg: float, resolution: int):
    """Load, capture rgb+seg, and unload one flattened scene on `sim`.

    Returns the RGB array. Nothing is saved server-side: `save=` made the one
    hub process PNG-encode both passes of every worker's frames (and write a
    seg PNG + labels JSON that were deleted right away); the caller encodes
    the RGB PNG in its own process instead.
    """
    robot = sim.load_urdf(str(tmp_urdf), name=tag)
    try:
        cam = sim.camera(pos=tuple(cam_pos), lookat=tuple(lookat),
                         fov=fov_deg, size=(resolution, resolution))
        result = cam.capture(passes=("rgb", "seg"), outline=_OUTLINE)
        seg = result.get("seg") if isinstance(result, dict) else None
        if seg is None or not bool((seg > 0).any()):
            raise RuntimeError(
                "LightPhyx rendered no foreground geometry "
                "(segmentation contains only background id 0)"
            )
        return result["rgb"]
    finally:
        sim.remove_robot(robot.name)


def _write_png(rgb, out_path: Path) -> None:
    """Atomically write an (H, W, 3) uint8 array as a PNG."""
    from PIL import Image  # pillow ships with the lightphyx SDK

    tmp = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        Image.fromarray(rgb, "RGB").save(tmp, format="PNG", compress_level=6)
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)


def render_scene(entries, out_path: Path, *, cam_pos, lookat,
                 fov_deg: float = _FOV_DEG, resolution: int = _RES) -> None:
    """Render (world_matrix, mesh_path) entries to a PNG at out_path.

    Shared low-level renderer: the oracle path and the assembly SDK's
    render_preview both flatten their scene into entries and call this.
    The RGB pass is saved, while a simultaneous segmentation pass is used as
    a geometry-presence contract.  This catches successful-looking transport
    responses whose render contains only background; RGB alone cannot do that
    reliably because an unmaterialed white mesh on the white background is a
    valid (but visually blank) beauty pass.

    Raises on failure (LightphyxUnavailable, transport error, or blank
    segmentation)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"__lp_{uuid.uuid4().hex[:12]}"
    tmp_urdf = out_path.parent / f"{tag}.urdf"
    try:
        _write_flat_urdf(tmp_urdf, entries)
        # A dead page/socket (renderer crash, server restart) must not poison
        # the Sim for the rest of the process: drop that connection on a
        # transport failure and retry once on a fresh one.
        for attempt in (0, 1):
            try:
                sim, generation = _lease()
            except _TRANSPORT_ERRORS as exc:  # server gone / refusing
                if attempt:
                    raise LightphyxUnavailable(
                        f"lightphyx transport failed twice: {exc}"
                    ) from exc
                continue
            healthy = False
            try:
                rgb = _capture_with(
                    sim, tmp_urdf, tag, cam_pos, lookat, fov_deg, resolution,
                )
                healthy = True
                break
            except _TRANSPORT_ERRORS as exc:
                if attempt:
                    raise LightphyxUnavailable(
                        f"lightphyx transport failed twice: {exc}"
                    ) from exc
            except Exception:
                healthy = True  # a render-level failure; the connection is fine
                raise
            finally:
                _give_back(sim, generation, healthy)
        _write_png(rgb, out_path)
    finally:
        tmp_urdf.unlink(missing_ok=True)


def sweep_frame(links, joints, base_dir: Path, parts_map: dict | None,
                joint_qs: list[dict]) -> tuple[tuple[float, float, float], float] | None:
    """One camera frame (center, extent) covering every pose in `joint_qs`.

    A camera fitted per pose follows the moving part: as a door swings the
    bounding box grows, the camera backs off, and the frames zoom and drift.
    A judge — or a viewer of the stitched video — then sees camera motion on
    top of joint motion. Fit the union envelope once and render every pose
    of the sweep through it.
    """
    from evo.oracle.render_motion import build_pose  # lazy: avoids cycle

    mins: list[float] | None = None
    maxs: list[float] | None = None
    for joint_q in joint_qs:
        entries = _mesh_entries(links, build_pose(joints, links, joint_q), base_dir, parts_map)
        aabb = _world_aabb(entries)
        if not entries or aabb is None:
            continue
        mn, mx = aabb
        mins = list(mn) if mins is None else [min(a, b) for a, b in zip(mins, mn)]
        maxs = list(mx) if maxs is None else [max(a, b) for a, b in zip(maxs, mx)]
    if mins is None or maxs is None:
        return None
    center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    extent = math.sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(3)))
    return center, (extent if extent > 0 else 1.0)


def render_pose_lightphyx(out_dir: Path, links, joints, joint_q: dict,
                          base_dir: Path, out_path: Path,
                          azimuth: float = 135.0,
                          parts_map: dict | None = None,
                          frame: tuple[tuple[float, float, float], float] | None = None) -> bool:
    """Signature-identical replacement for render_motion.render_pose().

    `frame` = (center, extent) from `sweep_frame` pins the camera across a
    sweep; without it the camera is fitted to this pose alone.
    """
    from evo.oracle.render_motion import build_pose  # lazy: avoids cycle

    poses = build_pose(joints, links, joint_q)
    entries = _mesh_entries(links, poses, base_dir, parts_map)
    aabb = _world_aabb(entries)
    if not entries or aabb is None:
        print(f"  [fail] {out_path.name}: no meshes loaded")
        return False

    if frame is not None:
        center, extent = frame
    else:
        mn, mx = aabb
        center = tuple((mn[i] + mx[i]) * 0.5 for i in range(3))
        extent = math.sqrt(sum((mx[i] - mn[i]) ** 2 for i in range(3)))
    if extent <= 0:
        extent = 1.0
    el = math.radians(_ELEV_DEG)
    az = math.radians(azimuth)
    dist = extent * _DIST_FACTOR
    cam_pos = (
        center[0] + dist * math.cos(el) * math.sin(az),
        center[1] - dist * math.cos(el) * math.cos(az),
        center[2] + dist * math.sin(el),
    )
    try:
        render_scene(entries, Path(out_path), cam_pos=cam_pos, lookat=center)
        return True
    except LightphyxUnavailable:
        raise
    except Exception as e:
        print(f"  [fail] {out_path.name}: {type(e).__name__}: {str(e)[-200:]}")
        return False
