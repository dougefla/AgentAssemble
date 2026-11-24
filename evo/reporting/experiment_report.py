#!/usr/bin/env python3
"""Generate a portable HTML report for an evo-assemble training run.

Metrics and videos are derived deterministically from persisted artifacts. A
separate Responses agent produces a short per-round skill-diff summary; that
summary is persisted in the report and never changes experiment selection or
reads ground truth.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import difflib
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPORT_MARKER = ".evo-assemble-report"
DEFAULT_SKILL_SUMMARY_MODEL = "gpt-6.1-sol"
DEFAULT_SKILL_SUMMARY_EFFORT = "high"
_ROUND_RE = re.compile(r"round_(\d+)$")
_MOTION_RE = re.compile(r"^(?P<name>.+)_(?P<step>\d{2})_\d{3}pct(?P<cam>_az\d{3})?\.png$")
_JOURNAL_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?\*\*(Observed|Investigated|Changed|Verified|Unresolved)\*\*\s*$"
    r"|^\s*#{1,6}\s*(Observed|Investigated|Changed|Verified|Unresolved)\s*$",
    re.IGNORECASE,
)


_FIXED_CAMERA_BLENDER_SCRIPT = r'''import bpy, json, math, mathutils, os
cfg = json.loads(open(os.environ["EVO_REPORT_RENDER_CFG"]).read())
base_dir = cfg["base_dir"]
links = cfg["links"]
frames = cfg["frames"]

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

objects_by_link = {}
for link_name, info in links.items():
    mesh_path = info.get("mesh_path")
    if not mesh_path:
        objects_by_link[link_name] = []
        continue
    full = os.path.join(base_dir, mesh_path)
    if not os.path.exists(full):
        objects_by_link[link_name] = []
        continue
    before = {obj.as_pointer() for obj in bpy.data.objects}
    try:
        bpy.ops.wm.obj_import(filepath=full, up_axis='Z', forward_axis='NEGATIVE_Y')
    except Exception:
        objects_by_link[link_name] = []
        continue
    objects_by_link[link_name] = [
        obj for obj in bpy.data.objects
        if obj.as_pointer() not in before and obj.type == 'MESH'
    ]

def matrix_for(link_name, pose):
    info = links[link_name]
    vx, vy, vz = info["visual_xyz"]
    vr, vp, vyaw = info["visual_rpy"]
    visual = (
        mathutils.Matrix.Translation((vx, vy, vz))
        @ mathutils.Euler((vr, vp, vyaw), 'XYZ').to_matrix().to_4x4()
    )
    return mathutils.Matrix(pose) @ visual

minimum = mathutils.Vector((math.inf, math.inf, math.inf))
maximum = mathutils.Vector((-math.inf, -math.inf, -math.inf))
for frame in frames:
    poses = frame["poses"]
    for link_name, objects in objects_by_link.items():
        pose = poses.get(link_name)
        if pose is None:
            continue
        world = matrix_for(link_name, pose)
        for obj in objects:
            obj.matrix_world = world
            for corner in obj.bound_box:
                point = world @ mathutils.Vector(corner)
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], point[axis])
                    maximum[axis] = max(maximum[axis], point[axis])

if not all(math.isfinite(v) for v in minimum):
    raise RuntimeError("no renderable URDF meshes")

# The camera is computed ONCE from the union motion envelope and is never
# touched inside the frame loop below.
center = (minimum + maximum) * 0.5
radius = max((maximum - minimum).length * 0.75, 0.5)
azimuth = math.radians(float(cfg.get("azimuth_deg", 45.0)))
elevation = math.radians(float(cfg.get("elevation_deg", 25.0)))
direction = mathutils.Vector((
    math.cos(elevation) * math.cos(azimuth),
    math.cos(elevation) * math.sin(azimuth),
    math.sin(elevation),
))

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 4
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = int(cfg.get("resolution", 512))
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False

world_data = scene.world or bpy.data.worlds.new("EvoReportWorld")
scene.world = world_data
world_data.use_nodes = True
background = world_data.node_tree.nodes.get("Background")
if background:
    background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs[1].default_value = 0.8

camera_data = bpy.data.cameras.new("EvoReportFixedCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = radius * 2.4
camera = bpy.data.objects.new("EvoReportFixedCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = center + direction * (radius * 3.0)
camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
scene.camera = camera

for index, location in enumerate((
    center + mathutils.Vector((radius, -radius, radius * 2.0)),
    center + mathutils.Vector((-radius, radius, radius)),
)):
    light_data = bpy.data.lights.new(f"EvoReportLight{index}", "AREA")
    light_data.energy = 900.0
    light_data.shape = "DISK"
    light_data.size = radius * 2.0
    light = bpy.data.objects.new(f"EvoReportLight{index}", light_data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (center - light.location).to_track_quat('-Z', 'Y').to_euler()

for frame in frames:
    poses = frame["poses"]
    for link_name, objects in objects_by_link.items():
        pose = poses.get(link_name)
        if pose is None:
            continue
        transform = matrix_for(link_name, pose)
        for obj in objects:
            obj.matrix_world = transform
    bpy.context.view_layer.update()
    scene.render.filepath = frame["path"]
    bpy.ops.render.render(write_still=True)

print(json.dumps({
    "camera_location": list(camera.location),
    "camera_rotation": list(camera.rotation_euler),
    "ortho_scale": camera_data.ortho_scale,
    "center": list(center),
    "radius": radius,
    "frames": len(frames),
}))
'''


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return default


def _round_number(path: Path) -> int | None:
    match = _ROUND_RE.match(path.name)
    return int(match.group(1)) if match else None


def _parse_rate(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.match(r"\s*(\d+)\s*/\s*(\d+)", value)
    if not match or int(match.group(2)) == 0:
        return None
    return int(match.group(1)) / int(match.group(2))


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return value or "artifact"


def _journal_sections(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return {}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        match = _JOURNAL_HEADING_RE.match(line)
        if match:
            current = (match.group(1) or match.group(2)).title()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return {
        name: "\n".join(body).strip()
        for name, body in sections.items()
        if "\n".join(body).strip()
    }


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_flatten_text(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(_flatten_text(v) for v in value.values())
    return str(value) if value is not None else ""


def _trajectory_summary(round_dir: Path) -> dict[str, Any]:
    response = _read_json(round_dir / "generator.responses.json", {}) or {}
    result = _read_json(round_dir / "generator.result.json", {}) or {}
    items = response.get("items") or []
    calls: collections.Counter[str] = collections.Counter()
    error_outputs = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            calls[str(item.get("name") or "unknown")] += 1
        elif item.get("type") == "function_call_output":
            text = _flatten_text(item.get("output")).lower()
            if any(token in text for token in (
                "traceback (most recent call last)",
                '"iserror": true',
                "tool_error",
                "blender error:",
            )):
                error_outputs += 1
    usage = response.get("usage") or result.get("usage") or {}
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = usage.get("cache_read", 0)
    input_tokens = usage.get("input_tokens", usage.get("fresh_input", 0)) or 0
    reasoning_tokens = (
        (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        or 0
    )
    return {
        "backend": response.get("backend") or result.get("backend") or "unknown",
        "model": response.get("model") or result.get("model") or "unknown",
        "provider": response.get("provider_base_url") or "unknown",
        "effort": response.get("effort") or result.get("effort") or "unknown",
        "service_tier": response.get("service_tier") or "unknown",
        "requests": usage.get("requests", usage.get("n_api_calls", len(calls))) or 0,
        "input_tokens": input_tokens,
        "cached_tokens": cached or 0,
        "output_tokens": usage.get("output_tokens", usage.get("output", 0)) or 0,
        "reasoning_tokens": reasoning_tokens,
        "tool_calls": sum(calls.values()),
        "tool_histogram": calls.most_common(),
        "tool_error_outputs": error_outputs,
        "final_output": response.get("final_output") or "",
        "files_modified": result.get("files_modified") or [],
        "error": result.get("error"),
    }


def _oracle_info(output_dir: Path) -> tuple[float | None, list[str]]:
    payload = _read_json(output_dir / "oracle.json", {}) or {}
    raw_checks = payload.get("checks") or []
    if isinstance(raw_checks, dict):
        checks = [
            {"name": name, **value}
            for name, value in raw_checks.items()
            if isinstance(value, dict)
        ]
    elif isinstance(raw_checks, list):
        checks = raw_checks
    else:
        checks = []
    scored_checks = [
        check for check in checks
        if isinstance(check, dict) and check.get("scored", True)
    ]
    if not scored_checks:
        return None, []
    passed = 0
    failed: list[str] = []
    for check in scored_checks:
        if check.get("passed"):
            passed += 1
        else:
            failed.append(str(check.get("name") or "unknown"))
    return passed / len(scored_checks), failed


def _artifact_output_dir(run_dir: Path, round_dir: Path, artifact: dict[str, Any]) -> Path:
    raw = artifact.get("output_dir")
    if raw:
        candidate = Path(raw)
        if candidate.is_dir():
            return candidate
    tag = str(artifact.get("tag") or "")
    if "/" in tag:
        task, instance = tag.split("/", 1)
        candidate = round_dir / "outputs" / task / instance
        if candidate.is_dir():
            return candidate
    return run_dir / "missing-output"


def _discover_artifacts(run_dir: Path, round_dir: Path) -> list[dict[str, Any]]:
    artifacts = _read_json(round_dir / "all_artifacts.json", []) or []
    if isinstance(artifacts, list) and artifacts:
        return [a for a in artifacts if isinstance(a, dict)]
    found: list[dict[str, Any]] = []
    outputs = round_dir / "outputs"
    if outputs.is_dir():
        for task_dir in sorted(outputs.iterdir()):
            if not task_dir.is_dir():
                continue
            for instance_dir in sorted(task_dir.iterdir()):
                if instance_dir.is_dir():
                    found.append({
                        "tag": f"{task_dir.name}/{instance_dir.name}",
                        "output_dir": str(instance_dir),
                        "oracle_verdict": "UNKNOWN",
                        "failed_oracle_checks": [],
                        "cost_summary": {},
                    })
    return found


def _find_poster(output_dir: Path) -> Path | None:
    preferred = (
        "render_assembled_crop.png",
        "render_assembled.png",
        "render_thumbnail.png",
    )
    for name in preferred:
        path = output_dir / name
        if path.is_file():
            return path
    motion = sorted(
        p for p in output_dir.glob("*_??_???pct.png")
        if not p.name.endswith("_crop.png")
    )
    return motion[-1] if motion else None


def _motion_frames_by_camera(output_dir: Path) -> dict[str, list[Path]]:
    """Executor sweep frames, ping-ponged per joint, grouped by camera.

    The SDK's render_joint_sweep writes the primary camera as
    `<joint>_<idx>_<pct>pct.png` and each extra camera with an `_az<NNN>`
    suffix; every camera is fixed across its own frames. Key "" is the
    primary camera, "_az225" the opposite side, and so on.
    """
    groups: dict[str, dict[str, list[tuple[int, Path]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for path in output_dir.glob("*.png"):
        if path.name.endswith("_crop.png"):
            continue
        match = _MOTION_RE.match(path.name)
        if match:
            cam = match.group("cam") or ""
            groups[cam][match.group("name")].append((int(match.group("step")), path))
    out: dict[str, list[Path]] = {}
    for cam, joints in groups.items():
        frames: list[Path] = []
        for name in sorted(joints):
            ordered = [path for _, path in sorted(joints[name])]
            if len(ordered) < 2:
                continue
            frames.extend(ordered)
            frames.extend(ordered[-2:0:-1])
        if frames:
            out[cam] = frames
    return out


def _motion_frames(output_dir: Path) -> list[Path]:
    return _motion_frames_by_camera(output_dir).get("", [])


def _ffmpeg_escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


_REPORT_SWEEP_AZIMUTHS = (45.0, 225.0)


def _fixed_camera_motion_frames(
    output_dir: Path,
    frame_dir: Path,
    *,
    blender_bin: str | None = None,
    steps: int = 4,
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    """Render every predicted movable joint with one immutable camera per
    azimuth, through the same lightphyx pipeline the executor and the oracle
    use — outlines, lighting and convention included.

    The camera frame is the union envelope over every pose of a joint, so
    articulation is preserved and camera translation/zoom is not. Returns
    {"": frames_at_45, "_az225": frames_at_225} ping-ponged per joint, plus
    camera metadata. `blender_bin` is accepted for compatibility and ignored:
    the Blender-era renderer produced frames a different renderer had lit,
    from a single azimuth, with no feature edges.
    """
    urdf_path = output_dir / "furniture.urdf"
    if not urdf_path.is_file():
        return {}, {}
    try:
        from evo.oracle.render_motion import parse_urdf_raw, render_pose
        from evo.oracle.render_lightphyx import sweep_frame

        links, joints = parse_urdf_raw(urdf_path)
    except Exception as error:
        return {}, {"error": str(error)}
    movable = {
        name: joint for name, joint in joints.items()
        if joint["type"] in {"REVOLUTE", "CONTINUOUS", "PRISMATIC"}
    }
    if not movable:
        return {}, {}
    parts_map: dict | None = None
    parts_map_path = output_dir / "parts_map.json"
    if parts_map_path.is_file():
        try:
            parts_map = json.loads(parts_map_path.read_text())
        except Exception:
            parts_map = None
    frame_dir.mkdir(parents=True, exist_ok=True)
    by_camera: dict[str, list[Path]] = {}
    cameras: dict[str, Any] = {"azimuths_deg": list(_REPORT_SWEEP_AZIMUTHS), "joints": {}}
    for joint_name, joint in sorted(movable.items()):
        lower, upper = float(joint["lower"]), float(joint["upper"])
        if joint["type"] == "CONTINUOUS" and lower == upper == 0.0:
            upper = 2.0 * math.pi
        poses = []
        for index in range(steps):
            positions = {name: 0.0 for name in movable}
            positions[joint_name] = lower + (upper - lower) * (index / (steps - 1))
            poses.append(positions)
        try:
            frame = sweep_frame(links, joints, output_dir, parts_map, poses)
        except Exception as error:
            return {}, {"error": str(error)}
        if frame is None:
            continue
        cameras["joints"][joint_name] = {"center": list(frame[0]), "extent": frame[1]}
        for azimuth in _REPORT_SWEEP_AZIMUTHS:
            cam = "" if azimuth == _REPORT_SWEEP_AZIMUTHS[0] else f"_az{round(azimuth) % 360:03d}"
            forward: list[Path] = []
            for index, positions in enumerate(poses):
                path = frame_dir / (
                    f"{_slug(joint_name)}_{index:02d}_"
                    f"{round(100 * index / (steps - 1)):03d}pct{cam}.png"
                )
                try:
                    ok = render_pose(frame_dir, links, joints, positions, output_dir,
                                     path, azimuth, parts_map=parts_map, frame=frame)
                except Exception as error:
                    return {}, {"error": str(error)}
                if not ok:
                    return {}, {"error": f"render failed for {path.name}"}
                forward.append(path)
            by_camera.setdefault(cam, []).extend(forward)
            by_camera[cam].extend(forward[-2:0:-1])
    return by_camera, cameras


def _encode_video(
    output_dir: Path,
    destination: Path,
    *,
    ffmpeg_bin: str,
    frames_override: list[Path] | None = None,
) -> str | None:
    frames = _motion_frames(output_dir) if frames_override is None else frames_override
    poster = _find_poster(output_dir)
    if not frames and poster is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    common = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
    ]
    video_filter = (
        "scale=640:640:force_original_aspect_ratio=decrease,"
        "pad=640:640:(ow-iw)/2:(oh-ih)/2:color=white,"
        "fps=12,format=yuv420p"
    )
    try:
        if frames:
            concat_path = destination.with_suffix(".concat.txt")
            lines: list[str] = []
            for frame in frames:
                lines.append(f"file '{_ffmpeg_escape(frame)}'")
                lines.append("duration 0.42")
            lines.append(f"file '{_ffmpeg_escape(frames[-1])}'")
            concat_path.write_text("\n".join(lines) + "\n")
            command = common + [
                "-f", "concat", "-safe", "0", "-i", str(concat_path),
                "-vf", video_filter, "-an", "-c:v", "libx264", "-crf", "24",
                "-movflags", "+faststart", str(destination),
            ]
        else:
            command = common + [
                "-loop", "1", "-i", str(poster), "-t", "2.5",
                "-vf", video_filter, "-an", "-c:v", "libx264", "-crf", "24",
                "-movflags", "+faststart", str(destination),
            ]
        subprocess.run(command, check=True, timeout=120)
        if frames:
            concat_path.unlink(missing_ok=True)
        return "joint_sweep" if frames else "static_preview"
    except (OSError, subprocess.SubprocessError):
        destination.unlink(missing_ok=True)
        return None


def _encode_fixed_camera_video(
    output_dir: Path,
    destination: Path,
    *,
    ffmpeg_bin: str,
    blender_bin: str | None,
) -> tuple[str | None, dict[str, Any], dict[str, Path]]:
    """Encode one fixed-camera video per camera; never let the camera drift.

    Order of preference:
      1. the executor's own sweep frames — the SDK renders them through one
         camera per azimuth, and they are what its validators judged;
      2. a lightphyx fixed-camera re-render (needs the render server);
      3. a still of the assembled render, which cannot drift.
    Returns (kind, camera_metadata, {"": primary_path, "_az225": path, ...}).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    by_camera = _motion_frames_by_camera(output_dir)
    kind = "executor_joint_sweep"
    camera: dict[str, Any] = {"source": "executor_render_joint_sweep"}
    if not by_camera and (output_dir / "furniture.urdf").is_file():
        with tempfile.TemporaryDirectory(prefix=".fixed-camera-", dir=destination.parent) as temporary:
            by_camera, camera = _fixed_camera_motion_frames(output_dir, Path(temporary), blender_bin=blender_bin)
            kind = "fixed_camera_joint_sweep"
            videos = _encode_by_camera(output_dir, destination, by_camera, ffmpeg_bin)
            if videos:
                return kind, camera, videos
            by_camera = {}
    if by_camera:
        videos = _encode_by_camera(output_dir, destination, by_camera, ffmpeg_bin)
        if videos:
            return kind, camera, videos
    still = _encode_video(output_dir, destination, ffmpeg_bin=ffmpeg_bin, frames_override=[])
    return ("fixed_camera_static_preview" if still else None), {}, ({"": destination} if still else {})


def _encode_by_camera(output_dir: Path, destination: Path, by_camera: dict[str, list[Path]],
                      ffmpeg_bin: str) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for cam in sorted(by_camera, key=lambda c: (c != "", c)):
        target = destination if cam == "" else destination.with_name(f"{destination.stem}{cam}{destination.suffix}")
        if _encode_video(output_dir, target, ffmpeg_bin=ffmpeg_bin, frames_override=by_camera[cam]):
            videos[cam] = target
    return videos


def _skill_files(directory: Path | None) -> dict[str, str]:
    if directory is None or not directory.is_dir():
        return {}
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".py"}:
            try:
                result[str(path.relative_to(directory))] = path.read_text(errors="replace")
            except OSError:
                pass
    return result


def _baseline_skill(round_dir: Path) -> Path | None:
    candidates = sorted(round_dir.glob("outputs/*/*/skill_snapshot/SKILL.md"))
    return candidates[0].parent if candidates else None


def _write_skill_diff(
    baseline_dir: Path | None,
    candidate_dir: Path | None,
    destination: Path,
) -> dict[str, Any]:
    before = _skill_files(baseline_dir)
    after = _skill_files(candidate_dir)
    changed: list[dict[str, Any]] = []
    diff_lines: list[str] = []
    total_additions = 0
    total_deletions = 0
    for name in sorted(set(before) | set(after)):
        if before.get(name) == after.get(name):
            continue
        lines = list(difflib.unified_diff(
            before.get(name, "").splitlines(),
            after.get(name, "").splitlines(),
            fromfile=f"before/{name}",
            tofile=f"after/{name}",
            lineterm="",
        ))
        additions = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        total_additions += additions
        total_deletions += deletions
        status = "modified" if name in before and name in after else (
            "added" if name in after else "deleted"
        )
        changed.append({
            "path": name,
            "status": status,
            "additions": additions,
            "deletions": deletions,
        })
        diff_lines.extend(lines)
        diff_lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(diff_lines))
    excerpt = "\n".join(diff_lines[:240])
    if len(diff_lines) > 240:
        excerpt += f"\n… {len(diff_lines) - 240} more diff lines; open the full diff."
    return {
        "files": changed,
        "additions": total_additions,
        "deletions": total_deletions,
        "excerpt": excerpt,
        "diff_path": str(destination),
    }


def _fallback_skill_change_summary(round_data: dict[str, Any]) -> str:
    """Return a compact artifact-only summary when the agent is disabled/fails."""
    changed = str((round_data.get("journal") or {}).get("Changed") or "").strip()
    if changed:
        changed = re.sub(r"\s+", " ", changed)
        return changed if len(changed) <= 360 else changed[:357].rstrip() + "…"
    diff = round_data.get("skill_diff") or {}
    files = [str(item.get("path")) for item in diff.get("files") or []]
    if not files:
        return "No textual skill change was detected for this round."
    return (
        f"Changed {len(files)} file(s): {', '.join(files[:5])}; "
        f"+{diff.get('additions', 0)} / −{diff.get('deletions', 0)} lines."
    )


def _parse_skill_summary_output(text: str) -> dict[int, str]:
    """Parse the agent's strict JSON response, tolerating a fenced wrapper."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("skill summary agent returned no JSON object")
    payload = json.loads(text[start:end + 1])
    rows = payload.get("summaries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("skill summary agent JSON has no summaries list")
    summaries: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        number = row.get("round")
        summary = row.get("summary")
        if isinstance(number, int) and isinstance(summary, str) and summary.strip():
            summaries[number] = re.sub(r"\s+", " ", summary).strip()
    if not summaries:
        raise ValueError("skill summary agent returned no usable summaries")
    return summaries


def _request_skill_change_summaries(
    rounds: list[dict[str, Any]],
    *,
    run_dir: Path,
    artifact_path: Path,
    model: str,
    effort: str,
) -> dict[int, str]:
    """Run one independent Responses agent over all round skill diffs."""
    inputs = []
    for round_data in rounds:
        diff = round_data["skill_diff"]
        inputs.append({
            "round": round_data["number"],
            "candidate_decision": round_data["candidate_decision"],
            "changed_files": diff["files"],
            "generator_changed_note": (
                round_data.get("journal") or {}
            ).get("Changed", ""),
            # Bound the prompt while retaining enough surrounding diff context
            # to summarize behavior rather than merely line counts.
            "unified_diff_excerpt": str(diff.get("excerpt") or "")[:18000],
        })

    instructions = """You summarize skill evolution for an experiment report.
Use only the supplied Generator journal and unified skill diffs. Do not infer
unseen evaluation results. For every round, write one or two concise Chinese
sentences describing the substantive workflow/tool/heuristic change and its
purpose. Mention rejection when candidate_decision says it was rejected. Avoid
line counts and generic phrases. Return JSON only in this exact shape:
{"summaries":[{"round":1,"summary":"..."}]}"""
    prompt = (
        "Summarize these evo-assemble skill changes:\n"
        + json.dumps({"rounds": inputs}, ensure_ascii=False)
    )

    async def invoke() -> str:
        from evo.engine.responses_backend import run_responses_agent

        outcome = await run_responses_agent(
            name="ExperimentReportSkillSummarizer",
            instructions=instructions,
            prompt=prompt,
            model=model,
            effort=effort,
            max_turns=2,
            cwd=run_dir,
            read_roots=[run_dir],
            write_root=None,
            artifact_path=artifact_path,
        )
        return outcome.final_output

    # Formal report generation runs inside the async training loop's finally
    # block. A dedicated thread owns this short-lived event loop, avoiding a
    # nested asyncio.run() while preserving the synchronous report API.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        text_output = pool.submit(lambda: asyncio.run(invoke())).result()
    return _parse_skill_summary_output(text_output)


def _attach_skill_change_summaries(
    rounds: list[dict[str, Any]],
    *,
    run_dir: Path,
    asset_dir: Path,
    enabled: bool,
    model: str,
    effort: str,
) -> dict[str, Any]:
    summaries: dict[int, str] = {}
    error: str | None = None
    artifact_rel = Path("skill-summary-agent.responses.json")
    if enabled and rounds:
        try:
            summaries = _request_skill_change_summaries(
                rounds,
                run_dir=run_dir,
                artifact_path=asset_dir / artifact_rel,
                model=model,
                effort=effort,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    for round_data in rounds:
        number = round_data["number"]
        agent_summary = summaries.get(number)
        round_data["skill_diff"]["summary"] = (
            agent_summary or _fallback_skill_change_summary(round_data)
        )
        round_data["skill_diff"]["summary_source"] = (
            "agent" if agent_summary else "artifact_fallback"
        )

    return {
        "enabled": enabled,
        "model": model if enabled else None,
        "effort": effort if enabled else None,
        "status": "complete" if summaries else (
            "fallback" if rounds else "no_rounds"
        ),
        "artifact": (
            f"assets/{artifact_rel.as_posix()}"
            if (asset_dir / artifact_rel).is_file() else None
        ),
        "error": error,
    }


def _copy_poster(source: Path | None, destination: Path) -> bool:
    if source is None or not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def collect_report_data(
    run_dir: Path,
    asset_dir: Path,
    *,
    create_videos: bool = True,
    ffmpeg_bin: str | None = None,
    blender_bin: str | None = None,
    summarize_skill_changes: bool = True,
    skill_summary_model: str = DEFAULT_SKILL_SUMMARY_MODEL,
    skill_summary_effort: str = DEFAULT_SKILL_SUMMARY_EFFORT,
) -> dict[str, Any]:
    """Collect persisted run artifacts and copy portable report media."""
    run_dir = run_dir.resolve()
    state = _read_json(run_dir / "loop_state.json", {}) or {}
    feedback_by_round = {
        int(item["round"]): item
        for item in state.get("feedback_history", [])
        if isinstance(item, dict) and isinstance(item.get("round"), int)
    }
    round_dirs = sorted(
        (path for path in run_dir.glob("round_*") if _round_number(path) is not None),
        key=lambda path: _round_number(path) or 0,
    )
    ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg")
    if create_videos and blender_bin is None:
        try:
            from evo.engine.envfile import load_env_chain

            load_env_chain()
            from evo.engine.paths import blender_bin as resolve_blender_bin

            blender_bin = str(resolve_blender_bin())
        except Exception:
            blender_bin = None
    rounds: list[dict[str, Any]] = []
    all_checks: set[str] = set()
    total_attempts = 0
    total_videos = 0

    for round_dir in round_dirs:
        number = _round_number(round_dir)
        assert number is not None
        feedback = feedback_by_round.get(number, {})
        artifacts: list[dict[str, Any]] = []
        for artifact in _discover_artifacts(run_dir, round_dir):
            tag = str(artifact.get("tag") or "unknown")
            output_dir = _artifact_output_dir(run_dir, round_dir, artifact)
            oracle_score, oracle_failed = _oracle_info(output_dir)
            failed_checks = artifact.get("failed_oracle_checks") or oracle_failed
            safe_name = f"r{number:02d}-{_slug(tag.replace('/', '-'))}"
            poster_source = _find_poster(output_dir)
            poster_rel = Path("media") / f"{safe_name}.png"
            has_poster = _copy_poster(poster_source, asset_dir / poster_rel)
            video_rel = Path("videos") / f"{safe_name}.mp4"
            video_kind = None
            camera: dict[str, Any] = {}
            video_alt_rel: Path | None = None
            if create_videos and ffmpeg_bin and poster_source is not None:
                video_kind, camera, videos = _encode_fixed_camera_video(
                    output_dir,
                    asset_dir / video_rel,
                    ffmpeg_bin=ffmpeg_bin,
                    blender_bin=blender_bin,
                )
                if video_kind:
                    total_videos += 1
                alt = next((p for cam, p in videos.items() if cam), None)
                if alt is not None:
                    video_alt_rel = alt.relative_to(asset_dir)
            cost = artifact.get("cost_summary") or {}
            artifacts.append({
                "tag": tag,
                "status": artifact.get("oracle_verdict") or "UNKNOWN",
                "failed_checks": list(failed_checks),
                "oracle_score": oracle_score,
                "api_calls": cost.get("n_api_calls", 0) or 0,
                "output_tokens": cost.get("output_tokens", 0) or 0,
                "poster": f"assets/{poster_rel.as_posix()}" if has_poster else None,
                "video": f"assets/{video_rel.as_posix()}" if video_kind else None,
                "video_alt": f"assets/{video_alt_rel.as_posix()}" if video_alt_rel else None,
                "video_kind": video_kind,
                "fixed_camera": camera,
                "has_urdf": (output_dir / "furniture.urdf").is_file(),
            })
        total_attempts += len(artifacts)

        per_check: dict[str, float | None] = {}
        for name, value in (feedback.get("per_check_pass_rate") or {}).items():
            per_check[str(name)] = _parse_rate(value)
            all_checks.add(str(name))

        trajectory = _trajectory_summary(round_dir)
        journal = _journal_sections(round_dir / "generator_journal.md")
        candidate_dir = round_dir / "skill"
        baseline_dir = _baseline_skill(round_dir)
        if baseline_dir is None and number > 0:
            previous = run_dir / f"round_{number - 1}" / "skill"
            baseline_dir = previous if previous.is_dir() else None
        diff_rel = Path("diffs") / f"round-{number + 1}.diff"
        skill_diff = _write_skill_diff(
            baseline_dir,
            candidate_dir if candidate_dir.is_dir() else None,
            asset_dir / diff_rel,
        )
        skill_diff["diff_path"] = f"assets/{diff_rel.as_posix()}"
        ab_record = _read_json(round_dir / "candidate_ab_test.json", {}) or {}
        rejected = feedback.get("generator_output_rejected")
        accepted = ab_record.get("accepted")
        if accepted is None:
            accepted = not bool(rejected or trajectory.get("error"))
        adaptive = feedback.get("adaptive_train_decision") or {}
        rounds.append({
            "number": number + 1,
            "index": number,
            "pass_rate": float(feedback.get("pass_rate", 0) or 0),
            "mean_oracle_score": float(feedback.get("mean_oracle_score", 0) or 0),
            "selection_fitness": float(feedback.get("selection_fitness", 0) or 0),
            "n_failures": int(feedback.get("n_failures", 0) or 0),
            "sample_count": len(artifacts),
            "per_check": per_check,
            "cost": feedback.get("batch_cost_summary") or {},
            "adaptive": adaptive,
            "artifacts": artifacts,
            "journal": journal,
            "trajectory": trajectory,
            "skill_diff": skill_diff,
            "candidate_accepted": bool(accepted),
            "candidate_decision": ab_record.get("action") or (
                "REJECTED" if not accepted else "ACCEPTED (legacy run)"
            ),
            "generator_rejection_reason": rejected,
            "skill_rolled_back": bool(feedback.get("skill_rolled_back")),
        })

    summary_agent = _attach_skill_change_summaries(
        rounds,
        run_dir=run_dir,
        asset_dir=asset_dir,
        enabled=summarize_skill_changes,
        model=skill_summary_model,
        effort=skill_summary_effort,
    )
    final_round = rounds[-1] if rounds else {}
    best_round = max(rounds, key=lambda item: item["selection_fitness"], default={})
    saved_at = state.get("saved_at")
    if isinstance(saved_at, (int, float)):
        saved_at = datetime.fromtimestamp(saved_at).isoformat(timespec="seconds")
    return {
        "run_id": run_dir.name,
        "run_path": str(run_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "saved_at": saved_at or datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
        "round_count": len(rounds),
        "total_attempts": total_attempts,
        "total_videos": total_videos,
        "best_fitness": float(state.get("best_fitness", best_round.get("selection_fitness", 0)) or 0),
        "best_round": best_round.get("number"),
        "final_pass_rate": final_round.get("pass_rate", 0),
        "final_mean_score": final_round.get("mean_oracle_score", 0),
        "all_checks": sorted(all_checks),
        "skill_summary_agent": summary_agent,
        "rounds": rounds,
    }


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def _num(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _markdownish(text: str) -> str:
    escaped = _esc(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"^[-*]\s+", "• ", escaped, flags=re.MULTILINE)
    return "<br>".join(escaped.splitlines())


def _usd(value: Any) -> str:
    """Format a priced amount. Runs before evo/engine/pricing.py, and models
    with no rate entry, have no figure — show a dash rather than $0.00, which
    would read as a free round."""
    if value is None:
        return "—"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _render_round_metric_rows(data: dict[str, Any]) -> str:
    rows = []
    for round_data in data["rounds"]:
        cost = round_data["cost"]
        rows.append(
            "<tr>"
            f"<td>R{round_data['number']}</td>"
            f"<td>{round_data['sample_count']}</td>"
            f"<td>{_pct(round_data['pass_rate'])}</td>"
            f"<td>{_pct(round_data['mean_oracle_score'])}</td>"
            f"<td>{round_data['selection_fitness']:.6f}</td>"
            f"<td>{round_data['n_failures']}</td>"
            f"<td>{_num(cost.get('total_api_calls', 0))}</td>"
            f"<td>{_num(cost.get('total_output_tokens', 0))}</td>"
            f"<td>{_usd(cost.get('executor_usd'))}</td>"
            f"<td>{_usd(cost.get('oracle_usd'))}</td>"
            f"<td>{_usd(cost.get('total_usd'))}</td>"
            f"<td>{_esc((round_data['adaptive'] or {}).get('reason', '—'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_check_table(data: dict[str, Any]) -> str:
    headers = "".join(f"<th>R{r['number']}</th>" for r in data["rounds"])
    rows: list[str] = []
    for check in data["all_checks"]:
        cells = []
        for round_data in data["rounds"]:
            value = round_data["per_check"].get(check)
            if value is None:
                cells.append("<td class=\"heat empty\">—</td>")
            else:
                cells.append(
                    f"<td class=\"heat\" style=\"--heat:{value:.3f}\">{_pct(value)}</td>"
                )
        rows.append(f"<tr><th>{_esc(check)}</th>{''.join(cells)}</tr>")
    return f"<thead><tr><th>Oracle check</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody>"


def _render_trajectory(round_data: dict[str, Any]) -> str:
    trajectory = round_data["trajectory"]
    journal = round_data["journal"]
    journal_html = "".join(
        f"<section class=\"journal-part\"><h4>{_esc(name)}</h4><p>{_markdownish(body)}</p></section>"
        for name, body in journal.items()
    ) or "<p class=\"muted\">No structured Generator journal was persisted.</p>"
    tools = ", ".join(
        f"{_esc(name)} × {count}" for name, count in trajectory["tool_histogram"][:8]
    ) or "none"
    modified = ", ".join(f"<code>{_esc(path)}</code>" for path in trajectory["files_modified"]) or "none"
    adaptive = round_data["adaptive"] or {}
    return f"""
    <details class="round-detail" {'open' if round_data['number'] == 1 else ''}>
      <summary><span>Round {round_data['number']}</span><span class="summary-meta">{round_data['sample_count']} samples · {_pct(round_data['mean_oracle_score'])} mean score · {_esc(round_data['candidate_decision'])}</span></summary>
      <div class="round-body">
        <div class="trajectory-facts">
          <div><span>Provider</span><strong>{_esc(trajectory['provider'])}</strong></div>
          <div><span>Model</span><strong>{_esc(trajectory['model'])} / {_esc(trajectory['effort'])}</strong></div>
          <div><span>Generator requests</span><strong>{_num(trajectory['requests'])}</strong></div>
          <div><span>Generator output</span><strong>{_num(trajectory['output_tokens'])} tokens</strong></div>
          <div><span>Tool calls / error outputs</span><strong>{_num(trajectory['tool_calls'])} / {_num(trajectory['tool_error_outputs'])}</strong></div>
          <div><span>Adaptive decision</span><strong>{_esc(adaptive.get('action', '—'))}: {_esc(adaptive.get('reason', '—'))}</strong></div>
        </div>
        <p class="tool-line"><strong>Top tools:</strong> {tools}</p>
        <p class="tool-line"><strong>Modified:</strong> {modified}</p>
        <div class="journal-grid">{journal_html}</div>
      </div>
    </details>"""


def _render_skill_change(round_data: dict[str, Any]) -> str:
    diff = round_data["skill_diff"]
    files = "".join(
        f"<tr><td><code>{_esc(item['path'])}</code></td><td>{_esc(item['status'])}</td>"
        f"<td>+{item['additions']} / −{item['deletions']}</td></tr>"
        for item in diff["files"]
    ) or "<tr><td colspan=\"3\">No textual skill change detected.</td></tr>"
    accepted = "accepted" if round_data["candidate_accepted"] else "rejected"
    return f"""
    <details class="skill-change">
      <summary><span>Round {round_data['number']}</span><span class="summary-meta {accepted}">{accepted} · {len(diff['files'])} files · +{diff['additions']} / −{diff['deletions']}</span></summary>
      <div class="round-body">
        <div class="skill-summary"><h4>Agent summary</h4><p>{_markdownish(diff.get('summary') or '—')}</p><small>{_esc(diff.get('summary_source') or 'unknown')}</small></div>
        <table class="compact"><thead><tr><th>File</th><th>Status</th><th>Lines</th></tr></thead><tbody>{files}</tbody></table>
        <p><a href="{_esc(diff['diff_path'])}" download>Download full unified diff</a></p>
        <pre><code>{_esc(diff['excerpt'])}</code></pre>
      </div>
    </details>"""


def _render_results(data: dict[str, Any]) -> str:
    blocks: list[str] = []
    for round_data in data["rounds"]:
        cards: list[str] = []
        for artifact in round_data["artifacts"]:
            poster = artifact.get("poster")
            video = artifact.get("video")
            media = ""
            if video:
                poster_attr = f" poster=\"{_esc(poster)}\"" if poster else ""
                media = f"<video controls preload=\"metadata\"{poster_attr}><source src=\"{_esc(video)}\" type=\"video/mp4\"></video>"
                video_alt = artifact.get("video_alt")
                if video_alt:
                    # The dataset has no shared front convention; the second
                    # camera is the same sweep from the opposite side.
                    media += f"<video controls preload=\"metadata\"><source src=\"{_esc(video_alt)}\" type=\"video/mp4\"></video>"
            elif poster:
                media = f"<img src=\"{_esc(poster)}\" loading=\"lazy\" alt=\"Rendered result for {_esc(artifact['tag'])}\">"
            else:
                media = "<div class=\"no-media\">No assembled render</div>"
            failed = ", ".join(artifact["failed_checks"]) or "none"
            status_class = "pass" if artifact["status"] == "PASS" else "fail"
            cards.append(f"""
            <article class="result-card" data-status="{_esc(artifact['status'])}">
              {media}
              <div class="result-meta">
                <h4>{_esc(artifact['tag'])}</h4>
                <span class="status {status_class}">{_esc(artifact['status'])}</span>
                <p>Check score: {_pct(artifact['oracle_score'])} · URDF: {'yes' if artifact['has_urdf'] else 'no'}</p>
                <p class="failed-checks">Failed: {_esc(failed)}</p>
              </div>
            </article>""")
        blocks.append(f"<section class=\"result-round\"><h3>Round {round_data['number']}</h3><div class=\"result-grid\">{''.join(cards)}</div></section>")
    return "".join(blocks)


def render_html(data: dict[str, Any]) -> str:
    chart_data = json.dumps({
        "rounds": [r["number"] for r in data["rounds"]],
        "passRate": [r["pass_rate"] for r in data["rounds"]],
        "meanScore": [r["mean_oracle_score"] for r in data["rounds"]],
        "samples": [r["sample_count"] for r in data["rounds"]],
        "apiCalls": [r["cost"].get("total_api_calls", 0) for r in data["rounds"]],
        "outputTokens": [r["cost"].get("total_output_tokens", 0) for r in data["rounds"]],
    }, ensure_ascii=False).replace("</", "<\\/")
    trajectories = "".join(_render_trajectory(r) for r in data["rounds"])
    skill_changes = "".join(_render_skill_change(r) for r in data["rounds"])
    results = _render_results(data)
    metric_rows = _render_round_metric_rows(data)
    check_table = _render_check_table(data)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>evo-assemble experiment report · {_esc(data['run_id'])}</title>
<style>
:root{{--bg:#f6f5f2;--paper:#fff;--ink:#20211f;--muted:#6d706b;--line:#d9d9d4;--accent:#376956;--accent2:#a8662d;--good:#287a4b;--bad:#a53e35;--soft:#e9efe9;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}
.shell{{max-width:1240px;margin:auto;padding:32px 24px 80px}} header{{border-bottom:1px solid var(--line);padding-bottom:24px}} h1{{font-size:clamp(28px,4vw,46px);font-weight:500;line-height:1.08;margin:8px 0}} h2{{font-size:24px;font-weight:500;margin:48px 0 16px}} h3{{font-size:18px;font-weight:500}} h4{{margin:0;font-size:15px}} p{{margin:8px 0}} code,pre{{font-family:var(--mono)}} code{{font-size:.9em}} .eyebrow,.muted{{color:var(--muted)}}
header p{{overflow-wrap:anywhere}} nav{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0 0}} a{{color:var(--accent)}} nav a{{text-decoration:none;border-bottom:1px solid transparent}} nav a:hover{{border-color:currentColor}}
.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}} .stat{{background:var(--paper);border:1px solid var(--line);padding:16px}} .stat span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}} .stat strong{{display:block;font-size:25px;font-weight:500;margin-top:4px}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}} .chart{{min-height:300px;background:var(--paper);border:1px solid var(--line);padding:16px}} .chart h3{{margin:0 0 12px}} svg{{display:block;width:100%;height:250px}} .axis{{stroke:var(--line);stroke-width:1}} .grid{{stroke:var(--line);stroke-width:1;opacity:.65}} .tick{{fill:var(--muted);font-size:11px}} .series-a{{stroke:var(--accent);fill:none;stroke-width:2.5}} .series-b{{stroke:var(--accent2);fill:none;stroke-width:2.5}} .dot-a{{fill:var(--accent)}} .dot-b{{fill:var(--accent2)}} .legend{{display:flex;gap:18px;color:var(--muted);font-size:13px}} .swatch{{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px;background:var(--accent)}} .swatch.b{{background:var(--accent2)}}
.table-wrap{{overflow-x:auto;background:var(--paper);border:1px solid var(--line)}} table{{border-collapse:collapse;width:100%}} th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}} th{{font-weight:500}} .compact th,.compact td{{padding:7px 9px}} .heat{{text-align:center;background:color-mix(in srgb,var(--accent) calc(var(--heat) * 42%),transparent)}} .heat.empty{{background:none;color:var(--muted)}}
details{{border-top:1px solid var(--line);background:var(--paper)}} details:last-child{{border-bottom:1px solid var(--line)}} summary{{cursor:pointer;display:flex;justify-content:space-between;gap:16px;padding:15px 16px;font-weight:500}} .summary-meta{{color:var(--muted);font-weight:400;text-align:right}} .summary-meta.accepted{{color:var(--good)}} .summary-meta.rejected{{color:var(--bad)}} .round-body{{padding:0 16px 20px}} .trajectory-facts{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--line)}} .trajectory-facts div{{background:var(--paper);padding:10px}} .trajectory-facts span{{display:block;color:var(--muted);font-size:12px}} .trajectory-facts strong{{font-weight:500;overflow-wrap:anywhere}} .tool-line{{overflow-wrap:anywhere}} .journal-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin-top:18px}} .journal-part p{{color:#3c3e3a}} pre{{max-height:420px;overflow:auto;background:#202421;color:#e9eee9;padding:14px;font-size:12px;white-space:pre-wrap}}
.skill-summary{{margin:0 0 14px;padding:12px 14px;border-left:3px solid var(--accent);background:var(--soft)}} .skill-summary p{{margin:4px 0}} .skill-summary small{{color:var(--muted)}}
.result-round{{margin:28px 0}} .result-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}} .result-card{{background:var(--paper);border:1px solid var(--line);min-width:0}} video,.result-card img{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#eee}} .no-media{{display:grid;place-items:center;aspect-ratio:1/1;color:var(--muted);background:#ecece8}} .result-meta{{padding:12px;position:relative}} .status{{position:absolute;right:12px;top:12px;font-size:12px;font-weight:500}} .status.pass{{color:var(--good)}} .status.fail{{color:var(--bad)}} .result-meta p{{font-size:13px;color:var(--muted)}} .failed-checks{{overflow-wrap:anywhere}}
footer{{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}}
@media(max-width:850px){{.stats,.chart-grid,.result-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.trajectory-facts{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:560px){{.shell{{padding:20px 12px 56px}}.stats,.chart-grid,.result-grid,.trajectory-facts,.journal-grid{{grid-template-columns:1fr}}summary{{display:block}}.summary-meta{{display:block;text-align:left;margin-top:4px}}.run-time span{{display:block}}}}
@media(prefers-color-scheme:dark){{:root{{--bg:#171916;--paper:#20231f;--ink:#f0f1ed;--muted:#aeb3aa;--line:#3b4039;--accent:#79b79c;--accent2:#dda56f;--good:#7dcc9a;--bad:#ed8d84;--soft:#29332d}}.journal-part p{{color:#d8dbd5}}pre{{background:#101210}}video,.result-card img,.no-media{{background:#2b2e29}}}}
</style>
</head>
<body><main class="shell">
<header><div class="eyebrow">EVO-ASSEMBLE · FORMAL TRAINING REPORT</div><h1>{_esc(data['run_id'])}</h1><p class="run-time"><span>Run completed {_esc(data['saved_at'])}</span><span> · report generated {_esc(data['generated_at'])}</span></p>
<nav><a href="#metrics">Metrics</a><a href="#checks">Oracle checks</a><a href="#trajectory">Trajectory</a><a href="#skills">Skill changes</a><a href="#results">Rendered results</a></nav></header>
<section class="stats" aria-label="Run summary">
<div class="stat"><span>Rounds</span><strong>{data['round_count']}</strong></div><div class="stat"><span>Train attempts</span><strong>{data['total_attempts']}</strong></div><div class="stat"><span>Best fitness</span><strong>{data['best_fitness']:.6f}</strong><small>Round {data['best_round'] or '—'}</small></div><div class="stat"><span>Result videos</span><strong>{data['total_videos']}</strong></div>
</section>
<section id="metrics"><h2>Metric curves</h2><div class="chart-grid"><div class="chart"><h3>Objective quality by round</h3><div class="legend"><span><i class="swatch"></i>Binary pass rate</span><span><i class="swatch b"></i>Mean oracle score</span></div><svg id="quality-chart" role="img" aria-label="Pass rate and mean oracle score by round"></svg></div><div class="chart"><h3>Train usage by round</h3><div class="legend"><span><i class="swatch"></i>Samples</span><span><i class="swatch b"></i>API calls</span></div><svg id="usage-chart" role="img" aria-label="Sample and API call count by round"></svg></div></div>
<div class="table-wrap"><table><thead><tr><th>Round</th><th>Samples</th><th>Pass rate</th><th>Mean score</th><th>Fitness</th><th>Failures</th><th>API calls</th><th>Output tokens</th><th>Executor $</th><th>Oracle $</th><th>Round $</th><th>Adaptive stop</th></tr></thead><tbody>{metric_rows}</tbody></table></div></section>
<section id="checks"><h2>Oracle check curves</h2><div class="table-wrap"><table>{check_table}</table></div></section>
<section id="trajectory"><h2>Round trajectory analysis</h2>{trajectories}</section>
<section id="skills"><h2>Skill changes</h2>{skill_changes}</section>
<section id="results"><h2>Rendered assembly results</h2><p class="muted">Motion videos are re-rendered from each predicted URDF with one camera locked to the union envelope of every joint pose; camera location, rotation, and orthographic scale remain identical across the entire video. Static videos indicate that no movable predicted joint was available. Failed or interrupted tasks without an assembled render retain their input thumbnail for diagnosis.</p>{results}</section>
<footer>Metrics and fixed-camera renders are generated deterministically from persisted run artifacts. Skill-change summaries are produced by a separate Responses agent and persisted with the report. Reporting never reads GT and does not change experiment selection.</footer>
</main>
<script id="report-data" type="application/json">{chart_data}</script>
<script>
const D=JSON.parse(document.getElementById('report-data').textContent);
const NS='http://www.w3.org/2000/svg';
function el(name,attrs={{}}){{const n=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n}}
function chart(id,a,b,maxValue,percent=false){{const svg=document.getElementById(id),W=620,H=250,m={{l:48,r:18,t:18,b:36}},w=W-m.l-m.r,h=H-m.t-m.b;svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);const max=Math.max(maxValue,...a,...b,1e-9);for(let i=0;i<=4;i++){{const y=m.t+h-i*h/4;svg.append(el('line',{{x1:m.l,y1:y,x2:W-m.r,y2:y,class:'grid'}}));const t=el('text',{{x:m.l-8,y:y+4,'text-anchor':'end',class:'tick'}});const value=max*i/4;t.textContent=percent?Math.round(value*100)+'%':Math.round(value);svg.append(t)}}const x=i=>D.rounds.length===1?m.l+w/2:m.l+i*w/(D.rounds.length-1),y=v=>m.t+h-(v/max)*h;D.rounds.forEach((r,i)=>{{const t=el('text',{{x:x(i),y:H-10,'text-anchor':'middle',class:'tick'}});t.textContent='R'+r;svg.append(t)}});function series(vals,cls,dot){{const points=vals.map((v,i)=>`${{x(i)}},${{y(v)}}`).join(' ');svg.append(el('polyline',{{points,class:cls}}));vals.forEach((v,i)=>{{const c=el('circle',{{cx:x(i),cy:y(v),r:4,class:dot}});const title=el('title');title.textContent=`R${{D.rounds[i]}}: ${{percent?(v*100).toFixed(1)+'%':v}}`;c.append(title);svg.append(c)}})}}series(a,'series-a','dot-a');series(b,'series-b','dot-b')}}
chart('quality-chart',D.passRate,D.meanScore,1,true);chart('usage-chart',D.samples,D.apiCalls,Math.max(...D.samples,...D.apiCalls),false);
</script></body></html>"""


def generate_experiment_report(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    create_videos: bool = True,
    ffmpeg_bin: str | None = None,
    blender_bin: str | None = None,
    summarize_skill_changes: bool = True,
    skill_summary_model: str = DEFAULT_SKILL_SUMMARY_MODEL,
    skill_summary_effort: str = DEFAULT_SKILL_SUMMARY_EFFORT,
) -> Path:
    """Build a portable report directory and return its index.html path."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    output = Path(output_dir).expanduser().resolve() if output_dir else run_dir / "report"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not (output / REPORT_MARKER).is_file():
        raise FileExistsError(
            f"refusing to replace non-report directory without {REPORT_MARKER}: {output}"
        )
    temporary = Path(tempfile.mkdtemp(prefix=".evo-report-", dir=output.parent))
    try:
        asset_dir = temporary / "assets"
        data = collect_report_data(
            run_dir,
            asset_dir,
            create_videos=create_videos,
            ffmpeg_bin=ffmpeg_bin,
            blender_bin=blender_bin,
            summarize_skill_changes=summarize_skill_changes,
            skill_summary_model=skill_summary_model,
            skill_summary_effort=skill_summary_effort,
        )
        (temporary / "report-data.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2)
        )
        (temporary / "index.html").write_text(render_html(data))
        (temporary / REPORT_MARKER).write_text("generated by evo.reporting.experiment_report\n")
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "index.html"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an evo-assemble HTML experiment report")
    parser.add_argument("run_dir", type=Path, help="logs/runs/multi_* directory")
    parser.add_argument("--output", type=Path, default=None, help="report directory (default: <run>/report)")
    parser.add_argument("--no-videos", action="store_true", help="copy posters but skip MP4 encoding")
    parser.add_argument("--ffmpeg", default=None, help="explicit ffmpeg executable")
    parser.add_argument("--blender", default=None, help="explicit Blender executable")
    parser.add_argument(
        "--no-skill-summaries", action="store_true",
        help="skip the Responses summary agent and use journal-based fallbacks",
    )
    parser.add_argument(
        "--summary-model", default=DEFAULT_SKILL_SUMMARY_MODEL,
        help=f"skill-summary agent model (default: {DEFAULT_SKILL_SUMMARY_MODEL})",
    )
    parser.add_argument(
        "--summary-effort", default=DEFAULT_SKILL_SUMMARY_EFFORT,
        help=f"skill-summary reasoning effort (default: {DEFAULT_SKILL_SUMMARY_EFFORT})",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    path = generate_experiment_report(
        args.run_dir,
        args.output,
        create_videos=not args.no_videos,
        ffmpeg_bin=args.ffmpeg,
        blender_bin=args.blender,
        summarize_skill_changes=not args.no_skill_summaries,
        skill_summary_model=args.summary_model,
        skill_summary_effort=args.summary_effort,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
