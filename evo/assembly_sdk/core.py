"""Pure contracts for the frozen Assembly SDK.

This module intentionally has no simulation-runtime dependency.  Mutators in
``runtime.py`` validate a candidate state here before persisting it, ensuring
that failed calls cannot leave a half-written kinematic graph behind.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Mapping, Sequence


SDK_VERSION = "1.4.0"
SCHEMA_VERSION = 1
JOINT_TYPES = frozenset({"FIXED", "REVOLUTE", "CONTINUOUS", "PRISMATIC"})
MOVABLE_TYPES = frozenset({"REVOLUTE", "CONTINUOUS", "PRISMATIC"})
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class ErrorCode(str, Enum):
    INVALID_ARGUMENT = "invalid_argument"
    UNKNOWN_HANDLE = "unknown_handle"
    AMBIGUOUS_GEOMETRY = "ambiguous_geometry"
    INVARIANT_VIOLATION = "invariant_violation"
    SIM_RUNTIME = "sim_runtime"
    IO_FAILURE = "io_failure"


class AssemblyError(RuntimeError):
    """Stable, machine-readable SDK failure."""

    def __init__(
        self,
        code: ErrorCode | str,
        operation: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.operation = operation
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"ASSEMBLY_SDK[{self.code.value}] {operation}: {message}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code.value,
                "operation": self.operation,
                "message": self.message,
                "details": self.details,
            },
        }


@dataclass(frozen=True)
class OperationResult:
    ok: bool
    value: Any = None
    error: dict[str, Any] | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        out = {"ok": self.ok, "value": self.value, "warnings": list(self.warnings)}
        if self.error is not None:
            out["error"] = self.error
        return out


def fail(
    code: ErrorCode,
    operation: str,
    message: str,
    **details: Any,
) -> None:
    raise AssemblyError(code, operation, message, details=details)


def vec3(value: Sequence[float], operation: str, field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        fail(
            ErrorCode.INVALID_ARGUMENT,
            operation,
            f"{field_name} must be a length-3 numeric sequence",
            field=field_name,
            received=repr(value),
        )
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        fail(
            ErrorCode.INVALID_ARGUMENT,
            operation,
            f"{field_name} contains a non-numeric value",
            field=field_name,
            received=repr(value),
            cause=str(exc),
        )
    if not all(math.isfinite(v) for v in result):
        fail(
            ErrorCode.INVALID_ARGUMENT,
            operation,
            f"{field_name} must contain finite values",
            field=field_name,
            received=repr(value),
        )
    return result  # type: ignore[return-value]


def normalize_axis(value: str | Sequence[float], operation: str = "normalize_axis") -> tuple[float, float, float]:
    if isinstance(value, str):
        mapped = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
            "-X": (-1.0, 0.0, 0.0),
            "-Y": (0.0, -1.0, 0.0),
            "-Z": (0.0, 0.0, -1.0),
        }.get(value.upper())
        if mapped is None:
            fail(ErrorCode.INVALID_ARGUMENT, operation, "unknown axis label", axis=value)
        return mapped
    axis = vec3(value, operation, "axis")
    norm = math.sqrt(sum(v * v for v in axis))
    if norm < 1e-9:
        fail(ErrorCode.INVALID_ARGUMENT, operation, "movable joint axis must be non-zero")
    return tuple(v / norm for v in axis)  # type: ignore[return-value]


def validate_identifier(value: Any, operation: str, field_name: str = "name") -> str:
    """Validate a ROS/URDF-safe link or joint identifier."""
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        fail(
            ErrorCode.INVALID_ARGUMENT,
            operation,
            f"{field_name} must match {IDENTIFIER_RE.pattern}",
            field=field_name,
            received=repr(value),
        )
    return value


def _audit_state(
    state: Mapping[str, Any], *, require_complete_tree: bool
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    def issue(code: str, message: str, **details: Any) -> None:
        issues.append({"code": code, "message": message, "details": details})

    if state.get("schema_version") != SCHEMA_VERSION:
        issue("schema_version", "unsupported state schema", found=state.get("schema_version"))
    parts = state.get("parts")
    links = state.get("links")
    joints = state.get("joints")
    if not isinstance(parts, dict) or not isinstance(links, dict) or not isinstance(joints, dict):
        issue("shape", "parts, links, and joints must all be dictionaries")
        return issues

    for part_handle, part in parts.items():
        if not isinstance(part_handle, str) or not part_handle:
            issue(
                "part_name",
                "part handle must be a non-empty string",
                part=repr(part_handle),
            )
        if not isinstance(part, dict):
            issue(
                "part_shape",
                "part registry entry must be a dictionary",
                part=part_handle,
            )
            continue
        try:
            scale = float(part.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = float("nan")
        if not math.isfinite(scale) or scale <= 0.0:
            issue(
                "part_scale",
                "part scale must be a finite positive uniform factor",
                part=part_handle,
                scale=part.get("scale"),
            )

    owners: dict[str, str] = {}
    if require_complete_tree and not links:
        issue("empty_assembly", "complete assembly must contain at least one link")
    for link_name, link in links.items():
        if not isinstance(link_name, str) or not IDENTIFIER_RE.fullmatch(link_name):
            issue("link_name", "link name is not a valid URDF identifier", name=repr(link_name))
        if not isinstance(link, dict):
            issue("link_shape", "link must be a dictionary", link=link_name)
            continue
        handles = link.get("part_handles", [])
        if not isinstance(handles, list):
            issue("part_handles_shape", "link part_handles must be a list", link=link_name)
            handles = []
        if not handles:
            issue("empty_link", "link owns no parts", link=link_name)
        for handle in handles:
            if not isinstance(handle, str) or not handle:
                issue(
                    "part_handle",
                    "part handle must be a non-empty string",
                    link=link_name,
                    part=repr(handle),
                )
                continue
            if handle not in parts:
                issue("unknown_part", "link references unknown part", link=link_name, part=handle)
            if handle in owners:
                issue("duplicate_part_owner", "part belongs to multiple links", part=handle, links=[owners[handle], link_name])
            owners[handle] = link_name
        for field_name in ("visual_xyz", "visual_rpy"):
            try:
                vec3(link.get(field_name), "validate_state", f"{link_name}.{field_name}")
            except AssemblyError as exc:
                issue("invalid_transform", exc.message, link=link_name, field=field_name)

    if require_complete_tree:
        orphans = sorted(set(parts) - set(owners))
        if orphans:
            issue(
                "unassigned_part",
                "imported part belongs to no link",
                parts=orphans,
            )

    child_owner: dict[str, str] = {}
    children_of: dict[str, list[str]] = {}
    for joint_name, joint in joints.items():
        if not isinstance(joint_name, str) or not IDENTIFIER_RE.fullmatch(joint_name):
            issue("joint_name", "joint name is not a valid URDF identifier", name=repr(joint_name))
        if not isinstance(joint, dict):
            issue("joint_shape", "joint must be a dictionary", joint=joint_name)
            continue
        parent, child = joint.get("parent"), joint.get("child")
        if not isinstance(parent, str) or not isinstance(child, str):
            issue(
                "joint_links",
                "joint parent and child must be link-name strings",
                joint=joint_name,
                parent=repr(parent),
                child=repr(child),
            )
            continue
        if parent not in links:
            issue("unknown_parent", "joint parent is unknown", joint=joint_name, parent=parent)
        if child not in links:
            issue("unknown_child", "joint child is unknown", joint=joint_name, child=child)
        if parent == child:
            issue("self_parent", "joint parent and child are identical", joint=joint_name)
        if child in child_owner:
            issue("multiple_parents", "link has more than one parent joint", child=child, joints=[child_owner[child], joint_name])
        child_owner[child] = joint_name
        children_of.setdefault(parent, []).append(child)
        joint_type = str(joint.get("type", "")).upper()
        if joint_type not in JOINT_TYPES:
            issue("joint_type", "unsupported joint type", joint=joint_name, joint_type=joint_type)
        if joint_type in MOVABLE_TYPES:
            try:
                normalize_axis(joint.get("axis"), "validate_state")
            except AssemblyError as exc:
                issue("joint_axis", exc.message, joint=joint_name)
        if joint_type in ("REVOLUTE", "PRISMATIC"):
            lower, upper = joint.get("limit_lower"), joint.get("limit_upper")
            try:
                valid_limits = (
                    lower is not None
                    and upper is not None
                    and math.isfinite(float(lower))
                    and math.isfinite(float(upper))
                    and float(lower) <= float(upper)
                )
            except (TypeError, ValueError):
                valid_limits = False
            if not valid_limits:
                issue("joint_limits", "bounded joint needs finite lower <= upper", joint=joint_name, lower=lower, upper=upper)
        for field_name in ("origin_xyz", "origin_rpy"):
            try:
                vec3(joint.get(field_name), "validate_state", f"{joint_name}.{field_name}")
            except AssemblyError as exc:
                issue("invalid_transform", exc.message, joint=joint_name, field=field_name)

    positions = state.get("joint_positions", {})
    if not isinstance(positions, dict):
        issue("joint_positions_shape", "joint_positions must be a dictionary")
        positions = {}
    stale_positions = sorted(set(positions) - set(joints))
    missing_positions = sorted(set(joints) - set(positions))
    if stale_positions:
        issue(
            "stale_joint_position",
            "joint_positions contains unknown joints",
            joints=stale_positions,
        )
    if missing_positions:
        issue(
            "missing_joint_position",
            "joint_positions is missing registered joints",
            joints=missing_positions,
        )
    for joint_name in sorted(set(positions) & set(joints)):
        try:
            q = float(positions[joint_name])
        except (TypeError, ValueError):
            issue("joint_position", "joint position must be numeric", joint=joint_name)
            continue
        if not math.isfinite(q):
            issue("joint_position", "joint position must be finite", joint=joint_name, value=q)
            continue
        joint = joints[joint_name]
        if not isinstance(joint, dict):
            continue
        joint_type = str(joint.get("type", "")).upper()
        if joint_type == "FIXED" and abs(q) > 1e-12:
            issue("joint_position", "fixed joint position must be zero", joint=joint_name, value=q)
        if joint_type in ("REVOLUTE", "PRISMATIC"):
            lower, upper = joint.get("limit_lower"), joint.get("limit_upper")
            try:
                if q < float(lower) - 1e-9 or q > float(upper) + 1e-9:
                    issue(
                        "joint_position",
                        "joint position is outside limits",
                        joint=joint_name,
                        value=q,
                        lower=lower,
                        upper=upper,
                    )
            except (TypeError, ValueError):
                pass

    # Detect cycles independently of root count.
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(link: str, path: list[str]) -> None:
        if link in visiting:
            issue("cycle", "kinematic graph contains a cycle", path=path + [link])
            return
        if link in visited:
            return
        visiting.add(link)
        for child in children_of.get(link, []):
            walk(child, path + [link])
        visiting.remove(link)
        visited.add(link)

    for link_name in links:
        if link_name not in visited:
            walk(link_name, [])
    if links and require_complete_tree:
        roots = sorted(set(links) - set(child_owner))
        if len(roots) != 1:
            issue("root_count", "kinematic graph must have exactly one root", roots=roots)
        else:
            reachable: set[str] = set()
            queue = [roots[0]]
            while queue:
                current = queue.pop(0)
                if current in reachable:
                    continue
                reachable.add(current)
                queue.extend(children_of.get(current, []))
            unreachable = sorted(set(links) - reachable)
            if unreachable:
                issue(
                    "disconnected",
                    "not every link is connected to the root",
                    root=roots[0],
                    unreachable=unreachable,
                )
    return issues


def validate_state(state: Mapping[str, Any], *, require_complete_tree: bool = False) -> list[dict[str, Any]]:
    """Return issues or raise when a complete tree is required."""
    if not isinstance(state, Mapping):
        issues = [{
            "code": "state_shape",
            "message": "state must be a mapping",
            "details": {"received": type(state).__name__},
        }]
        if require_complete_tree:
            raise AssemblyError(
                ErrorCode.INVARIANT_VIOLATION,
                "validate_state",
                "assembly state failed validation",
                details={"issues": issues},
            )
        return issues
    issues = _audit_state(state, require_complete_tree=require_complete_tree)
    if issues and require_complete_tree:
        raise AssemblyError(
            ErrorCode.INVARIANT_VIOLATION,
            "validate_state",
            "assembly state failed validation",
            details={"issues": issues},
        )
    return issues
