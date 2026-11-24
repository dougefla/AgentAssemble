from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest

from evo.oracle.check_urdf import (
    CheckResult,
    JointInfo,
    LinkInfo,
    OracleResult,
    PoseDescriptor,
    SimilarityAlignment,
    Thresholds,
    URDFModel,
    _canonical_reference_position,
    _descriptor_distance,
    _link_world_descriptor,
    _mat4_identity,
    _mat4_rotate_rpy,
    _match_links_detailed,
    _minimum_cost_assignment,
    check_joint_axes,
    check_joint_limits,
    check_joint_pivot_alignment,
    check_kinematic_equivalence,
    check_model_integrity,
    check_rest_pose_alignment,
    compute_signature,
    parse_urdf,
)
from evo.engine.loop import _selection_fitness


def _write_obj(path: Path, vertices: list[tuple[float, float, float]]) -> None:
    path.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in vertices))


def _visual(name: str, mesh: str, *, xyz=(0.0, 0.0, 0.0)) -> LinkInfo:
    return LinkInfo(
        name=name,
        mesh_paths=[mesh],
        has_visual=True,
        visual_xyz=xyz,
    )


def test_hungarian_assignment_is_global_not_greedy() -> None:
    # Row-wise greedy returns [0, 1] with cost 101.  Global assignment is
    # [1, 0] with cost 3.1.
    assert _minimum_cost_assignment([[1.0, 2.0], [1.1, 100.0]]) == [1, 0]


def test_selection_fitness_prioritizes_binary_success() -> None:
    assert _selection_fitness(0.1, 0.0) > _selection_fitness(0.0, 1.0)
    assert _selection_fitness(0.0, 0.8) > _selection_fitness(0.0, 0.2)
    assert _selection_fitness(1.0, 0.0) == 1.0


def test_non_scored_diagnostic_is_recorded_but_excluded_from_score() -> None:
    result = OracleResult(
        passed=True,
        checks=[
            CheckResult("valid_tree", True),
            CheckResult(
                "debug_diagnostic", False, "3.8 cm drift", scored=False,
            ),
            CheckResult("kinematic_equivalence", True),
        ],
    )
    assert result.score == 1.0
    payload = result.to_dict()
    assert payload["checks"]["debug_diagnostic"] == {
        "passed": False,
        "scored": False,
        "detail": "3.8 cm drift",
    }


def test_default_kinematic_tolerance_is_doubled() -> None:
    thresholds = Thresholds()
    assert thresholds.kinematic_abs_tol == pytest.approx(0.02)
    assert thresholds.kinematic_rel_tol == pytest.approx(0.04)


def test_rest_pose_and_revolute_pivot_are_distinct_diagnostics(
    tmp_path: Path,
) -> None:
    _write_obj(
        tmp_path / "base.obj",
        [(0, 0, 0), (0.2, 0, 0), (0, 0.3, 0), (0, 0, 0.4)],
    )
    _write_obj(
        tmp_path / "door.obj",
        [(0, 0, 0), (0.4, 0, 0), (0, 0.1, 0), (0, 0, 0.2)],
    )
    gt = URDFModel("gt", base_dir=tmp_path)
    gt.links = {
        "base": _visual("base", "base.obj"),
        "door": _visual("door", "door.obj", xyz=(1.0, 0.0, 0.0)),
    }
    gt.joints = {
        "hinge": JointInfo(
            "hinge", "REVOLUTE", "base", "door",
            origin_xyz=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
            limit_lower=0.0, limit_upper=math.pi / 2,
        )
    }

    pred = URDFModel("pred", base_dir=tmp_path)
    pred.links = {
        "base": _visual("base", "base.obj"),
        # The opposite visual offset hides the 0.5 m pivot error at q=0.
        "door": _visual("door", "door.obj", xyz=(0.5, 0.0, 0.0)),
    }
    pred.joints = {
        "hinge": JointInfo(
            "hinge", "REVOLUTE", "base", "door",
            origin_xyz=(0.5, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
            limit_lower=0.0, limit_upper=math.pi / 2,
        )
    }
    matching = {"base": "base", "door": "door"}
    thresholds = Thresholds(kinematic_tol=1e-4)

    rest = check_rest_pose_alignment(gt, pred, matching, thresholds)
    pivot = check_joint_pivot_alignment(gt, pred, matching, thresholds)
    assert rest.passed, rest.detail
    assert rest.scored
    assert not pivot.passed
    assert pivot.scored
    assert "line_distance=0.500m" in pivot.detail


def _axis_models(
    tmp_path: Path,
    *,
    pred_axis: tuple[float, float, float],
    pred_origin_rpy: tuple[float, float, float],
) -> tuple[URDFModel, URDFModel, dict[str, str]]:
    _write_obj(
        tmp_path / "axis_part.obj",
        [(0, 0, 0), (0.2, 0, 0), (0, 0.1, 0), (0, 0, 0.1)],
    )
    gt = URDFModel("gt", base_dir=tmp_path)
    gt.links = {
        "base": _visual("base", "axis_part.obj"),
        "door": _visual("door", "axis_part.obj", xyz=(1, 0, 0)),
    }
    gt.joints = {
        "hinge": JointInfo(
            "hinge", "REVOLUTE", "base", "door",
            axis=(0, 0, 1), limit_lower=0.0, limit_upper=1.0,
        )
    }
    pred = URDFModel("pred", base_dir=tmp_path)
    pred.links = {
        "base": _visual("base", "axis_part.obj"),
        "door": _visual("door", "axis_part.obj", xyz=(1, 0, 0)),
    }
    pred.joints = {
        "hinge": JointInfo(
            "hinge", "REVOLUTE", "base", "door",
            origin_rpy=pred_origin_rpy,
            axis=pred_axis, limit_lower=0.0, limit_upper=1.0,
        )
    }
    return gt, pred, {"base": "base", "door": "door"}


def test_joint_axes_compares_world_axis_not_local_axis(tmp_path: Path) -> None:
    gt, pred, matching = _axis_models(
        tmp_path,
        pred_axis=(1, 0, 0),
        # Ry(-pi/2) maps the local +X axis onto world +Z.
        pred_origin_rpy=(0, -math.pi / 2, 0),
    )
    result = check_joint_axes(gt, pred, matching, Thresholds())
    assert result.passed, result.detail
    assert "world axes" in result.detail


def test_joint_axes_detects_origin_rotation_of_same_local_axis(
    tmp_path: Path,
) -> None:
    gt, pred, matching = _axis_models(
        tmp_path,
        pred_axis=(0, 0, 1),
        # The same local axis is physically world +X after this rotation.
        pred_origin_rpy=(0, math.pi / 2, 0),
    )
    result = check_joint_axes(gt, pred, matching, Thresholds())
    assert not result.passed
    assert "world-axis cos=0.000" in result.detail


def test_pivot_position_is_blocked_when_world_axis_mismatches(
    tmp_path: Path,
) -> None:
    gt, pred, matching = _axis_models(
        tmp_path,
        pred_axis=(0, 0, 1),
        pred_origin_rpy=(0, math.pi / 2, 0),
    )
    result = check_joint_pivot_alignment(
        gt, pred, matching, Thresholds(kinematic_tol=1e-4),
    )
    assert not result.passed
    assert not result.scored
    assert "BLOCKED" in result.detail
    assert "world axis mismatch" in result.detail
    assert "axis angle" not in result.detail
    assert str(result).startswith("  [BLOCKED]")


def test_reference_q_zero_survives_lower_limit_canonicalization() -> None:
    assert _canonical_reference_position(
        "REVOLUTE", (1.0, 0.0, 0.0), -math.pi / 2, 0.0,
    ) == pytest.approx(math.pi / 2)
    assert _canonical_reference_position(
        "REVOLUTE", (-1.0, 0.0, 0.0), -math.pi / 2, 0.0,
    ) == pytest.approx(0.0)


def test_reversed_limits_are_repaired_only_for_reference_data(tmp_path: Path) -> None:
    urdf = tmp_path / "scene.urdf"
    urdf.write_text(
        """<robot name="test">
        <link name="base"/><link name="arm"/>
        <joint name="hinge" type="revolute">
          <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
          <limit lower="0.5" upper="0.0"/>
        </joint></robot>"""
    )
    prediction = parse_urdf(urdf)
    assert not check_model_integrity(prediction).passed

    reference = parse_urdf(urdf, repair_reversed_limits=True)
    assert check_model_integrity(reference).passed
    assert reference.joints["hinge"].limit_lower == pytest.approx(0.0)
    assert reference.joints["hinge"].limit_upper == pytest.approx(0.5)
    assert reference.joints["hinge"].reference_position == pytest.approx(0.0)


def test_mesh_signature_is_rigid_transform_invariant(tmp_path: Path) -> None:
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 3.0),
        (0.4, 0.2, 0.7),
    ]
    transformed = [(-y + 10.0, x - 4.0, z + 2.0) for x, y, z in points]
    _write_obj(tmp_path / "a.obj", points)
    _write_obj(tmp_path / "b.obj", transformed)
    a = compute_signature("a.obj", tmp_path)
    b = compute_signature("b.obj", tmp_path)
    assert a.distance(b) == pytest.approx(0.0, abs=1e-12)


def test_mesh_signature_is_uniform_scale_invariant(tmp_path: Path) -> None:
    points = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0), (0.0, 0.0, 3.0), (0.4, 0.2, 0.7),
    ]
    _write_obj(tmp_path / "metric.obj", points)
    _write_obj(tmp_path / "anonymous.obj", [
        (3.0 * x, 3.0 * y, 3.0 * z) for x, y, z in points
    ])
    metric = compute_signature("metric.obj", tmp_path)
    anonymous = compute_signature("anonymous.obj", tmp_path)
    assert metric.distance(anonymous) == pytest.approx(0.0, abs=1e-12)


def test_descriptor_distance_applies_similarity_dimensions() -> None:
    gt = PoseDescriptor(
        centroid=(4.0, 6.0, 8.0),
        covariance=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        third_moment=(8.0,) * 27,
        scale=2.0,
    )
    pred = PoseDescriptor(
        centroid=(1.0, 2.0, 3.0),
        covariance=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        third_moment=(1.0,) * 27,
        scale=1.0,
    )
    alignment = SimilarityAlignment(scale=2.0, translation=(2.0, 2.0, 2.0))
    assert _descriptor_distance(gt, pred, alignment) == pytest.approx(0.0)


def test_pose_descriptor_ignores_baked_local_rigid_transform(tmp_path: Path) -> None:
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 3.0),
        (0.4, 0.2, 0.7),
    ]
    transformed = [(-y + 10.0, x - 4.0, z + 2.0) for x, y, z in points]
    _write_obj(tmp_path / "a.obj", points)
    _write_obj(tmp_path / "b.obj", transformed)
    original = _visual("a", "a.obj")
    # b = Trans(10,-4,2) * Rz(90deg) * a; apply the inverse as visual pose.
    restored = LinkInfo(
        name="b", mesh_paths=["b.obj"], has_visual=True,
        visual_xyz=(4.0, 10.0, -2.0), visual_rpy=(0.0, 0.0, -math.pi / 2),
    )
    a_desc = _link_world_descriptor(original, _mat4_identity(), tmp_path, {})
    b_desc = _link_world_descriptor(restored, _mat4_identity(), tmp_path, {})
    assert a_desc is not None and b_desc is not None
    assert _descriptor_distance(a_desc, b_desc, (0.0, 0.0, 0.0)) < 1e-5


def test_pose_descriptor_detects_asymmetric_half_turn(tmp_path: Path) -> None:
    _write_obj(
        tmp_path / "asym.obj",
        [(0, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 0.2), (0.1, 0.1, 1.5)],
    )
    link = _visual("part", "asym.obj")
    rest = _link_world_descriptor(link, _mat4_identity(), tmp_path, {})
    flipped = _link_world_descriptor(
        link, _mat4_rotate_rpy((0.0, 0.0, math.pi)), tmp_path, {},
    )
    assert rest is not None and flipped is not None
    assert _descriptor_distance(rest, flipped, (0.0, 0.0, 0.0)) > 0.1


def test_symmetric_meshes_use_global_rest_pose_tie_break(tmp_path: Path) -> None:
    _write_obj(
        tmp_path / "part.obj",
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    gt = URDFModel("gt", base_dir=tmp_path)
    gt.links = {
        "root": LinkInfo("root"),
        "left": _visual("left", "part.obj"),
        "right": _visual("right", "part.obj"),
    }
    gt.joints = {
        "jl": JointInfo("jl", "FIXED", "root", "left", origin_xyz=(-1, 0, 0)),
        "jr": JointInfo("jr", "FIXED", "root", "right", origin_xyz=(1, 0, 0)),
    }

    pred = URDFModel("pred", base_dir=tmp_path)
    pred.links = {
        "root": LinkInfo("root"),
        # Lexical order is deliberately opposite to spatial identity.
        "a_right": _visual("a_right", "part.obj"),
        "z_left": _visual("z_left", "part.obj"),
    }
    pred.joints = {
        "ja": JointInfo("ja", "FIXED", "root", "a_right", origin_xyz=(1, 0, 0)),
        "jz": JointInfo("jz", "FIXED", "root", "z_left", origin_xyz=(-1, 0, 0)),
    }

    result = _match_links_detailed(gt, pred, max_geometry_cost=0.08)
    assert result.matching == {"left": "z_left", "right": "a_right"}
    assert result.rejected == []


def test_missing_movable_joint_cannot_vacuously_pass_axis_or_limits() -> None:
    gt = URDFModel("gt")
    gt.links = {"base": LinkInfo("base"), "door": LinkInfo("door")}
    gt.joints = {
        "hinge": JointInfo(
            "hinge", "REVOLUTE", "base", "door",
            axis=(0, 0, 1), limit_lower=0.0, limit_upper=1.0,
        )
    }
    pred = URDFModel("pred")
    pred.links = {"base": LinkInfo("base"), "door": LinkInfo("door")}
    pred.joints = {"fixed": JointInfo("fixed", "FIXED", "base", "door")}
    matching = {"base": "base", "door": "door"}
    assert not check_joint_axes(gt, pred, matching, Thresholds()).passed
    assert not check_joint_limits(gt, pred, matching, Thresholds()).passed


def test_prismatic_limits_scale_but_revolute_limits_do_not(tmp_path: Path) -> None:
    _write_obj(
        tmp_path / "gt.obj",
        [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)],
    )
    _write_obj(
        tmp_path / "pred.obj",
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    gt = URDFModel("gt", base_dir=tmp_path)
    gt.links = {
        "base": _visual("base", "gt.obj"),
        "drawer": _visual("drawer", "gt.obj"),
        "door": _visual("door", "gt.obj"),
    }
    gt.joints = {
        "slide": JointInfo(
            "slide", "PRISMATIC", "base", "drawer",
            axis=(1, 0, 0), limit_lower=0.0, limit_upper=0.4,
        ),
        "hinge": JointInfo(
            "hinge", "REVOLUTE", "base", "door",
            axis=(0, 0, 1), limit_lower=0.0, limit_upper=1.2,
        ),
    }
    pred = URDFModel("pred", base_dir=tmp_path)
    pred.links = {
        "base": _visual("base", "pred.obj"),
        "drawer": _visual("drawer", "pred.obj"),
        "door": _visual("door", "pred.obj"),
    }
    pred.joints = {
        "slide": JointInfo(
            "slide", "PRISMATIC", "base", "drawer",
            axis=(1, 0, 0), limit_lower=0.0, limit_upper=0.2,
        ),
        "hinge": JointInfo(
            "hinge", "REVOLUTE", "base", "door",
            axis=(0, 0, 1), limit_lower=0.0, limit_upper=1.2,
        ),
    }
    matching = {"base": "base", "drawer": "drawer", "door": "door"}
    result = check_joint_limits(gt, pred, matching, Thresholds())
    assert result.passed, result.detail

    pred.joints["hinge"].limit_upper = 0.6
    result = check_joint_limits(gt, pred, matching, Thresholds())
    assert not result.passed
    assert "door" in result.detail


def test_rest_and_prismatic_fk_are_globally_scale_invariant(
    tmp_path: Path,
) -> None:
    _write_obj(
        tmp_path / "gt.obj",
        [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (0.4, 0.2, 0.8)],
    )
    _write_obj(
        tmp_path / "pred.obj",
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.1, 0.4)],
    )
    gt = URDFModel("gt", base_dir=tmp_path)
    gt.links = {
        "base": _visual("base", "gt.obj"),
        "drawer": _visual("drawer", "gt.obj", xyz=(2, 0, 0)),
    }
    gt.joints = {
        "slide": JointInfo(
            "slide", "PRISMATIC", "base", "drawer",
            axis=(1, 0, 0), limit_lower=0.0, limit_upper=0.4,
        )
    }
    pred = URDFModel("pred", base_dir=tmp_path)
    pred.links = {
        "base": _visual("base", "pred.obj"),
        "drawer": _visual("drawer", "pred.obj", xyz=(1, 0, 0)),
    }
    pred.joints = {
        "slide": JointInfo(
            "slide", "PRISMATIC", "base", "drawer",
            axis=(1, 0, 0), limit_lower=0.0, limit_upper=0.2,
        )
    }
    matching = {"base": "base", "drawer": "drawer"}
    thresholds = Thresholds(kinematic_tol=1e-6, kinematic_samples=5)
    rest = check_rest_pose_alignment(gt, pred, matching, thresholds)
    motion = check_kinematic_equivalence(gt, pred, matching, thresholds)
    assert rest.passed, rest.detail
    assert motion.passed, motion.detail


def test_kinematic_trace_accepts_reversed_parameter_direction(tmp_path: Path) -> None:
    _write_obj(
        tmp_path / "part.obj",
        [(0, 0, 0), (0.2, 0, 0), (0, 0.1, 0), (0, 0, 0.1)],
    )
    gt = URDFModel("gt", base_dir=tmp_path)
    gt.links = {
        "base": _visual("base", "part.obj"),
        "arm": _visual("arm", "part.obj", xyz=(1, 0, 0)),
    }
    gt.joints = {
        "hinge": JointInfo(
            "hinge", "REVOLUTE", "base", "arm",
            axis=(0, 0, 1), limit_lower=0.0, limit_upper=math.pi / 2,
        )
    }

    pred = URDFModel("pred", base_dir=tmp_path)
    pred.links = {
        "base": _visual("base", "part.obj"),
        "arm": _visual("arm", "part.obj", xyz=(1, 0, 0)),
    }
    pred.joints = {
        # T_pred(q) = Rz(pi/2) * Rz(-q), so pred(1-t) == gt(t).
        "hinge": JointInfo(
            "hinge", "REVOLUTE", "base", "arm",
            origin_rpy=(0, 0, math.pi / 2),
            axis=(0, 0, -1), limit_lower=0.0, limit_upper=math.pi / 2,
        )
    }
    result = check_kinematic_equivalence(
        gt, pred, {"base": "base", "arm": "arm"},
        Thresholds(kinematic_tol=1e-5, kinematic_samples=5),
    )
    assert result.passed, result.detail
    assert "reverse_matches=1" in result.detail


# --- subagent oracle judge backend -----------------------------------------

def test_subagent_oracle_judges_on_the_shared_responses_provider() -> None:
    from evo.oracle import subagent

    assert subagent.DEFAULT_ORACLE_BACKEND == "responses"
    assert subagent.oracle_backend() == "responses"
    assert subagent.oracle_model("responses") == "gpt-6.1-sol"


def test_oracle_backend_and_model_come_from_env(monkeypatch) -> None:
    from evo.oracle import subagent

    monkeypatch.setenv("EVO_ORACLE_BACKEND", "claude")
    monkeypatch.setenv("EVO_ORACLE_MODEL", "claude-opus-4-7")
    assert subagent.oracle_backend() == "claude"
    assert subagent.oracle_model("claude") == "claude-opus-4-7"
    # An explicit argument still wins over the environment.
    assert subagent.oracle_backend("responses") == "responses"
    assert subagent.oracle_model("responses", "gpt-5.6-luna") == "gpt-5.6-luna"


def test_unknown_oracle_backend_is_rejected(monkeypatch) -> None:
    from evo.oracle import subagent

    monkeypatch.setenv("EVO_ORACLE_BACKEND", "gemini")
    with pytest.raises(ValueError, match="EVO_ORACLE_BACKEND"):
        subagent.oracle_backend()


def test_judge_call_routes_to_the_responses_backend(monkeypatch, tmp_path) -> None:
    from evo.engine import responses_backend
    from evo.oracle import subagent

    seen: dict = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        return responses_backend.ResponsesRunOutcome(
            final_output='{"verdict":"PASS","summary":"phi=90"}',
            response_id="resp_1",
            usage_summary={},
            artifact_path=None,
        )

    monkeypatch.setattr(responses_backend, "run_responses_agent", fake_run)
    vote, cost = asyncio.run(subagent._query_subagent(
        "compare these", backend="responses", model="gpt-5.6-luna", cwd=tmp_path,
    ))
    assert vote["verdict"] == "PASS"
    assert cost is None
    assert seen["model"] == "gpt-5.6-luna"
    # Read-only, and scoped to the prediction directory holding the renders.
    assert seen["write_root"] is None
    assert seen["read_roots"] == [tmp_path.resolve()]
    assert seen["cwd"] == tmp_path.resolve()
