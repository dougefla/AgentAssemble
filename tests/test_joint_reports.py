"""Per-GT-joint reports: oracle side, A/B pairing, and generator formatting."""
from __future__ import annotations

import math
from pathlib import Path

from evo.engine.agents import _format_joint_report, _format_joint_summary
from evo.engine.loop import _paired_joint_scores, _paired_task_scores
from evo.oracle.check_urdf import (
    CheckResult, JointInfo, JointReport, LinkInfo, OracleResult, Thresholds,
    URDFModel, check_joint_axes, check_joint_types, sanitize_oracle_dict,
)


def _model(name: str, axis: tuple[float, float, float], jtype: str = "REVOLUTE") -> URDFModel:
    m = URDFModel(name, base_dir=Path("."))
    m.links = {
        "base": LinkInfo(name="base", mesh_paths=["base.obj"], has_visual=True),
        "door": LinkInfo(name="door", mesh_paths=["door.obj"], has_visual=True),
    }
    m.joints = {
        "hinge": JointInfo(
            "hinge", jtype, "base", "door",
            origin_xyz=(0.5, 0.0, 0.0), axis=axis, limit_lower=0.0, limit_upper=math.pi / 2,
        )
    }
    return m


def _reports(gt: URDFModel, matching: dict[str, str]) -> dict[str, JointReport]:
    return {
        jname: JointReport(gt_joint=jname, gt_child=gj.child, gt_type=gj.type,
                           matched=gj.child in matching)
        for jname, gj in gt.movable_joints.items()
    }


def test_axis_check_records_angle_error_per_joint() -> None:
    gt = _model("gt", (0.0, 0.0, 1.0))
    matching = {"base": "base", "door": "door"}
    for pred_axis, expect_deg, expect_ok in (((0.0, 0.0, 1.0), 0.0, True), ((1.0, 0.0, 0.0), 90.0, False)):
        reports = _reports(gt, matching)
        check = check_joint_axes(gt, _model("pred", pred_axis), matching, Thresholds(), reports=reports)
        rep = reports["hinge"]
        assert check.passed is expect_ok
        assert rep.axis_ok is expect_ok
        assert math.isclose(rep.axis_angle_deg, expect_deg, abs_tol=1e-6)


def test_type_check_records_pred_joint_and_type() -> None:
    gt = _model("gt", (0.0, 0.0, 1.0), "REVOLUTE")
    matching = {"base": "base", "door": "door"}
    reports = _reports(gt, matching)
    check_joint_types(gt, _model("pred", (0.0, 0.0, 1.0), "PRISMATIC"), matching, reports=reports)
    rep = reports["hinge"]
    assert rep.pred_joint == "hinge" and rep.pred_child == "door" and rep.pred_type == "PRISMATIC"
    assert rep.type_ok is False
    reports = _reports(gt, matching)
    check_joint_types(gt, _model("pred", (0.0, 0.0, 1.0), "CONTINUOUS"), matching, reports=reports)
    assert reports["hinge"].type_ok is True  # rotational family is equivalent


def test_joint_score_and_redaction() -> None:
    rep = JointReport(gt_joint="j", gt_child="door", gt_type="REVOLUTE", matched=True,
                      pred_joint="hinge", pred_type="REVOLUTE",
                      type_ok=True, axis_ok=True, pivot_ok=None, limits_ok=False, motion_ok=False)
    assert rep.score() == 0.5          # pivot not comparable → excluded from the denominator
    assert JointReport(gt_joint="j", gt_child="door", gt_type="REVOLUTE").score() == 0.0
    full = OracleResult(False, [CheckResult("x", False)], joints={"j": rep}).to_dict()
    assert full["joints"]["j"]["gt_type"] == "REVOLUTE"
    safe = sanitize_oracle_dict(full)
    assert "gt_type" not in safe["joints"]["j"] and "gt_child" not in safe["joints"]["j"]
    assert safe["joints"]["j"]["pred_type"] == "REVOLUTE" and safe["joints"]["j"]["score"] == 0.5


def _result(joints: dict[str, float]) -> OracleResult:
    reps = {}
    for name, score in joints.items():
        ok = score >= 0.5
        reps[name] = JointReport(gt_joint=name, gt_child=name, gt_type="REVOLUTE", matched=True,
                                 pred_joint="p", pred_type="REVOLUTE",
                                 type_ok=ok, axis_ok=ok, limits_ok=ok, motion_ok=ok)
    return OracleResult(all(s >= 0.5 for s in joints.values()), [CheckResult("c", True)], joints=reps)


def test_joint_pairing_yields_one_sample_per_shared_gt_joint() -> None:
    base = [("t", "0", None, _result({"j0": 1.0, "j1": 0.0}), Path("."), Path(".")),
            ("t", "1", None, _result({"j0": 1.0}), Path("."), Path("."))]
    cand = [("t", "0", None, _result({"j0": 1.0, "j1": 1.0}), Path("."), Path(".")),
            ("t", "1", None, None, Path("."), Path("."))]
    pairs = _paired_joint_scores(base, cand)
    assert [k for k, _b, _c in pairs] == ["t/0::j0", "t/0::j1"]
    assert pairs[1][1] == 0.0 and pairs[1][2] == 1.0
    # A judge-only result carries no joints → no joint pairs, task pairing still works.
    judge = OracleResult(True, [CheckResult("subagent_verdict", True)])
    assert _paired_joint_scores([("t", "0", None, judge, Path("."), Path("."))],
                                [("t", "0", None, judge, Path("."), Path("."))]) == []
    assert len(_paired_task_scores([("t", "0", None, judge, Path("."), Path("."))],
                                   [("t", "0", None, judge, Path("."), Path("."))])) == 1


def test_generator_formatting_is_compact_and_grouped_by_pred_type() -> None:
    rep = {"matched": True, "pred_joint": "lid_hinge", "pred_type": "REVOLUTE", "type_ok": True,
           "axis_ok": False, "axis_angle_deg": 90.0, "pivot_ok": None, "limits_ok": False,
           "limit_ratio": 1.375, "motion_ok": False, "trace_drift_m": 0.2615, "score": 0.25}
    line = _format_joint_report(rep)
    assert "pred=lid_hinge (REVOLUTE)" in line and "axis✗ 90°" in line
    assert "limits✗ ×1.38" in line and "motion✗ drift 26.2cm" in line and "pivot?" in line
    assert _format_joint_report({"matched": False}) .startswith("unmatched")
    summary = _format_joint_summary([{"joint_reports": {"joint_0": rep}},
                                     {"joint_reports": {"joint_0": {**rep, "pred_type": "PRISMATIC", "axis_ok": True}}}])
    assert "2 GT joints" in summary and "REVOLUTE" in summary and "PRISMATIC" in summary
    assert "axis 0/1" in summary and "axis 1/1" in summary
    assert _format_joint_summary([{"joint_reports": {}}]) == ""
