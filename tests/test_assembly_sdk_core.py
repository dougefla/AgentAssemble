from __future__ import annotations

import math

import pytest

from evo.assembly_sdk.core import (
    AssemblyError,
    ErrorCode,
    normalize_axis,
    validate_identifier,
    validate_state,
)


def _state() -> dict:
    return {
        "schema_version": 1,
        "parts": {
            "part:a.obj": {"object_name": "a"},
            "part:b.obj": {"object_name": "b"},
        },
        "links": {
            "body": {"part_handles": ["part:a.obj"], "visual_xyz": [0, 0, 0], "visual_rpy": [0, 0, 0]},
            "door": {"part_handles": ["part:b.obj"], "visual_xyz": [0, 0, 0], "visual_rpy": [0, 0, 0]},
        },
        "joints": {
            "hinge": {
                "type": "REVOLUTE", "parent": "body", "child": "door",
                "origin_xyz": [0, 0, 0], "origin_rpy": [0, 0, 0],
                "axis": [0, 0, 1], "limit_lower": 0, "limit_upper": 1.57,
            }
        },
        "joint_positions": {"hinge": 0.0},
    }


def test_normalize_axis_accepts_labels_and_normalizes_vectors() -> None:
    assert normalize_axis("-Y") == (0.0, -1.0, 0.0)
    axis = normalize_axis([1, 2, 2])
    assert math.isclose(sum(v * v for v in axis), 1.0)


def test_zero_axis_has_machine_readable_error() -> None:
    with pytest.raises(AssemblyError) as caught:
        normalize_axis([0, 0, 0], "create_joint")
    assert caught.value.code is ErrorCode.INVALID_ARGUMENT
    assert caught.value.operation == "create_joint"
    assert caught.value.as_dict()["error"]["code"] == "invalid_argument"


def test_valid_complete_tree_passes() -> None:
    assert validate_state(_state(), require_complete_tree=True) == []


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), "not-a-number"])
def test_part_scale_must_be_finite_and_positive(scale) -> None:
    state = _state()
    state["parts"]["part:a.obj"]["scale"] = scale
    assert "part_scale" in {
        issue["code"] for issue in validate_state(state)
    }


def test_legacy_part_without_scale_defaults_to_one() -> None:
    state = _state()
    assert validate_state(state, require_complete_tree=True) == []


def test_complete_tree_rejects_empty_assembly() -> None:
    state = {
        "schema_version": 1,
        "parts": {},
        "links": {},
        "joints": {},
        "joint_positions": {},
    }
    with pytest.raises(AssemblyError) as caught:
        validate_state(state, require_complete_tree=True)
    assert "empty_assembly" in {
        issue["code"] for issue in caught.value.details["issues"]
    }


def test_cycle_and_multiple_parent_are_rejected() -> None:
    state = _state()
    state["joints"]["back"] = {
        "type": "FIXED", "parent": "door", "child": "body",
        "origin_xyz": [0, 0, 0], "origin_rpy": [0, 0, 0],
        "axis": [0, 0, 1], "limit_lower": None, "limit_upper": None,
    }
    state["joint_positions"]["back"] = 0.0
    with pytest.raises(AssemblyError) as caught:
        validate_state(state, require_complete_tree=True)
    issue_codes = {issue["code"] for issue in caught.value.details["issues"]}
    assert "cycle" in issue_codes


def test_partial_forest_validation_still_rejects_cycles() -> None:
    state = _state()
    state["joints"]["back"] = {
        "type": "FIXED", "parent": "door", "child": "body",
        "origin_xyz": [0, 0, 0], "origin_rpy": [0, 0, 0],
        "axis": [0, 0, 1], "limit_lower": None, "limit_upper": None,
    }
    state["joint_positions"]["back"] = 0.0
    assert "cycle" in {
        issue["code"] for issue in validate_state(
            state, require_complete_tree=False
        )
    }


def test_duplicate_part_ownership_is_rejected_at_mutator_boundary() -> None:
    state = _state()
    state["links"]["door"]["part_handles"] = ["part:a.obj"]
    issues = validate_state(state)
    assert "duplicate_part_owner" in {issue["code"] for issue in issues}


def test_malformed_link_is_reported_without_attribute_error() -> None:
    state = _state()
    state["links"]["door"] = "not a dictionary"
    issues = validate_state(state)
    assert "link_shape" in {issue["code"] for issue in issues}


def test_malformed_nested_handles_and_joint_links_are_structured() -> None:
    state = _state()
    state["links"]["door"]["part_handles"] = [["unhashable"]]
    state["joints"]["hinge"]["parent"] = ["unhashable"]
    codes = {issue["code"] for issue in validate_state(state)}
    assert "part_handle" in codes
    assert "joint_links" in codes


def test_malformed_part_registry_entry_is_structured() -> None:
    state = _state()
    state["parts"]["part:a.obj"] = "not a dictionary"
    assert "part_shape" in {issue["code"] for issue in validate_state(state)}


def test_complete_tree_rejects_unassigned_parts() -> None:
    state = _state()
    state["parts"]["part:orphan.obj"] = {"object_name": "orphan"}
    with pytest.raises(AssemblyError) as caught:
        validate_state(state, require_complete_tree=True)
    assert "unassigned_part" in {
        issue["code"] for issue in caught.value.details["issues"]
    }


def test_joint_position_keys_and_values_are_part_of_state_contract() -> None:
    state = _state()
    state["joint_positions"] = {"stale": 0.0}
    codes = {issue["code"] for issue in validate_state(state)}
    assert "stale_joint_position" in codes
    assert "missing_joint_position" in codes


def test_complete_validation_does_not_duplicate_issues() -> None:
    state = _state()
    state["links"]["door"]["part_handles"] = []
    with pytest.raises(AssemblyError) as caught:
        validate_state(state, require_complete_tree=True)
    codes = [issue["code"] for issue in caught.value.details["issues"]]
    assert codes.count("empty_link") == 1


def test_urdf_identifier_contract_is_machine_readable() -> None:
    assert validate_identifier("door_joint.1", "create_joint") == "door_joint.1"
    with pytest.raises(AssemblyError) as caught:
        validate_identifier('bad name "quoted"', "create_link")
    assert caught.value.code is ErrorCode.INVALID_ARGUMENT
