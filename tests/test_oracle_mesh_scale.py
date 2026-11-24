"""Oracle 2.0: a URDF ``<mesh scale>`` is honoured when the geometry is loaded.

Before 2.0 the oracle read link geometry from the OBJ files as written and the
scale attribute only reached it through the loop's finalisation step, which
bakes the scale into staged vertex copies and is skipped for a URDF that already
carries visual origins and resolvable meshes. A prediction that filled the task
template in place with a non-unit scale was therefore scored on unit-size
meshes.

The unit tests here need no data. The end-to-end tests use the
``artic300_rotscale`` dataset (``EVO_ROTSCALE300_ROOT``, default
``~/Dataset/artic300_rotscale``) and skip when it is absent.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from evo.oracle import check_urdf as cu
from evo.oracle.check_urdf import (
    ORACLE_VERSION,
    JointReport,
    OracleResult,
    Thresholds,
    compute_signature,
    configure_scored_checks,
    link_signature,
    parse_urdf,
    run_oracle,
)

DATASET = Path(os.environ.get("EVO_ROTSCALE300_ROOT", "~/Dataset/artic300_rotscale")).expanduser()
# 1-joint tasks of artic300_rotscale (one root part + one moving part).
ONE_JOINT_TASKS = ("task_000", "task_001", "task_002")

needs_dataset = pytest.mark.skipif(
    not (DATASET / "gt" / ONE_JOINT_TASKS[0] / "0" / "scene.urdf").is_file(),
    reason=f"artic300_rotscale not found under {DATASET}",
)


# --------------------------------------------------------------------------
# Unit (no data)
# --------------------------------------------------------------------------

def _write_obj(path: Path, verts) -> None:
    path.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in verts) + "f 1 2 3\n")


def test_compute_signature_applies_scale(tmp_path):
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0), (1.0, 2.0, 3.0)]
    _write_obj(tmp_path / "a.obj", verts)
    plain = compute_signature("a.obj", tmp_path)
    unit = compute_signature("a.obj", tmp_path, (1.0, 1.0, 1.0))
    assert unit is plain, "unit scale must hit the same cache entry as no scale"
    s = 0.37
    scaled = compute_signature("a.obj", tmp_path, (s, s, s))
    assert scaled.vertex_count == plain.vertex_count
    assert math.isclose(scaled.rms_radius, s * plain.rms_radius, rel_tol=1e-12)
    assert all(math.isclose(a, s * b, rel_tol=1e-12, abs_tol=1e-15)
               for a, b in zip(scaled.centroid, plain.centroid))
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
               for a, b in zip(scaled.radial_quantiles, plain.radial_quantiles)), \
        "normalised radial moments are invariant under uniform scale"
    aniso = compute_signature("a.obj", tmp_path, (1.0, 1.0, 2.0))
    assert aniso.bbox_max == (1.0, 2.0, 6.0)


def test_parse_urdf_reads_mesh_scale(tmp_path):
    (tmp_path / "m.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (tmp_path / "r.urdf").write_text("""<robot name="r">
      <link name="a"><visual><origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry><mesh filename="m.obj" scale="0.5 0.5 0.5"/></geometry></visual></link>
      <link name="b"><visual><geometry><mesh filename="m.obj"/></geometry></visual></link>
      <joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>
    </robot>""")
    model = parse_urdf(tmp_path / "r.urdf")
    assert model.links["a"].mesh_scales == [(0.5, 0.5, 0.5)]
    assert model.links["a"].mesh_scale == (0.5, 0.5, 0.5)
    assert model.links["b"].mesh_scale == (1.0, 1.0, 1.0)
    sa = link_signature(model.links["a"], tmp_path)
    sb = link_signature(model.links["b"], tmp_path)
    assert math.isclose(sa.rms_radius, 0.5 * sb.rms_radius, rel_tol=1e-12)


def test_oracle_result_records_version():
    d = OracleResult(True, []).to_dict()
    assert d["oracle_version"] == ORACLE_VERSION == "3.0-surface"


# --------------------------------------------------------------------------
# End-to-end on artic300_rotscale (skips when the dataset is absent)
# --------------------------------------------------------------------------

@pytest.fixture
def rotscale_variant():
    """Score exactly as the loop does for this dataset, then restore the defaults."""
    variant = json.loads((DATASET / "dataset_variant.json").read_text())
    old_checks, old_dims = cu.SCORED_CHECKS, JointReport.DIMS
    configure_scored_checks(variant["scored_checks"])
    JointReport.configure_dims(variant["scored_joint_dims"])
    try:
        yield variant
    finally:
        cu.SCORED_CHECKS = old_checks
        JointReport.DIMS = old_dims


def _copy_gt_as_pred(task: str, dst: Path) -> Path:
    """gt/<task>/0 copied as a prediction dir: furniture.urdf + assets/ (visual origins
    present, meshes resolvable, so the loop's finalisation step would leave it alone)."""
    src = DATASET / "gt" / task / "0"
    dst.mkdir(parents=True)
    shutil.copytree(src / "assets", dst / "assets")
    shutil.copy2(src / "scene.urdf", dst / "furniture.urdf")
    return dst


def _unit_scale_meshes(pred_dir: Path, scales: dict[str, float], *, write_attr: bool) -> None:
    """Divide every non-root link's vertices by ``scales[link]`` and, with ``write_attr``,
    set ``<mesh scale="s s s">`` so the URDF describes the same object."""
    urdf = pred_dir / "furniture.urdf"
    root = ET.parse(urdf).getroot()
    rewritten: set[Path] = set()  # visual and collision reference the same OBJ
    for link_el in root.findall("link"):
        s = scales.get(link_el.attrib["name"])
        if s is None:
            continue
        for mesh_el in link_el.iter("mesh"):
            obj = pred_dir / mesh_el.attrib["filename"]
            if obj not in rewritten:
                rewritten.add(obj)
                lines = []
                for line in obj.read_text().splitlines(keepends=True):
                    if line.startswith("v "):
                        p = line.split()
                        lines.append("v %.9g %.9g %.9g\n" % tuple(float(c) / s for c in p[1:4]))
                    else:
                        lines.append(line)
                obj.write_text("".join(lines))
            if write_attr:
                mesh_el.set("scale", f"{s} {s} {s}")
            else:
                mesh_el.attrib.pop("scale", None)
    ET.ElementTree(root).write(urdf, encoding="unicode")


def _summary(result: OracleResult) -> dict:
    return {
        "passed": result.passed,
        "score": round(result.score, 3),
        "checks": {c.name: c.passed for c in result.checks},
    }


@needs_dataset
@pytest.mark.parametrize("task", ONE_JOINT_TASKS)
def test_gt_scores_perfectly_against_itself(task, rotscale_variant):
    gt_dir = DATASET / "gt" / task / "0"
    result = run_oracle(gt_dir, gt_dir, Thresholds())
    assert result.passed, result.report()
    assert result.score == 1.0
    assert result.to_dict()["oracle_version"] == ORACLE_VERSION


@needs_dataset
@pytest.mark.parametrize("task", ONE_JOINT_TASKS)
def test_mesh_scale_attribute_scores_like_baked_geometry(task, tmp_path, rotscale_variant):
    gt_dir = DATASET / "gt" / task / "0"
    gt = parse_urdf(gt_dir / "scene.urdf", allow_duplicate_joint_names=True)
    root = gt.root_link()
    moving = sorted(n for n in gt.visual_links if n != root)
    assert moving, "need at least one non-root part"
    scales = {name: 0.3 + 0.2 * i for i, name in enumerate(moving)}  # 0.3, 0.5, ...

    original = run_oracle(gt_dir, _copy_gt_as_pred(task, tmp_path / "orig"), Thresholds())
    assert original.passed and original.score == 1.0, original.report()

    # Unit-scaled meshes + <mesh scale="s s s">: the same object, so the same verdict,
    # score and per-check outcome, with no staging step in between.
    scaled_dir = _copy_gt_as_pred(task, tmp_path / "scaled")
    _unit_scale_meshes(scaled_dir, scales, write_attr=True)
    scaled = run_oracle(gt_dir, scaled_dir, Thresholds())
    assert _summary(scaled) == _summary(original), scaled.report()
    for rep in (scaled.joints or {}).values():
        if rep.scale_ratio is not None:
            assert abs(rep.scale_ratio - 1.0) < 1e-6

    # Control: the same unit-scaled meshes without the attribute really are off-size,
    # so the equality above comes from the attribute being honoured.
    bare_dir = _copy_gt_as_pred(task, tmp_path / "bare")
    _unit_scale_meshes(bare_dir, scales, write_attr=False)
    bare = run_oracle(gt_dir, bare_dir, Thresholds())
    assert not bare.passed
    assert _summary(bare)["checks"]["link_scale"] is False
