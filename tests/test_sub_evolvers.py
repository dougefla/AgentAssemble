"""EVO_SUB_EVOLVERS: cluster selection, cooling, notes and proposal diffs."""
from __future__ import annotations

from pathlib import Path

from evo.engine.sub_evolvers import (
    cooled_clusters, integrator_note, journal_section, proposal_record, select_clusters,
    skill_diff, sub_evolver_count, sub_note,
)


def _clusters():
    return [
        {"id": "C1", "label": "pose·orientation · REVOLUTE · flipped", "key": ("pose·orientation", "REVOLUTE", "flipped"),
         "n_tasks": 7, "tasks": {f"t{i}/0": [] for i in range(7)}},
        {"id": "C2", "label": "scale", "key": ("scale", "PRISMATIC", "small"), "n_tasks": 3,
         "tasks": {f"u{i}/0": [] for i in range(3)}},
        {"id": "C3", "label": "pose·position", "key": ("pose·position", "PRISMATIC", "off"), "n_tasks": 5,
         "tasks": {f"v{i}/0": [] for i in range(5)}},
    ]


def test_selection_needs_unseen_members_and_respects_k_and_cooling() -> None:
    trial = {"C1": ["t0/0", "t1/0", "t2/0"], "C2": ["u0/0", "u1/0", "u2/0"], "C3": ["v0/0", "v1/0", "v2/0"]}
    sel = select_clusters(_clusters(), trial, 4)
    assert [c["id"] for c in sel] == ["C1", "C3"]                       # C2 has no member left to verify on
    assert sel[0]["verify"][:3] == ["t3/0", "t4/0", "t5/0"] and len(sel[0]["verify"]) <= 6 and not set(sel[0]["verify"]) & set(sel[0]["dev"])
    assert [c["id"] for c in select_clusters(_clusters(), trial, 1)] == ["C1"]
    cooled = {"pose·orientation|REVOLUTE|flipped"}
    assert [c["id"] for c in select_clusters(_clusters(), trial, 4, cooled=cooled)] == ["C3"]


def test_cooling_after_two_unverified_rounds_lasts_two_rounds() -> None:
    miss = {"sub_evolvers": [{"key": "a", "verified": False}]}
    hit = {"sub_evolvers": [{"key": "a", "verified": True}]}
    assert cooled_clusters([miss]) == set()
    assert cooled_clusters([miss, miss]) == {"a"}
    assert cooled_clusters([miss, miss, {}]) == {"a"}
    assert cooled_clusters([miss, miss, {}, {}]) == set()
    assert cooled_clusters([miss, hit]) == set()


def test_notes_and_diff(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base"; new = tmp_path / "new"
    for d in (base, new):
        d.mkdir()
    (base / "SKILL.md").write_text("a\nb\nc\n")
    (new / "SKILL.md").write_text("a\nB\nc\nd\n")
    diff, added, removed = skill_diff(base, new)
    assert added == 2 and removed == 1 and "+B" in diff
    (tmp_path / "j.md").write_text("## Observed\nx\n## Changed\nC1 — rule — kept\n## Verified\ny\n")
    assert journal_section(tmp_path / "j.md", "Changed") == "C1 — rule — kept"
    c = select_clusters(_clusters(), {"C1": ["t0/0", "t1/0", "t2/0"]}, 1)[0]
    assert "SUB-EVOLVER" in sub_note(c) and 'try_skill(cluster="C1")' in sub_note(c)
    p = {"cluster": "C1", "label": c["label"], "key": c["key"], "n_tasks": 7, "status": "verified on unseen cluster members",
         "skill_dir": str(new), "journal": str(tmp_path / "j.md"), "changed": True, "verified": True,
         "changed_text": "C1 — rule — kept", "diff": diff, "added": added, "removed": removed, "cost_usd": 1.23456,
         "dev_gate": {"ok": True, "improved": True, "fitness_before": 0.2, "fitness_after": 0.5, "pass_before": 0,
                      "pass_after": 1, "n_tasks": 3},
         "verify_gate": {"ok": True, "improved": False, "fitness_before": 0.3, "fitness_after": 0.3, "pass_before": 0,
                         "pass_after": 0, "n_tasks": 3}}
    note = integrator_note([p], skill_lines=120)
    assert "INTEGRATOR" in note and "NOT improved" in note and "fitness 0.2 → 0.5" in note
    rec = proposal_record(p)
    assert rec["verified"] and rec["dev_gate"]["improved"] and rec["cost_usd"] == 1.2346
    monkeypatch.setenv("EVO_SUB_EVOLVERS", "4")
    assert sub_evolver_count() == 4
    monkeypatch.setenv("EVO_SUB_EVOLVERS", "x")
    assert sub_evolver_count() == 0


def test_is_verified_uses_gain_and_tolerates_missing_diff() -> None:
    from evo.engine.sub_evolvers import is_verified
    base = {"ok": True, "improved": True, "fitness_before": 0.212, "fitness_after": 0.344, "n_tasks": 6}
    assert is_verified(base)                                   # no fitness_diff: recomputed
    assert is_verified({**base, "fitness_diff": 0.132})
    assert not is_verified({**base, "fitness_diff": 0.02})
    assert not is_verified({**base, "improved": False})
    assert not is_verified({"ok": False, "error": "x"})
    assert not is_verified(None)
    assert not is_verified({**base, "n_tasks": 2})              # too few unseen members to verify on
