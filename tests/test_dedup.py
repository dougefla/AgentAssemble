from pathlib import Path

from evo.engine.dedup import hardlink_duplicate_meshes


def test_identical_meshes_are_hardlinked_and_content_kept(tmp_path: Path) -> None:
    a, b, c = tmp_path / "r0" / "a.obj", tmp_path / "r0" / "rep_1" / "a.obj", tmp_path / "r1" / "a.obj"
    for p in (a, b, c):
        p.parent.mkdir(parents=True, exist_ok=True)
    a.write_bytes(b"v 0 0 0\n" * 1000)
    b.write_bytes(b"v 0 0 0\n" * 1000)
    c.write_bytes(b"v 1 0 0\n" * 1000)                      # same size, different content
    index: dict = {}
    assert hardlink_duplicate_meshes([tmp_path / "r0"], index) == (1, 8000)
    assert a.stat().st_ino == b.stat().st_ino and b.read_bytes() == b"v 0 0 0\n" * 1000
    d = tmp_path / "r1" / "d.obj"
    d.write_bytes(b"v 0 0 0\n" * 1000)
    assert hardlink_duplicate_meshes([tmp_path / "r1"], index) == (1, 8000)   # links against round 0
    assert d.stat().st_ino == a.stat().st_ino and c.stat().st_ino != a.stat().st_ino
    assert c.read_bytes() == b"v 1 0 0\n" * 1000
