"""Lossless disk saving for run directories.

Every session exports its part meshes next to the URDF, and with K sessions per
task per round the same OBJ bytes are written hundreds of times (97% of a run
directory). :func:`hardlink_duplicate_meshes` replaces byte-identical ``*.obj``
files with hardlinks to one copy; the content of every path is unchanged.
"""
from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path

MIN_BYTES = 4096


def _digest(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hardlink_duplicate_meshes(roots: list[Path], index: dict[tuple[int, str], str] | None = None) -> tuple[int, int]:
    """Hardlink identical ``*.obj`` files under ``roots``. ``index`` maps
    ``(size, sha1)`` to a kept path and can be carried across calls so later
    rounds link against earlier ones. Returns ``(files linked, bytes freed)``."""
    index = {} if index is None else index
    by_size: dict[int, list[str]] = defaultdict(list)
    for root in roots:
        if not Path(root).is_dir():
            continue
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                if fn.endswith(".obj"):
                    p = os.path.join(dp, fn)
                    try:
                        st = os.lstat(p)
                        # Already-linked files were handled by an earlier call (their
                        # key is in ``index`` when it is carried over): skip the rehash.
                        if os.path.islink(p) or st.st_size < MIN_BYTES or (index and st.st_nlink > 1):
                            continue
                        by_size[st.st_size].append(p)
                    except OSError:
                        pass
    linked = freed = 0
    for size, paths in by_size.items():
        for p in paths:
            try:
                key = (size, _digest(p))
                keep = index.get(key)
                if keep is None or not os.path.exists(keep):
                    index[key] = p
                    continue
                if os.stat(keep).st_ino == os.stat(p).st_ino:
                    continue
                tmp = p + ".lnk_tmp"
                os.link(keep, tmp)
                os.replace(tmp, p)
                linked += 1
                freed += size
            except OSError:
                continue
    return linked, freed
