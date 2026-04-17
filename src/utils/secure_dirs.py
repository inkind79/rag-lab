"""
Helper for creating directories that hold cached or derived user data.

``os.makedirs(..., mode=0o700)`` is filtered through the process umask, so
typical default umasks (002 or 022) silently widen the bits to 0o755 or
worse. This helper does the makedirs first, then chmods every newly-created
directory to ``mode`` so it survives the umask.

Use for caches that may contain document text, embeddings, OCR output, or
session-derived artifacts. Don't use for shared static asset directories.
"""

import os
from pathlib import Path
from typing import Union


def secure_makedirs(path: Union[str, os.PathLike], mode: int = 0o700) -> None:
    """Create ``path`` (and parents) and chmod every newly-created dir to ``mode``.

    Existing parents are left alone so we don't widen-by-tightening a
    directory that the user may have set permissions on intentionally.
    """
    p = Path(path)
    # Walk parents from shallowest → deepest, recording which segments don't
    # exist yet. Anything that already exists keeps its current perms.
    new_segments = []
    cur = p
    while not cur.exists():
        new_segments.append(cur)
        if cur.parent == cur:  # reached root
            break
        cur = cur.parent
    new_segments.reverse()

    p.mkdir(parents=True, exist_ok=True)

    for segment in new_segments:
        try:
            os.chmod(segment, mode)
        except OSError:
            # Best-effort on platforms / filesystems without POSIX perms.
            pass
