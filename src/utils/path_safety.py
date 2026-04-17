"""
Path-traversal-safe filename and join helpers for user-supplied paths.

Used to defend file upload, view, and delete endpoints against attempts like
``../../etc/passwd`` or absolute paths in user-supplied filenames.
"""

import os
import re
from pathlib import Path
from typing import Iterable

ALLOWED_DOC_EXTENSIONS = frozenset({
    '.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp',
})

MAX_FILENAME_LENGTH = 255

_UNSAFE_CHARS = re.compile(r'[\x00-\x1f\x7f]')


class UnsafePathError(ValueError):
    """Raised when a user-supplied path fails safety validation."""


def safe_filename(name: str, allowed_extensions: Iterable[str] = ALLOWED_DOC_EXTENSIONS) -> str:
    """Validate that ``name`` is a plain filename with an allowed extension.

    Returns the basename. Raises ``UnsafePathError`` on path separators,
    null bytes, control characters, parent-dir refs, or disallowed extensions.
    """
    if not name or not name.strip():
        raise UnsafePathError("Empty filename")

    if len(name) > MAX_FILENAME_LENGTH:
        raise UnsafePathError(f"Filename exceeds {MAX_FILENAME_LENGTH} chars")

    if _UNSAFE_CHARS.search(name):
        raise UnsafePathError("Filename contains control characters")

    # Reject any path components — accept only a bare filename.
    if name != os.path.basename(name) or '/' in name or '\\' in name or name.startswith('.'):
        raise UnsafePathError(f"Invalid filename: {name!r}")

    if name in {'.', '..'}:
        raise UnsafePathError(f"Invalid filename: {name!r}")

    ext = os.path.splitext(name)[1].lower()
    allowed = {e.lower() for e in allowed_extensions}
    if ext not in allowed:
        raise UnsafePathError(f"Extension {ext!r} not allowed")

    return name


def safe_join(base_dir: str | os.PathLike, *parts: str) -> Path:
    """Join ``parts`` onto ``base_dir`` and verify the result stays within base.

    Resolves symlinks/`..` and raises ``UnsafePathError`` if the final path
    escapes ``base_dir``. ``base_dir`` itself does not need to exist yet, but
    its parent must (so resolution can canonicalize the prefix).
    """
    base = Path(base_dir).resolve()
    candidate = base.joinpath(*parts).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise UnsafePathError(f"Path escapes base directory: {candidate}") from exc
    return candidate
