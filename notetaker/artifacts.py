"""Crash-safe artifact writes for NoteTaker.

Notes and re-written transcripts must never be left half-written: a laptop
dying mid-save should lose nothing, not corrupt the file. The standard trick
is write-to-a-sibling-temp-file then atomic rename (os.replace), which either
leaves the old file intact or the new one complete -- never a mix.

See docs/BUILD_PLAN.md phase 5.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _fsync_directory(directory: Path) -> None:
    """Persist the rename itself, where the platform supports it.

    os.replace is atomic, but the rename only becomes durable after the
    directory entry is synced. Not every OS/filesystem allows opening a
    directory, so failures here are deliberately ignored: the rename has
    already happened and this is belt-and-braces.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except (OSError, PermissionError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_text_atomic(path: Path | str, text: str) -> None:
    """Replace `path` with `text` atomically, or leave the old file untouched.

    The temporary file is created in the same directory as the target so the
    final rename is an atomic same-filesystem operation. On any failure the
    temp file is removed and the original content is preserved.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=target.name + ".", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _fsync_directory(target.parent)
