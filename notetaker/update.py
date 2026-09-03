"""Update the NoteTaker checkout without overwriting local work."""

from __future__ import annotations

import subprocess
from pathlib import Path


class UpdateError(RuntimeError):
    """Raised when an update cannot be completed safely."""


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def update_checkout(app_dir: Path, python_path: Path) -> tuple[bool, str]:
    """Fast-forward the checkout and refresh its Python dependencies.

    Returns ``(changed, message)``. A dirty or divergent checkout is never
    modified, and dependency failures are reported after the code update.
    """
    status = _run(["git", "status", "--porcelain"], app_dir)
    if status.returncode != 0:
        raise UpdateError(status.stderr.strip() or "this folder is not a Git checkout")
    if status.stdout.strip():
        raise UpdateError("local changes are present; commit or stash them before updating")

    pull = _run(["git", "pull", "--ff-only"], app_dir)
    if pull.returncode != 0:
        detail = pull.stderr.strip() or pull.stdout.strip()
        raise UpdateError(detail or "Git could not fast-forward this checkout")

    install = _run(
        [str(python_path), "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        app_dir,
    )
    if install.returncode != 0:
        detail = install.stderr.strip() or install.stdout.strip()
        raise UpdateError(f"code updated, but Python dependencies could not be refreshed: {detail}")

    changed = "Already up to date" not in pull.stdout
    return changed, pull.stdout.strip() or "Update complete"