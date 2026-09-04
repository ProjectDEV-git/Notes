"""Update the NoteTaker checkout without overwriting local work."""

from __future__ import annotations

import subprocess
from pathlib import Path


# An update runs on the way into a lesson, so it must never be the reason a
# class is not being recorded. School wifi behind a captive portal makes
# `git pull` hang until TCP gives up, which is far longer than anyone will
# wait with a teacher already talking.
NETWORK_TIMEOUT = 10.0
INSTALL_TIMEOUT = 120.0


class UpdateError(RuntimeError):
    """Raised when an update cannot be completed safely."""


def _run(
    command: list[str], cwd: Path, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(
            f"{command[0]} took longer than {timeout:.0f}s and was stopped. "
            "NoteTaker is unchanged, so you can carry on recording."
        ) from exc


def update_checkout(app_dir: Path, python_path: Path) -> tuple[bool, str]:
    """Fast-forward the checkout and refresh its Python dependencies.

    Returns ``(changed, message)``. A dirty or divergent checkout is never
    modified, and dependency failures are reported after the code update.

    Every step is time-bounded: an update that cannot finish promptly is
    abandoned rather than allowed to delay the start of a lesson.
    """
    status = _run(["git", "status", "--porcelain"], app_dir, timeout=NETWORK_TIMEOUT)
    if status.returncode != 0:
        raise UpdateError(status.stderr.strip() or "this folder is not a Git checkout")
    if status.stdout.strip():
        raise UpdateError("local changes are present; commit or stash them before updating")

    pull = _run(["git", "pull", "--ff-only"], app_dir, timeout=NETWORK_TIMEOUT)
    if pull.returncode != 0:
        detail = pull.stderr.strip() or pull.stdout.strip()
        raise UpdateError(detail or "Git could not fast-forward this checkout")

    changed = "Already up to date" not in pull.stdout
    if not changed:
        # Nothing was downloaded, so the installed dependencies still match.
        # Skipping pip here is what keeps the common launch fast.
        return False, pull.stdout.strip() or "Update complete"

    install = _run(
        [str(python_path), "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        app_dir,
        timeout=INSTALL_TIMEOUT,
    )
    if install.returncode != 0:
        detail = install.stderr.strip() or install.stdout.strip()
        raise UpdateError(f"code updated, but Python dependencies could not be refreshed: {detail}")

    return True, pull.stdout.strip() or "Update complete"
