from __future__ import annotations

from pathlib import Path

import pytest

from notetaker import update


def test_update_refuses_dirty_checkout(monkeypatch, tmp_path):
    def fake_run(command, cwd, **kwargs):
        return type("Result", (), {"returncode": 0, "stdout": " M notes.md\n", "stderr": ""})()

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    with pytest.raises(update.UpdateError, match="local changes"):
        update.update_checkout(tmp_path, Path("python"))


def test_update_fast_forwards_and_refreshes_dependencies(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, cwd, **kwargs):
        commands.append(command)
        if command[:3] == ["git", "status", "--porcelain"]:
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("Result", (), {"returncode": 0, "stdout": "Updating abc..def\n", "stderr": ""})()

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    changed, message = update.update_checkout(tmp_path, Path("python"))

    assert changed
    assert "Updating" in message
    assert commands == [
        ["git", "status", "--porcelain"],
        ["git", "pull", "--ff-only"],
        ["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    ]