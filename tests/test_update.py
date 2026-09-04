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

# ------------------------------------------------------ never delay a lesson
def test_a_hanging_network_cannot_block_the_start_of_a_class(monkeypatch, tmp_path):
    """School wifi behind a captive portal makes git pull hang forever."""
    def fake_run(command, cwd, **kwargs):
        if command[:3] == ["git", "status", "--porcelain"]:
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        raise update.subprocess.TimeoutExpired(command, kwargs.get("timeout", 10))

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    with pytest.raises(update.UpdateError, match="carry on recording"):
        update.update_checkout(tmp_path, Path("python"))


def test_every_network_step_is_time_bounded(monkeypatch, tmp_path):
    """A step with no timeout is a step that can hang indefinitely."""
    timeouts = []

    def fake_run(command, cwd, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        if command[:3] == ["git", "status", "--porcelain"]:
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("R", (), {"returncode": 0, "stdout": "Updating a..b\n", "stderr": ""})()

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    update.update_checkout(tmp_path, Path("python"))

    assert timeouts, "no commands were run"
    assert all(t for t in timeouts), f"a step had no timeout: {timeouts}"


def test_an_up_to_date_checkout_does_not_pay_for_pip(monkeypatch, tmp_path):
    """The common launch must stay fast: nothing changed, nothing to install."""
    commands = []

    def fake_run(command, cwd, **kwargs):
        commands.append(command)
        if command[:3] == ["git", "status", "--porcelain"]:
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("R", (), {"returncode": 0, "stdout": "Already up to date.\n", "stderr": ""})()

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    changed, _ = update.update_checkout(tmp_path, Path("python"))

    assert changed is False
    assert not any("pip" in c for c in commands), "reinstalled dependencies for nothing"
