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


def test_a_failed_update_never_stops_the_app_from_running(tmp_path):
    """A student with bad wifi must still be able to record."""
    import subprocess as sp
    from pathlib import Path

    launcher = Path(__file__).resolve().parent.parent / "scripts" / "notes"
    traced = launcher.read_text().replace('exec "$PY" -m notetaker.cli', "echo CLI")
    # Force the update step to fail outright.
    traced = traced.replace('"$PY" -m notetaker.cli update --quiet', "false")
    # The dirty-checkout guard would otherwise return before the update runs,
    # and this test is about what happens when the update itself fails.
    traced = traced.replace("git status --porcelain 2>/dev/null", "true")
    script = tmp_path / "notes"
    script.write_text(traced)

    result = sp.run(["bash", str(script), "all"], capture_output=True, text=True, timeout=60)
    assert "CLI list" in result.stdout, "a failed update blocked the app"
    assert "notes update" in result.stderr, "the user was not told how to retry"
