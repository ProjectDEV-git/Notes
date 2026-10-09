"""Windows support that can be verified without a Windows machine.

The PowerShell installer is parse-checked when pwsh is available, and always
checked for the mistakes that hurt a beginner most.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from notetaker import config, shortcuts

REPO = Path(__file__).resolve().parent.parent
PS1 = REPO / "install.ps1"


def test_installer_never_calls_exit():
    """`irm | iex` runs in the user's own window: exit would close it."""
    code = "\n".join(l for l in PS1.read_text().splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"^\s*exit\b|;\s*exit\b|\{\s*exit\b", code, re.M)


def test_installer_avoids_powershell_7_only_syntax():
    """Windows 10/11 ship PowerShell 5.1, which rejects these."""
    code = "\n".join(l for l in PS1.read_text().splitlines() if not l.lstrip().startswith("#"))
    assert " ?? " not in code
    assert " && " not in code and " || " not in code


def test_installer_uses_official_winget_packages():
    text = PS1.read_text()
    for package in ("Python.Python.3.12", "Gyan.FFmpeg", "Git.Git", "Ollama.Ollama"):
        assert package in text


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell not installed")
def test_installer_parses():
    script = (
        "$e=$null; [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{PS1}', [ref]$null, [ref]$e) | Out-Null; exit $e.Count"
    )
    assert subprocess.run(["pwsh", "-NoProfile", "-c", script]).returncode == 0


def test_double_click_installer_runs_the_powershell_script():
    cmd = (REPO / "Install NoteTaker (Windows).cmd").read_bytes()
    assert b"\r\n" in cmd, "batch files need CRLF line endings"
    assert b"-ExecutionPolicy Bypass" in cmd and b"install.ps1" in cmd


def test_windows_data_lives_in_local_app_data(monkeypatch, tmp_path):
    import importlib

    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr("sys.platform", "win32")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.DATA_DIR == tmp_path / "NoteTaker"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_shortcuts_pass_unknown_words_to_the_cli():
    assert shortcuts.translate(["export", "physics", "-o", "x.md"]) == ["export", "physics", "-o", "x.md"]
    assert shortcuts.translate([]) == ["menu"]
    assert shortcuts.translate(["help"]) is None


def test_class_length_can_be_changed_on_windows(monkeypatch):
    monkeypatch.setenv("NOTES_CLASS_MINUTES", "45")
    assert shortcuts.translate(["class"])[-2:] == ["--minutes", "45"]
