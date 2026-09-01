"""The installer: the very first thing a user runs.

These tests execute install.sh against a sandboxed PATH of stub commands, so
nothing is installed and no real package manager is invoked. What matters is
that it never damages the machine silently, never hangs, and always tells the
user the correct command for their system.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
INSTALL = REPO / "install.sh"

# Utilities the script genuinely needs to run at all.
_REAL_TOOLS = [
    "bash", "sh", "sed", "grep", "cat", "printf", "head", "tail", "cut", "tr",
    "basename", "dirname", "mkdir", "chmod", "sleep", "seq", "uname", "kill",
    "wait", "env", "sort", "ls", "rm", "cp", "mv", "python3",
]


def _stub(path: Path, name: str, body: str = "") -> Path:
    script = path / name
    script.write_text(f"#!/bin/sh\n{body or f'echo \"[stub {name}] $*\"'}\n")
    script.chmod(0o755)
    return script


@pytest.fixture
def sandbox(tmp_path):
    """A PATH containing only what we choose, plus a throwaway HOME."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in _REAL_TOOLS:
        real = shutil.which(tool)
        if real:
            (bin_dir / tool).symlink_to(real)
    home = tmp_path / "home"
    home.mkdir()
    return bin_dir, home


def run_installer(sandbox, *args, stdin: str = "", extra_env: dict | None = None):
    bin_dir, home = sandbox
    env = {
        "PATH": str(bin_dir),
        "HOME": str(home),
        "SHELL": "/bin/bash",
        # Point at a port nothing is listening on, so a real local Ollama
        # cannot make these tests pass by accident.
        "OLLAMA_URL": "http://127.0.0.1:9",
    }
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(INSTALL), *args],
        input=stdin, capture_output=True, text=True, timeout=300, env=env,
    )


# ---------------------------------------------------------------- basics
def test_installer_is_valid_shell():
    assert subprocess.run(["bash", "-n", str(INSTALL)]).returncode == 0


def test_help_does_not_touch_the_machine(sandbox):
    result = run_installer(sandbox, "--help")
    assert result.returncode == 0
    assert "--yes" in result.stdout


def test_no_install_mode_installs_nothing(sandbox):
    """The escape hatch for users who want to install things themselves.

    --yes is passed too: --no-install must win over it, or a user combining
    them would get software installed they explicitly refused.
    """
    bin_dir, _ = sandbox
    _stub(bin_dir, "apt-get", 'echo "APT WAS CALLED" >&2; exit 1')
    _stub(bin_dir, "sudo", 'echo "SUDO WAS CALLED" >&2; exit 1')

    result = run_installer(sandbox, "--no-install", "--yes")
    assert "APT WAS CALLED" not in result.stderr
    assert "SUDO WAS CALLED" not in result.stderr


def test_never_hangs_without_a_terminal(sandbox):
    """Piped into a shell there is nobody to answer prompts; it must not block."""
    bin_dir, _ = sandbox
    _stub(bin_dir, "apt-get")
    _stub(bin_dir, "sudo", 'echo "SUDO WAS CALLED" >&2')

    result = run_installer(sandbox, stdin="")  # closed stdin
    # Declined by default rather than installing unattended.
    assert "SUDO WAS CALLED" not in result.stderr


def test_declining_installs_nothing_but_says_how(sandbox):
    bin_dir, _ = sandbox
    _stub(bin_dir, "apt-get")
    _stub(bin_dir, "sudo", 'echo "SUDO WAS CALLED" >&2')

    result = run_installer(sandbox, stdin="n\nn\nn\nn\nn\n")
    assert "SUDO WAS CALLED" not in result.stderr
    assert "sudo apt install ffmpeg" in result.stdout


# ------------------------------------------------------ package managers
@pytest.mark.parametrize(
    "manager,expected",
    [
        ("apt-get", "apt-get install -y ffmpeg"),
        ("dnf", "dnf install -y ffmpeg"),
        ("pacman", "pacman -S --noconfirm ffmpeg"),
        ("zypper", "zypper install -y ffmpeg"),
        ("apk", "apk add ffmpeg"),
    ],
)
def test_uses_the_right_package_manager(sandbox, manager, expected):
    """A Fedora user must not be told to run apt."""
    bin_dir, _ = sandbox
    _stub(bin_dir, manager)
    _stub(bin_dir, "sudo", 'echo "sudo $*"')

    result = run_installer(sandbox, "--yes")
    assert expected in result.stdout


@pytest.mark.parametrize(
    "manager,expected",
    [
        ("apt-get", "pulseaudio-utils"),
        ("pacman", "libpulse"),
    ],
)
def test_pulse_package_name_differs_per_distro(sandbox, manager, expected):
    """`pactl` lives in libpulse on Arch, pulseaudio-utils on Debian."""
    bin_dir, _ = sandbox
    _stub(bin_dir, manager)
    _stub(bin_dir, "sudo", 'echo "sudo $*"')

    result = run_installer(sandbox, "--yes")
    assert expected in result.stdout


def test_missing_package_manager_is_explained_not_crashed(sandbox):
    result = run_installer(sandbox, "--yes")
    assert result.returncode == 0
    assert "no known package manager" in result.stdout


# ------------------------------------------------------------- commands
def test_root_commands_are_printed_before_running(sandbox):
    """Nothing should touch the system without the user seeing the command."""
    bin_dir, _ = sandbox
    _stub(bin_dir, "apt-get")
    _stub(bin_dir, "sudo", 'echo "sudo $*"')

    result = run_installer(sandbox, "--yes")
    assert "$ sudo apt-get install -y ffmpeg" in result.stdout


def test_ollama_is_installed_from_the_official_script_on_linux(sandbox):
    bin_dir, _ = sandbox
    _stub(bin_dir, "apt-get")
    _stub(bin_dir, "sudo", 'echo "sudo $*"')
    _stub(bin_dir, "curl", 'echo "[stub curl] $*"; exit 1')

    result = run_installer(sandbox, "--yes")
    assert "ollama.com/install.sh" in result.stdout


def test_model_is_not_pulled_when_ollama_is_absent(sandbox):
    """Pulling would fail confusingly; say why instead."""
    bin_dir, _ = sandbox
    _stub(bin_dir, "curl", "exit 1")

    result = run_installer(sandbox, "--yes")
    assert "skipped: Ollama is not installed" in result.stdout


def test_model_pull_is_skipped_when_the_server_is_down(sandbox):
    """Installed but not running is a distinct failure with a distinct fix."""
    bin_dir, _ = sandbox
    _stub(bin_dir, "ollama", 'echo "[stub ollama] $*"; exit 0')
    _stub(bin_dir, "curl", "exit 1")  # server unreachable

    result = run_installer(sandbox, "--yes")
    assert "ollama pull llama3.2:3b" in result.stdout


# ------------------------------------------------------------ notes cmd
def test_notes_launcher_is_installed_and_points_at_this_checkout(sandbox):
    _, home = sandbox
    run_installer(sandbox, "--no-install")

    launcher = home / ".local" / "bin" / "notes"
    assert launcher.exists() and os.access(launcher, os.X_OK)
    assert f'APP_DIR="{REPO}"' in launcher.read_text()


def test_path_is_configured_for_the_users_shell(sandbox):
    """Telling a beginner to 'edit your rc file' is where installs die."""
    _, home = sandbox
    # --yes answers the PATH question; --no-install still blocks any install.
    run_installer(sandbox, "--no-install", "--yes",
                  extra_env={"SHELL": "/usr/bin/zsh"})

    rc = home / ".zshrc"
    assert rc.exists() and str(home / ".local" / "bin") in rc.read_text()


def test_fish_gets_fish_syntax(sandbox):
    """`export PATH=` in config.fish is a syntax error, not a path change."""
    _, home = sandbox
    run_installer(sandbox, "--no-install", "--yes",
                  extra_env={"SHELL": "/usr/bin/fish"})

    rc = home / ".config" / "fish" / "config.fish"
    assert "fish_add_path" in rc.read_text()
    assert "export PATH" not in rc.read_text()


def test_path_entry_is_not_added_twice(sandbox):
    """Running the installer repeatedly is normal and must stay clean."""
    _, home = sandbox
    for _ in range(3):
        run_installer(sandbox, "--no-install", "--yes")

    rc = home / ".bashrc"
    assert rc.read_text().count(str(home / ".local" / "bin")) == 1


def test_rerunning_is_safe(sandbox):
    first = run_installer(sandbox, "--no-install")
    second = run_installer(sandbox, "--no-install")
    assert first.returncode == 0 and second.returncode == 0


def test_finishes_by_telling_the_user_the_one_word(sandbox):
    result = run_installer(sandbox, "--no-install")
    assert "notes" in result.stdout.splitlines()[-5:][0] or "notes" in result.stdout[-200:]
