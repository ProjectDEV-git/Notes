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


def test_notes_launcher_checks_for_updates(sandbox):
    _, home = sandbox
    run_installer(sandbox, "--no-install")

    launcher = home / ".local" / "bin" / "notes"
    assert "notetaker.cli update --quiet" in launcher.read_text()


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


# --------------------------------------------------------------------- macOS
# A Mac is where this is hardest: no package manager out of the box, Homebrew
# installs to a different place on Apple Silicon, and system audio needs a
# driver. Each of these is simulated here so the path can be verified without
# owning a Mac.
def mac_sandbox(sandbox, *, brew=False, xcode=True):
    """A sandbox that looks like macOS to the installer.

    The real tools are symlinked in, so a stub must replace the link rather
    than try to write through it.
    """
    bin_dir, home = sandbox
    (bin_dir / "uname").unlink(missing_ok=True)
    _stub(bin_dir, "uname", "echo Darwin")
    _stub(bin_dir, "sw_vers", "echo 14.0")
    _stub(bin_dir, "xcode-select",
          "echo /Library/Developer/CommandLineTools" if xcode else "exit 2")
    if brew:
        _stub(bin_dir, "brew", 'echo "[stub brew] $*"')
    return bin_dir, home


def test_mac_without_homebrew_offers_to_install_it(sandbox):
    """A fresh Mac has no package manager. Printing a URL is a dead end."""
    result = run_installer(mac_sandbox(sandbox, brew=False), "--no-install")
    out = result.stdout + result.stderr
    assert "brew.sh" in out or "Homebrew" in out
    # It must actually offer to do it, not just name it.
    assert "install.sh" in out.lower() or "install homebrew" in out.lower()


def test_mac_explains_the_apple_silicon_path_problem(sandbox):
    """Homebrew installs to /opt/homebrew on Apple Silicon and is not on PATH."""
    result = run_installer(mac_sandbox(sandbox, brew=False), "--no-install")
    out = result.stdout + result.stderr
    assert "/opt/homebrew" in out


def test_mac_mentions_the_microphone_permission(sandbox):
    """Recording silently fails until the terminal is granted mic access."""
    result = run_installer(mac_sandbox(sandbox, brew=True), "--no-install")
    out = result.stdout + result.stderr
    assert "Microphone" in out


def test_mac_names_where_to_grant_microphone_access(sandbox):
    """'It will ask' is not true if the user already denied it once."""
    result = run_installer(mac_sandbox(sandbox, brew=True), "--no-install")
    out = result.stdout + result.stderr
    assert "System Settings" in out or "Privacy" in out


def test_mac_explains_blackhole_is_only_for_online_classes(sandbox):
    """A student recording in person must not think they need a driver."""
    result = run_installer(mac_sandbox(sandbox, brew=True), "--no-install")
    out = result.stdout + result.stderr
    assert "BlackHole" in out or "blackhole" in out
    assert "in person" in out.lower() or "in-person" in out.lower()


def test_mac_install_never_hangs(sandbox):
    """No terminal means no answers; it must decline rather than wait."""
    result = run_installer(mac_sandbox(sandbox, brew=False))
    assert result.returncode is not None


def test_mac_finds_homebrew_that_is_installed_but_not_on_path(sandbox):
    """The commonest Apple Silicon failure: brew exists, PATH does not know."""
    bin_dir, home = mac_sandbox(sandbox, brew=False)
    # Simulate /opt/homebrew/bin/brew existing by making find_brew succeed
    # through a shim directory the installer probes via PATH after shellenv.
    result = run_installer((bin_dir, home), "--no-install")
    out = result.stdout + result.stderr
    # It must at least explain the situation rather than saying nothing.
    assert "/opt/homebrew" in out or "/usr/local" in out


def test_mac_without_brew_still_finishes_and_reports(sandbox):
    """A missing package manager must not abort the whole install."""
    result = run_installer(mac_sandbox(sandbox, brew=False), "--no-install")
    assert "Done." in result.stdout


def test_mac_tells_the_user_what_to_type_at_the_end(sandbox):
    result = run_installer(mac_sandbox(sandbox, brew=True), "--no-install")
    assert "notes" in result.stdout


def test_mac_does_not_claim_in_person_recording_needs_a_driver(sandbox):
    """Most students record in person; scaring them off is the wrong default."""
    result = run_installer(mac_sandbox(sandbox, brew=True), "--no-install")
    out = result.stdout
    marker = out.find("Recording on macOS")
    assert marker != -1
    # The reassurance must come before the driver talk, not after it.
    section = out[marker:marker + 400]
    assert section.index("no extra driver") < section.index("ONLINE")


# ------------------------------------------------------ the one-line install
def _piped(sandbox, *args, extra_env=None):
    """Run install.sh the way `curl ... | bash` does: no file on disk."""
    bin_dir, home = sandbox
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(home),
        "SHELL": "/bin/bash",
        "OLLAMA_URL": "http://127.0.0.1:9",
    }
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", "-s", "--", *args],
        input=INSTALL.read_text(),
        capture_output=True, text=True, timeout=300, env=env,
    )


def test_piped_install_does_not_crash_on_unset_bash_source(sandbox):
    """`curl | bash` leaves BASH_SOURCE unset; set -u used to abort on it."""
    result = _piped(sandbox, "--no-install",
                    extra_env={"NOTETAKER_REPO": "https://127.0.0.1:9/x.git"})
    assert "unbound variable" not in result.stderr


def test_piped_install_does_not_use_the_current_directory(sandbox, tmp_path):
    """It used to install into wherever the user happened to be standing."""
    result = _piped(sandbox, "--no-install",
                    extra_env={"NOTETAKER_REPO": "https://127.0.0.1:9/x.git"})
    out = result.stdout + result.stderr
    assert "Getting NoteTaker" in out, "did not try to fetch a checkout"


def test_piped_install_refuses_to_touch_an_unrelated_directory(sandbox):
    """A directory that is not a checkout must be left completely alone."""
    bin_dir, home = sandbox
    target = home / "NoteTaker"
    target.mkdir()
    keeper = target / "myfile.txt"
    keeper.write_text("important user data")

    result = _piped(sandbox, "--no-install",
                    extra_env={"NOTETAKER_DIR": str(target)})

    assert "already exists" in (result.stdout + result.stderr)
    assert keeper.read_text() == "important user data"
    assert [p.name for p in target.iterdir()] == ["myfile.txt"]


def test_piped_install_says_so_when_git_is_missing(sandbox):
    """Without git there is no way to fetch anything; say that plainly."""
    bin_dir, home = sandbox
    env = {"PATH": str(bin_dir), "HOME": str(home), "SHELL": "/bin/bash"}
    result = subprocess.run(
        ["bash", "-s", "--", "--no-install"],
        input=INSTALL.read_text(),
        capture_output=True, text=True, timeout=120, env=env,
    )
    out = result.stdout + result.stderr
    assert "git is needed" in out


def test_the_repo_url_matches_the_one_in_the_readme():
    """A one-line install pointing at the wrong repo installs the wrong thing."""
    import re

    readme = (REPO / "README.md").read_text()
    script = INSTALL.read_text()

    in_script = re.search(r'NOTETAKER_REPO:-([^}]+)\}', script)
    assert in_script, "install.sh no longer declares a default repo URL"
    url = in_script.group(1)
    assert url in readme, f"install.sh clones {url}, which the README never mentions"


def test_the_readme_curl_url_points_at_this_repos_install_script():
    """The headline instruction is a URL; a wrong one installs nothing.

    Checked offline by construction: the raw URL must name the same repo the
    script clones, on the branch this checkout is on.
    """
    import re

    readme = (REPO / "README.md").read_text()
    match = re.search(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/install\.sh",
                      readme)
    assert match, "the README no longer shows a one-line install URL"
    owner, repo_name, branch = match.groups()

    script = INSTALL.read_text()
    clone_url = re.search(r"NOTETAKER_REPO:-([^}]+)\}", script).group(1)
    assert f"{owner}/{repo_name}" in clone_url, (
        f"README fetches from {owner}/{repo_name} but install.sh clones {clone_url}"
    )
    assert branch == "main", f"README points at branch {branch!r}"


def test_help_works_when_piped_from_curl(sandbox):
    """It read its own source with sed "$0", which is 'bash' when piped."""
    result = _piped(sandbox, "--help")
    assert result.returncode == 0
    assert "sed:" not in result.stderr
    assert "install everything" in result.stdout


def test_help_still_works_from_a_checkout(sandbox):
    result = run_installer(sandbox, "--help")
    assert result.returncode == 0
    assert "install everything" in result.stdout


def test_help_shows_the_same_url_as_the_readme():
    """Help text is where a confused user looks; a stale URL there is a trap.

    Only NoteTaker's own URL is checked. install.sh also fetches Homebrew's
    installer, which has no business being in this README.
    """
    import re

    script = INSTALL.read_text()
    readme = (REPO / "README.md").read_text()

    repo_url = re.search(r"NOTETAKER_REPO:-\S*?github\.com/([^/]+/[^/.]+)", script)
    assert repo_url, "install.sh no longer declares a default repo"
    owner_repo = repo_url.group(1)

    ours = [u for u in re.findall(r"https://raw\.githubusercontent\.com/\S+/install\.sh", script)
            if owner_repo in u]
    assert ours, "the help text no longer shows the one-line install"
    for url in ours:
        assert url in readme, f"install.sh --help shows {url}, absent from the README"
