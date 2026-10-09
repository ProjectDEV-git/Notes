#!/usr/bin/env bash
# Set up NoteTaker completely: dependencies, Ollama, the summary model, and
# the one-word `notes` command.
#
#   ./install.sh              install everything, asking before each step
#   ./install.sh --yes        install everything without asking
#   ./install.sh --no-install only check, never install (the old behaviour)
#
# Anything that needs root is run with sudo and printed first, so nothing
# happens to the machine without the user seeing the exact command.

set -euo pipefail

BIN_DIR="$HOME/.local/bin"
MODEL="llama3.2:3b"
REPO_URL="${NOTETAKER_REPO:-https://github.com/ProjectDEV-git/Notes.git}"

# Where this script lives. Piped through curl there is no file on disk, so
# BASH_SOURCE is unset and the checkout has to be fetched first. Without this
# the one-line install would silently set itself up in whatever directory the
# user happened to be standing in.
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    APP_DIR=""
fi

ASSUME_YES=0
NO_INSTALL=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes)        ASSUME_YES=1 ;;
        --no-install)    NO_INSTALL=1 ;;
        -h|--help)
            # Printed literally rather than read back out of this file: piped
            # from curl there is no file, and $0 is just "bash".
            cat <<'USAGE'
Set up NoteTaker completely: dependencies, Ollama, the summary model, and
the one-word `notes` command.

  ./install.sh              install everything, asking before each step
  ./install.sh --yes        install everything without asking
  ./install.sh --no-install only check, never install

Anything that needs root is run with sudo and printed first, so nothing
happens to the machine without the user seeing the exact command.

Run without a checkout, NoteTaker is downloaded to ~/NoteTaker first:

  curl -fsSL https://raw.githubusercontent.com/ProjectDEV-git/Notes/main/install.sh | bash

  NOTETAKER_DIR=~/elsewhere   put the checkout somewhere else
USAGE
            exit 0
            ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

IS_MAC=0
[[ "$(uname -s)" == "Darwin" ]] && IS_MAC=1

say()  { printf '%s\n' "$*"; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }

# Ask before changing the machine. A non-interactive shell (piped installer,
# CI) must never block waiting for an answer, so it declines instead.
#
# `curl ... | bash` makes stdin the script itself, so questions are read from
# the terminal directly. Without that, the one-line install quietly declined
# everything and a beginner ended up with nothing installed.
TTY=""
if [[ -t 0 ]]; then
    TTY=/dev/stdin
elif [[ -z "${NOTETAKER_NO_TTY:-}" ]] && { : </dev/tty; } 2>/dev/null; then
    TTY=/dev/tty
fi

ask_user() {
    [[ -n "$TTY" ]] || return 1
    local reply
    read -r -p "$1 [Y/n] " reply <"$TTY" || return 1
    [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

confirm() {
    (( ASSUME_YES )) && return 0
    [[ -n "$TTY" ]] || return 1
    local reply
    read -r -p "$1 [Y/n] " reply <"$TTY" || return 1
    [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

# As confirm, but for anything that installs software. --no-install suppresses
# these while still allowing harmless setup like adding the PATH entry.
confirm_install() {
    (( NO_INSTALL )) && return 1
    confirm "$1"
}

# Run a command that needs root, showing it first.
as_root() {
    say "  \$ sudo $*"
    sudo "$@"
}

# --------------------------------------------------------------------------
# Package manager detection
# --------------------------------------------------------------------------
PKG=""
if (( IS_MAC )); then
    command -v brew >/dev/null && PKG="brew"
else
    for candidate in apt-get dnf pacman zypper apk; do
        if command -v "$candidate" >/dev/null; then PKG="$candidate"; break; fi
    done
fi

# Homebrew installs somewhere different on Apple Silicon than on Intel, and in
# neither case does a fresh shell have it on PATH. A Mac that has Homebrew but
# cannot see it looks exactly like a Mac with no package manager, which is the
# single most common way this install goes wrong.
find_brew() {
    command -v brew >/dev/null && { echo "brew"; return 0; }
    local candidate
    for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
        [[ -x "$candidate" ]] && { echo "$candidate"; return 0; }
    done
    return 1
}

if (( IS_MAC )) && [[ -z "$PKG" ]]; then
    if BREW_BIN="$(find_brew)"; then
        # Found it where the installer leaves it, just not on PATH yet.
        eval "$("$BREW_BIN" shellenv 2>/dev/null)" || true
        command -v brew >/dev/null && PKG="brew"
    fi
fi

# Offering to run the official installer is the difference between a student
# finishing setup and giving up at a link to another website.
install_homebrew() {
    say "Homebrew is Apple's usual way to install command line software."
    say "It is the official installer from https://brew.sh:"
    say '  $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    confirm_install "Install Homebrew now?" || {
        say "  Install it later from https://brew.sh, then run ./install.sh again."
        return 1
    }

    command -v curl >/dev/null || { warn "curl is missing, cannot download Homebrew"; return 1; }
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
        warn "the Homebrew installer did not finish"
        return 1
    }

    local found
    found="$(find_brew)" || { warn "Homebrew installed but could not be found"; return 1; }
    eval "$("$found" shellenv 2>/dev/null)" || true
    command -v brew >/dev/null || { warn "Homebrew is installed but not on PATH"; return 1; }
    PKG="brew"
    ok "Homebrew installed"
    return 0
}

# --------------------------------------------------------------------------
# macOS: one plain-English question instead of a string of tool names.
# A student does not know what ffmpeg or Homebrew is, and should not have to.
# --------------------------------------------------------------------------
python_ok() {
    # /usr/bin/python3 on a Mac without developer tools is a stub that pops up
    # an install dialog, so "it exists" is not enough: it has to run.
    "$1" -c 'import sys; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1
}

PYTHON="python3"
if (( IS_MAC )); then
    MAC_NEEDS=()
    [[ -z "$PKG" ]] && MAC_NEEDS+=("Homebrew   installs the free tools below (asks for your Mac password)")
    python_ok python3 || MAC_NEEDS+=("Python     runs NoteTaker")
    command -v ffmpeg >/dev/null || MAC_NEEDS+=("FFmpeg     records the sound")
    command -v ollama >/dev/null || MAC_NEEDS+=("Ollama     the app that writes your notes, offline")
    if (( ${#MAC_NEEDS[@]} )) && (( ! NO_INSTALL )); then
        step "NoteTaker needs a few free helper apps"
        for item in "${MAC_NEEDS[@]}"; do say "  • $item"; done
        say "  • the notes model, a one-time download of about 2 GB"
        say
        say "This takes 10-20 minutes. Keep the Mac plugged in and online."
        say "If a window asks to install \"command line developer tools\", click Install"
        say "and wait for it to finish; Homebrew needs them."
        say "When asked for a password, type your Mac login password. Nothing shows"
        say "while you type; that is normal. Press Return when done."
        if (( ! ASSUME_YES )) && ask_user "Install all of these now?"; then
            ASSUME_YES=1
        fi
    fi
fi

if (( IS_MAC )) && [[ -z "$PKG" ]]; then
    step "Homebrew (needed to install everything else)"
    say "Nothing to install software with was found on this Mac."
    say "Homebrew lives in /opt/homebrew on Apple Silicon and /usr/local on Intel;"
    say "neither is on PATH until it has been set up."
    install_homebrew || true
fi

# Package names differ per distro; only these three are ever needed.
pkg_name() {
    case "$1:$PKG" in
        ffmpeg:*)              echo "ffmpeg" ;;
        pactl:apt-get)         echo "pulseaudio-utils" ;;
        pactl:dnf)             echo "pulseaudio-utils" ;;
        pactl:pacman)          echo "libpulse" ;;
        pactl:zypper)          echo "pulseaudio-utils" ;;
        pactl:apk)             echo "pulseaudio-utils" ;;
        venv:apt-get)          echo "python3-venv" ;;
        *)                     echo "$1" ;;
    esac
}

install_pkg() {
    local pkg; pkg="$(pkg_name "$1")"
    case "$PKG" in
        brew)     say "  \$ brew install $pkg"; brew install "$pkg" ;;
        apt-get)  as_root apt-get update -qq && as_root apt-get install -y "$pkg" ;;
        dnf)      as_root dnf install -y "$pkg" ;;
        pacman)   as_root pacman -S --noconfirm "$pkg" ;;
        zypper)   as_root zypper install -y "$pkg" ;;
        apk)      as_root apk add "$pkg" ;;
        *)        return 1 ;;
    esac
}

# Ensure a command exists, installing it when the user agrees.
# Returns non-zero if it is still missing afterwards.
ensure_command() {
    local cmd="$1" why="$2"
    command -v "$cmd" >/dev/null && { ok "$cmd already installed"; return 0; }

    if [[ -z "$PKG" ]]; then
        warn "$cmd is missing ($why) and no known package manager was found."
        # Homebrew was already offered above; repeating a bare link here just
        # tells the user to go somewhere else after they declined.
        (( IS_MAC )) && say "  Install Homebrew (offered above), then run ./install.sh again."
        return 1
    fi
    if confirm_install "Install $cmd? ($why)"; then
        install_pkg "$cmd" || { warn "could not install $cmd"; return 1; }
        command -v "$cmd" >/dev/null && { ok "$cmd installed"; return 0; }
        warn "$cmd still not on PATH after installing"
        return 1
    fi
    warn "skipped $cmd. Install it later with: $(manual_hint "$cmd")"
    return 1
}

manual_hint() {
    local pkg; pkg="$(pkg_name "$1")"
    case "$PKG" in
        brew)     echo "brew install $pkg" ;;
        apt-get)  echo "sudo apt install $pkg" ;;
        dnf)      echo "sudo dnf install $pkg" ;;
        pacman)   echo "sudo pacman -S $pkg" ;;
        zypper)   echo "sudo zypper install $pkg" ;;
        apk)      echo "sudo apk add $pkg" ;;
        *)        echo "your package manager's install command for $pkg" ;;
    esac
}

# --------------------------------------------------------------------------
# Bootstrap: piped from curl, so there is no checkout yet. Clone one and hand
# over to the copy inside it, which is the ordinary path from then on.
# --------------------------------------------------------------------------
if [[ -z "$APP_DIR" ]]; then
    TARGET="${NOTETAKER_DIR:-$HOME/NoteTaker}"
    step "Getting NoteTaker"

    if ! command -v git >/dev/null; then
        warn "git is needed to download NoteTaker, and is not installed."
        say "  Install git, then run this again."
        exit 1
    fi

    if [[ -d "$TARGET/.git" ]]; then
        say "  already downloaded to $TARGET, updating it"
        say "  \$ git -C $TARGET pull --ff-only"
        git -C "$TARGET" pull --ff-only || warn "could not update; using what is there"
    elif [[ -e "$TARGET" ]]; then
        warn "$TARGET already exists and is not a NoteTaker checkout."
        say "  Move it, or choose somewhere else:"
        say "    NOTETAKER_DIR=~/somewhere-else  (then run this again)"
        exit 1
    else
        say "  \$ git clone $REPO_URL $TARGET"
        git clone --quiet "$REPO_URL" "$TARGET" || {
            warn "could not download NoteTaker from $REPO_URL"
            exit 1
        }
        ok "downloaded to $TARGET"
    fi

    # Continue in the real script, so there is only one install path to keep
    # working rather than two that can drift apart.
    exec bash "$TARGET/install.sh" "$@"
fi

say "Installing NoteTaker from $APP_DIR"
(( NO_INSTALL )) && warn "--no-install: checking only, nothing will be installed"

# --------------------------------------------------------------------------
# 1. Audio and transcoding
# --------------------------------------------------------------------------
step "1/5  Audio tools"
ensure_command ffmpeg "records the lecture audio" || true
if (( ! IS_MAC )); then
    ensure_command pactl "finds your microphone and system audio" || true
fi

# --------------------------------------------------------------------------
# 2. Python environment
# --------------------------------------------------------------------------
step "2/5  Python environment"
if [[ ! -x "$APP_DIR/.venv/bin/python" ]] && ! python_ok "$PYTHON"; then
    if [[ "$PKG" == "brew" ]] && confirm_install "Install Python? (runs NoteTaker)"; then
        say "  \$ brew install python@3.12"
        brew install python@3.12 && PYTHON="$(brew --prefix)/bin/python3.12"
    fi
    python_ok "$PYTHON" || warn "a working Python 3.9 or newer was not found"
fi
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
    # --system-site-packages so a preinstalled faster-whisper is reused.
    if ! "$PYTHON" -m venv "$APP_DIR/.venv" --system-site-packages 2>/dev/null; then
        # Debian and Ubuntu ship python3 without venv, which is the single
        # most common first-run failure on those systems.
        warn "python3 venv is unavailable."
        if [[ "$PKG" == "apt-get" ]] && confirm_install "Install python3-venv?"; then
            install_pkg venv
            "$PYTHON" -m venv "$APP_DIR/.venv" --system-site-packages
        else
            warn "install it with: $(manual_hint venv)"
            exit 1
        fi
    fi
    ok "created .venv"
else
    ok "virtualenv already exists"
fi
say "  installing Python dependencies..."
# Skip the network round-trip when the venv is already complete: re-running
# the installer is normal and should be fast.
if "$APP_DIR/.venv/bin/python" -c "import faster_whisper, rich" >/dev/null 2>&1; then
    ok "Python dependencies already installed"
else
    "$APP_DIR/.venv/bin/pip" install -q --upgrade pip >/dev/null 2>&1 || true
    "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"
    ok "Python dependencies ready"
fi

# --------------------------------------------------------------------------
# 3. Ollama, which writes the notes
# --------------------------------------------------------------------------
step "3/5  Ollama (writes the notes)"
install_ollama() {
    if (( IS_MAC )); then
        if [[ "$PKG" == "brew" ]]; then
            say "  \$ brew install --cask ollama"
            brew install --cask ollama
        else
            warn "install Ollama from https://ollama.com/download"
            return 1
        fi
    else
        # The official script is the supported path on Linux and handles
        # every distro, including the systemd service.
        say "  \$ curl -fsSL https://ollama.com/install.sh | sh"
        curl -fsSL https://ollama.com/install.sh | sh
    fi
}

if command -v ollama >/dev/null; then
    ok "ollama already installed"
elif confirm_install "Install Ollama? (needed to turn transcripts into notes)"; then
    install_ollama || true
else
    warn "skipped Ollama. Recording and transcription still work; notes will not."
fi

# Ollama must be *running*, not merely installed, before a model can be pulled.
ollama_up() { curl -fsS --max-time 3 "${OLLAMA_URL:-http://localhost:11434}/api/tags" >/dev/null 2>&1; }

SERVER_PID=""
if command -v ollama >/dev/null && ! ollama_up; then
    say "  starting Ollama..."
    if (( IS_MAC )) && open -a Ollama >/dev/null 2>&1; then
        # The menu-bar app keeps itself running and starts at login, so the
        # student never has to think about it again.
        :
    else
        # Started detached so this script can talk to it; on Linux the
        # installer normally leaves a systemd service running already.
        ollama serve >/dev/null 2>&1 &
        SERVER_PID=$!
    fi
    for _ in $(seq 1 40); do
        ollama_up && break
        sleep 0.5
    done
fi
if command -v ollama >/dev/null; then
    if ollama_up; then ok "Ollama is running"; else warn "Ollama is installed but not running (start it with: ollama serve)"; fi
fi

# --------------------------------------------------------------------------
# 4. The summary model
# --------------------------------------------------------------------------
step "4/5  Summary model ($MODEL)"
if ! command -v ollama >/dev/null; then
    warn "skipped: Ollama is not installed"
elif ollama list 2>/dev/null | grep -q "${MODEL%%:*}"; then
    ok "$MODEL already downloaded"
elif ! ollama_up; then
    warn "cannot download the model while Ollama is not running."
    say "  Later, run:  ollama serve   then:  ollama pull $MODEL"
elif confirm_install "Download $MODEL now? (about 2 GB, one time)"; then
    ollama pull "$MODEL" && ok "$MODEL ready"
else
    warn "skipped. Download it later with: ollama pull $MODEL"
fi

# The background server was only needed to pull the model.
if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
fi

# --------------------------------------------------------------------------
# 5. The `notes` command
# --------------------------------------------------------------------------
step "5/5  The 'notes' command"
mkdir -p "$BIN_DIR"
sed "s|^APP_DIR=.*|APP_DIR=\"$APP_DIR\"|" "$APP_DIR/scripts/notes" > "$BIN_DIR/notes"
chmod +x "$BIN_DIR/notes"
ok "installed $BIN_DIR/notes"

# Put it on PATH for the user, rather than telling them to edit a dotfile.
add_to_path() {
    local shell_name rc line="export PATH=\"\$PATH:$BIN_DIR\""
    shell_name="$(basename "${SHELL:-bash}")"
    case "$shell_name" in
        fish)
            rc="$HOME/.config/fish/config.fish"
            line="fish_add_path $BIN_DIR"
            ;;
        zsh)  rc="$HOME/.zshrc" ;;
        *)    rc="$HOME/.bashrc" ;;
    esac
    mkdir -p "$(dirname "$rc")"
    if [[ -f "$rc" ]] && grep -qF "$BIN_DIR" "$rc"; then
        ok "$BIN_DIR is already configured in $rc"
        return 0
    fi
    printf '\n# added by NoteTaker install.sh\n%s\n' "$line" >> "$rc"
    ok "added $BIN_DIR to $rc"
    warn "open a new terminal (or run: source $rc) before typing 'notes'"
}

case ":$PATH:" in
    *":$BIN_DIR:"*) ok "$BIN_DIR is already on your PATH" ;;
    *)
        if confirm "Add $BIN_DIR to your PATH so 'notes' just works?"; then
            add_to_path
        else
            warn "run NoteTaker with: $BIN_DIR/notes"
        fi
        ;;
esac

# --------------------------------------------------------------------------
# macOS: system audio needs a loopback driver
# --------------------------------------------------------------------------
if (( IS_MAC )); then
    step "Recording on macOS"
    say "Recording a class you are sitting in works now, with no extra driver."
    say
    say "Only ONLINE classes need one, because macOS cannot record what the"
    say "speakers are playing on its own. Skip this if you record in person."
    say "When you need it, NoteTaker walks you through it step by step:"
    say "  notes setup-online"
    say "(it installs BlackHole, a free helper, and opens the right windows for you)."
    if ask_user "Do you record online classes? Set them up now?"; then
        "$APP_DIR/.venv/bin/python" -m notetaker.cli setup-online <"$TTY" || true
    fi
    say
    step "Microphone permission"
    say "macOS asks once whether NoteTaker may use the Microphone. Click Allow,"
    say "or every class records perfect silence. The check below asks now, so"
    say "it is out of the way before your first class."
    say
    say "If you clicked Don't Allow before, NoteTaker opens the exact page for you:"
    say "  System Settings > Privacy & Security > Microphone"
    say "Turn on your terminal (Terminal or iTerm), then quit it with Cmd-Q and reopen."

    # A double-clickable icon, so the next time needs no terminal knowledge.
    if confirm "Put a NoteTaker icon on your Desktop?"; then
        desktop="$HOME/Desktop/NoteTaker.command"
        printf '#!/bin/bash\n"%s" menu\n' "$BIN_DIR/notes" > "$desktop"
        chmod +x "$desktop"
        ok "double-click NoteTaker on your Desktop to start"
    fi
fi

# --------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------
step "Checking the installation"
if (( IS_MAC )) && [[ -n "$TTY" ]]; then
    # Recording a few seconds is what makes macOS ask for the microphone.
    "$APP_DIR/.venv/bin/python" -m notetaker.cli check --listen <"$TTY" || true
else
    "$APP_DIR/.venv/bin/python" -m notetaker.cli check || true
fi

say
say "Done. Type one word to begin:"
say
say "  notes"
say
say "It shows a short list. Pick a number, or just press Enter to record."
