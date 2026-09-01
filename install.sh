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

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"
MODEL="llama3.2:3b"

ASSUME_YES=0
NO_INSTALL=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes)        ASSUME_YES=1 ;;
        --no-install)    NO_INSTALL=1 ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
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
confirm() {
    (( ASSUME_YES )) && return 0
    [[ -t 0 ]] || return 1
    local reply
    read -r -p "$1 [Y/n] " reply || return 1
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
        (( IS_MAC )) && say "  Install Homebrew first: https://brew.sh"
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
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
    # --system-site-packages so a preinstalled faster-whisper is reused.
    if ! python3 -m venv "$APP_DIR/.venv" --system-site-packages 2>/dev/null; then
        # Debian and Ubuntu ship python3 without venv, which is the single
        # most common first-run failure on those systems.
        warn "python3 venv is unavailable."
        if [[ "$PKG" == "apt-get" ]] && confirm_install "Install python3-venv?"; then
            install_pkg venv
            python3 -m venv "$APP_DIR/.venv" --system-site-packages
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
    say "  starting the Ollama server..."
    # Started detached so this script can talk to it; on Linux the installer
    # normally leaves a systemd service running already.
    ollama serve >/dev/null 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 20); do
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
    step "Online lectures on macOS"
    if command -v ffmpeg >/dev/null &&
       ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 |
         grep -qiE "blackhole|soundflower|loopback"; then
        ok "a loopback device is present, online lectures will work"
    else
        # macOS exposes no monitor of the output, so without this `notes
        # online` records silence.
        say "macOS cannot record what your speakers play without a loopback driver."
        if [[ "$PKG" == "brew" ]] && confirm_install "Install BlackHole (free, open source)?"; then
            say "  \$ brew install --cask blackhole-2ch"
            brew install --cask blackhole-2ch && ok "BlackHole installed"
            say
            say "One manual step remains, because macOS cannot script it:"
            say "  1. Open Audio MIDI Setup"
            say "  2. Create a Multi-Output Device with BlackHole 2ch + your speakers"
            say "  3. Select it as your sound output during online lectures"
        else
            say "  Install it later with: brew install --cask blackhole-2ch"
        fi
    fi
    say
    say "The first recording will ask for Microphone permission for your terminal."
fi

# --------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------
step "Checking the installation"
"$APP_DIR/.venv/bin/python" -m notetaker.cli check || true

say
say "Done. Type one word to begin:"
say
say "  notes"
say
say "It shows a short list. Pick a number, or just press Enter to record."
