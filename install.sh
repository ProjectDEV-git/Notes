#!/usr/bin/env bash
# Set up NoteTaker and install the one-word `notes` command.
#
#   ./install.sh
#   notes

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"

echo "Installing NoteTaker from $APP_DIR"

# 1. virtualenv (system site packages so a preinstalled faster-whisper is reused)
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
    echo "  creating virtualenv..."
    python3 -m venv "$APP_DIR/.venv" --system-site-packages
fi
echo "  installing dependencies..."
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

# 2. the `notes` launcher
mkdir -p "$BIN_DIR"
sed "s|^APP_DIR=.*|APP_DIR=\"$APP_DIR\"|" "$APP_DIR/scripts/notes" > "$BIN_DIR/notes"
chmod +x "$BIN_DIR/notes"
echo "  installed $BIN_DIR/notes"

# 3. checks the user cannot skip silently
missing=()
command -v ffmpeg  >/dev/null || missing+=("ffmpeg")
command -v ollama  >/dev/null || missing+=("ollama (https://ollama.com)")

if [[ "$(uname -s)" == "Darwin" ]]; then
    if (( ${#missing[@]} )); then
        echo
        echo "Still needed: ${missing[*]}"
        echo "  brew install ffmpeg"
        echo "  brew install --cask ollama"
    fi
    # macOS cannot capture system audio without a loopback driver: CoreAudio
    # exposes no monitor of the output. Without one, `notes online` is dead.
    if command -v ffmpeg >/dev/null &&
       ! ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 |
         grep -qiE "blackhole|soundflower|loopback"; then
        echo
        echo "For online lectures (system audio) macOS needs a loopback device:"
        echo "  brew install blackhole-2ch"
        echo "Then in Audio MIDI Setup create a Multi-Output Device with"
        echo "BlackHole + your speakers, so you can still hear the lecture."
    fi
    echo
    echo "The first recording will ask for Microphone permission for your terminal."
else
    command -v pactl >/dev/null || missing+=("pulseaudio-utils")
    if (( ${#missing[@]} )); then
        echo
        echo "Still needed: ${missing[*]}"
        echo "  sudo apt install ffmpeg pulseaudio-utils"
    fi
fi

if command -v ollama >/dev/null && ! ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
    echo
    echo "The summarizer model is not installed yet. Run:"
    echo "  ollama pull llama3.2:3b"
fi

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo
        echo "$BIN_DIR is not on your PATH. Add it:"
        echo "  fish:  fish_add_path $BIN_DIR"
        echo "  bash:  echo 'export PATH=\"\$PATH:$BIN_DIR\"' >> ~/.bashrc"
        ;;
esac

echo
echo "Done. Type one word to begin:"
echo
echo "  notes"
echo
echo "It shows a short list. Pick a number, or just press Enter to record."
