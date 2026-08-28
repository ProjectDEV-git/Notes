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
command -v pactl   >/dev/null || missing+=("pulseaudio-utils")
command -v ollama  >/dev/null || missing+=("ollama (https://ollama.com)")

if (( ${#missing[@]} )); then
    echo
    echo "Still needed: ${missing[*]}"
    echo "  sudo apt install ffmpeg pulseaudio-utils"
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
echo "Done. Start recording a lecture with:  notes"
