#!/usr/bin/env bash
# NoteTaker launcher. Uses the project virtualenv without needing activation.
#
#   ./run.sh devices
#   ./run.sh record --source mic --live-notes
#   ./run.sh list

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$here/.venv/bin/python"

if [[ ! -x "$python" ]]; then
    echo "virtualenv missing. Create it with:" >&2
    echo "  python3 -m venv .venv --system-site-packages" >&2
    echo "  .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

exec "$python" -m notetaker.cli "$@"
