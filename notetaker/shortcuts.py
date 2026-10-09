"""The one-word `notes` commands, for systems without bash (Windows).

Mirrors scripts/notes so `notes class`, `notes later` and friends behave the
same everywhere. Run as:  python -m notetaker.shortcuts <word> [options]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent


def _class_minutes() -> str:
    return os.environ.get("NOTES_CLASS_MINUTES") or "60"


VERBS: dict[str, list[str]] = {}
for _words, _argv in [
    (("now", "start", "record", "go", "mic", "lecture"), ["record", "--source", "mic", "--live-notes"]),
    (("class", "period", "lesson"), ["record", "--source", "mic", "--live-notes", "--minutes", "{minutes}"]),
    (("later", "quick", "save"), ["record", "--source", "mic", "--later", "--minutes", "{minutes}"]),
    (("catchup", "writeup", "finish", "homework"), ["catchup"]),
    (("online", "zoom", "teams", "system", "meet", "youtube"), ["record", "--source", "system", "--live-notes"]),
    (("all", "ls", "list"), ["list"]),
    (("check", "test", "doctor", "setup"), ["check", "--listen"]),
    (("update", "upgrade"), ["update"]),
    (("menu",), ["menu"]),
]:
    for _word in _words:
        VERBS[_word] = _argv


HELP = """\
notes              open the menu - pick what you want from a list
notes class        record a class; stops by itself after 60 minutes
notes later        record the sound only; write the notes after school
notes catchup      write notes for every class that does not have them yet
notes now          record until you press Ctrl-C (a long lecture)
notes online       record an online class (Zoom/Teams/YouTube)
notes setup-online one-time setup so online classes can be recorded
notes last         show the notes from your last class
notes all          list every class
notes check        confirm the microphone and notes writer work
notes update       update NoteTaker
"""


def translate(argv: list[str]) -> list[str] | None:
    """Turn `notes <word> ...` into CLI arguments. None means print help."""
    if not argv:
        return ["menu"]
    word, rest = argv[0], argv[1:]
    if word in ("help", "-h", "--help"):
        return None
    if word in ("last", "latest", "read"):
        from . import store

        sessions = store.list_sessions(limit=1)
        return ["show", sessions[0].id] if sessions else ["list"]
    if word in VERBS:
        return [a.replace("{minutes}", _class_minutes()) for a in VERBS[word]] + rest
    return argv


def _auto_update(word: str) -> None:
    """Same promise as the bash launcher: never block opening the app."""
    if os.environ.get("NOTES_AUTO_UPDATE", "1") == "0" or word in ("update", "upgrade"):
        return
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=APP_DIR,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if status.returncode != 0 or status.stdout.strip():
        return
    from .update import UpdateError, update_checkout

    try:
        update_checkout(APP_DIR, Path(sys.executable))
    except UpdateError as exc:
        print(f"{exc}\nContinuing without updating. Run 'notes update' later to retry.",
              file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _auto_update(argv[0] if argv else "")
    translated = translate(argv)
    if translated is None:
        print(HELP)
        return 0
    from .cli import main as cli_main

    return cli_main(translated)


if __name__ == "__main__":
    raise SystemExit(main())
