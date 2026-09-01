"""Interactive, no-typing interface for NoteTaker.

Most people recording a lecture are not going to remember
`notetaker record --source system --live-notes`. Running `notes` with no
arguments therefore opens a numbered menu where every action is one keypress,
and pressing Enter does the most common thing (record this lecture now).

Design rules:
  * Never require the user to type an id, a path, or a flag.
  * Every prompt has a safe default, so Enter is always a valid answer.
  * Nothing here is required: every action has an equivalent CLI command,
    which is printed after it runs so the interface teaches itself.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from typing import Callable

from . import config, store

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console: Console | None = Console()
except ImportError:  # pragma: no cover - rich is a listed dependency
    console = None


def out(message: str = "") -> None:
    if console:
        console.print(message)
    else:
        # Strip rich markup so the plain fallback stays readable.
        import re

        print(re.sub(r"\[/?[a-z ]+\]", "", message))


def ask(prompt: str, default: str = "") -> str:
    """Read one line. EOF (piped input, Ctrl-D) falls back to the default.

    Without this, running the menu in a non-interactive context would raise
    EOFError and look like a crash.
    """
    try:
        answer = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer or default


def confirm(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    answer = ask(prompt + suffix, "y" if default else "n").lower()
    return answer.startswith("y")


# --------------------------------------------------------------------------
# Setup check ("does this machine have what NoteTaker needs?")
# --------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _pkg_hint(linux: str, mac: str) -> str:
    from .audio import IS_MACOS

    return mac if IS_MACOS else linux


def run_checks() -> list[Check]:
    """Everything that must be true before a lecture can be recorded.

    Checked up front because each failure is silent and expensive otherwise:
    a missing loopback device means an hour of silence, a stopped Ollama
    means no notes at the end of the lecture.
    """
    from . import audio, summarize

    checks: list[Check] = []

    have_ffmpeg = shutil.which(config.FFMPEG_BIN) is not None
    checks.append(
        Check(
            "ffmpeg (records the audio)",
            have_ffmpeg,
            "found" if have_ffmpeg else "not installed",
            _pkg_hint("sudo apt install ffmpeg", "brew install ffmpeg"),
        )
    )

    mics: list = []
    systems: list = []
    try:
        sources = audio.list_sources()
        mics = [s for s in sources if s.kind == config.SOURCE_MIC]
        systems = [s for s in sources if s.kind == config.SOURCE_SYSTEM]
        detail = f"{len(mics)} microphone(s), {len(systems)} system-audio source(s)"
        checks.append(Check("audio devices", bool(sources), detail))
    except audio.AudioError as exc:
        checks.append(
            Check(
                "audio devices",
                False,
                str(exc).splitlines()[0],
                _pkg_hint("sudo apt install pulseaudio-utils", "brew install ffmpeg"),
            )
        )

    checks.append(
        Check(
            "in-person lectures (microphone)",
            bool(mics),
            mics[0].description if mics else "no microphone found",
        )
    )
    checks.append(
        Check(
            "online lectures (system audio)",
            bool(systems),
            systems[0].description if systems else "not available",
            "" if systems else _pkg_hint(
                "your machine exposes no .monitor source; check `notes devices`",
                "brew install blackhole-2ch, then make a Multi-Output Device "
                "in Audio MIDI Setup",
            ),
        )
    )

    running = summarize.ollama_available()
    checks.append(
        Check(
            "Ollama (writes the notes)",
            running,
            "running" if running else "not running",
            "" if running else "start it with: ollama serve",
        )
    )

    if running:
        models = summarize.installed_models()
        have_model = any(m.startswith(config.SUMMARY_MODEL.split(":")[0]) for m in models)
        checks.append(
            Check(
                f"summary model ({config.SUMMARY_MODEL})",
                have_model,
                "installed" if have_model else "not downloaded",
                "" if have_model else f"ollama pull {config.SUMMARY_MODEL}",
            )
        )

    return checks


def print_checks(checks: list[Check]) -> bool:
    """Render the checks. Returns True when everything needed is present."""
    if console:
        table = Table(title="NoteTaker setup", header_style="bold", show_lines=False)
        table.add_column("")
        table.add_column("what")
        table.add_column("status")
        for check in checks:
            mark = "[green]✓[/green]" if check.ok else "[red]✗[/red]"
            table.add_row(mark, check.name, check.detail)
        console.print(table)
    else:
        for check in checks:
            print(f"{'ok ' if check.ok else 'MISSING'} {check.name}: {check.detail}")

    fixes = [c for c in checks if not c.ok and c.fix]
    if fixes:
        out("\n[bold]To fix:[/bold]")
        for check in fixes:
            out(f"  {check.name}\n    [cyan]{check.fix}[/cyan]")
    return all(c.ok for c in checks)


# --------------------------------------------------------------------------
# Pickers
# --------------------------------------------------------------------------
def pick_session(action: str = "open", limit: int = 10) -> store.Session | None:
    """Choose a past lecture from a numbered list instead of typing an id."""
    sessions = store.list_sessions(limit=limit)
    if not sessions:
        out("[yellow]No lectures recorded yet.[/yellow]")
        return None
    if len(sessions) == 1:
        return sessions[0]

    out(f"\n[bold]Which lecture do you want to {action}?[/bold]")
    for index, session in enumerate(sessions, start=1):
        when = session.started_at.replace("T", " ")[:16]
        notes = "notes ready" if session.has_notes else "no notes yet"
        marker = " [dim](most recent)[/dim]" if index == 1 else ""
        out(f"  [bold]{index}[/bold]. {session.title} [dim]— {when}, {notes}[/dim]{marker}")

    answer = ask("\nNumber (Enter = most recent, 0 = cancel): ", "1")
    if answer == "0":
        return None
    try:
        chosen = sessions[int(answer) - 1]
    except (ValueError, IndexError):
        out("[yellow]Not a valid number, using the most recent lecture.[/yellow]")
        return sessions[0]
    return chosen


def pick_language() -> str:
    """Ask for the lecture language, defaulting to autodetect."""
    codes = list(config.supported_languages())
    out("\n[bold]What language is the lecture in?[/bold]")
    out("  [bold]1[/bold]. Detect automatically [dim](recommended)[/dim]")
    for index, code in enumerate(codes, start=2):
        out(f"  [bold]{index}[/bold]. {code}")
    answer = ask("\nNumber (Enter = detect automatically): ", "1")
    if answer == "1":
        return "auto"
    try:
        return codes[int(answer) - 2]
    except (ValueError, IndexError):
        return "auto"


# --------------------------------------------------------------------------
# Menu
# --------------------------------------------------------------------------
@dataclass
class Option:
    key: str
    label: str
    hint: str
    run: Callable[[], int]


def _cli(*argv: str) -> int:
    """Run a normal CLI command, then show it so the user learns the shortcut."""
    from .cli import main as cli_main

    code = cli_main(list(argv))
    out(f"\n[dim]Same thing, typed:  notes {' '.join(argv)}[/dim]")
    return code


def _record(source: str) -> int:
    title = ask("\nWhat is this lecture called? (Enter to skip): ", "")
    argv = ["record", "--source", source, "--live-notes"]
    if title:
        argv += ["--title", title]
    out("\n[dim]Recording starts now. Press Ctrl-C when the lecture ends.[/dim]")
    return _cli(*argv)


def _record_with_options() -> int:
    language = pick_language()
    title = ask("\nWhat is this lecture called? (Enter to skip): ", "")
    online = confirm("\nIs this an online lecture (Zoom/Teams/YouTube)?", default=False)
    argv = [
        "record",
        "--source", config.SOURCE_SYSTEM if online else config.SOURCE_MIC,
        "--live-notes",
        "--lang", language,
    ]
    if title:
        argv += ["--title", title]
    out("\n[dim]Recording starts now. Press Ctrl-C when the lecture ends.[/dim]")
    return _cli(*argv)


def _open_last() -> int:
    session = pick_session("read")
    return 0 if session is None else _cli("show", session.id)


def _make_notes() -> int:
    session = pick_session("summarize")
    if session is None:
        return 0
    argv = ["summarize", session.id]
    if session.has_notes and confirm("\nNotes already exist. Write them again?", default=False):
        argv.append("--rerun")
    return _cli(*argv)


def _save_to_file() -> int:
    session = pick_session("save")
    if session is None:
        return 0
    destination = ask(
        f"\nSave as (Enter for {session.id}.md): ", f"{session.id}.md"
    )
    return _cli("export", session.id, "-o", destination)


def _check() -> int:
    ok = print_checks(run_checks())
    if ok:
        out("\n[green]Everything is ready. You can record a lecture.[/green]")
    out("\n[dim]Same thing, typed:  notes check[/dim]")
    return 0 if ok else 1


def options() -> list[Option]:
    return [
        Option("1", "Record a lecture I am attending", "uses the microphone",
               lambda: _record(config.SOURCE_MIC)),
        Option("2", "Record an online lecture", "Zoom, Teams, YouTube",
               lambda: _record(config.SOURCE_SYSTEM)),
        Option("3", "Read my notes", "from a past lecture", _open_last),
        Option("4", "Write notes for a past lecture", "if they are missing", _make_notes),
        Option("5", "Save notes to a file", "to share or print", _save_to_file),
        Option("6", "Record with more options", "language, title, source",
               _record_with_options),
        Option("7", "Check that everything works", "microphone, notes writer", _check),
    ]


def show_menu() -> int:
    if console:
        console.print(
            Panel(
                "Record a lecture and get the key ideas written down.\n"
                "[dim]Everything runs on this computer. Nothing is uploaded.[/dim]",
                title="NoteTaker",
                border_style="blue",
            )
        )
    else:
        print("NoteTaker - record a lecture and get the key ideas written down.\n")

    items = options()
    for option in items:
        out(f"  [bold]{option.key}[/bold]. {option.label} [dim]— {option.hint}[/dim]")
    out("  [bold]q[/bold]. Quit")

    answer = ask("\nWhat would you like to do? (Enter = 1, record now): ", "1").lower()
    if answer in ("q", "quit", "exit"):
        return 0

    for option in items:
        if answer == option.key:
            return option.run()

    out("[yellow]I did not understand that. Nothing was changed.[/yellow]")
    out("[dim]Run `notes` again and press a number from the list.[/dim]")
    return 1


def main() -> int:
    if not sys.stdin.isatty():
        # Piped or scripted: a menu would hang or read garbage. Do the
        # predictable thing instead.
        from .cli import main as cli_main

        return cli_main(["record", "--source", config.SOURCE_MIC, "--live-notes"])
    try:
        return show_menu()
    except KeyboardInterrupt:
        out("\ncancelled")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
