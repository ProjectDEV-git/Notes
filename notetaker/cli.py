"""Command line interface for NoteTaker.

    notetaker devices
    notetaker record [--source mic|system] [--title T] [--live-notes]
    notetaker list
    notetaker show <id> [--transcript]
    notetaker summarize <id> [--rerun] [--hq]
    notetaker export <id> [--md|--txt] [-o FILE]

See docs/BUILD_PLAN.md phase 8b.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from . import config, store, summarize
from .asr import Transcriber, TranscriptWriter
from .audio import AudioError, list_sources, resolve_source
from .pipeline import RecordingPipeline

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    HAVE_RICH = True
except ImportError:  # pragma: no cover - rich is a listed dependency
    console = None
    HAVE_RICH = False


def echo(message: str = "") -> None:
    if console:
        console.print(message)
    else:
        print(message)


def fail(message: str, code: int = 1) -> int:
    if console:
        console.print(f"[red]error:[/red] {message}")
    else:
        print(f"error: {message}", file=sys.stderr)
    return code


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# --------------------------------------------------------------------------
# devices
# --------------------------------------------------------------------------
def cmd_menu(args: argparse.Namespace) -> int:
    """The friendly front door: a numbered list, no flags to remember."""
    from .menu import main as menu_main

    return menu_main()


def cmd_check(args: argparse.Namespace) -> int:
    """Verify the machine can record and summarize before a lecture starts."""
    from .menu import print_checks, run_checks

    ok = print_checks(run_checks())
    if ok:
        echo("\n[green]Everything is ready. You can record a lecture.[/green]")
    return 0 if ok else 1


def cmd_devices(args: argparse.Namespace) -> int:
    try:
        sources = list_sources()
    except AudioError as exc:
        return fail(str(exc))

    if not sources:
        return fail("no capture devices found")

    if HAVE_RICH:
        table = Table(title="Audio sources", header_style="bold")
        table.add_column("use with")
        table.add_column("kind")
        table.add_column("device")
        for src in sources:
            alias = "--source system" if src.kind == config.SOURCE_SYSTEM else "--source mic"
            kind = "system audio" if src.kind == config.SOURCE_SYSTEM else "microphone"
            label = src.description + (" [dim](default)[/dim]" if src.is_default else "")
            table.add_row(alias, kind, label)
        console.print(table)
        console.print(
            "\n[dim]mic    = in-person lecture (microphone)\n"
            "system = online lecture (Zoom/Teams/YouTube playing through your speakers)[/dim]"
        )
    else:
        for src in sources:
            print(f"{src.kind:8} {src.description}")
    return 0


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------
def _render_live(state, source_label: str, live_notes: bool):
    """Build the live display for a recording in progress."""
    header = Text()
    header.append("● REC ", style="bold red")
    header.append(format_duration(state.elapsed), style="bold")
    header.append(f"   {source_label}", style="dim")
    if state.language:
        header.append(f"   lang={state.language}", style="dim")

    bars = int(min(state.level * 60, 20))
    meter = Text()
    meter.append("level ", style="dim")
    meter.append("█" * bars, style="green" if bars else "dim")
    meter.append("░" * (20 - bars), style="dim")

    body = Table.grid(padding=(0, 1))
    body.add_row(header)
    body.add_row(meter)
    body.add_row(Text(f"chunks transcribed: {state.chunks_done}", style="dim"))
    if state.warning:
        body.add_row(Text(f"⚠ {state.warning}", style="yellow"))

    if state.recent_text:
        body.add_row(Text("\nTranscript", style="bold"))
        for line in state.recent_text:
            body.add_row(Text(f"  {line}", style="white"))

    if live_notes:
        body.add_row(Text("\nKey ideas so far", style="bold cyan"))
        if state.live_points:
            for point in state.live_points[-8:]:
                body.add_row(Text(f"  • {point}", style="cyan"))
        else:
            body.add_row(Text("  (waiting for the first interval...)", style="dim"))

    if state.error:
        body.add_row(Text(f"\n{state.error}", style="red"))

    return Panel(body, title="NoteTaker", subtitle="press Ctrl-C to stop", border_style="blue")


def cmd_record(args: argparse.Namespace) -> int:
    config.ensure_dirs()

    try:
        source = resolve_source(args.source)
    except AudioError as exc:
        return fail(str(exc))

    if args.live_notes and not summarize.ollama_available():
        echo("[yellow]warning:[/yellow] Ollama is not running, live notes disabled")
        args.live_notes = False

    title = args.title or "Lecture"
    session = store.create_session(title, source.kind, source.name)

    language = None if args.lang in (None, "auto") else args.lang
    pipeline = RecordingPipeline(
        source=source,
        session=session,
        model=args.model,
        language=language,
        live_notes=args.live_notes,
        summary_model=args.summary_model,
        chunk_seconds=args.chunk_seconds,
    )

    echo(f"[dim]loading {args.model} model...[/dim]")
    try:
        pipeline.start()
    except (AudioError, ValueError) as exc:
        return fail(str(exc))

    stopping = threading.Event()

    def handle_interrupt(signum, frame):
        stopping.set()

    signal.signal(signal.SIGINT, handle_interrupt)

    label = f"{source.description} ({'system audio' if source.kind == config.SOURCE_SYSTEM else 'microphone'})"
    try:
        if HAVE_RICH and not args.plain:
            with Live(console=console, refresh_per_second=4, transient=False) as live:
                while not stopping.is_set():
                    live.update(_render_live(pipeline.state, label, args.live_notes))
                    stopping.wait(0.25)
        else:
            echo(f"recording from {label}. Press Ctrl-C to stop.")
            while not stopping.is_set():
                stopping.wait(1.0)
    finally:
        pending = pipeline.state.chunks_pending
        if pending:
            echo(
                f"\n[yellow]transcription is {pending} chunks behind, catching up now.[/yellow] "
                "[dim]This is normal for Thai, which runs slower than real time on CPU.[/dim]"
            )
        else:
            echo("\n[dim]finishing up, transcribing the last chunk...[/dim]")
        pipeline.stop()
        duration = pipeline.finish()

    segments = store.load_transcript(session)
    echo(f"[green]saved[/green] {len(segments)} segments · {format_duration(duration)} · {session.id}")

    if not segments:
        echo("[yellow]no speech detected, skipping summary[/yellow]")
        return 0

    if args.no_summary:
        echo(f"[dim]run: notetaker summarize {session.id}[/dim]")
        return 0

    return _summarize_session(session, model=args.summary_model)


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------
def _summarize_session(session: store.Session, model: str, quiet: bool = False) -> int:
    segments = store.load_transcript(session)
    if not segments:
        return fail(f"session {session.id} has no transcript")

    if not summarize.ollama_available():
        return fail(
            "Ollama is not running, so notes cannot be generated. "
            f"Start it with 'ollama serve', then: notetaker summarize {session.id}\n"
            f"Your transcript is safe at {session.transcript_path}"
        )

    def progress(index: int, total: int) -> None:
        if not quiet:
            echo(f"[dim]summarizing window {index}/{total}...[/dim]")

    try:
        notes = summarize.summarize_segments(
            segments,
            title=session.title,
            language=session.language,
            model=model,
            duration=session.duration,
            progress=progress,
        )
    except summarize.SummarizerError as exc:
        return fail(str(exc))

    store.write_notes(session, notes.markdown)
    echo()
    echo(notes.markdown)
    echo(f"[green]notes saved:[/green] {session.notes_path}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    try:
        session = store.resolve_session(args.session)
    except KeyError as exc:
        return fail(str(exc))

    if session.has_notes and not args.rerun:
        echo(session.notes_path.read_text(encoding="utf-8"))
        echo("[dim]use --rerun to regenerate[/dim]")
        return 0

    if args.hq:
        if not session.audio_path.exists():
            return fail("the original recording is gone, cannot re-transcribe")
        echo(f"[dim]re-transcribing with {config.ASR_MODEL_HQ} (slower, more accurate)...[/dim]")
        transcriber = Transcriber(config.ASR_MODEL_HQ, language=session.language)
        segments = transcriber.transcribe_file(session.audio_path)
        session.transcript_path.write_text("", encoding="utf-8")
        with TranscriptWriter(session.transcript_path) as writer:
            writer.write(segments)
        echo(f"[dim]{len(segments)} segments[/dim]")

    return _summarize_session(session, model=args.model)


# --------------------------------------------------------------------------
# list / show / export
# --------------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    sessions = store.list_sessions(limit=args.limit)
    if not sessions:
        echo("no recordings yet. Start one with: notetaker record")
        return 0

    if HAVE_RICH:
        table = Table(header_style="bold")
        table.add_column("id")
        table.add_column("title")
        table.add_column("when")
        table.add_column("length", justify="right")
        table.add_column("lang")
        table.add_column("notes")
        for session in sessions:
            flag = "✓" if session.has_notes else ("…" if not session.is_complete else "-")
            table.add_row(
                session.id,
                session.title,
                session.started_at.replace("T", " ")[:16],
                format_duration(session.duration),
                session.language or "?",
                flag,
            )
        console.print(table)
        if any(not s.is_complete for s in sessions):
            console.print("[dim]… = recording was interrupted; the transcript is still usable[/dim]")
    else:
        for session in sessions:
            print(f"{session.id}  {session.title}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    try:
        session = store.resolve_session(args.session)
    except KeyError as exc:
        return fail(str(exc))

    if args.transcript:
        segments = store.load_transcript(session)
        if not segments:
            return fail("no transcript for this session")
        for seg in segments:
            minutes, secs = divmod(int(seg.start), 60)
            echo(f"[dim]{minutes:02d}:{secs:02d}[/dim] {seg.text}")
        return 0

    if session.has_notes and session.notes_path.exists():
        echo(session.notes_path.read_text(encoding="utf-8"))
        return 0

    echo(f"no notes yet. Run: notetaker summarize {session.id}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    try:
        session = store.resolve_session(args.session)
    except KeyError as exc:
        return fail(str(exc))

    if args.txt:
        segments = store.load_transcript(session)
        if not segments:
            return fail("no transcript to export")
        content = "\n".join(s.text for s in segments)
        suffix = ".txt"
    else:
        if not session.notes_path.exists():
            return fail(f"no notes yet. Run: notetaker summarize {session.id}")
        content = session.notes_path.read_text(encoding="utf-8")
        suffix = ".md"

    destination = Path(args.output) if args.output else Path.cwd() / f"{session.id}{suffix}"
    destination.write_text(content, encoding="utf-8")
    echo(f"[green]exported:[/green] {destination}")
    return 0


# --------------------------------------------------------------------------
# lang
# --------------------------------------------------------------------------
# Whisper's own language codes. Transcription works for all of these; a
# language pack only decides what language the NOTES are written in.
WHISPER_LANGUAGES = {
    "en": "English", "th": "Thai", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "hi": "Hindi", "id": "Indonesian", "vi": "Vietnamese", "nl": "Dutch",
    "pl": "Polish", "tr": "Turkish", "sv": "Swedish", "uk": "Ukrainian",
    "he": "Hebrew", "fa": "Persian", "ms": "Malay", "ta": "Tamil",
    "my": "Burmese", "km": "Khmer", "lo": "Lao", "bn": "Bengali",
}

# Scripts written without spaces between words. Grounding and dedupe use
# character n-grams for these, as they already do for Thai.
UNSPACED_SCRIPTS = {
    "th": ["\u0e00", "\u0e7f"],
    "zh": ["\u4e00", "\u9fff"],
    "ja": ["\u3040", "\u30ff"],
    "km": ["\u1780", "\u17ff"],
    "lo": ["\u0e80", "\u0eff"],
    "my": ["\u1000", "\u109f"],
}


def cmd_lang(args: argparse.Namespace) -> int:
    from . import languages

    if args.lang_command == "list":
        installed = languages.load_all(refresh=True)
        if HAVE_RICH:
            table = Table(title="Note languages", header_style="bold")
            table.add_column("code")
            table.add_column("notes written in")
            table.add_column("source")
            for code in sorted(installed):
                lang = installed[code]
                custom = (languages.LANGUAGE_DIR / f"{code}.json").exists()
                table.add_row(code, lang.name, "custom" if custom else "built-in")
            console.print(table)
            console.print(
                "\n[dim]Add one with:  notetaker lang add <code>\n"
                "Transcription already works for ~100 languages; a pack only decides\n"
                "what language your NOTES are written in.[/dim]"
            )
        else:
            for code in sorted(installed):
                print(f"{code}\t{installed[code].name}")
        return 0

    if args.lang_command == "add":
        code = args.code.lower()
        if code not in WHISPER_LANGUAGES and not args.force:
            close = ", ".join(sorted(WHISPER_LANGUAGES)[:12])
            return fail(
                f"{code!r} is not a Whisper language code, so speech could not be "
                f"transcribed for it.\nKnown codes include: {close} ...\n"
                "Use --force if you are sure."
            )

        path = languages.LANGUAGE_DIR / f"{code}.json"
        if path.exists() and not args.force:
            return fail(f"{path} already exists. Edit it, or pass --force to reset it.")

        data = languages.template(code, args.name or WHISPER_LANGUAGES.get(code, code))
        if code in UNSPACED_SCRIPTS:
            data["script_range"] = UNSPACED_SCRIPTS[code]

        path = languages.save(code, data)
        echo(f"[green]created[/green] {path}")
        echo()
        echo("The prompts are in English so the file works immediately.")
        echo("Translate the two prompts into your language for the best results:")
        echo(f"  [bold]{path}[/bold]")
        echo()
        echo("Keep [bold]{text}[/bold] in both prompts: that is where the transcript goes.")
        echo(f"Then record with:  notetaker record --lang {code}")
        return 0

    if args.lang_command == "edit":
        code = args.code.lower()
        path = languages.LANGUAGE_DIR / f"{code}.json"
        if not path.exists():
            builtin = languages.BUILTIN.get(code)
            if not builtin:
                return fail(f"no language {code!r}. Create it with: notetaker lang add {code}")
            # Copy the built-in out so it can be customised.
            path = languages.save(code, builtin.to_dict())
            echo(f"[dim]copied built-in {code} to {path} for editing[/dim]")

        editor = os.environ.get("EDITOR", "nano")
        subprocess.call([editor, str(path)])

        try:
            languages.load_all(refresh=True)
            data = json.loads(path.read_text(encoding="utf-8"))
            languages.Language.from_dict(code, data)
            echo(f"[green]ok[/green] {code} loads correctly")
        except (json.JSONDecodeError, ValueError) as exc:
            return fail(f"{path} is not valid: {exc}")
        return 0

    if args.lang_command == "remove":
        code = args.code.lower()
        path = languages.LANGUAGE_DIR / f"{code}.json"
        if not path.exists():
            return fail(f"no custom language pack for {code!r}")
        path.unlink()
        languages.load_all(refresh=True)
        echo(f"[green]removed[/green] {path}")
        if code in languages.BUILTIN:
            echo(f"[dim]the built-in {code} language is still available[/dim]")
        return 0

    return fail("unknown lang command")


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notetaker",
        description="Record a lecture, transcribe it, and summarize the key ideas. Runs offline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("menu", help="the simple menu (no commands to remember)")
    subparsers.add_parser("check", help="check the microphone, devices and notes writer")
    subparsers.add_parser("devices", help="list microphones and system-audio sources")

    record = subparsers.add_parser("record", help="record a lecture and summarize it")
    record.add_argument(
        "--source", default=None,
        help="mic (in-person) or system (online lecture). Default: system default input",
    )
    record.add_argument("--title", "-t", default=None, help="lecture title")
    record.add_argument(
        "--live-notes", action="store_true",
        help="show key ideas while recording (extra CPU load)",
    )
    record.add_argument(
        "--lang", default="auto",
        choices=["auto", *config.supported_languages()],
        help="lecture language. Default: autodetect. Add more with 'notetaker lang add'",
    )
    record.add_argument("--model", default=config.ASR_MODEL, help="whisper model")
    record.add_argument("--summary-model", default=config.SUMMARY_MODEL, help="Ollama model")
    record.add_argument("--chunk-seconds", type=int, default=config.CHUNK_SECONDS)
    record.add_argument("--no-summary", action="store_true", help="transcribe only")
    record.add_argument("--plain", action="store_true", help="disable the live display")

    listing = subparsers.add_parser("list", help="list past recordings")
    listing.add_argument("--limit", "-n", type=int, default=20)

    show = subparsers.add_parser("show", help="show notes for a recording")
    show.add_argument("session", help="session id or part of the title")
    show.add_argument("--transcript", action="store_true", help="show the transcript instead")

    summarize_cmd = subparsers.add_parser("summarize", help="generate notes for a recording")
    summarize_cmd.add_argument("session", help="session id or part of the title")
    summarize_cmd.add_argument("--rerun", action="store_true", help="regenerate existing notes")
    summarize_cmd.add_argument(
        "--hq", action="store_true",
        help=f"re-transcribe with {config.ASR_MODEL_HQ} first (slow, better for Thai)",
    )
    summarize_cmd.add_argument("--model", default=config.SUMMARY_MODEL)

    export = subparsers.add_parser("export", help="write notes or transcript to a file")
    export.add_argument("session", help="session id or part of the title")
    export.add_argument("--md", action="store_true", help="export notes as markdown (default)")
    export.add_argument("--txt", action="store_true", help="export the raw transcript")
    export.add_argument("--output", "-o", default=None)

    lang = subparsers.add_parser("lang", help="add or edit the languages notes are written in")
    lang_sub = lang.add_subparsers(dest="lang_command", required=True)
    lang_sub.add_parser("list", help="show installed note languages")

    lang_add = lang_sub.add_parser("add", help="add a new note language")
    lang_add.add_argument("code", help="language code, e.g. ja, zh, de")
    lang_add.add_argument("--name", default=None, help="how to label it in the notes")
    lang_add.add_argument("--force", action="store_true", help="overwrite an existing pack")

    lang_edit = lang_sub.add_parser("edit", help="edit a language's prompts in $EDITOR")
    lang_edit.add_argument("code")

    lang_remove = lang_sub.add_parser("remove", help="delete a custom language pack")
    lang_remove.add_argument("code")

    return parser


COMMANDS = {
    "menu": cmd_menu,
    "check": cmd_check,
    "devices": cmd_devices,
    "record": cmd_record,
    "list": cmd_list,
    "show": cmd_show,
    "summarize": cmd_summarize,
    "export": cmd_export,
    "lang": cmd_lang,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Bare `notetaker` used to print a usage error. A non-technical user has
    # nothing to go on at that point, so open the menu instead.
    if not argv:
        argv = ["menu"]
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        echo("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
