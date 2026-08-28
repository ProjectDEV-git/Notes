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
import signal
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
# argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notetaker",
        description="Record a lecture, transcribe it, and summarize the key ideas. Runs offline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        "--lang", default="auto", choices=["auto", "en", "th"],
        help="lecture language. Default: autodetect",
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

    return parser


COMMANDS = {
    "devices": cmd_devices,
    "record": cmd_record,
    "list": cmd_list,
    "show": cmd_show,
    "summarize": cmd_summarize,
    "export": cmd_export,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        echo("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
