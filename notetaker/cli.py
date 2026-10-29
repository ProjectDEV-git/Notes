"""Command line interface for NoteTaker.

    notetaker devices
    notetaker record [--source mic|system] [--title T] [--live-notes]
    notetaker list
    notetaker show <id> [--transcript]
    notetaker video <link-or-file> [--title T] [--lang L] [--later] [--hq]
    notetaker summarize <id> [--rerun] [--hq]
    notetaker export <id> [--md|--txt] [-o FILE]

See docs/BUILD_PLAN.md phase 8b.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

from . import config, store, summarize
from .asr import Transcriber
from .audio import AudioError, list_sources, resolve_source
from .pipeline import RecordingPipeline
from .update import UpdateError, update_checkout

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


def non_negative_int(value: str) -> int:
    """Parse a duration/count that may be zero, but never negative."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def positive_int(value: str) -> int:
    """Parse a setting that must be greater than zero."""
    parsed = non_negative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def cmd_update(args: argparse.Namespace) -> int:
    """Update the checkout without overwriting local changes."""
    app_dir = Path(__file__).resolve().parent.parent
    python_path = Path(sys.executable)
    try:
        changed, message = update_checkout(app_dir, python_path)
    except UpdateError as exc:
        return fail(str(exc))

    if not args.quiet:
        if changed:
            echo("[green]NoteTaker updated.[/green]")
        else:
            echo("[green]NoteTaker is already up to date.[/green]")
        echo(message)
    return 0


# --------------------------------------------------------------------------
# devices
# --------------------------------------------------------------------------
def cmd_menu(args: argparse.Namespace) -> int:
    """The friendly front door: a numbered list, no flags to remember."""
    from .menu import main as menu_main

    return menu_main()


def cmd_check(args: argparse.Namespace) -> int:
    """Verify the machine can record and summarize before a lecture starts."""
    from .menu import ask, print_checks, run_checks
    from .setup_help import listen_test

    ok = print_checks(run_checks())
    if getattr(args, "listen", False):
        ok = listen_test(echo, ask) and ok
    if ok:
        echo("\n[green]Everything is ready. You can record a class.[/green]")
    return 0 if ok else 1


def cmd_setup_online(args: argparse.Namespace) -> int:
    """Guided one-time setup so online classes record real sound."""
    from .menu import ask, confirm
    from .setup_help import setup_online

    return 0 if setup_online(echo, ask, lambda q: confirm(q, default=True)) else 1


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
def _render_live(state, source_label: str, live_notes: bool, remaining: float | None = None):
    """Build the live display for a recording in progress."""
    header = Text()
    header.append("● REC ", style="bold red")
    header.append(format_duration(state.elapsed), style="bold")
    if remaining is not None:
        # A student wants to know the class is still being recorded and how
        # much is left, without doing arithmetic during a lesson.
        header.append(f"  ({format_duration(max(remaining, 0))} left)", style="bold cyan")
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
    if state.silent_input:
        body.add_row(Text(f"⚠ {state.silent_input}", style="bold red"))

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

    later = getattr(args, "later", False)
    if later:
        # Nothing is transcribed during class, so live notes cannot exist.
        args.live_notes = False
    elif args.live_notes and not summarize.ollama_available():
        echo("[yellow]warning:[/yellow] Ollama is not running, live notes disabled")
        args.live_notes = False

    # An untitled recording is a class, not a lecture, in the default register.
    title = args.title or ("Lecture" if getattr(args, "level", None) == "university" else "Class")
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
        transcribe=not later,
        level=getattr(args, "level", None),
    )

    if later:
        echo("[dim]recording sound only. Write the notes afterwards with: notes catchup[/dim]")
    else:
        echo(f"[dim]loading {args.model} model...[/dim]")
    try:
        pipeline.start()
    except (AudioError, ValueError, OSError, ImportError) as exc:
        # The database row is created before device/model startup so crashes
        # cannot leave a recording that never actually started in the history.
        store.delete_session(session.id)
        return fail(str(exc))

    stopping = threading.Event()

    def handle_interrupt(signum, frame):
        stopping.set()

    signal.signal(signal.SIGINT, handle_interrupt)

    # A school period has a known length, so the recording can end on its own
    # and the student never has to remember to stop it. Ctrl-C still works.
    minutes = max(getattr(args, "minutes", 0) or 0, 0)
    limit = minutes * 60 + config.CLASS_OVERRUN_SECONDS if minutes else None

    def remaining() -> float | None:
        return None if limit is None else limit - pipeline.state.elapsed

    label = f"{source.description} ({'system audio' if source.kind == config.SOURCE_SYSTEM else 'microphone'})"
    if limit is not None:
        # Only mention the overrun when it rounds to a whole minute, otherwise
        # a short grace period reads as the nonsensical "plus 0 min".
        extra = config.CLASS_OVERRUN_SECONDS // 60
        grace = f" (plus {extra} min in case the class runs over)" if extra else ""
        echo(
            f"[dim]recording for {minutes} minutes, then stopping by itself{grace}. "
            "Ctrl-C stops it sooner.[/dim]"
        )
    try:
        if HAVE_RICH and not args.plain:
            with Live(console=console, refresh_per_second=4, transient=False) as live:
                while not stopping.is_set():
                    left = remaining()
                    if (left is not None and left <= 0) or pipeline.state.capture_stopped:
                        break
                    live.update(_render_live(pipeline.state, label, args.live_notes, left))
                    stopping.wait(0.25)
        else:
            echo(f"recording from {label}. Press Ctrl-C to stop.")
            while not stopping.is_set():
                left = remaining()
                if (left is not None and left <= 0) or pipeline.state.capture_stopped:
                    break
                stopping.wait(1.0)
    finally:
        if pipeline.state.capture_stopped:
            echo(
                "\n[red]recording stopped by itself: the microphone or audio "
                "device was lost.[/red] Everything before that is saved.\n"
                "[dim]Check the device with:  notes check[/dim]"
            )
        pending = pipeline.state.chunks_pending
        if later:
            # Nothing is being transcribed, so do not claim otherwise.
            echo("\n[dim]saving the recording...[/dim]")
        elif pending:
            # Only mention Thai when it is actually the language in play;
            # otherwise the explanation is noise that reads like a bug.
            slow_language = (pipeline.state.language or "") == "th"
            why = (
                " [dim]This is normal for Thai, which runs slower than real time "
                "on CPU.[/dim]" if slow_language else ""
            )
            echo(
                f"\n[yellow]transcription is {pending} chunks behind, "
                f"catching up now.[/yellow]{why}"
            )
        else:
            echo("\n[dim]finishing up, transcribing the last chunk...[/dim]")
        drained = pipeline.stop()
        duration = pipeline.finish()
        if not drained:
            # The chunks are still on disk, so this is recoverable. Say so
            # rather than leaving the student thinking the class was lost.
            echo(
                f"[yellow]transcription did not finish in time[/yellow] "
                f"({pipeline.state.undrained_chunks} chunks left). Nothing is lost.\n"
                f"[dim]Finish it later with:  notes catchup[/dim]"
            )

    # Re-read the row: this object was created before recording started, so its
    # duration and language are still empty. Without this the notes header
    # silently loses the class length.
    session = store.get_session(session.id) or session
    segments = store.load_transcript(session)
    echo(f"[green]saved[/green] {len(segments)} segments · {format_duration(duration)} · {session.id}")

    if later:
        # Expected: nothing was transcribed on purpose. The audio is the record.
        echo(
            f"[green]sound saved[/green] for {session.title}. "
            "Nothing has been transcribed yet.\n"
            "[dim]Write the notes when you have time:  notes catchup[/dim]"
        )
        return 0

    if not segments:
        echo("[yellow]no speech detected, skipping summary[/yellow]")
        return 0

    if args.no_summary:
        echo(f"[dim]write the notes later with:  notes catchup[/dim]")
        return 0

    # The live-notes thread already summarized most of the class while it was
    # happening. Reuse that instead of paying for the same model calls twice.
    state = pipeline.state
    premapped = None
    if state.live_consumed:
        premapped = (
            list(state.live_key_points),
            list(state.live_admin_points),
            state.live_consumed,
        )
    return _summarize_session(
        session, model=args.summary_model, premapped=premapped,
        level=getattr(args, "level", None), notes_lang=args.notes_lang,
    )


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------
def _summarize_session(
    session: store.Session,
    model: str,
    quiet: bool = False,
    premapped: tuple[list[str], list[str], int] | None = None,
    level: str | None = None,
    notes_lang: str | None = None,
) -> int:
    segments = store.load_transcript(session)
    if not segments:
        return fail(f"session {session.id} has no transcript")

    if not summarize.ollama_available():
        return fail(
            "Ollama is not running, so notes cannot be generated. "
            f"To start it, {summarize.start_ollama_hint()}, then run:  notes catchup\n"
            f"Your transcript is safe at {session.transcript_path}"
        )

    def progress(index: int, total: int) -> None:
        if not quiet:
            echo(f"[dim]summarizing window {index}/{total}...[/dim]")

    if premapped and premapped[2]:
        echo(
            f"[dim]reusing the notes taken during class "
            f"({premapped[2]} of {len(segments)} parts already done)[/dim]"
        )

    try:
        notes = summarize.summarize_segments(
            segments,
            title=session.title,
            language=session.language,
            model=model,
            duration=session.duration,
            progress=progress,
            premapped=premapped,
            level=level,
        )
    except summarize.SummarizerError as exc:
        # Losing the notes is recoverable; losing the transcript is not. Say
        # where it is, so a failure here never looks like a lost class.
        return fail(f"{exc}\n  transcript: {session.transcript_path}")

    markdown = notes.markdown
    if session.source_kind == config.SOURCE_VIDEO:
        markdown = _annotate_video_notes(session, markdown, segments)
    target = notes_lang or config.NOTES_LANGUAGE
    if target and target != notes.language:
        markdown = _translated(markdown, notes.language, target, model, quiet)
    store.write_notes(session, markdown)
    echo()
    echo(markdown)
    echo(f"[green]notes saved:[/green] {session.notes_path}")
    return 0


def _translated(markdown: str, source: str, target: str, model: str, quiet: bool) -> str:
    """The notes in another language, or as they are if that fails.

    Written and checked in the spoken language first (see translate.py);
    a translation failure must never cost the notes themselves.
    """
    from . import languages, translate

    if not quiet:
        echo(f"[dim]writing the notes in {languages.get(target).name}...[/dim]")
    try:
        return translate.translate_notes(markdown, source, target, model=model)
    except summarize.SummarizerError as exc:
        echo(f"[yellow]could not translate the notes ({exc}); "
             f"keeping them in {languages.get(source).name}[/yellow]")
        return markdown


def _annotate_video_notes(session: store.Session, markdown: str, segments) -> str:
    """Give each note the time it was said, linked into the video.

    Only for videos: there, a click replays the moment. Best-effort, because
    a timestamp is a nicety and must never cost the student their notes.
    """
    from . import timestamps, video

    try:
        meta = video.load_meta(session.directory)
        return timestamps.add_timestamps(
            markdown, segments,
            link=video.link_for(meta),
            offset=float(meta.get("offset") or 0),
            chapters=meta.get("chapters"),
        )
    except Exception:
        return markdown


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
        try:
            store.write_transcript(session, segments)
        except OSError as exc:
            return fail(
                f"could not save transcript: {exc}\n"
                f"Check free disk space and permissions for {session.directory}, "
                "then retry with --hq. The original recording is unchanged."
            )
        echo(f"[dim]{len(segments)} segments[/dim]")

    return _summarize_session(session, model=args.model, level=getattr(args, "level", None),
                              notes_lang=args.notes_lang)


# --------------------------------------------------------------------------
# video
# --------------------------------------------------------------------------
def _previous_import(origin: str, start: float = 0.0, end: float | None = None) -> store.Session | None:
    """The class already made from this video (and the same part of it)."""
    from . import video

    for session in store.list_sessions():
        if session.source_kind != config.SOURCE_VIDEO or session.source_name != origin:
            continue
        meta = video.load_meta(session.directory)
        if float(meta.get("offset") or 0) == start and meta.get("end") == end:
            return session
    return None


def video_time(text: str) -> float:
    from .video import parse_time

    try:
        return parse_time(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _transcription_progress():
    """Print a line every 10% so an hour of transcribing never looks stuck."""
    shown = {"step": 0}

    def report(done: float, total: float) -> None:
        if total <= 0:
            return
        step = min(int(done / total * 10), 10)
        if step > shown["step"]:
            shown["step"] = step
            echo(f"[dim]  transcribed {step * 10}% "
                 f"({format_duration(done)} of {format_duration(total)})[/dim]")

    return report


def cmd_video(args: argparse.Namespace) -> int:
    """Make notes from a YouTube (or other) link, or a video/audio file."""
    from . import video

    config.ensure_dirs()
    pasted = video.clean_pasted(args.source)
    if not pasted:
        return fail("give a video link or a file, for example:  notes video https://youtu.be/...")

    path = config.user_path(pasted)
    is_file = path.is_file()
    if not is_file and not video.looks_like_url(pasted):
        return fail(
            f"{pasted!r} is not a link, and there is no file with that name.\n"
            "Paste the whole link (it starts with https://), or the path to a video file."
        )
    if shutil.which(config.FFMPEG_BIN) is None:
        return fail("ffmpeg is not installed, so the sound cannot be read. Run:  notes check")

    language = None if args.lang in (None, "auto") else args.lang
    info = None
    try:
        if is_file:
            origin = str(path.resolve())
            title = args.title or video.title_from_filename(path)
        else:
            url = video.normalize_url(pasted)
            if not video.have_downloader():
                video.install_downloader(echo)
            echo("[dim]looking up the video...[/dim]")
            info = video.fetch_info(url, echo, language=language)
            if isinstance(info, video.Playlist):
                return _import_playlist(args, info, language)
            origin = info.url
            title = args.title or info.title
    except video.VideoError as exc:
        return fail(str(exc))

    return _import_one(
        args, path=path if is_file else None, url=None if is_file else url,
        info=info, title=title, origin=origin, language=language,
    )


def _import_one(
    args: argparse.Namespace,
    *,
    path: Path | None,
    url: str | None,
    info,
    title: str,
    origin: str,
    language: str | None,
    quiet_repeat: bool = False,
) -> int:
    """Turn one video or file into a class, then transcribe and summarize it."""
    from . import video
    from .audio import wav_duration

    is_file = path is not None

    start, end = args.start or 0.0, args.end
    known_length = info.duration if info else 0
    if known_length and start >= known_length:
        return fail(f"--from {format_duration(start)} is after the end of the video "
                    f"({format_duration(known_length)})")
    if start or end is not None:
        span = f"{format_duration(start)}–{format_duration(end) if end is not None else 'end'}"
        title = f"{title} ({span})"

    earlier = None if args.again else _previous_import(origin, start, end)
    if earlier is not None:
        if quiet_repeat:
            echo(f"[dim]already done: {earlier.title}[/dim]")
            return 0
        echo(
            f"[yellow]You already have this video:[/yellow] {earlier.title} "
            f"[dim]({earlier.id})[/dim]\n"
            f"[dim]Read its notes with:  notes show {earlier.id}\n"
            "Make them again from scratch with --again.[/dim]"
        )
        return 0

    session = store.create_session(title, config.SOURCE_VIDEO, origin)
    video.save_meta(
        session.directory,
        url=None if is_file else origin,
        offset=start,
        end=end,
        chapters=info.chapters if info else [],
    )
    download_dir = session.directory / "download"
    captions: list = []
    use_captions = bool(info and info.caption_key and not args.whisper)
    try:
        if is_file:
            echo(f"[dim]reading the sound from {path.name}...[/dim]")
            video.to_wav(path, session.audio_path, start, end)
        else:
            echo(f"[bold]{title}[/bold]" + (
                f" [dim]({format_duration(info.duration)})[/dim]" if info.duration else ""))
            media, vtt = video.download(
                url, download_dir, echo,
                caption_key=info.caption_key if use_captions else None,
            )
            echo("[dim]preparing the sound...[/dim]")
            video.to_wav(media, session.audio_path, start, end)
            if vtt is not None:
                captions = video.trim_captions(video.parse_vtt(
                    vtt.read_text(encoding="utf-8", errors="replace"),
                    info.language or "en",
                ), start, end)
    except video.VideoError as exc:
        # Nothing usable arrived, so do not leave an empty class in the list.
        store.delete_session(session.id)
        return fail(str(exc))
    except KeyboardInterrupt:
        store.delete_session(session.id)
        raise
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)

    try:
        duration = wav_duration(session.audio_path)
    except Exception:
        duration = info.duration if info else 0.0
    if (start or end is not None) and duration < 1:
        store.delete_session(session.id)
        return fail("nothing is left after trimming. Check the --from and --to times.")

    if captions:
        store.write_transcript(session, captions)
        store.finish_session(session.id, duration=duration, language=info.language)
        echo(
            f"[green]used the video's own subtitles[/green] "
            f"[dim]({len(captions)} lines), so nothing needs transcribing. "
            "Prefer Whisper's transcript? Add --whisper.[/dim]"
        )
    else:
        store.finish_session(session.id, duration=duration, language=language)
        if args.later:
            echo(
                f"[green]sound saved[/green] for {title} "
                f"[dim]({format_duration(duration)})[/dim]. Nothing has been transcribed yet.\n"
                "[dim]Write the notes when you have time:  notes catchup[/dim]"
            )
            return 0

        model = config.ASR_MODEL_HQ if args.hq else args.model
        echo(
            f"[dim]transcribing {format_duration(duration)} of sound with {model}. "
            "On a laptop this can take about as long as the video.[/dim]"
        )
        try:
            transcriber = Transcriber(model, language=language)
            segments = transcriber.transcribe_file(
                session.audio_path, progress=_transcription_progress()
            )
            store.write_transcript(session, segments)
        except KeyboardInterrupt:
            echo("\n[yellow]stopped.[/yellow] The sound is saved, so nothing is lost.\n"
                 "[dim]Finish it later with:  notes catchup[/dim]")
            return 130
        except Exception as exc:
            return fail(
                f"could not transcribe: {exc}\n"
                "The sound is saved. Try again later with:  notes catchup"
            )
        store.finish_session(session.id, duration=duration, language=transcriber.language)
        if not segments:
            echo("[yellow]no speech found in this video, so there is nothing to "
                 "write notes from[/yellow]")
            return 0

    session = store.get_session(session.id) or session
    return _summarize_session(session, model=args.summary_model, level=args.level,
                              notes_lang=args.notes_lang)



def _import_playlist(args: argparse.Namespace, playlist, language: str | None) -> int:
    """Every video in a playlist, one after another, each its own class.

    Re-running the same playlist picks up where it stopped: videos already
    done are skipped, so an interrupted course costs nothing to resume.
    """
    from . import video

    entries = playlist.entries[: args.limit] if args.limit else playlist.entries
    if not entries:
        return fail("that playlist has no videos in it")
    if args.start or args.end is not None:
        return fail("--from and --to work on one video at a time, not a playlist")
    count = len(entries)
    more = f" (the first {count} of {len(playlist.entries)})" if count < len(playlist.entries) else ""
    echo(f"[bold]{playlist.title}[/bold]: {count} videos{more}")
    if not args.later:
        echo("[dim]Transcribing every video takes a long time. To only download now and "
             "write the notes later, stop with Ctrl-C and add --later.[/dim]")

    done, failed = 0, []
    for index, entry in enumerate(entries, start=1):
        echo(f"\n[bold]({index}/{count})[/bold] {entry.title}")
        try:
            info = video.fetch_info(entry.url, echo, language=language)
            if isinstance(info, video.Playlist):
                raise video.VideoError("this entry is a playlist of its own")
        except video.VideoError as exc:
            echo(f"  [yellow]skipped:[/yellow] {exc}")
            failed.append(entry.title)
            continue
        title = f"{args.title}: {info.title}" if args.title else info.title
        code = _import_one(
            args, path=None, url=entry.url, info=info, title=title,
            origin=info.url, language=language, quiet_repeat=True,
        )
        if code == 0:
            done += 1
        else:
            failed.append(info.title)

    echo(f"\n[green]finished {done} of {count}.[/green]")
    if failed:
        echo("[yellow]Not finished:[/yellow] " + ", ".join(failed))
        echo("[dim]Run the same command again to retry; finished videos are skipped. "
             "Videos that were saved but have no notes yet:  notes catchup[/dim]")
    return 0 if not failed else 1

# --------------------------------------------------------------------------
# list / show / export
# --------------------------------------------------------------------------
def cmd_catchup(args: argparse.Namespace) -> int:
    """Finish every class that was recorded but never written up.

    This is the other half of `record --later`, and the safety net for a class
    where transcription could not keep up. It is deliberately a separate step:
    it is slow and CPU-heavy, and the right time to run it is after school,
    not between periods.
    """
    pending = store.pending_sessions()
    if not pending:
        echo("[green]Nothing to catch up on.[/green] Every class has notes.")
        return 0

    if args.limit:
        pending = pending[: args.limit]

    echo(f"[bold]{len(pending)} class(es) to write up.[/bold] This takes a while.\n")
    if not summarize.ollama_available():
        return fail(
            "Ollama is not running, so notes cannot be written. "
            f"To start it, {summarize.start_ollama_hint()}, then run: notes catchup"
        )

    failed = 0
    for index, session in enumerate(pending, start=1):
        echo(f"[bold]({index}/{len(pending)})[/bold] {session.title} [dim]{session.id}[/dim]")

        segments = store.load_transcript(session)
        have_audio = session.audio_path.exists()
        # A transcript cut short is worse than none: summarizing it looks like
        # success while quietly dropping the rest of the class. The full audio
        # is always written, so redo the transcription from that.
        partial = session.transcript_is_partial

        if not segments and not have_audio:
            echo("  [yellow]no sound and no transcript, skipping[/yellow]")
            failed += 1
            continue

        if partial and not have_audio:
            # Nothing better survives; say so rather than implying it is whole.
            echo("  [yellow]the recording is gone, so only the part that was "
                 "transcribed can be written up[/yellow]")
        elif not segments or partial:
            if partial:
                echo(f"  [yellow]transcription was cut short last time "
                     f"({len(segments)} segments); redoing it from the full "
                     f"recording[/yellow]")
            echo(f"  [dim]transcribing the recording ({args.model})...[/dim]")
            try:
                transcriber = Transcriber(args.model, language=session.language)
                segments = transcriber.transcribe_file(session.audio_path)
                # Replace rather than append: the old transcript is a prefix of
                # this one, so appending would duplicate the start of the class.
                try:
                    store.write_transcript(session, segments)
                except OSError as exc:
                    echo(
                        f"  [red]could not save transcript:[/red] {exc}\n"
                        f"  Check free disk space and permissions for {session.directory}, "
                        "then retry: notes catchup. The original recording is unchanged."
                    )
                    failed += 1
                    continue
                # A session recorded with --later has no detected language yet.
                store.finish_session(
                    session.id,
                    duration=session.duration,
                    language=transcriber.language,
                )
                session = store.get_session(session.id) or session
                # The chunks have now been superseded by a full transcript.
                shutil.rmtree(session.chunks_dir, ignore_errors=True)
            except Exception as exc:
                echo(f"  [red]could not transcribe:[/red] {exc}")
                failed += 1
                continue
            echo(f"  [dim]{len(segments)} segments[/dim]")

        if _summarize_session(
            session, model=args.summary_model, quiet=True,
            level=getattr(args, "level", None), notes_lang=args.notes_lang,
        ) != 0:
            failed += 1
            continue
        echo(f"  [green]notes written[/green] {session.notes_path}\n")

    done = len(pending) - failed
    echo(f"[green]finished {done} of {len(pending)}.[/green]")
    if failed:
        echo("[dim]The ones that failed still have their sound and transcript.[/dim]")
    return 0 if not failed else 1


def cmd_search(args: argparse.Namespace) -> int:
    """Where was this said? Every class, with times (and links for videos)."""
    from . import search, timestamps, video

    try:
        from rich.markup import escape
    except ImportError:  # pragma: no cover - rich is a listed dependency
        def escape(text: str) -> str:
            return text

    query = " ".join(args.query).strip()
    wanted = search.terms(query)
    if not wanted:
        return fail("type what to look for, for example:  notes search photosynthesis")
    results = search.search(query)
    if not results:
        echo(f"Nothing found for [bold]{escape(query)}[/bold].")
        echo("[dim]Try fewer words, or another spelling. Every word typed must appear.[/dim]")
        return 1

    total = sum(len(r.hits) for r in results)
    echo(f"[bold]{escape(query)}[/bold]: {total} mention(s) in {len(results)} class(es)\n")
    for found in results:
        session = found.session
        when = session.started_at.replace("T", " ")[:16]
        echo(f"[bold]{escape(session.title)}[/bold] [dim]{when} · {session.id}[/dim]")
        link, offset = None, 0.0
        if session.source_kind == config.SOURCE_VIDEO:
            meta = video.load_meta(session.directory)
            link, offset = video.link_for(meta), float(meta.get("offset") or 0)
        for hit in found.hits[: args.limit]:
            at = hit.start + offset
            label = timestamps.format_time(at)
            line = escape(search.snippet(hit.text, wanted))
            echo(f"  [cyan]{label:>7}[/cyan]  {line}")
            if link:
                echo(f"           [dim]{link(max(at - timestamps.LEAD_IN_SECONDS, 0))}[/dim]")
        hidden = len(found.hits) - args.limit
        if hidden > 0:
            echo(f"  [dim]… and {hidden} more. See them all with:  "
                 f"notes show {session.id} --transcript[/dim]")
        if not found.hits and found.in_notes:
            echo("  [dim]in the notes[/dim]")
        echo()
    return 0


def _latest_with_notes() -> store.Session | None:
    for session in store.list_sessions():
        if session.has_notes and session.notes_path.exists():
            return session
    return None


def cmd_quiz(args: argparse.Namespace) -> int:
    """Practice questions and flashcards from a class's notes."""
    import random

    from . import quiz

    try:
        session = store.resolve_session(args.session) if args.session else _latest_with_notes()
    except KeyError as exc:
        return fail(str(exc))
    if session is None or not session.notes_path.exists():
        return fail("no notes to practise from yet. Write them first with:  notes catchup")

    cards = None if args.new else quiz.load(session.directory)
    if cards is None:
        markdown = session.notes_path.read_text(encoding="utf-8")
        use_model = summarize.ollama_available()
        if not use_model:
            echo("[yellow]Ollama is not running[/yellow], so only the 'words to know' "
                 f"cards can be made. To start it, {summarize.start_ollama_hint()}.")
        else:
            echo("[dim]writing practice questions from your notes...[/dim]")
        try:
            cards = quiz.build(markdown, session.language, model=args.model, use_model=use_model)
        except summarize.SummarizerError as exc:
            echo(f"[yellow]could not write questions ({exc})[/yellow]")
            use_model = False
            cards = quiz.build(markdown, session.language, use_model=False)
        if cards and use_model:
            quiz.save(session.directory, cards)  # instant next time
    if not cards:
        return fail("no practice questions could be made from these notes")

    if args.anki is not None:
        destination = (config.user_path(args.anki) if args.anki
                       else config.caller_dir() / f"{session.id}-flashcards.txt")
        destination.write_text(quiz.to_anki(cards, session.title), encoding="utf-8")
        echo(f"[green]{len(cards)} flashcards saved:[/green] {destination}\n"
             "[dim]In Anki: File > Import, then pick that file.[/dim]")
        return 0

    from rich.markup import escape

    if args.list or not sys.stdin.isatty():
        for number, card in enumerate(cards, start=1):
            echo(f"[bold]{number}. {escape(card.question)}[/bold]")
            echo(f"   {escape(card.answer)}" + (f" [dim]({card.time})[/dim]" if card.time else ""))
        return 0

    from .menu import ask

    order = list(cards)
    if not args.in_order:
        random.shuffle(order)
    echo(f"[bold]{escape(session.title)}[/bold]: {len(order)} questions. "
         "[dim]Answer in your head, then press Enter. Type q to stop.[/dim]")
    score = asked = 0
    for number, card in enumerate(order, start=1):
        echo(f"\n[bold]{number}/{len(order)}.[/bold] {escape(card.question)}")
        if ask("  (Enter to see the answer) ", "").lower() == "q":
            break
        when = f" [dim](said at {card.time})[/dim]" if card.time else ""
        echo(f"  [cyan]{escape(card.answer)}[/cyan]{when}")
        reply = ask("  Did you get it? [y/N/q] ", "n").lower()
        if reply == "q":
            break
        asked += 1
        score += reply.startswith("y")
    if asked:
        echo(f"\n[green]You got {score} of {asked}.[/green]")
        if score < asked:
            echo(f"[dim]Read the notes again with:  notes show {session.id}[/dim]")
    return 0


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

    echo("no notes for this class yet. Write them with:  notes catchup")
    return 0


def _export_printable(session: store.Session, args: argparse.Namespace) -> int:
    """A page to print or share, and a PDF when a browser can make one."""
    from . import printable, quiz

    if not session.notes_path.exists():
        return fail("no notes for this class yet. Write them with:  notes catchup")
    markdown = session.notes_path.read_text(encoding="utf-8")
    page_html = printable.to_html(
        markdown, session.title, quiz.notes_language(markdown, session.language))

    wanted = config.user_path(args.output) if args.output else None
    base = (wanted.with_suffix("") if wanted
            else config.caller_dir() / session.id)
    page = base.with_suffix(".html")
    page.write_text(page_html, encoding="utf-8")
    if not args.pdf:
        echo(f"[green]exported:[/green] {page}\n"
             "[dim]Open it in a browser to read, print, or send.[/dim]")
        return 0

    pdf = base.with_suffix(".pdf")
    browser = printable.find_browser()
    if browser and printable.html_to_pdf(page, pdf, browser):
        page.unlink(missing_ok=True)
        echo(f"[green]exported:[/green] {pdf}")
        return 0
    echo(
        f"[yellow]No browser could make the PDF here[/yellow], so the notes were saved as "
        f"a page instead:\n  {page}\n"
        "[dim]Open it, press Ctrl-P (Cmd-P on a Mac), and choose 'Save as PDF'.[/dim]"
    )
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
    elif args.html or args.pdf:
        return _export_printable(session, args)
    else:
        if not session.notes_path.exists():
            return fail("no notes for this class yet. Write them with:  notes catchup")
        content = session.notes_path.read_text(encoding="utf-8")
        suffix = ".md"

    # Resolved against where the student ran `notes`, not the app folder the
    # launcher runs from: a file there is hard to find and blocks updates.
    destination = (
        config.user_path(args.output) if args.output
        else config.caller_dir() / f"{session.id}{suffix}"
    )
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


_HELP_EPILOG = """\
Start here (run from the project directory):

  .venv/bin/python -m notetaker.cli menu
      The easiest way in: pick what to do from a numbered list.

  .venv/bin/python -m notetaker.cli record --title Physics --source mic --minutes 60
      Record the microphone in person; stops by itself after about an hour
      (--minutes can run a couple of minutes past the limit so the ending
      is not cut off). Omit --minutes to record until Ctrl-C.

  .venv/bin/python -m notetaker.cli record --later --title Physics
      Record only, to save battery; transcribe and write the notes later.
      With no --minutes it keeps recording until you press Ctrl-C;
      write the notes afterwards with `catchup`.

  .venv/bin/python -m notetaker.cli catchup
      Write notes for every recording that does not have them yet.

  .venv/bin/python -m notetaker.cli video https://www.youtube.com/watch?v=...
  .venv/bin/python -m notetaker.cli video ~/Downloads/lesson.mp4
      Make notes from a YouTube (or other) link, or a video/audio file.

  .venv/bin/python -m notetaker.cli list
  .venv/bin/python -m notetaker.cli show Physics
      See past recordings, and read the notes for one. `show Physics`
      matches part of the title; if you have more than one Physics
      class, use the id shown by `list` instead.
"""


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def _add_notes_lang(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--notes-lang", default=config.NOTES_LANGUAGE,
        choices=list(config.supported_languages()),
        help="write the notes in this language, e.g. th for Thai notes from an "
             "English lesson. Default: the language spoken",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notetaker",
        description="Record a lecture, transcribe it, and summarize the key ideas. Runs offline.",
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("menu", help="the simple menu (no commands to remember)")
    check = subparsers.add_parser("check", help="check the microphone, devices and notes writer")
    check.add_argument(
        "--listen", action="store_true",
        help="also record a few seconds to prove the microphone really works",
    )
    subparsers.add_parser(
        "setup-online", help="one-time guided setup for recording online classes"
    )
    subparsers.add_parser("devices", help="list microphones and system-audio sources")
    update = subparsers.add_parser(
        "update", help="update NoteTaker without overwriting local changes"
    )
    update.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)

    record = subparsers.add_parser("record", help="record a lecture and summarize it")
    record.add_argument(
        "--source", default=None,
        help="mic (in-person) or system (online lecture). Default: system default input",
    )
    record.add_argument("--title", "-t", default=None, help="lecture title")
    record.add_argument(
        "--minutes", "-m", type=non_negative_int, default=0,
        help="stop by itself after this many minutes (a class period). "
             "0 = keep recording until Ctrl-C",
    )
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
    record.add_argument(
        "--level", default=config.NOTES_LEVEL, choices=list(config.NOTES_LEVELS),
        help="who the notes are written for. Default: school",
    )
    record.add_argument("--chunk-seconds", type=positive_int, default=config.CHUNK_SECONDS)
    record.add_argument("--no-summary", action="store_true", help="transcribe only")
    record.add_argument(
        "--later", action="store_true",
        help="just record the sound; transcribe and write the notes later with "
             "'notes catchup'. Uses far less battery during class",
    )
    record.add_argument("--plain", action="store_true", help="disable the live display")

    video_cmd = subparsers.add_parser(
        "video",
        help="make notes from a YouTube (or other) link, or a video/audio file",
    )
    video_cmd.add_argument(
        "source", help="a link (YouTube, Vimeo, ...) or a video/audio file on this computer"
    )
    video_cmd.add_argument("--title", "-t", default=None,
                           help="what to call it. Default: the video's own title")
    video_cmd.add_argument(
        "--lang", default="auto",
        choices=["auto", *config.supported_languages()],
        help="language spoken in the video. Default: autodetect",
    )
    video_cmd.add_argument("--model", default=config.ASR_MODEL, help="whisper model")
    video_cmd.add_argument(
        "--hq", action="store_true",
        help=f"transcribe with {config.ASR_MODEL_HQ} (slower, more accurate; better for Thai)",
    )
    video_cmd.add_argument(
        "--whisper", action="store_true",
        help="transcribe the sound even when the video has its own subtitles",
    )
    video_cmd.add_argument(
        "--later", action="store_true",
        help="only download the sound now; write the notes later with 'notes catchup'",
    )
    video_cmd.add_argument(
        "--again", action="store_true",
        help="make a new copy even if this video was done before",
    )
    video_cmd.add_argument(
        "--from", dest="start", type=video_time, default=0.0, metavar="TIME",
        help="start here, e.g. 2:30 to skip an intro",
    )
    video_cmd.add_argument(
        "--to", dest="end", type=video_time, default=None, metavar="TIME",
        help="stop here, e.g. 45:00 to skip the end",
    )
    video_cmd.add_argument(
        "--limit", "-n", type=non_negative_int, default=0,
        help="for a playlist: only the first N videos (0 = all of them)",
    )
    video_cmd.add_argument("--summary-model", default=config.SUMMARY_MODEL, help="Ollama model")
    video_cmd.add_argument(
        "--level", default=config.NOTES_LEVEL, choices=list(config.NOTES_LEVELS),
        help="who the notes are written for. Default: school",
    )

    _add_notes_lang(record)
    _add_notes_lang(video_cmd)

    listing = subparsers.add_parser("list", help="list past recordings")
    listing.add_argument("--limit", "-n", type=positive_int, default=20)

    catchup = subparsers.add_parser(
        "catchup", help="write notes for every class that does not have them yet"
    )
    catchup.add_argument(
        "--limit", "-n", type=non_negative_int, default=0,
        help="only do this many classes (0 = all of them)",
    )
    catchup.add_argument("--model", default=config.ASR_MODEL, help="whisper model")
    catchup.add_argument("--summary-model", default=config.SUMMARY_MODEL)
    catchup.add_argument(
        "--level", default=config.NOTES_LEVEL, choices=list(config.NOTES_LEVELS),
    )

    _add_notes_lang(catchup)

    search_cmd = subparsers.add_parser(
        "search", help="find where something was said, across every class"
    )
    search_cmd.add_argument("query", nargs="+", help="words to look for")
    search_cmd.add_argument(
        "--limit", "-n", type=positive_int, default=5,
        help="mentions to show per class (default 5)",
    )

    quiz_cmd = subparsers.add_parser(
        "quiz", help="practise with questions and flashcards from a class's notes"
    )
    quiz_cmd.add_argument("session", nargs="?", default=None,
                          help="class id or part of the title. Default: the latest with notes")
    quiz_cmd.add_argument("--list", action="store_true",
                          help="print every question and answer instead of asking")
    quiz_cmd.add_argument("--anki", nargs="?", const="", default=None, metavar="FILE",
                          help="save flashcards for Anki (File > Import)")
    quiz_cmd.add_argument("--in-order", action="store_true", help="do not shuffle")
    quiz_cmd.add_argument("--new", action="store_true", help="write fresh questions")
    quiz_cmd.add_argument("--model", default=config.SUMMARY_MODEL, help="Ollama model")

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
    summarize_cmd.add_argument(
        "--level", default=config.NOTES_LEVEL, choices=list(config.NOTES_LEVELS),
        help="who the notes are written for. Default: school",
    )

    _add_notes_lang(summarize_cmd)

    export = subparsers.add_parser("export", help="write notes or transcript to a file")
    export.add_argument("session", help="session id or part of the title")
    export.add_argument("--md", action="store_true", help="export notes as markdown (default)")
    export.add_argument("--txt", action="store_true", help="export the raw transcript")
    export.add_argument("--html", action="store_true",
                        help="a page to print or share (opens in any browser)")
    export.add_argument("--pdf", action="store_true",
                        help="a PDF, made with a browser already on this computer")
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
    "setup-online": cmd_setup_online,
    "devices": cmd_devices,
    "update": cmd_update,
    "record": cmd_record,
    "catchup": cmd_catchup,
    "video": cmd_video,
    "list": cmd_list,
    "search": cmd_search,
    "quiz": cmd_quiz,
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "end", None) is not None and args.end <= (args.start or 0):
        parser.error("--to must be later than --from")
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        echo("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
