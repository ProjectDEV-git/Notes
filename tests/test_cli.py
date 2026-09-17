"""Phase 8b verification: CLI behaviour.

Covers argument parsing, output formatting, and the guarantee that a failure
to summarize never costs the user their transcript.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from notetaker import cli, config, store
from notetaker.asr import Segment, TranscriptWriter


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Point the CLI at a temporary data directory."""
    data = tmp_path / "data"
    sessions = data / "sessions"
    sessions.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")
    return data / "notetaker.db"


def make_session(title="Physics 101", with_transcript=True, with_notes=False):
    session = store.create_session(title, config.SOURCE_MIC, "test-device")
    if with_transcript:
        with TranscriptWriter(session.transcript_path) as writer:
            writer.write([
                Segment(0.0, 3.0, "energy is conserved in a closed system", "en", 0),
                Segment(3.0, 6.0, "potential converts to kinetic", "en", 0),
            ])
    if with_notes:
        store.write_notes(session, "## Key ideas\n- energy is conserved\n")
    store.finish_session(session.id, duration=6.0, language="en")
    return store.get_session(session.id)


# ------------------------------------------------------------------- parsing
def test_record_defaults():
    args = cli.build_parser().parse_args(["record"])
    assert args.source is None  # falls back to the system default input
    assert args.live_notes is False  # opt-in, since it costs CPU
    assert args.lang == "auto"


def test_negative_recording_minutes_are_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["record", "--minutes", "-1"])


def test_zero_recording_minutes_means_until_stopped():
    assert cli.build_parser().parse_args(["record", "--minutes", "0"]).minutes == 0


def test_zero_chunk_seconds_are_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["record", "--chunk-seconds", "0"])


def test_source_alias_is_accepted():
    assert cli.build_parser().parse_args(["record", "--source", "system"]).source == "system"


def test_live_notes_flag():
    assert cli.build_parser().parse_args(["record", "--live-notes"]).live_notes is True


def test_language_restricted_to_supported_values():
    for lang in ("auto", "en", "th"):
        assert cli.build_parser().parse_args(["record", "--lang", lang]).lang == lang
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["record", "--lang", "fr"])


def test_summarize_flags():
    args = cli.build_parser().parse_args(["summarize", "abc", "--rerun", "--hq"])
    assert args.rerun and args.hq


def test_command_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_update_command_is_available():
    assert cli.build_parser().parse_args(["update"]).command == "update"


def test_quiet_update_flag_is_available():
    assert cli.build_parser().parse_args(["update", "--quiet"]).quiet is True


# ---------------------------------------------------------------- formatting
@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0:00"), (59, "0:59"), (60, "1:00"), (3599, "59:59"), (3600, "1:00:00")],
)
def test_duration_formatting(seconds, expected):
    assert cli.format_duration(seconds) == expected


def test_duration_of_a_typical_lecture():
    assert cli.format_duration(50 * 60) == "50:00"


# ------------------------------------------------------------------ commands
def test_devices_lists_sources(db, capsys):
    if cli.cmd_devices(cli.build_parser().parse_args(["devices"])) != 0:
        pytest.skip("no audio devices in this environment")
    assert "mic" in capsys.readouterr().out.lower()


def test_list_is_friendly_when_empty(db, capsys):
    cli.cmd_list(cli.build_parser().parse_args(["list"]))
    assert "no recordings yet" in capsys.readouterr().out


def test_list_shows_a_session(db, capsys):
    make_session("Thermodynamics")
    cli.cmd_list(cli.build_parser().parse_args(["list"]))
    assert "Thermodynamics" in capsys.readouterr().out


def test_show_transcript(db, capsys):
    session = make_session()
    cli.cmd_show(cli.build_parser().parse_args(["show", session.id, "--transcript"]))
    assert "energy is conserved" in capsys.readouterr().out


def test_show_without_notes_says_how_to_get_them(db, capsys):
    """The point is actionable advice, not one particular word.

    It used to name 'notetaker summarize <id>', the internal entry point; the
    launcher a student installs is 'notes'.
    """
    session = make_session()
    cli.cmd_show(cli.build_parser().parse_args(["show", session.id]))
    out = capsys.readouterr().out
    assert "no notes" in out.lower()
    assert "notes catchup" in out


def test_show_existing_notes(db, capsys):
    session = make_session(with_notes=True)
    cli.cmd_show(cli.build_parser().parse_args(["show", session.id]))
    assert "Key ideas" in capsys.readouterr().out


def test_show_accepts_a_title_substring(db, capsys):
    make_session("Quantum Field Theory", with_notes=True)
    cli.cmd_show(cli.build_parser().parse_args(["show", "quantum"]))
    assert "Key ideas" in capsys.readouterr().out


def test_unknown_session_fails_cleanly(db, capsys):
    assert cli.cmd_show(cli.build_parser().parse_args(["show", "nope"])) == 1
    assert "no session" in capsys.readouterr().out.lower()


def test_failed_recording_start_does_not_leave_a_ghost_session(db, monkeypatch, capsys):
    from notetaker.audio import AudioSource

    source = AudioSource("stub", "Test microphone", config.SOURCE_MIC)
    monkeypatch.setattr(cli, "resolve_source", lambda _: source)

    class BrokenPipeline:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise cli.AudioError("test device is unavailable")

    monkeypatch.setattr(cli, "RecordingPipeline", BrokenPipeline)

    assert cli.cmd_record(cli.build_parser().parse_args(["record", "--title", "Broken"])) == 1
    assert store.list_sessions() == []
    assert "test device is unavailable" in capsys.readouterr().out


# -------------------------------------------------------------------- export
def test_export_notes_as_markdown(db, tmp_path, capsys):
    session = make_session(with_notes=True)
    out = tmp_path / "notes.md"
    cli.cmd_export(cli.build_parser().parse_args(["export", session.id, "-o", str(out)]))
    assert "Key ideas" in out.read_text(encoding="utf-8")


def test_export_transcript_as_text(db, tmp_path):
    session = make_session()
    out = tmp_path / "transcript.txt"
    cli.cmd_export(cli.build_parser().parse_args(["export", session.id, "--txt", "-o", str(out)]))
    assert "energy is conserved" in out.read_text(encoding="utf-8")


def test_export_without_notes_fails_cleanly(db, tmp_path, capsys):
    session = make_session()
    out = tmp_path / "x.md"
    assert cli.cmd_export(cli.build_parser().parse_args(["export", session.id, "-o", str(out)])) == 1
    printed = capsys.readouterr().out
    assert "no notes" in printed.lower()
    assert "notes catchup" in printed


# ------------------------------------------------------- transcript is sacred
def test_summarize_failure_preserves_the_transcript(db, monkeypatch, capsys):
    """Losing a summary is recoverable; losing an hour of lecture is not."""
    session = make_session()
    monkeypatch.setattr("notetaker.summarize.ollama_available", lambda *a, **k: False)

    assert cli.cmd_summarize(cli.build_parser().parse_args(["summarize", session.id])) == 1

    output = capsys.readouterr().out
    assert "ollama serve" in output.lower()
    assert session.transcript_path.exists()
    assert "energy is conserved" in store.transcript_text(session)


def test_summarize_without_transcript_fails_cleanly(db, capsys):
    session = make_session(with_transcript=False)
    assert cli.cmd_summarize(cli.build_parser().parse_args(["summarize", session.id])) == 1


def test_existing_notes_are_not_regenerated_by_default(db, monkeypatch, capsys):
    session = make_session(with_notes=True)

    def explode(*args, **kwargs):
        raise AssertionError("should not re-summarize without --rerun")

    monkeypatch.setattr("notetaker.summarize.summarize_segments", explode)
    assert cli.cmd_summarize(cli.build_parser().parse_args(["summarize", session.id])) == 0
    assert "--rerun" in capsys.readouterr().out


# ------------------------------------------------------------- ASR backlog
def test_backlog_flag_is_off_when_keeping_up():
    from notetaker.pipeline import PipelineState

    assert not PipelineState(chunks_done=5, chunks_pending=0).is_falling_behind
    assert not PipelineState(chunks_done=5, chunks_pending=2).is_falling_behind


def test_backlog_flag_trips_when_behind():
    """Thai runs ~5x slower than real time, so a backlog must be visible."""
    from notetaker.pipeline import PipelineState

    assert PipelineState(chunks_done=1, chunks_pending=3).is_falling_behind


# ------------------------------------------------------------ 'Start here' guide


def _epilog() -> str:
    return cli.build_parser().format_help()


def test_help_contains_start_here_guide_with_copyable_examples():
    text = _epilog()
    assert "Start here" in text
    # Every example must be a copyable module invocation, not bare `notetaker`.
    assert ".venv/bin/python -m notetaker.cli record --title Physics --source mic --minutes 60" in text
    assert ".venv/bin/python -m notetaker.cli record --later --title Physics" in text
    assert ".venv/bin/python -m notetaker.cli catchup" in text
    assert ".venv/bin/python -m notetaker.cli list" in text
    assert ".venv/bin/python -m notetaker.cli show Physics" in text


def test_examples_do_not_promise_an_exact_stop():
    text = _epilog()
    example = re.search(r"--minutes 60.*?\n\s*\n", text, re.S)
    assert example and "about an hour" in example.group(0)


def test_epilog_examples_validate_against_the_parser():
    """Parse the actual displayed examples, so documentation cannot drift."""
    prefix = ".venv/bin/python -m notetaker.cli "
    examples = [
        shlex.split(line.strip()[len(prefix):])
        for line in _epilog().splitlines() if line.strip().startswith(prefix)
    ]
    parsed = [cli.build_parser().parse_args(argv) for argv in examples]
    assert {args.command for args in parsed} >= {"menu", "record", "catchup", "list", "show"}
    assert any(args.command == "record" and args.later for args in parsed)
    assert any(args.command == "record" and args.minutes > 0 for args in parsed)


def test_help_subprocess_exits_zero_without_recording(tmp_path):
    """Help must not create sessions or require recording/model services."""
    home, data, settings = (tmp_path / name for name in ("home", "data", "config"))
    for directory in (home, data, settings):
        directory.mkdir()
    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "XDG_DATA_HOME": str(data),
        "XDG_CONFIG_HOME": str(settings),
        "OLLAMA_URL": "http://127.0.0.1:1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    result = subprocess.run(
        [sys.executable, "-m", "notetaker.cli", "--help"],
        capture_output=True, text=True, timeout=15, env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert "Start here" in result.stdout
    assert "--minutes" in result.stdout
    for directory in (home, data, settings):
        assert list(directory.iterdir()) == []


def test_help_does_not_dispatch_a_command(monkeypatch, capsys):
    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("help must not dispatch a command")

    monkeypatch.setattr(cli, "COMMANDS", dict.fromkeys(cli.COMMANDS, unexpected_dispatch))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "Start here" in capsys.readouterr().out
