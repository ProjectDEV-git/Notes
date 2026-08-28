"""Phase 8b verification: CLI behaviour.

Covers argument parsing, output formatting, and the guarantee that a failure
to summarize never costs the user their transcript.
"""

from __future__ import annotations

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


def test_show_without_notes_suggests_summarize(db, capsys):
    session = make_session()
    cli.cmd_show(cli.build_parser().parse_args(["show", session.id]))
    assert "summarize" in capsys.readouterr().out


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
    assert "summarize" in capsys.readouterr().out


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
