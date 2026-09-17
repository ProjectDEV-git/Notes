"""Atomic artifact replacement, including failures before publication."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from notetaker import artifacts, store
from notetaker.asr import Segment


def fail_io(*args, **kwargs):
    raise OSError(errno.ENOSPC, "injected disk full")


@pytest.mark.parametrize("operation", ["fsync", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_failure_preserves_original_and_cleans_temp(tmp_path, monkeypatch, operation, existing):
    target = tmp_path / "notes.md"
    if existing:
        target.write_text("original", encoding="utf-8")
    monkeypatch.setattr(artifacts.os, operation, fail_io)

    with pytest.raises(OSError, match="injected disk full"):
        artifacts.write_text_atomic(target, "replacement")

    if existing:
        assert target.read_text(encoding="utf-8") == "original"
    else:
        assert not target.exists()
    assert list(tmp_path.iterdir()) == ([target] if existing else [])


def test_encoding_failure_preserves_original_and_cleans_temp(tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(UnicodeEncodeError):
        artifacts.write_text_atomic(target, "cannot encode \ud800")
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_uses_synced_sibling_temp(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "notes.md"
    text = "## แนวคิดสำคัญ\n- พลังงานคงที่\n"
    events = []
    original_fsync = artifacts.os.fsync
    original_replace = artifacts.os.replace

    def fsync(fd):
        events.append("fsync")
        original_fsync(fd)

    def replace(source, destination):
        source = Path(source)
        assert source.parent == target.parent
        assert source != target
        assert source.read_bytes() == text.encode("utf-8")
        assert events == ["fsync"]
        events.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "fsync", fsync)
    monkeypatch.setattr(artifacts.os, "replace", replace)
    artifacts.write_text_atomic(target, text)
    assert target.read_bytes() == text.encode("utf-8")
    assert list(target.parent.iterdir()) == [target]
    if artifacts.os.name == "posix":
        assert events == ["fsync", "replace", "fsync"]


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_notes_flag_not_set_when_write_fails(tmp_path, monkeypatch, operation):
    db = tmp_path / "notetaker.db"
    session = store.create_session("Physics", "mic", db_path=db)
    session.notes_path.write_text("previous notes", encoding="utf-8")
    monkeypatch.setattr(artifacts.os, operation, fail_io)

    with pytest.raises(OSError, match="injected disk full"):
        store.write_notes(session, "replacement notes", db_path=db)

    assert not store.get_session(session.id, db_path=db).has_notes
    assert session.notes_path.read_text(encoding="utf-8") == "previous notes"
    assert list(session.directory.iterdir()) == [session.notes_path]


def test_transcript_roundtrip_replaces_instead_of_appending(tmp_path):
    session = store.create_session("Physics", "mic", db_path=tmp_path / "notetaker.db")
    store.write_transcript(session, [Segment(0, 1, "old transcript", "en", 0)])
    segments = [
        Segment(0.0, 2.0, "พลังงานคงที่", "th", 0),
        Segment(2.0, 4.0, 'line with "quotes"\nand newline', "en", 1),
    ]
    assert store.write_transcript(session, iter(segments)) == session.transcript_path
    assert store.load_transcript(session) == segments
    assert "พลังงานคงที่" in session.transcript_path.read_text(encoding="utf-8")
    assert len(session.transcript_path.read_text(encoding="utf-8").splitlines()) == 2
    assert list(session.directory.iterdir()) == [session.transcript_path]


def test_empty_transcript_replaces_previous_content(tmp_path):
    session = store.create_session("Physics", "mic", db_path=tmp_path / "notetaker.db")
    store.write_transcript(session, [Segment(0, 1, "old", "en", 0)])
    store.write_transcript(session, [])
    assert session.transcript_path.read_bytes() == b""
    assert store.load_transcript(session) == []


def test_transcript_failure_preserves_previous_content(tmp_path, monkeypatch):
    session = store.create_session("Physics", "mic", db_path=tmp_path / "notetaker.db")
    previous = [Segment(0, 1, "old", "en", 0)]
    store.write_transcript(session, previous)
    monkeypatch.setattr(artifacts.os, "replace", fail_io)
    with pytest.raises(OSError):
        store.write_transcript(session, [Segment(0, 2, "new", "en", 0)])
    assert store.load_transcript(session) == previous
    assert list(session.directory.iterdir()) == [session.transcript_path]


@pytest.mark.parametrize("command", ["hq", "catchup"])
def test_cli_publication_failure_preserves_recovery_artifacts(tmp_path, monkeypatch, command):
    from notetaker import cli, config

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "notetaker.db")
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path / "sessions")
    session = store.create_session("Physics", "mic")
    previous = [Segment(0, 1, "previous transcript", "en", 0)]
    store.write_transcript(session, previous)
    session.audio_path.write_bytes(b"original audio")
    session.chunks_dir.mkdir()
    chunk = session.chunks_dir / "chunk_000.wav"
    chunk.write_bytes(b"recovery audio")
    messages = []
    monkeypatch.setattr(cli, "echo", messages.append)
    monkeypatch.setattr(cli, "console", None)
    monkeypatch.setattr(cli.summarize, "ollama_available", lambda: True)

    class FakeTranscriber:
        language = "en"

        def __init__(self, *args, **kwargs):
            pass

        def transcribe_file(self, path):
            assert path == session.audio_path
            return [Segment(0, 2, "replacement transcript", "en", 0)]

    def unexpected_summary(*args, **kwargs):
        pytest.fail("summarization must not run after failed publication")

    monkeypatch.setattr(cli, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(cli, "_summarize_session", unexpected_summary)
    monkeypatch.setattr(artifacts.os, "replace", fail_io)
    argv = ["summarize", session.id, "--hq"] if command == "hq" else ["catchup"]
    assert cli.main(argv) == 1
    assert store.load_transcript(session) == previous
    assert session.audio_path.read_bytes() == b"original audio"
    assert chunk.read_bytes() == b"recovery audio"
    assert not store.get_session(session.id).has_notes
    assert not store.get_session(session.id).is_complete
    assert store.pending_sessions()[0].id == session.id
    assert not list(session.directory.glob("*.tmp"))
    if command == "catchup":
        assert "could not save transcript" in " ".join(messages)
