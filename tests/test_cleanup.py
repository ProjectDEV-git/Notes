"""Freeing space without losing anything that is still needed."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from notetaker import cli, config, shortcuts, store
from notetaker.asr import Segment


@pytest.fixture()
def db(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "sessions").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", data / "sessions")
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")


def _class(title, days_ago, notes=True, transcript=True, partial=False, audio=b"x" * 2048):
    session = store.create_session(title, config.SOURCE_MIC)
    when = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
    with store.connect() as conn:
        conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (when, session.id))
        conn.commit()
    if audio is not None:
        session.audio_path.write_bytes(audio)
    if transcript:
        store.write_transcript(session, [Segment(0, 3, "energy", "en", 0)])
    if notes:
        store.write_notes(session, "## Key ideas\n- energy\n")
    if partial:
        session.chunks_dir.mkdir()
        (session.chunks_dir / "chunk_00001.wav").write_bytes(b"")
    store.finish_session(session.id, duration=3)
    return store.get_session(session.id)


def test_only_finished_old_classes_lose_their_recording(db):
    old = _class("Old physics", 40)
    _class("Recent", 5)
    _class("Old, no notes yet", 40, notes=False)
    _class("Old, recorded with --later", 40, notes=False, transcript=False)
    _class("Old, transcript cut short", 40, partial=True)
    _class("Old, already cleaned", 40, audio=None)
    assert [s.id for s in store.audio_to_free(30)] == [old.id]


def test_cleanup_deletes_audio_and_keeps_notes(db, capsys):
    old = _class("Old physics", 40)
    assert cli.main(["cleanup", "--yes"]) == 0
    assert not old.audio_path.exists()
    assert old.notes_path.exists() and old.transcript_path.exists()
    out = capsys.readouterr().out
    assert "Freed 2.0 KB" in out and "Old physics" in out


def test_dry_run_deletes_nothing(db):
    old = _class("Old physics", 40)
    assert cli.main(["cleanup", "--dry-run"]) == 0
    assert old.audio_path.exists()


def test_it_asks_first_and_no_is_the_default(db):
    old = _class("Old physics", 40)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=[""]):
        assert cli.main(["cleanup"]) == 0
    assert old.audio_path.exists()
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["y"]):
        assert cli.main(["cleanup"]) == 0
    assert not old.audio_path.exists()


def test_a_script_must_say_yes(db, capsys):
    old = _class("Old physics", 40)
    with patch("sys.stdin.isatty", return_value=False):
        assert cli.main(["cleanup"]) == 1
    assert old.audio_path.exists() and "--yes" in capsys.readouterr().out


def test_the_age_can_be_changed(db):
    week_old = _class("Last week", 8)
    assert store.audio_to_free(30) == []
    assert [s.id for s in store.audio_to_free(7)] == [week_old.id]


def test_nothing_to_do_says_so(db, capsys):
    _class("Recent", 1)
    assert cli.main(["cleanup", "--yes"]) == 0
    assert "Nothing to clean up" in capsys.readouterr().out


def test_cleaned_classes_are_not_sent_back_to_catchup(db):
    _class("Old physics", 40)
    cli.main(["cleanup", "--yes"])
    assert store.pending_sessions() == []


def test_cleanup_is_a_launcher_word():
    assert shortcuts.translate(["cleanup"]) == ["cleanup"]
    assert shortcuts.translate(["tidy", "--dry-run"]) == ["cleanup", "--dry-run"]
