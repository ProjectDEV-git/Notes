"""Phase 5 verification: session index and crash recovery.

All tests use a temporary database, so the user's real recordings in
~/.local/share/notetaker are never touched.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from notetaker import store
from notetaker.asr import Segment, TranscriptWriter


@pytest.fixture()
def db(tmp_path):
    """A temporary database; session dirs are derived from its location."""
    return tmp_path / "notetaker.db"


def make_session(db, title="Physics 101", kind="mic"):
    return store.create_session(title, kind, "test-device", db_path=db)


# -------------------------------------------------------------------- slugs
def test_slugify_handles_spaces_and_case():
    assert store.slugify("Intro To Machine Learning") == "intro-to-machine-learning"


def test_slugify_keeps_thai_characters():
    """Thai titles must survive; stripping non-ASCII would leave an empty slug."""
    slug = store.slugify("ฟิสิกส์ เบื้องต้น")
    assert "ฟิสิกส์" in slug


def test_slugify_strips_path_separators():
    """A '/' in a title must not escape the sessions directory."""
    assert "/" not in store.slugify("a/b/c")
    assert "\\" not in store.slugify("a\\b")


def test_slugify_never_returns_empty():
    assert store.slugify("!!!") == "lecture"


def test_session_ids_sort_chronologically():
    early = store.new_session_id("x", datetime(2026, 1, 1, 9, 0))
    later = store.new_session_id("x", datetime(2026, 1, 1, 14, 0))
    assert early < later


# ------------------------------------------------------------------ lifecycle
def test_create_then_get_roundtrip(db):
    created = make_session(db)
    fetched = store.get_session(created.id, db_path=db)
    assert fetched is not None
    assert fetched.title == "Physics 101"
    assert fetched.source_kind == "mic"


def test_create_makes_the_session_directory(db):
    session = make_session(db)
    assert session.directory.is_dir()


def test_session_dir_is_isolated_from_real_user_data(db):
    """Guard: a test must never write into ~/.local/share/notetaker."""
    session = make_session(db)
    assert str(db.parent) in str(session.directory)


def test_new_session_is_incomplete_until_finished(db):
    session = make_session(db)
    assert not session.is_complete

    store.finish_session(session.id, duration=123.4, language="en", db_path=db)
    done = store.get_session(session.id, db_path=db)
    assert done.is_complete
    assert done.duration == pytest.approx(123.4)
    assert done.language == "en"


def test_same_title_in_same_minute_does_not_collide(db):
    a = make_session(db, "Repeated Lecture")
    b = make_session(db, "Repeated Lecture")
    assert a.id != b.id
    assert store.get_session(a.id, db_path=db) is not None
    assert store.get_session(b.id, db_path=db) is not None


def test_list_returns_most_recent_first(db):
    make_session(db, "First")
    make_session(db, "Second")
    titles = [s.title for s in store.list_sessions(db_path=db)]
    assert set(titles) == {"First", "Second"}


def test_get_unknown_session_returns_none(db):
    assert store.get_session("does-not-exist", db_path=db) is None


def test_delete_removes_row_and_directory(db):
    session = make_session(db)
    path = session.directory
    store.delete_session(session.id, db_path=db)
    assert store.get_session(session.id, db_path=db) is None
    assert not path.exists()


# ------------------------------------------------------------------ resolving
def test_resolve_by_exact_id(db):
    session = make_session(db)
    assert store.resolve_session(session.id, db_path=db).id == session.id


def test_resolve_by_title_substring(db):
    """Users should not have to type a full timestamped id."""
    session = make_session(db, "Quantum Mechanics")
    assert store.resolve_session("quantum", db_path=db).id == session.id


def test_resolve_reports_ambiguity(db):
    make_session(db, "Calculus Lecture One")
    make_session(db, "Calculus Lecture Two")
    with pytest.raises(KeyError, match="ambiguous"):
        store.resolve_session("calculus", db_path=db)


def test_resolve_unknown_raises(db):
    with pytest.raises(KeyError):
        store.resolve_session("nothing-like-this", db_path=db)


# ------------------------------------------------------------------ artifacts
def test_transcript_roundtrip(db):
    session = make_session(db)
    with TranscriptWriter(session.transcript_path) as w:
        w.write([
            Segment(0.0, 2.0, "energy is conserved", "en", 0),
            Segment(2.0, 4.0, "in a closed system", "en", 0),
        ])

    segments = store.load_transcript(session)
    assert len(segments) == 2
    assert "energy is conserved" in store.transcript_text(session)


def test_write_notes_sets_the_flag(db):
    session = make_session(db)
    assert not session.has_notes

    store.write_notes(session, "## Key ideas\n- energy is conserved\n", db_path=db)
    assert store.get_session(session.id, db_path=db).has_notes
    assert session.notes_path.read_text(encoding="utf-8").startswith("## Key ideas")


def test_notes_preserve_thai(db):
    session = make_session(db, "ฟิสิกส์")
    store.write_notes(session, "## แนวคิดสำคัญ\n- พลังงานคงที่\n", db_path=db)
    assert "พลังงานคงที่" in session.notes_path.read_text(encoding="utf-8")


def test_missing_transcript_reads_as_empty(db):
    assert store.load_transcript(make_session(db)) == []


# --------------------------------------------------------------- crash safety
def test_interrupted_session_is_recoverable(db):
    """A laptop dying mid-lecture must leave a summarizable transcript."""
    session = make_session(db, "Interrupted Lecture")
    writer = TranscriptWriter(session.transcript_path)
    writer.write([Segment(0.0, 3.0, "the first ten minutes survived", "en", 0)])
    # Process dies here: finish_session never runs, writer never closed.

    incomplete = store.recover_incomplete(db_path=db)
    assert session.id in [s.id for s in incomplete]

    recovered = store.load_transcript(store.get_session(session.id, db_path=db))
    assert recovered[0].text == "the first ten minutes survived"
    writer.close()


def test_torn_transcript_line_does_not_break_recovery(db):
    session = make_session(db)
    with TranscriptWriter(session.transcript_path) as w:
        w.write([Segment(0.0, 1.0, "good line", "en", 0)])
    with session.transcript_path.open("a", encoding="utf-8") as fh:
        fh.write('{"start": 5.0, "en')  # killed mid-write

    segments = store.load_transcript(session)
    assert len(segments) == 1
    assert segments[0].text == "good line"


def test_completed_sessions_are_not_flagged_for_recovery(db):
    session = make_session(db)
    store.finish_session(session.id, duration=10.0, db_path=db)
    assert session.id not in [s.id for s in store.recover_incomplete(db_path=db)]


def test_transcript_is_valid_jsonl(db):
    session = make_session(db)
    with TranscriptWriter(session.transcript_path) as w:
        w.write([Segment(0.0, 1.0, "line one", "en", 0), Segment(1.0, 2.0, "line two", "en", 1)])

    for line in session.transcript_path.read_text(encoding="utf-8").splitlines():
        json.loads(line)  # raises if malformed
