"""Session storage for NoteTaker.

A SQLite table indexes sessions; the bulky artifacts live on disk beside it:

    ~/.local/share/notetaker/
        notetaker.db
        sessions/<id>/
            audio.wav          full recording (enables --hq re-runs)
            chunks/            transient, removed after processing
            transcript.jsonl   appended live, crash-safe
            notes.md           key-idea summary

Session ids are timestamp-prefixed so they sort chronologically.
See docs/BUILD_PLAN.md phase 5.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from . import config
from .asr import Segment, read_transcript

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    source_kind  TEXT NOT NULL,
    source_name  TEXT,
    language     TEXT,
    duration     REAL DEFAULT 0,
    has_notes    INTEGER DEFAULT 0
);
"""


def sessions_dir(db_path: Path | None = None) -> Path:
    """Directory holding session artifacts for a given database.

    Derived from the db location so that passing a temporary db_path keeps
    tests fully isolated from the user's real recordings.
    """
    if db_path is None:
        return config.SESSIONS_DIR
    return Path(db_path).parent / "sessions"


@dataclass
class Session:
    id: str
    title: str
    started_at: str
    ended_at: str | None
    source_kind: str
    source_name: str | None
    language: str | None
    duration: float
    has_notes: bool
    base_dir: Path | None = None  # set when using a non-default database

    @property
    def directory(self) -> Path:
        return (self.base_dir or config.SESSIONS_DIR) / self.id

    @property
    def audio_path(self) -> Path:
        return self.directory / "audio.wav"

    @property
    def transcript_path(self) -> Path:
        return self.directory / "transcript.jsonl"

    @property
    def notes_path(self) -> Path:
        return self.directory / "notes.md"

    @property
    def is_complete(self) -> bool:
        """False for a session interrupted by a crash or power loss."""
        return self.ended_at is not None


def slugify(text: str, max_length: int = 40) -> str:
    """Filesystem-safe slug that keeps Thai characters intact."""
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = re.sub(r"[\s/\\]+", "-", text)
    text = re.sub(r"[^\w\-\u0E00-\u0E7F]", "", text, flags=re.UNICODE)
    return text.strip("-")[:max_length] or "lecture"


def new_session_id(title: str, when: datetime | None = None) -> str:
    stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H%M")
    return f"{stamp}_{slugify(title)}"


# --------------------------------------------------------------------------
# Database access
# --------------------------------------------------------------------------
@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the session database, creating the schema on first use.

    `with conn:` only commits/rolls back, it does NOT close, so the connection
    is closed explicitly in the finally block.
    """
    path = Path(db_path) if db_path else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        yield conn
    finally:
        conn.close()


def _row_to_session(row: sqlite3.Row, base_dir: Path | None = None) -> Session:
    return Session(
        id=row["id"],
        title=row["title"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        source_kind=row["source_kind"],
        source_name=row["source_name"],
        language=row["language"],
        duration=row["duration"] or 0.0,
        has_notes=bool(row["has_notes"]),
        base_dir=base_dir,
    )


def create_session(
    title: str,
    source_kind: str,
    source_name: str | None = None,
    db_path: Path | None = None,
) -> Session:
    """Register a new session and create its directory."""
    started = datetime.now()
    session_id = new_session_id(title, started)

    with connect(db_path) as conn:
        # Collision only happens if two recordings start in the same minute
        # with the same title; suffix rather than overwrite.
        existing = {r["id"] for r in conn.execute("SELECT id FROM sessions")}
        unique_id, suffix = session_id, 2
        while unique_id in existing:
            unique_id = f"{session_id}-{suffix}"
            suffix += 1
        session_id = unique_id

        conn.execute(
            "INSERT INTO sessions (id, title, started_at, source_kind, source_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, title, started.isoformat(timespec="seconds"), source_kind, source_name),
        )
        conn.commit()

    (sessions_dir(db_path) / session_id).mkdir(parents=True, exist_ok=True)
    return get_session(session_id, db_path=db_path)  # type: ignore[return-value]


def finish_session(
    session_id: str,
    duration: float,
    language: str | None = None,
    db_path: Path | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ?, duration = ?, language = COALESCE(?, language) "
            "WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), duration, language, session_id),
        )
        conn.commit()


def mark_notes(session_id: str, has_notes: bool = True, db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE sessions SET has_notes = ? WHERE id = ?", (int(has_notes), session_id))
        conn.commit()


def get_session(session_id: str, db_path: Path | None = None) -> Session | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_session(row, sessions_dir(db_path)) if row else None


def list_sessions(limit: int | None = None, db_path: Path | None = None) -> list[Session]:
    """Most recent first."""
    query = "SELECT * FROM sessions ORDER BY started_at DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    base = sessions_dir(db_path)
    with connect(db_path) as conn:
        return [_row_to_session(r, base) for r in conn.execute(query)]


def resolve_session(selector: str, db_path: Path | None = None) -> Session:
    """Look up by exact id, then by unique id/title substring."""
    exact = get_session(selector, db_path=db_path)
    if exact:
        return exact

    needle = selector.lower()
    matches = [
        s for s in list_sessions(db_path=db_path)
        if needle in s.id.lower() or needle in s.title.lower()
    ]
    if not matches:
        raise KeyError(f"no session matching {selector!r}")
    if len(matches) > 1:
        ids = ", ".join(m.id for m in matches[:5])
        raise KeyError(f"{selector!r} is ambiguous, matches: {ids}")
    return matches[0]


def delete_session(session_id: str, db_path: Path | None = None) -> None:
    import shutil

    with connect(db_path) as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    shutil.rmtree(sessions_dir(db_path) / session_id, ignore_errors=True)


# --------------------------------------------------------------------------
# Artifact helpers
# --------------------------------------------------------------------------
def load_transcript(session: Session) -> list[Segment]:
    """Read a session's transcript, tolerating a torn final line."""
    return read_transcript(session.transcript_path)


def transcript_text(session: Session) -> str:
    return " ".join(s.text for s in load_transcript(session))


def write_notes(session: Session, markdown: str, db_path: Path | None = None) -> Path:
    session.directory.mkdir(parents=True, exist_ok=True)
    session.notes_path.write_text(markdown, encoding="utf-8")
    mark_notes(session.id, True, db_path=db_path)
    return session.notes_path


def recover_incomplete(db_path: Path | None = None) -> list[Session]:
    """Sessions that never got an ended_at, i.e. interrupted recordings.

    Their transcripts are still on disk and can be summarized normally.
    """
    return [s for s in list_sessions(db_path=db_path) if not s.is_complete]
