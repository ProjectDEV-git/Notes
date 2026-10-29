"""Find where something was said, across every class.

`notes search photosynthesis` lists each class it came up in, with the time,
so revising a topic does not mean reopening every lesson. For a video the
time is a link straight to that moment.

Plain substring matching, deliberately: it works the same for Thai, which has
no spaces between words, and every word typed must appear, so a search can be
narrowed by adding words. Words may be split across two transcript lines,
because a sentence often is.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from . import store
from .asr import Segment


@dataclass
class Hit:
    start: float
    text: str


@dataclass
class ClassHits:
    session: store.Session
    hits: list[Hit] = field(default_factory=list)
    in_title: bool = False
    in_notes: bool = False


def fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def terms(query: str) -> list[str]:
    return [t for t in fold(query).split() if t]


def matches(text: str, wanted: list[str]) -> bool:
    folded = fold(text)
    return bool(wanted) and all(t in folded for t in wanted)


def search_segments(segments: list[Segment], wanted: list[str]) -> list[Hit]:
    """Transcript lines containing every term, alone or with the next line.

    A hit spanning two lines is reported once, at the first line. The next
    line is skipped so one sentence is not listed twice.
    """
    hits: list[Hit] = []
    skip_next = False
    for index, seg in enumerate(segments):
        if skip_next:
            skip_next = False
            continue
        if matches(seg.text, wanted):
            hits.append(Hit(seg.start, seg.text))
            continue
        if index + 1 < len(segments):
            following = segments[index + 1].text
            # Only a real split: this line holds part of the search, and the
            # next line does not match on its own (then it is its own hit).
            here = fold(seg.text)
            split = any(t in here for t in wanted) and not matches(following, wanted)
            joined = f"{seg.text} {following}"
            if split and matches(joined, wanted):
                hits.append(Hit(seg.start, joined))
                skip_next = True
    return hits


def search(query: str, sessions: list[store.Session] | None = None) -> list[ClassHits]:
    """Every class mentioning the query, most recent first."""
    wanted = terms(query)
    if not wanted:
        return []
    results: list[ClassHits] = []
    for session in sessions if sessions is not None else store.list_sessions():
        found = ClassHits(session)
        found.in_title = matches(session.title, wanted)
        try:
            found.hits = search_segments(store.load_transcript(session), wanted)
        except OSError:
            found.hits = []
        try:
            found.in_notes = session.notes_path.exists() and matches(
                session.notes_path.read_text(encoding="utf-8"), wanted)
        except OSError:
            found.in_notes = False
        if found.hits or found.in_title or found.in_notes:
            results.append(found)
    return results


def snippet(text: str, wanted: list[str], width: int = 110) -> str:
    """The part of a line around the first term, so long lines stay readable."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    at = max(fold(text).find(wanted[0]), 0) if wanted else 0
    start = max(0, min(at - width // 3, len(text) - width))
    clipped = text[start:start + width]
    return ("…" if start else "") + clipped + ("…" if start + width < len(text) else "")
