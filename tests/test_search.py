"""Finding where something was said, across every class."""

from __future__ import annotations

from pathlib import Path

import pytest

from notetaker import cli, config, search, shortcuts, store, video
from notetaker.asr import Segment, read_transcript

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "sessions").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", data / "sessions")
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")


def _class(title, segments, kind=config.SOURCE_MIC, notes=None, meta=None):
    session = store.create_session(title, kind)
    store.write_transcript(session, segments)
    if notes:
        store.write_notes(session, notes)
    if meta:
        video.save_meta(session.directory, **meta)
    return session


def S(start, text, lang="en"):
    return Segment(start, start + 3, text, lang, 0)


def test_every_word_typed_must_appear():
    segs = [S(0, "Kinetic energy is motion"), S(5, "Potential energy is stored")]
    assert [h.start for h in search.search_segments(segs, search.terms("energy"))] == [0, 5]
    assert [h.start for h in search.search_segments(segs, search.terms("STORED energy"))] == [5]
    assert search.search_segments(segs, search.terms("energy photosynthesis")) == []


def test_a_sentence_split_over_two_lines_is_found_once():
    segs = [S(0, "thereby converting gravitational potential energy"),
            S(3, "into kinetic energy,"), S(6, "and that way you demolish a building")]
    hits = search.search_segments(segs, search.terms("gravitational kinetic"))
    assert [h.start for h in hits] == [0]


def test_thai_is_found_without_word_spaces():
    thai = read_transcript(FIXTURES / "lecture_th.jsonl")
    word = thai[len(thai) // 2].text.strip()[:6]
    hits = search.search_segments(thai, search.terms(word))
    assert hits and all(word in h.text for h in hits)


def test_classes_are_found_by_transcript_title_or_notes(db):
    _class("Physics", [S(0, "energy is conserved")])
    _class("Energy revision", [S(0, "nothing relevant")])
    _class("Biology", [S(0, "cells")], notes="## Key ideas\n- energy comes from food\n")
    _class("History", [S(0, "the war ended")])
    found = {r.session.title: r for r in search.search("energy")}
    assert set(found) == {"Physics", "Energy revision", "Biology"}
    assert found["Biology"].in_notes and not found["Biology"].hits
    assert found["Energy revision"].in_title


def test_long_lines_are_clipped_around_the_word():
    text = "word " * 50 + "photosynthesis happens here " + "word " * 50
    clip = search.snippet(text, ["photosynthesis"], width=60)
    assert "photosynthesis" in clip and clip.startswith("…") and clip.endswith("…")
    assert len(clip) <= 62


def test_the_command_shows_times_and_video_links(db, capsys):
    _class("Physics", [S(0, "intro"), S(754, "energy is conserved")])
    _class("Energy video", [S(40, "kinetic energy")], kind=config.SOURCE_VIDEO,
           meta={"url": "https://youtu.be/abc", "offset": 60})
    assert cli.main(["search", "energy"]) == 0
    out = capsys.readouterr().out
    assert "2 mention(s) in 2 class(es)" in out
    assert "12:34" in out and "energy is conserved" in out
    assert "1:40" in out, "a trimmed video shows full-video time"
    assert "https://youtu.be/abc?t=97s" in out


def test_many_mentions_are_capped_per_class(db, capsys):
    _class("Physics", [S(i * 10, f"energy point {i}") for i in range(9)])
    assert cli.main(["search", "energy", "-n", "3"]) == 0
    out = capsys.readouterr().out
    assert "energy point 2" in out and "energy point 3" not in out
    assert "and 6 more" in out and "notes show" in out


def test_nothing_found_says_so(db, capsys):
    _class("Physics", [S(0, "energy")])
    assert cli.main(["search", "photosynthesis"]) == 1
    assert "Nothing found" in capsys.readouterr().out


def test_brackets_in_a_transcript_do_not_break_the_display(db, capsys):
    _class("Maths", [S(0, "the set [x for energy] is [bold]closed")])
    assert cli.main(["search", "energy"]) == 0
    assert "[x for energy] is [bold]closed" in capsys.readouterr().out


def test_search_is_a_launcher_word_everywhere():
    assert shortcuts.translate(["search", "energy"]) == ["search", "energy"]
    assert shortcuts.translate(["find", "energy", "kinetic"]) == ["search", "energy", "kinetic"]
