"""Each note points back to the moment in the video it came from."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from notetaker import cli, config, store, summarize, timestamps, video
from notetaker.asr import Segment, read_transcript

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LECTURE = read_transcript(FIXTURES / "lecture_en.jsonl")
THAI = read_transcript(FIXTURES / "lecture_th.jsonl")


# ---------------------------------------------------------------- matching
@pytest.mark.parametrize("bullet, said_at", [
    ("A wrecking ball demolishes buildings by converting potential energy to kinetic energy",
     47.86),
    ("Lifting a 15 kilogram object one meter increases its potential energy", 10.16),
    ("150 joules is enough energy to kill a person", 40.73),
    ("A released pendulum bob can never come back higher than where it started", 98.4),
])
def test_a_note_is_traced_to_where_it_was_said(bullet, said_at):
    found = timestamps.locate(bullet, LECTURE)
    assert found is not None
    assert abs(found - said_at) <= 10


def test_a_note_nobody_said_gets_no_time():
    """A wrong link is worse than none: the student would trust it."""
    assert timestamps.locate("Photosynthesis happens in the chloroplasts of plant cells",
                             LECTURE) is None


def test_a_thai_note_is_traced_too():
    """Thai has no word spaces, so matching must use character n-grams."""
    middle = THAI[len(THAI) // 2]
    assert timestamps.locate(middle.text, THAI) == middle.start


def test_times_read_like_a_video_player():
    assert timestamps.format_time(5) == "0:05"
    assert timestamps.format_time(754) == "12:34"
    assert timestamps.format_time(3723) == "1:02:03"


# --------------------------------------------------------------- rendering
NOTES = """# Physics

*2026-10-29 · 3m 23s · English*

## Key ideas
- 150 joules is enough energy to kill a person
- Photosynthesis happens in the chloroplasts
1. A wrecking ball demolishes buildings by converting potential energy to kinetic energy
"""


def test_bullets_get_a_link_into_the_video():
    link = lambda s: video.timestamp_url("https://www.youtube.com/watch?v=abc", s)
    out = timestamps.add_timestamps(NOTES, LECTURE, link=link)
    lines = out.splitlines()
    assert lines[5] == ("- 150 joules is enough energy to kill a person "
                        "([0:40](https://www.youtube.com/watch?v=abc&t=37s))")
    assert lines[6] == "- Photosynthesis happens in the chloroplasts"
    # Said between 0:44 and 1:23: any moment in that stretch is right.
    assert re.search(r"\(\[[01]:\d\d\]\(https://www\.youtube\.com/watch\?v=abc&t=\d+s\)\)$",
                     lines[7])
    bullet = "A wrecking ball demolishes buildings by converting potential energy to kinetic energy"
    assert 44 <= timestamps.locate(bullet, LECTURE) <= 83
    assert lines[0] == "# Physics" and out.endswith("\n")


def test_a_file_gets_plain_times():
    out = timestamps.add_timestamps(NOTES, LECTURE)
    assert "- 150 joules is enough energy to kill a person (0:40)" in out


def test_a_trimmed_video_links_into_the_full_video():
    out = timestamps.add_timestamps(NOTES, LECTURE, offset=600)
    assert "kill a person (10:40)" in out


def test_notes_are_not_stamped_twice():
    once = timestamps.add_timestamps(NOTES, LECTURE, link=lambda s: f"https://x/#t={int(s)}")
    assert timestamps.add_timestamps(once, LECTURE, link=lambda s: "y") == once


# ------------------------------------------------------------------- links
@pytest.mark.parametrize("url, expected", [
    ("https://www.youtube.com/watch?v=abc", "https://www.youtube.com/watch?v=abc&t=90s"),
    ("https://www.youtube.com/watch?v=abc&t=5s&list=PL1",
     "https://www.youtube.com/watch?v=abc&list=PL1&t=90s"),
    ("https://youtu.be/abc", "https://youtu.be/abc?t=90s"),
    ("https://vimeo.com/123", "https://vimeo.com/123#t=90s"),
    ("https://example.com/lesson.mp4", "https://example.com/lesson.mp4#t=90"),
])
def test_links_open_the_video_at_the_right_moment(url, expected):
    assert video.timestamp_url(url, 90.7) == expected


def test_only_web_addresses_become_links():
    assert video.link_for({"url": "https://youtu.be/abc"})(3) == "https://youtu.be/abc?t=3s"
    assert video.link_for({"url": None}) is None
    assert video.link_for({}) is None


# --------------------------------------------------------------- end to end
@pytest.fixture()
def db(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "sessions").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", data / "sessions")
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")


def _summarized(source_kind, meta=None):
    session = store.create_session("Physics", source_kind)
    store.write_transcript(session, LECTURE)
    if meta is not None:
        video.save_meta(session.directory, **meta)
    fake = summarize.Notes(markdown=NOTES)
    with patch.object(summarize, "ollama_available", return_value=True), \
            patch.object(summarize, "summarize_segments", return_value=fake):
        assert cli._summarize_session(session, model="stub") == 0
    return session.notes_path.read_text()


def test_video_notes_are_linked(db):
    notes = _summarized(config.SOURCE_VIDEO, {"url": "https://youtu.be/abc", "offset": 0})
    assert "([0:40](https://youtu.be/abc?t=37s))" in notes


def test_recorded_classes_are_left_as_they_were(db):
    assert _summarized(config.SOURCE_MIC) == NOTES


def test_broken_video_info_never_costs_the_notes(db):
    session_notes = _summarized(config.SOURCE_VIDEO, None)
    assert "(0:40)" in session_notes  # no link without an address, times still help
