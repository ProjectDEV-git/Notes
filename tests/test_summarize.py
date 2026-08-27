"""Phase 7 verification: key-idea summarization.

Pure text-processing tests run offline with no model. The end-to-end tests use
the tiny TEST_MODEL so the suite stays fast, and skip when Ollama is down.
"""

from __future__ import annotations

import pytest

from notetaker import config, summarize as S
from notetaker.asr import Segment, read_transcript

FIXTURE = "tests/fixtures/lecture_en.jsonl"

needs_ollama = pytest.mark.skipif(
    not S.ollama_available(), reason="Ollama not running"
)


def seg(text, start=0.0, end=1.0, lang="en"):
    return Segment(start=start, end=end, text=text, lang=lang, chunk=0)


# ------------------------------------------------------------- think stripping
def test_strips_think_block():
    assert S.strip_think("<think>reasoning</think>Answer") == "Answer"


def test_strips_multiline_think_block():
    assert "reasoning" not in S.strip_think("<think>\nlots\nof\nreasoning\n</think>\n- point")


def test_strips_orphan_think_tags():
    """An unclosed tag must not leak into notes.md."""
    assert "<think>" not in S.strip_think("<think>dangling\n- a real point")


def test_leaves_clean_text_alone():
    assert S.strip_think("- a clean bullet") == "- a clean bullet"


# -------------------------------------------------------------------- bullets
def test_parses_dash_bullets():
    assert S.parse_bullets("- first\n- second") == ["first", "second"]


def test_parses_numbered_and_star_bullets():
    assert S.parse_bullets("1. first\n* second") == ["first", "second"]


def test_ignores_model_preamble_and_headings():
    text = "Here are the key points:\n\n## Notes\n- the only real bullet"
    assert S.parse_bullets(text) == ["the only real bullet"]


def test_parses_thai_bullets():
    assert S.parse_bullets("- พลังงานคงที่") == ["พลังงานคงที่"]


# --------------------------------------------------------------------- dedupe
def test_drops_identical_points():
    assert len(S.dedupe_points(["energy is conserved", "energy is conserved"])) == 1


def test_drops_points_differing_only_by_case_or_punctuation():
    assert len(S.dedupe_points(["Energy is conserved.", "energy is conserved"])) == 1


def test_drops_a_point_contained_in_another():
    points = ["gravitational potential energy converts to kinetic energy", "converts to kinetic energy"]
    assert len(S.dedupe_points(points)) == 1


def test_keeps_genuinely_distinct_points():
    points = ["energy is conserved", "momentum is conserved separately"]
    assert len(S.dedupe_points(points)) == 2


def test_dedupes_thai_points():
    """Character-based comparison: Thai has no word spaces."""
    assert len(S.dedupe_points(["พลังงานถูกอนุรักษ์ไว้", "พลังงานถูกอนุรักษ์ไว้"])) == 1


def test_keeps_first_wording_of_a_duplicate():
    assert S.dedupe_points(["Energy is conserved", "energy is conserved"])[0] == "Energy is conserved"


# -------------------------------------------------------------- empty sections
def test_removes_heading_with_no_content():
    messy = "## Key ideas\n- something\n\n## Action items\n"
    assert "Action items" not in S.strip_empty_sections(messy)


def test_keeps_heading_that_has_content():
    ok = "## Key ideas\n- a\n\n## Action items\n- exam on friday"
    assert "Action items" in S.strip_empty_sections(ok)


def test_removes_several_empty_headings():
    messy = "## Key ideas\n- a\n\n## Terms & definitions\n\n## Action items\n"
    result = S.strip_empty_sections(messy)
    assert "Terms" not in result and "Action" not in result
    assert "- a" in result


# -------------------------------------------------------------------- windows
def test_groups_segments_into_windows():
    segments = [seg(f"sentence {i}", start=i * 60, end=i * 60 + 59) for i in range(7)]
    windows = S.build_windows(segments, window_seconds=180)
    assert len(windows) > 1


def test_short_transcript_is_a_single_window():
    segments = [seg("a", 0, 10), seg("b", 10, 20)]
    assert len(S.build_windows(segments, window_seconds=180)) == 1


def test_windows_cover_all_text():
    segments = [seg(f"word{i}", start=i * 60, end=i * 60 + 59) for i in range(6)]
    combined = " ".join(w.text for w in S.build_windows(segments, window_seconds=180))
    for i in range(6):
        assert f"word{i}" in combined


def test_empty_transcript_produces_no_windows():
    assert S.build_windows([]) == []


def test_window_timestamp_is_readable():
    assert S.Window(125.0, 130.0, "x").timestamp == "02:05"


# -------------------------------------------------------------------- errors
def test_empty_transcript_raises():
    with pytest.raises(S.SummarizerError, match="empty"):
        S.summarize_segments([])


def test_unreachable_ollama_gives_actionable_error():
    """The transcript is already saved; the message must say so."""
    with pytest.raises(S.SummarizerError) as excinfo:
        S.chat("hi", url="http://localhost:59999", timeout=2)
    message = str(excinfo.value)
    assert "ollama serve" in message.lower()
    assert "saved" in message.lower()


# ------------------------------------------------------------------ rendering
def test_render_includes_title_and_language():
    out = S._render("Physics 101", "## Key ideas\n- a", "en", 203.0, 45)
    assert out.startswith("# Physics 101")
    assert "English" in out


def test_render_thai_metadata():
    assert "ภาษาไทย" in S._render("ฟิสิกส์", "## แนวคิดสำคัญ\n- a", "th", 60.0, 5)


def test_fallback_body_uses_thai_headings():
    body = S._fallback_body(["จุดสำคัญ"], [], "th")
    assert "แนวคิดสำคัญ" in body


def test_fallback_omits_admin_section_when_empty():
    assert "Action items" not in S._fallback_body(["a point"], [], "en")


def test_compression_ratio():
    assert S.compression_ratio("x" * 100, "y" * 15) == pytest.approx(0.15)


# ------------------------------------------------------------------ end-to-end
@needs_ollama
def test_summarizes_real_lecture_transcript():
    """The core promise: real lecture in, short accurate key ideas out."""
    segments = read_transcript(FIXTURE)
    if not segments:
        pytest.skip("fixture transcript missing")

    notes = S.summarize_segments(
        segments, title="Conservation of Mechanical Energy",
        model=config.TEST_MODEL, duration=203.0,
    )

    assert notes.key_points, "no key ideas extracted"
    assert "<think>" not in notes.markdown.lower()
    assert notes.markdown.startswith("# Conservation")

    transcript = " ".join(s.text for s in segments)
    ratio = S.compression_ratio(transcript, notes.markdown)
    assert ratio < 0.6, f"notes are {ratio:.0%} of the transcript, not a summary"


@needs_ollama
def test_notes_mention_the_lectures_actual_subject():
    """Guards against fluent but content-free output."""
    segments = read_transcript(FIXTURE)
    if not segments:
        pytest.skip("fixture transcript missing")

    notes = S.summarize_segments(segments, model=config.TEST_MODEL)
    text = notes.markdown.lower()
    assert any(term in text for term in ("energy", "kinetic", "potential", "pendulum"))


@needs_ollama
def test_no_empty_sections_in_output():
    segments = read_transcript(FIXTURE)
    if not segments:
        pytest.skip("fixture transcript missing")

    notes = S.summarize_segments(segments, model=config.TEST_MODEL)
    lines = notes.markdown.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("##"):
            rest = [l for l in lines[index + 1:] if l.strip()]
            assert rest and not rest[0].startswith("##"), f"empty section: {line}"
