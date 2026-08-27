"""Phase 3 verification: transcript segments, overlap dedupe, persistence.

The dedupe tests are the important ones. Duplicated text at chunk seams is the
most likely correctness bug in the streaming design, and Thai exercises the
path that naive whitespace tokenization gets wrong.

These tests do not load a Whisper model, so they run fast and offline.
"""

from __future__ import annotations

import json

from notetaker.asr import (
    Transcriber,
    Segment,
    TranscriptWriter,
    dedupe_overlap,
    normalize,
    read_transcript,
)


def seg(text: str, start: float = 0.0, end: float = 1.0, lang: str = "en", chunk: int = 0) -> Segment:
    return Segment(start=start, end=end, text=text, lang=lang, chunk=chunk)


# ------------------------------------------------------------- normalisation
def test_normalize_folds_case_and_punctuation():
    assert normalize("Hello, World!") == normalize("hello world")


def test_normalize_strips_whitespace_entirely():
    """Whitespace is removed, not used as a separator, so Thai compares correctly."""
    assert " " not in normalize("the mitochondria is the powerhouse")


def test_normalize_handles_thai():
    thai = "สวัสดีครับ นักศึกษาทุกคน"
    assert normalize(thai)
    assert " " not in normalize(thai)


# -------------------------------------------------------------------- dedupe
def test_no_previous_returns_incoming_unchanged():
    incoming = [seg("first thing said")]
    assert dedupe_overlap([], incoming) == incoming


def test_exact_repeat_at_seam_is_dropped():
    previous = [seg("photosynthesis converts light into chemical energy")]
    incoming = [
        seg("photosynthesis converts light into chemical energy"),
        seg("and it happens in the chloroplast"),
    ]
    result = dedupe_overlap(previous, incoming)
    assert len(result) == 1
    assert "chloroplast" in result[0].text


def test_phrase_appears_exactly_once_after_dedupe():
    """The core guarantee: no sentence is transcribed twice."""
    previous = [seg("the derivative of a constant is zero")]
    incoming = [
        seg("the derivative of a constant is zero"),
        seg("now consider the product rule"),
    ]
    combined = previous + dedupe_overlap(previous, incoming)
    joined = " ".join(s.text for s in combined)
    assert joined.count("derivative of a constant") == 1


def test_thai_overlap_is_deduped():
    """Thai has no word spaces: splitting on whitespace would fail here."""
    repeated = "การสังเคราะห์แสงเปลี่ยนพลังงานแสงเป็นพลังงานเคมี"
    previous = [seg(repeated, lang="th")]
    incoming = [
        seg(repeated, lang="th"),
        seg("ต่อไปเราจะพูดถึงคลอโรพลาสต์", lang="th"),
    ]
    result = dedupe_overlap(previous, incoming)
    assert len(result) == 1, "Thai duplicate at seam was not removed"
    assert "คลอโรพลาสต์" in result[0].text


def test_thai_distinct_content_is_kept():
    previous = [seg("การสังเคราะห์แสงเปลี่ยนพลังงานแสง", lang="th")]
    incoming = [seg("ไมโทคอนเดรียสร้างพลังงานให้เซลล์", lang="th")]
    assert len(dedupe_overlap(previous, incoming)) == 1


def test_distinct_content_is_never_dropped():
    previous = [seg("today we cover neural networks")]
    incoming = [seg("a perceptron computes a weighted sum")]
    assert dedupe_overlap(previous, incoming) == incoming


def test_short_common_words_do_not_trigger_dedupe():
    """'the' overlapping must not delete a real sentence."""
    previous = [seg("and so the")]
    incoming = [seg("the experiment showed a clear increase in yield")]
    result = dedupe_overlap(previous, incoming)
    assert len(result) == 1
    assert "experiment" in result[0].text


def test_punctuation_and_case_differences_still_dedupe():
    previous = [seg("Gradient descent minimises the loss function.")]
    incoming = [
        seg("gradient descent minimises the loss function"),
        seg("the learning rate controls step size"),
    ]
    result = dedupe_overlap(previous, incoming)
    assert len(result) == 1


def test_empty_segments_are_discarded():
    result = dedupe_overlap([seg("intro")], [seg("   "), seg("real content here")])
    assert len(result) == 1
    assert result[0].text == "real content here"


# --------------------------------------------------------------- persistence
def test_writer_appends_one_json_object_per_line(tmp_path):
    path = tmp_path / "transcript.jsonl"
    with TranscriptWriter(path) as w:
        w.write([seg("first", 0, 1), seg("second", 1, 2)])

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "first"


def test_writer_flushes_so_a_crash_loses_at_most_one_chunk(tmp_path):
    """Content must be on disk before the process exits, not buffered."""
    path = tmp_path / "transcript.jsonl"
    writer = TranscriptWriter(path)
    writer.write([seg("survives a crash")])
    # Deliberately not closing: simulates the process dying here.
    assert "survives a crash" in path.read_text(encoding="utf-8")
    writer.close()


def test_roundtrip_preserves_timings_and_language(tmp_path):
    path = tmp_path / "transcript.jsonl"
    original = [seg("สวัสดี", 1.5, 2.5, lang="th", chunk=3)]
    with TranscriptWriter(path) as w:
        w.write(original)

    loaded = read_transcript(path)
    assert loaded[0].text == "สวัสดี"
    assert loaded[0].lang == "th"
    assert loaded[0].start == 1.5
    assert loaded[0].chunk == 3


def test_torn_final_line_is_skipped(tmp_path):
    """A hard crash mid-write must not make the whole transcript unreadable."""
    path = tmp_path / "transcript.jsonl"
    with TranscriptWriter(path) as w:
        w.write([seg("complete line one"), seg("complete line two")])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"start": 9.0, "end": 10.0, "te')  # torn write

    loaded = read_transcript(path)
    assert len(loaded) == 2
    assert loaded[-1].text == "complete line two"


def test_missing_transcript_returns_empty(tmp_path):
    assert read_transcript(tmp_path / "nope.jsonl") == []


def test_thai_is_not_mangled_by_ascii_escaping(tmp_path):
    path = tmp_path / "transcript.jsonl"
    with TranscriptWriter(path) as w:
        w.write([seg("ทดสอบภาษาไทย", lang="th")])
    assert "ทดสอบภาษาไทย" in path.read_text(encoding="utf-8")


# ----------------------------------------------------------- language policy
class _FakeTranscriber:
    """Exercises language logic without loading a Whisper model."""

    def __init__(self, forced=None):
        from collections import Counter

        self.forced_language = forced
        self._detected = Counter()
        self._locked_language = forced

    language = Transcriber.language
    _update_language = Transcriber._update_language


def test_language_is_reported_before_lock_threshold():
    """Regression: a short lecture ending before the lock must still have a language.

    Segments written with lang=None would break prompt selection, which is
    keyed by language, silently producing English notes for a Thai lecture.
    """
    t = _FakeTranscriber()
    t._update_language("th")
    assert t.language == "th", "language must be known after the very first chunk"


def test_language_locks_to_majority():
    t = _FakeTranscriber()
    for detected in ("th", "th", "en"):
        t._update_language(detected)
    assert t.language == "th"


def test_a_single_bad_chunk_cannot_flip_a_locked_language():
    t = _FakeTranscriber()
    for _ in range(3):
        t._update_language("th")
    t._update_language("en")  # one noisy chunk
    assert t.language == "th"


def test_forced_language_overrides_detection():
    t = _FakeTranscriber(forced="th")
    t._update_language("en")
    assert t.language == "th"


def test_english_only_model_is_rejected():
    """small.en cannot transcribe Thai; the requirement is English AND Thai."""
    import pytest

    with pytest.raises(ValueError, match="English-only"):
        Transcriber("small.en")
