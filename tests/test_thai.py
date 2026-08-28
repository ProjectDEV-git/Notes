"""Thai language verification.

The requirement is English *and* Thai, and Thai is the harder case: it has no
spaces between words, and it is where the pipeline most easily degrades
silently (English notes for a Thai lecture, or dedupe failing to match).

Slow tests that load a Whisper model are marked and can be deselected with
    pytest -m "not slow"
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notetaker import config, summarize as S
from notetaker.asr import Transcriber, dedupe_overlap, normalize, read_transcript

TH_AUDIO = Path("tests/fixtures/lecture_th_20s.wav")
TH_TRANSCRIPT = Path("tests/fixtures/lecture_th.jsonl")

needs_ollama = pytest.mark.skipif(not S.ollama_available(), reason="Ollama not running")


# ------------------------------------------------------------------- fixture
def test_thai_transcript_fixture_exists():
    assert TH_TRANSCRIPT.exists(), "Thai fixture missing; regenerate it"


def test_thai_fixture_is_tagged_thai():
    segments = read_transcript(TH_TRANSCRIPT)
    assert segments
    assert all(s.lang == "th" for s in segments)


def test_thai_fixture_contains_thai_script():
    text = " ".join(s.text for s in read_transcript(TH_TRANSCRIPT))
    assert any("\u0e00" <= ch <= "\u0e7f" for ch in text)


# ---------------------------------------------------------------- text logic
def test_thai_normalization_removes_spacing_differences():
    """Thai is written without word spaces, so spacing must not affect matching."""
    assert normalize("การสังเคราะห์แสง") == normalize("การสังเคราะห์ แสง")


def test_thai_dedupe_on_real_transcript_text():
    """Replaying a real Thai segment must not duplicate it."""
    segments = read_transcript(TH_TRANSCRIPT)
    if len(segments) < 2:
        pytest.skip("fixture too short")

    previous = segments[:1]
    incoming = [segments[0], segments[1]]  # chunk seam repeats the first segment
    result = dedupe_overlap(previous, incoming)
    assert len(result) == 1
    assert result[0].text == segments[1].text


def test_thai_prompts_are_actually_thai():
    prompts = config.prompts_for("th")
    assert any("\u0e00" <= ch <= "\u0e7f" for ch in prompts["map"])
    assert any("\u0e00" <= ch <= "\u0e7f" for ch in prompts["reduce"])


def test_thai_and_english_prompts_differ():
    assert config.prompts_for("th")["map"] != config.prompts_for("en")["map"]


# ------------------------------------------------------------- transcription
@pytest.mark.slow
def test_thai_audio_is_detected_as_thai():
    """Guard against a Thai lecture being transcribed as English."""
    if not TH_AUDIO.exists():
        pytest.skip("Thai audio fixture missing")

    transcriber = Transcriber(config.ASR_MODEL)
    segments = transcriber.transcribe_file(TH_AUDIO)

    assert transcriber.language == "th"
    assert segments
    text = " ".join(s.text for s in segments)
    assert any("\u0e00" <= ch <= "\u0e7f" for ch in text), "no Thai script in output"


# -------------------------------------------------------------- summarization
@needs_ollama
def test_thai_transcript_produces_thai_notes():
    """A Thai lecture must not silently produce English notes."""
    segments = read_transcript(TH_TRANSCRIPT)
    if not segments:
        pytest.skip("fixture missing")

    notes = S.summarize_segments(segments, title="การสังเคราะห์แสง", duration=45.6)

    assert notes.language == "th"
    assert any("\u0e00" <= ch <= "\u0e7f" for ch in notes.markdown), "notes are not Thai"
    assert "แนวคิดสำคัญ" in notes.markdown, "Thai heading missing"


@needs_ollama
def test_thai_notes_are_shorter_than_the_transcript():
    segments = read_transcript(TH_TRANSCRIPT)
    if not segments:
        pytest.skip("fixture missing")

    notes = S.summarize_segments(segments, title="บทเรียน", duration=45.6)
    transcript = " ".join(s.text for s in segments)
    assert S.compression_ratio(transcript, notes.markdown) < 1.0
