"""Notes in a different language from the one spoken."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from notetaker import cli, config, languages, menu, store, summarize, translate
from notetaker.asr import Segment

EN = languages.get("en")
TH = languages.get("th")

NOTES = f"""# Physics week 4

*2026-10-29 · 3m 23s · English · 45 segments*

{EN.heading_for_key_ideas("school")}
- Energy is conserved ([0:40](https://youtu.be/abc?t=37s))
- Lifting 15 kg by 1 m stores 150 J (0:10)

### Wrecking balls
- A wrecking ball turns potential energy into kinetic energy

{EN.heading_for_actions("school")}
- Read chapter 4
"""

THAI = {
    "Energy is conserved": "พลังงานไม่สูญหาย",
    "Lifting 15 kg by 1 m stores 150 J": "การยก 15 kg ขึ้น 1 m เก็บพลังงาน 150 J",
    "Wrecking balls": "ลูกตุ้มทำลาย",
    "A wrecking ball turns potential energy into kinetic energy": "ลูกตุ้มเปลี่ยนพลังงานศักย์เป็นพลังงานจลน์",
    "Read chapter 4": "อ่านบทที่ 4",
}


def fake_chat(answers=THAI, drop=(), mangle=()):
    calls = []

    def chat(prompt, model=None, max_tokens=None):
        calls.append(prompt)
        lines = []
        for raw in prompt.splitlines():
            match = re.match(r"^(\d+)\| (.*)$", raw)
            if not match or match.group(2) in drop:
                continue
            text = match.group(2)
            reply = answers.get(text, text)
            if text in mangle:
                reply = re.sub(r"\d+", "9", reply)
            lines.append(f"{match.group(1)}| {reply}")
        return "\n".join(lines)

    chat.calls = calls
    return chat


def test_notes_are_translated_line_by_line_keeping_the_layout():
    out = translate.translate_notes(NOTES, "en", "th", chat=fake_chat()).splitlines()
    assert out[0] == "# Physics week 4", "the title is the class's name"
    assert out[2] == f"*2026-10-29 · 3m 23s · English → {TH.name} · 45 segments*"
    assert out[4] == TH.heading_for_key_ideas("school")
    assert out[5] == "- พลังงานไม่สูญหาย ([0:40](https://youtu.be/abc?t=37s))", "links survive"
    assert out[6] == "- การยก 15 kg ขึ้น 1 m เก็บพลังงาน 150 J (0:10)"
    assert out[8] == "### ลูกตุ้มทำลาย"
    assert out[11] == TH.heading_for_actions("school")
    assert out[12] == "- อ่านบทที่ 4"


def test_the_model_is_told_the_language_in_english_too():
    chat = fake_chat()
    translate.translate_notes(NOTES, "en", "th", chat=chat)
    assert f"into Thai ({TH.name})" in chat.calls[0]


def test_only_words_are_sent_to_the_model():
    """A small model cannot break a link or a heading it never sees."""
    chat = fake_chat()
    translate.translate_notes(NOTES, "en", "th", chat=chat)
    sent = "\n".join(chat.calls)
    assert "youtu.be" not in sent and "(0:10)" not in sent
    assert EN.heading_for_key_ideas("school") not in sent


def test_a_line_that_loses_its_numbers_stays_in_english():
    """A mistranslated '150 J' is worse than an English line."""
    out = translate.translate_notes(
        NOTES, "en", "th", chat=fake_chat(mangle={"Lifting 15 kg by 1 m stores 150 J"}))
    assert "- Lifting 15 kg by 1 m stores 150 J (0:10)" in out


def test_a_line_the_model_skipped_stays_as_it_was():
    out = translate.translate_notes(NOTES, "en", "th",
                                    chat=fake_chat(drop={"Read chapter 4"}))
    assert "- Read chapter 4" in out
    assert "พลังงานไม่สูญหาย" in out


def test_long_notes_are_sent_in_batches(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATE_BATCH_LINES", 2)
    chat = fake_chat()
    translate.translate_notes(NOTES, "en", "th", chat=chat)
    assert len(chat.calls) == 3  # five lines of words, two per call


def test_same_language_is_left_alone():
    chat = fake_chat()
    assert translate.translate_notes(NOTES, "en", "en", chat=chat) == NOTES
    assert chat.calls == []


# ------------------------------------------------------------ the command
@pytest.fixture()
def db(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "sessions").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", data / "sessions")
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")


def _session():
    session = store.create_session("Physics", config.SOURCE_MIC)
    store.write_transcript(session, [Segment(0, 5, "energy is conserved", "en", 0)])
    return session


def _run(session, notes_lang, chat=None):
    fake = summarize.Notes(markdown=NOTES, language="en")
    with patch.object(summarize, "ollama_available", return_value=True), \
            patch.object(summarize, "summarize_segments", return_value=fake), \
            patch.object(summarize, "chat", chat or fake_chat()):
        assert cli._summarize_session(session, model="stub", notes_lang=notes_lang) == 0
    return session.notes_path.read_text()


def test_notes_lang_gives_thai_notes_from_an_english_class(db):
    assert "พลังงานไม่สูญหาย" in _run(_session(), "th")


def test_without_it_the_notes_stay_in_the_spoken_language(db, monkeypatch):
    monkeypatch.setattr(config, "NOTES_LANGUAGE", None)
    assert _run(_session(), None) == NOTES


def test_a_default_notes_language_can_be_set_once(db, monkeypatch):
    monkeypatch.setattr(config, "NOTES_LANGUAGE", "th")
    assert "พลังงานไม่สูญหาย" in _run(_session(), None)


def test_a_failed_translation_keeps_the_notes(db, capsys):
    def broken(*a, **k):
        raise summarize.SummarizerError("Ollama timed out")

    assert _run(_session(), "th", chat=broken) == NOTES
    assert "keeping them in English" in capsys.readouterr().out


@pytest.mark.parametrize("command", [
    ["record"], ["video", "x.mp4"], ["catchup"], ["summarize", "physics"],
])
def test_every_note_writing_command_takes_notes_lang(command):
    args = cli.build_parser().parse_args([*command, "--notes-lang", "th"])
    assert args.notes_lang == "th"


def test_the_menu_asks_and_defaults_to_the_spoken_language():
    with patch("builtins.input", side_effect=[""]):
        assert menu.pick_notes_language() is None
    with patch("builtins.input", side_effect=["3"]):
        assert menu.pick_notes_language() == list(config.supported_languages())[1]
    with patch("builtins.input", side_effect=["banana"]):
        assert menu.pick_notes_language() is None


def test_the_video_menu_passes_the_notes_language():
    th = list(config.supported_languages()).index("th") + 2
    with patch("builtins.input", side_effect=["https://youtu.be/abc", str(th)]), \
            patch("notetaker.cli.main", return_value=0) as run:
        menu._from_video()
    run.assert_called_once_with(["video", "https://youtu.be/abc", "--notes-lang", "th"])
