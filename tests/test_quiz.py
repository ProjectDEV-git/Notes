"""Practice questions and flashcards from a class's notes."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from notetaker import cli, config, languages, menu, quiz, shortcuts, store, summarize

EN = languages.get("en")

NOTES = f"""# Physics

*2026-10-29 · 3m · English*

{EN.heading_for_key_ideas("school")}

### Lifting
- Lifting 15 kg by 1 m stores 150 J of potential energy ([0:10](https://youtu.be/a?t=7s))
- A wrecking ball turns potential energy into kinetic energy (0:47)
- Energy is conserved

{EN.heading_for_terms("school")}
- **Kinetic energy** — the energy something has because it is moving
- Potential energy: stored energy, for example from height

{EN.heading_for_actions("school")}
- Read chapter 4 before Friday
"""

QUESTIONS = {
    "Lifting 15 kg by 1 m stores 150 J of potential energy":
        "How much potential energy is stored by lifting 15 kg by 1 m?",
    "A wrecking ball turns potential energy into kinetic energy":
        "What energy change does a wrecking ball use?",
    "Energy is conserved": "Energy is conserved",  # gives the answer away
}


def fake_chat(questions=QUESTIONS):
    calls = []

    def chat(prompt, model=None, max_tokens=None):
        calls.append(prompt)
        out = []
        for raw in prompt.splitlines():
            match = re.match(r"^(\d+)\| (.*)$", raw)
            if match and match.group(2) in questions:
                out.append(f"{match.group(1)}| {questions[match.group(2)]}")
        return "\n".join(out)

    chat.calls = calls
    return chat


def test_notes_are_split_into_ideas_words_and_homework():
    found = quiz.sections(NOTES)
    assert [t for t, _ in found["ideas"]] == list(QUESTIONS)
    assert found["ideas"][0][1] == "0:10" and found["ideas"][1][1] == "0:47"
    assert len(found["words"]) == 2
    assert found["admin"] == [("Read chapter 4 before Friday", "")]


def test_words_to_know_become_flashcards_without_a_model():
    cards = quiz.build(NOTES, "en", use_model=False)
    assert [(c.question, c.answer, c.kind) for c in cards] == [
        ("Kinetic energy", "the energy something has because it is moving", "word"),
        ("Potential energy", "stored energy, for example from height", "word"),
    ]


def test_answers_are_the_students_own_notes():
    """The model writes only questions; it cannot invent an answer."""
    cards = [c for c in quiz.build(NOTES, "en", chat=fake_chat()) if c.kind == "idea"]
    assert [c.answer for c in cards] == [
        "Lifting 15 kg by 1 m stores 150 J of potential energy",
        "A wrecking ball turns potential energy into kinetic energy",
    ]
    assert cards[0].question.startswith("How much potential energy")
    assert cards[0].time == "0:10"


def test_homework_is_never_asked_about():
    chat = fake_chat()
    quiz.build(NOTES, "en", chat=chat)
    assert "chapter 4" not in "\n".join(chat.calls)


@pytest.mark.parametrize("question, ok", [
    ("What does a wrecking ball convert?", True),
    ("A wrecking ball turns potential energy into kinetic energy?", False),  # the answer
    ("Who won the war?", False),  # nothing to do with the note
    ("Why?", False),
    ("", False),
])
def test_bad_questions_are_dropped(question, ok):
    answer = "A wrecking ball turns potential energy into kinetic energy"
    assert quiz.good_question(question, answer) is ok


def test_thai_notes_get_thai_questions():
    thai = NOTES.replace("Energy is conserved", "พลังงานไม่สูญหาย")
    chat = fake_chat()
    quiz.build(thai, "en", chat=chat)
    assert "Thai (ภาษาไทย)" in chat.calls[0], "notes may be translated from English"


def test_anki_file_is_ready_to_import():
    cards = [quiz.Card("What\tis it?", "a\nthing", "idea"), quiz.Card("Osmosis", "water", "word")]
    text = quiz.to_anki(cards, "Biology week 2")
    lines = text.splitlines()
    assert lines[:3] == ["#separator:tab", "#html:false", "#deck:Biology week 2"]
    assert lines[4] == "What is it?\ta thing\tnotetaker_idea"
    assert lines[5] == "Osmosis\twater\tnotetaker_word"


# ------------------------------------------------------------ the command
@pytest.fixture()
def db(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "sessions").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", data / "sessions")
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")
    monkeypatch.delenv("NOTES_CALLER_DIR", raising=False)


def _with_notes(title="Physics"):
    session = store.create_session(title, config.SOURCE_MIC)
    store.write_notes(session, NOTES)
    return session


def _ollama(chat):
    return patch.object(summarize, "ollama_available", return_value=True), \
        patch.object(summarize, "chat", chat)


def test_listing_every_question(db, capsys):
    _with_notes()
    up, chat = _ollama(fake_chat())
    with up, chat:
        assert cli.main(["quiz", "--list"]) == 0
    out = capsys.readouterr().out
    assert "How much potential energy" in out and "(0:10)" in out
    assert "Kinetic energy" in out


def test_questions_are_saved_so_next_time_is_instant(db):
    session = _with_notes()
    chat = fake_chat()
    up, patched = _ollama(chat)
    with up, patched:
        cli.main(["quiz", "--list"])
        cli.main(["quiz", "--list"])
        assert len(chat.calls) == 1
        cli.main(["quiz", "--list", "--new"])
        assert len(chat.calls) == 2
    assert (session.directory / quiz.QUIZ_NAME).exists()


def test_without_ollama_the_word_cards_still_work(db, capsys):
    session = _with_notes()
    with patch.object(summarize, "ollama_available", return_value=False):
        assert cli.main(["quiz", "--list"]) == 0
    out = capsys.readouterr().out
    assert "Kinetic energy" in out and "only the 'words to know'" in out
    assert not (session.directory / quiz.QUIZ_NAME).exists(), "do not cache a partial quiz"


def test_anki_export_lands_where_the_student_is(db, tmp_path, monkeypatch, capsys):
    session = _with_notes()
    monkeypatch.setenv("NOTES_CALLER_DIR", str(tmp_path))
    up, chat = _ollama(fake_chat())
    with up, chat:
        assert cli.main(["quiz", "physics", "--anki"]) == 0
    saved = tmp_path / f"{session.id}-flashcards.txt"
    assert saved.read_text().startswith("#separator:tab")
    assert "File > Import" in capsys.readouterr().out


def test_practising_one_question_at_a_time(db, capsys):
    _with_notes()
    up, chat = _ollama(fake_chat())
    # 4 cards: reveal + "y", reveal + "n", then quit.
    with up, chat, patch("sys.stdin.isatty", return_value=True), \
            patch("builtins.input", side_effect=["", "y", "", "n", "q"]):
        assert cli.main(["quiz", "--in-order"]) == 0
    out = capsys.readouterr().out
    assert "1/4" in out and "You got 1 of 2" in out and "notes show" in out


def test_no_notes_yet_is_explained(db, capsys):
    store.create_session("Physics", config.SOURCE_MIC)
    assert cli.main(["quiz"]) == 1
    assert "notes catchup" in capsys.readouterr().out


def test_quiz_is_in_the_menu_and_the_launchers():
    by_key = {o.key: o for o in menu.options()}
    assert "test" in by_key["11"].label.lower()
    for word in ("quiz", "practise", "revise", "flashcards"):
        assert shortcuts.translate([word]) == ["quiz"]
