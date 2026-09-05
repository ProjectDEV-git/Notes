"""Notes written for a secondary school student, not a postgraduate.

The facts must be identical at either level. What changes is the wording, the
fact that new vocabulary gets defined, and that the admin section is called
homework. The safety guarantees must survive the change: an invented deadline
is worse than no deadline, and that is enforced by heading name, so a new
heading that the stripper does not recognise would quietly reintroduce the bug.
"""

from __future__ import annotations

import pytest

from notetaker import cli, config, languages, summarize as S
from notetaker.asr import Segment


# ------------------------------------------------------------------- config
def test_school_is_the_default_audience():
    assert config.NOTES_LEVEL == "school"


def test_university_level_is_still_available():
    assert "university" in config.NOTES_LEVELS


def test_level_can_be_chosen_per_run():
    for command in (["record"], ["summarize", "x"], ["catchup"]):
        args = cli.build_parser().parse_args(command + ["--level", "university"])
        assert args.level == "university"


# ------------------------------------------------------------------ prompts
def test_school_prompts_differ_from_university_ones():
    english = languages.get("en")
    assert english.prompts("school")["map"] != english.prompts("university")["map"]


def test_school_prompt_does_not_address_a_university_student():
    prompt = languages.get("en").prompts("school")["map"].lower()
    assert "university" not in prompt
    assert "school" in prompt


def test_school_prompt_asks_for_new_words_to_be_explained():
    """A 14-year-old meets the vocabulary for the first time."""
    prompt = languages.get("en").prompts("school")["map"].lower()
    assert "means" in prompt or "simple" in prompt


def test_school_prompt_still_demands_exact_numbers():
    """Simpler wording must not mean vaguer facts."""
    prompt = languages.get("en").prompts("school")["map"]
    assert "numbers" in prompt and "units" in prompt


def test_school_prompt_still_forbids_invention():
    reduce_prompt = languages.get("en").prompts("school")["reduce"].lower()
    assert "never make anything up" in reduce_prompt


def test_thai_school_prompts_are_written_in_thai():
    """A model follows instructions best in the language it must answer in."""
    prompt = languages.get("th").prompts("school")["map"]
    assert any("\u0e00" <= ch <= "\u0e7f" for ch in prompt)


def test_every_prompt_keeps_the_transcript_placeholder():
    for code in ("en", "th"):
        for level in config.NOTES_LEVELS:
            for key, prompt in languages.get(code).prompts(level).items():
                assert "{text}" in prompt, f"{code}/{level}/{key} lost its placeholder"


def test_university_prompts_are_unchanged():
    """Existing behaviour must still be available exactly as it was."""
    english = languages.get("en")
    assert english.prompts("university")["map"] == english.map_prompt
    assert english.prompts("university")["reduce"] == english.reduce_prompt


# ------------------------------------------------- packs written before this
def test_a_pack_without_school_prompts_still_works():
    """Every language pack a user has already written must keep working."""
    pack = languages.Language.from_dict("xx", {
        "name": "Test",
        "prompts": {"map": "map {text}", "reduce": "reduce {text}"},
    })
    assert pack.prompts("school")["map"] == "map {text}"
    assert pack.prompts("university")["map"] == "map {text}"


def test_a_pack_without_school_prompts_keeps_its_own_heading():
    pack = languages.Language.from_dict("xx", {
        "name": "Test",
        "headings": {"action_items": "## Todo"},
        "prompts": {"map": "m {text}", "reduce": "r {text}"},
    })
    assert pack.heading_for_actions("school") == "## Todo"


def test_a_school_prompt_without_the_placeholder_is_rejected():
    """Silently producing empty notes is worse than refusing to load."""
    with pytest.raises(ValueError):
        languages.Language.from_dict("xx", {
            "name": "Test",
            "prompts": {
                "map": "m {text}", "reduce": "r {text}",
                "map_school": "no placeholder here",
            },
        })


def test_school_prompts_survive_a_save_and_reload():
    data = languages.BUILTIN["en"].to_dict()
    restored = languages.Language.from_dict("en", data)
    assert restored.prompts("school")["map"] == languages.BUILTIN["en"].map_prompt_school
    assert restored.heading_for_actions("school") == "## Homework & reminders"


# --------------------------------------------------- invented homework guard
def test_the_homework_heading_is_recognised_as_an_action_section():
    """Missing from this list means invented homework stops being stripped."""
    headings = languages.action_headings()
    assert "homework & reminders" in headings
    assert "การบ้านและสิ่งที่ต้องทำ" in headings


def test_both_registers_are_recognised():
    headings = languages.action_headings()
    assert "action items" in headings  # university wording still handled


def test_invented_homework_is_removed_when_nobody_set_any():
    """The whole point: a made-up deadline is worse than no deadline."""
    markdown = (
        "## What we learned\n- plants use sunlight\n\n"
        "## Homework & reminders\n- finish page 40 for Friday\n"
    )
    cleaned = S.drop_unbacked_actions(markdown, admin_points=[])
    assert "page 40" not in cleaned
    assert "plants use sunlight" in cleaned


def test_real_homework_is_kept():
    markdown = (
        "## What we learned\n- plants use sunlight\n\n"
        "## Homework & reminders\n- finish page 40 for Friday\n"
    )
    kept = S.drop_unbacked_actions(markdown, admin_points=["finish page 40 for Friday"])
    assert "page 40" in kept


def test_thai_invented_homework_is_removed():
    markdown = (
        "## สิ่งที่เรียนวันนี้\n- พืชใช้แสงแดด\n\n"
        "## การบ้านและสิ่งที่ต้องทำ\n- ทำแบบฝึกหัดหน้า 40\n"
    )
    cleaned = S.drop_unbacked_actions(markdown, admin_points=[])
    assert "40" not in cleaned


# -------------------------------------------------------- level is respected
def test_the_chosen_level_reaches_the_model(monkeypatch):
    seen: list[str] = []

    def fake_chat(prompt, **kwargs):
        seen.append(prompt)
        return "- a point about photosynthesis"

    monkeypatch.setattr(S, "chat", fake_chat)
    window = S.Window(0.0, 60.0, "plants use sunlight to make food")

    S.map_window(window, "en", "stub", level="school")
    assert "school student" in seen[-1]

    S.map_window(window, "en", "stub", level="university")
    assert "university lecture" in seen[-1]


def test_the_default_level_is_school_when_none_is_given(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(S, "chat", lambda prompt, **k: seen.append(prompt) or "- p")

    S.map_window(S.Window(0.0, 60.0, "text"), "en", "stub")
    assert "school student" in seen[-1]


def test_live_notes_use_the_same_level_as_the_final_notes():
    """Live points become the final notes, so a mismatch would show up mid-page."""
    from notetaker.audio import AudioSource
    from notetaker.pipeline import RecordingPipeline
    from notetaker import store

    source = AudioSource(name="stub", description="stub", kind=config.SOURCE_MIC)
    session = store.Session(
        id="x", title="t", started_at="", ended_at=None,
        source_kind=config.SOURCE_MIC, source_name=None,
        language="en", duration=0.0, has_notes=False,
    )
    pipe = RecordingPipeline(source=source, session=session, level="university")
    assert pipe.level == "university"


# ------------------------------------------------- narration is not a note
def test_narration_bullets_are_dropped():
    """'The teacher talked about X' says a topic came up, not what was taught.

    Every prompt forbids this and a small model produces it anyway, so it is
    enforced in code. Observed in a real run against llama3.2:3b.
    """
    text = (
        "- The teacher talked about Apostles, a series of books.\n"
        "- The teacher also mentioned the Oxford Bill.\n"
        "- First cause refers to an explanation for how something began.\n"
    )
    kept = S.parse_bullets(text)
    assert kept == ["First cause refers to an explanation for how something began."]


@pytest.mark.parametrize("bullet", [
    "The teacher talked about energy",
    "The lecturer discussed momentum",
    "The professor explained the formula",
    "The speaker mentioned a deadline",
    "the class covered photosynthesis",
    "The teacher also mentioned the Oxford Bill",
])
def test_every_narration_phrasing_is_caught(bullet):
    assert S.parse_bullets(f"- {bullet}\n") == []


@pytest.mark.parametrize("bullet", [
    "Teachers are paid monthly under the new contract",
    "The lesson plan for photosynthesis has three stages",
    "Speakers convert electrical energy into sound",
    "Class size affects how much feedback a student gets",
])
def test_real_content_about_teaching_is_not_dropped(bullet):
    """The filter must not eat genuine subject matter that names a teacher."""
    assert S.parse_bullets(f"- {bullet}\n") == [bullet]


def test_narration_is_also_stripped_from_final_notes():
    """The reduce stage can reintroduce narration the map stage never emitted."""
    markdown = (
        "## What we learned\n"
        "- The teacher talked about photosynthesis\n"
        "- Plants use sunlight to make food\n"
    )
    cleaned = S.apply_grounding(
        markdown,
        ["The teacher talked about photosynthesis", "Plants use sunlight to make food"],
    )
    assert "teacher talked" not in cleaned
    assert "Plants use sunlight to make food" in cleaned


def test_reduce_is_told_to_drop_bad_definitions():
    """A wrong definition is worse than a missing one, in both languages.

    Transcription errors reach the notes as invented vocabulary ("Neers — book
    with a hard cover" came from a mis-heard word in a real run). The map stage
    guard alone does not catch these, so reduce must prune them too.
    """
    for code in ("en", "th"):
        reduce_prompt = languages.get(code).prompts("school")["reduce"]
        assert "{text}" in reduce_prompt
    assert "worse than a missing one" in languages.get("en").prompts("school")["reduce"]
    assert "แย่กว่าไม่มีนิยาม" in languages.get("th").prompts("school")["reduce"]


# ------------------------------------------------ narration in Thai as well
@pytest.mark.parametrize("bullet", [
    "ผู้สอนพูดถึงเรื่องพลังงาน",        # the instructor talked about energy
    "อาจารย์อธิบายเรื่องแรงโน้มถ่วง",    # the teacher explained gravity
    "ครูบอกว่าให้อ่านหน้า 40",          # the teacher said to read page 40
    "วิทยากรกล่าวถึงการทดลอง",          # the speaker mentioned the experiment
])
def test_thai_narration_is_dropped(bullet):
    """Thai is a first-class language here, so it needs the same guard.

    The Thai prompts forbid opening a bullet with ผู้สอน or อาจารย์ in the same
    way the English ones do, and a small model ignores it in the same way.
    """
    assert S.parse_bullets(f"- {bullet}\n") == []


@pytest.mark.parametrize("bullet", [
    "พลังงานศักย์เปลี่ยนเป็นพลังงานจลน์เมื่อวัตถุตกลง",  # real physics content
    "ครูใหญ่คือผู้บริหารสูงสุดของโรงเรียน",              # 'headmaster' as subject matter
])
def test_real_thai_content_survives(bullet):
    """Thai has no word spaces, so an over-eager pattern would eat real notes."""
    assert S.parse_bullets(f"- {bullet}\n") == [bullet]


def test_narration_check_is_exposed_for_both_languages():
    assert S.is_narration("The teacher talked about energy")
    assert S.is_narration("ผู้สอนพูดถึงเรื่องพลังงาน")
    assert not S.is_narration("Plants use sunlight to make food")
    assert not S.is_narration("พลังงานศักย์เปลี่ยนเป็นพลังงานจลน์")


def test_both_built_in_languages_get_the_same_guards():
    """A guard added only in English silently leaves Thai classes broken.

    That is exactly what happened with the narration filter, which shipped
    English-only and let 'ผู้สอนพูดถึง...' straight through.
    """
    en = languages.get("en").prompts("school")
    th = languages.get("th").prompts("school")

    assert "NOT a word to define" in en["map"]
    assert "ไม่ใช่คำศัพท์ที่ต้องนิยาม" in th["map"]
    assert "worse than a missing one" in en["reduce"]
    assert "แย่กว่าไม่มีนิยาม" in th["reduce"]


def test_thai_school_headings_match_the_reduce_prompt():
    """A heading the prompt never asks for cannot be produced or stripped."""
    thai = languages.get("th")
    reduce_prompt = thai.prompts("school")["reduce"]
    assert thai.heading_for_actions("school").lstrip("#").strip() in reduce_prompt
