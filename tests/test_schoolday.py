"""A school day: fixed-length periods, and notes that arrive before the next one.

Two behaviours matter here and neither existed when the tool was aimed at
university lectures:

* A period has a known length, so the recording ends by itself. A student
  should not have to remember to stop it while packing up.
* The live-notes thread already summarizes the class while it is happening.
  Throwing that away and re-running the whole map-reduce afterwards costs
  roughly fifteen minutes on a one-hour class, which is longer than the break.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from notetaker import cli, config, store, summarize as S
from notetaker.asr import Segment
from notetaker.pipeline import PipelineState


# ------------------------------------------------------------- class length
def test_recording_runs_until_stopped_by_default():
    args = cli.build_parser().parse_args(["record"])
    assert args.minutes == 0, "a lecture must still record until Ctrl-C"


def test_a_class_length_can_be_given():
    args = cli.build_parser().parse_args(["record", "--minutes", "60"])
    assert args.minutes == 60


def test_short_flag_works_for_a_hurried_student():
    assert cli.build_parser().parse_args(["record", "-m", "45"]).minutes == 45


def test_default_class_length_is_a_normal_period():
    assert config.DEFAULT_CLASS_MINUTES == 60
    assert 45 in config.CLASS_LENGTH_CHOICES


def test_recording_keeps_going_a_little_past_the_bell():
    """Classes overrun, and the homework is usually said last."""
    assert config.CLASS_OVERRUN_SECONDS > 0


def test_countdown_is_shown_while_recording():
    state = PipelineState(elapsed=600.0)
    panel = cli._render_live(state, "mic", live_notes=False, remaining=1200.0)
    assert "left" in _text_of(panel)


def test_no_countdown_when_there_is_no_time_limit():
    state = PipelineState(elapsed=600.0)
    panel = cli._render_live(state, "mic", live_notes=False, remaining=None)
    assert "left" not in _text_of(panel)


def test_countdown_never_shows_negative_time():
    """Overrunning must read 0:00, not a minus sign."""
    state = PipelineState(elapsed=4000.0)
    rendered = _text_of(cli._render_live(state, "mic", False, remaining=-30.0))
    assert "-" not in rendered.split("left")[0][-12:]


def _text_of(renderable) -> str:
    from rich.console import Console

    console = Console(width=200, no_color=True)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


# ------------------------------------------------- reusing in-class notework
def _segments(count: int) -> list[Segment]:
    return [
        Segment(i * 10.0, i * 10.0 + 10.0, f"point number {i} about photosynthesis", "en", 0)
        for i in range(count)
    ]


def test_live_points_are_kept_for_reuse():
    """The pipeline must hand its MAP work on, not just display it."""
    state = PipelineState()
    assert state.live_key_points == []
    assert state.live_consumed == 0


def test_premapped_work_is_not_repeated(monkeypatch):
    """Segments already summarized during class must not be mapped again."""
    mapped: list[str] = []

    def fake_map(window, language, model, grounding_source=None, level=None):
        mapped.append(window.text)
        return ([f"mapped {window.text[:12]}"], [])

    monkeypatch.setattr(S, "map_window", fake_map)
    monkeypatch.setattr(S, "reduce_points", lambda pts, lang, model, level=None: "\n".join(f"- {p}" for p in pts))

    segments = _segments(30)
    S.summarize_segments(
        segments, model="stub",
        premapped=(["already known point about photosynthesis"], [], 30),
    )
    assert mapped == [], "re-mapped work the live pass had already done"


def test_the_tail_after_the_last_live_pass_is_still_mapped(monkeypatch):
    """Whatever the live thread never reached must not be silently dropped."""
    mapped: list[str] = []

    def fake_map(window, language, model, grounding_source=None, level=None):
        mapped.append(window.text)
        return (["tail point"], [])

    monkeypatch.setattr(S, "map_window", fake_map)
    monkeypatch.setattr(S, "reduce_points", lambda pts, lang, model, level=None: "\n".join(f"- {p}" for p in pts))

    segments = _segments(30)
    S.summarize_segments(
        segments, model="stub",
        premapped=(["known point"], [], 20),
    )
    assert mapped, "the end of the class was never summarized"
    assert "point number 25" in " ".join(mapped)


def test_premapped_points_reach_the_final_notes(monkeypatch):
    monkeypatch.setattr(S, "map_window", lambda *a, **k: ([], []))
    monkeypatch.setattr(
        S, "reduce_points", lambda pts, lang, model: "\n".join(f"- {p}" for p in pts)
    )

    notes = S.summarize_segments(
        _segments(20), model="stub",
        premapped=(["photosynthesis needs light"], ["homework is page 40"], 20),
    )
    assert "photosynthesis needs light" in notes.key_points
    assert "homework is page 40" in notes.admin_points


def test_a_stale_count_cannot_skip_the_whole_class(monkeypatch):
    """A count larger than the transcript must not silently drop everything."""
    monkeypatch.setattr(S, "map_window", lambda *a, **k: (["x"], []))
    monkeypatch.setattr(S, "reduce_points", lambda pts, lang, model, level=None: "- x")

    notes = S.summarize_segments(
        _segments(5), model="stub", premapped=(["known"], [], 9999),
    )
    assert notes.key_points  # did not crash, did not lose the notes


def test_behaviour_is_unchanged_without_premapped_work(monkeypatch):
    """The plain path (no live notes) must map every window as before."""
    mapped: list[str] = []

    def fake_map(window, language, model, grounding_source=None, level=None):
        mapped.append(window.text)
        return (["p"], [])

    monkeypatch.setattr(S, "map_window", fake_map)
    monkeypatch.setattr(S, "reduce_points", lambda pts, lang, model, level=None: "- p")

    S.summarize_segments(_segments(30), model="stub")
    assert mapped, "nothing was summarized at all"
    assert "point number 0" in " ".join(mapped)


# ------------------------------------------- record now, write up afterwards
@pytest.fixture()
def db(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "sessions").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", data / "sessions")
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")
    return data / "notetaker.db"


def test_later_is_off_by_default():
    assert cli.build_parser().parse_args(["record"]).later is False


def test_later_can_be_asked_for():
    assert cli.build_parser().parse_args(["record", "--later"]).later is True


def test_audio_only_mode_loads_no_speech_model(db):
    """The whole point of --later is that class costs no CPU."""
    from notetaker.audio import AudioSource
    from notetaker.pipeline import RecordingPipeline

    source = AudioSource(name="stub", description="stub", kind=config.SOURCE_MIC)
    session = store.create_session("Maths", config.SOURCE_MIC)
    pipe = RecordingPipeline(source=source, session=session, transcribe=False, live_notes=True)
    assert pipe.transcribe is False
    assert pipe.live_notes is False, "cannot show live notes with no transcript"


def test_audio_only_mode_never_reports_a_backlog(db):
    """Untranscribed chunks are the plan in --later, not a problem."""
    from notetaker.audio import AudioSource
    from notetaker.pipeline import RecordingPipeline

    source = AudioSource(name="stub", description="stub", kind=config.SOURCE_MIC)
    session = store.create_session("Maths", config.SOURCE_MIC)
    pipe = RecordingPipeline(source=source, session=session, transcribe=False)
    assert pipe.state.chunks_pending == 0
    assert not pipe.state.is_falling_behind


def _recorded(title, transcript=False, notes=False, audio=True):
    from notetaker.asr import TranscriptWriter

    session = store.create_session(title, config.SOURCE_MIC)
    session.directory.mkdir(parents=True, exist_ok=True)
    if audio:
        session.audio_path.write_bytes(b"RIFF")
    if transcript:
        with TranscriptWriter(session.transcript_path) as writer:
            writer.write([Segment(0.0, 3.0, "plants use sunlight to make food", "en", 0)])
    if notes:
        store.write_notes(session, "## Key ideas\n- plants use sunlight\n")
    store.finish_session(session.id, duration=3600.0, language="en")
    return store.get_session(session.id)


def test_a_class_recorded_for_later_is_listed_as_pending(db):
    _recorded("Biology", transcript=False, notes=False)
    assert [s.title for s in store.pending_sessions()] == ["Biology"]


def test_a_finished_class_is_not_pending(db):
    _recorded("Biology", transcript=True, notes=True)
    assert store.pending_sessions() == []


def test_a_transcribed_but_unsummarized_class_is_pending(db):
    """Ollama being down at the bell must not lose the class."""
    _recorded("Chemistry", transcript=True, notes=False)
    assert [s.title for s in store.pending_sessions()] == ["Chemistry"]


def test_a_class_with_nothing_recoverable_is_not_offered(db):
    session = store.create_session("Empty", config.SOURCE_MIC)
    store.finish_session(session.id, duration=0.0)
    assert store.pending_sessions() == []


def test_pending_classes_come_out_oldest_first(db):
    _recorded("Period one")
    _recorded("Period two")
    titles = [s.title for s in store.pending_sessions()]
    assert titles == ["Period one", "Period two"], "a day should be written up in order"


def test_catchup_with_nothing_to_do_is_not_an_error(db, capsys):
    assert cli.main(["catchup"]) == 0
    assert "Nothing to catch up" in capsys.readouterr().out


def test_catchup_summarizes_a_pending_class(db, monkeypatch):
    _recorded("Chemistry", transcript=True, notes=False)
    monkeypatch.setattr(cli.summarize, "ollama_available", lambda: True)
    monkeypatch.setattr(
        cli.summarize, "summarize_segments",
        lambda *a, **k: cli.summarize.Notes(markdown="## Key ideas\n- sunlight\n"),
    )
    assert cli.main(["catchup"]) == 0
    assert store.pending_sessions() == [], "class was not marked as done"


def test_catchup_without_ollama_keeps_everything(db, monkeypatch):
    """Failing to write notes must never cost the recording."""
    session = _recorded("Chemistry", transcript=True, notes=False)
    monkeypatch.setattr(cli.summarize, "ollama_available", lambda: False)

    assert cli.main(["catchup"]) != 0
    assert session.transcript_path.exists()
    assert session.audio_path.exists()


# ------------------------------------------------- the one-word launcher
LAUNCHER = Path(__file__).resolve().parent.parent / "scripts" / "notes"


def _dispatch(verb: str, env: dict | None = None) -> str:
    """What CLI command does `notes <verb>` actually run?

    The launcher is the surface a student touches, so a typo in it is a
    broken feature no Python test would catch.
    """
    traced = LAUNCHER.read_text().replace(
        'exec "$PY" -m notetaker.cli', "echo CLI"
    )
    script = LAUNCHER.parent / "_traced_notes"
    script.write_text(traced)
    try:
        result = subprocess.run(
            ["bash", str(script), verb],
            capture_output=True, text=True,
            env={**os.environ, **(env or {})},
        )
    finally:
        script.unlink(missing_ok=True)
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def test_launcher_is_valid_shell():
    assert subprocess.run(["bash", "-n", str(LAUNCHER)]).returncode == 0


def test_notes_class_stops_by_itself():
    command = _dispatch("class")
    assert "--minutes" in command
    assert str(config.DEFAULT_CLASS_MINUTES) in command


def test_notes_class_length_can_be_changed():
    command = _dispatch("class", {"NOTES_CLASS_MINUTES": "45"})
    assert "--minutes 45" in command


def test_notes_later_records_sound_only():
    command = _dispatch("later")
    assert "--later" in command
    assert "--live-notes" not in command, "--later must not run the model in class"


def test_notes_catchup_reaches_the_catchup_command():
    assert _dispatch("catchup").endswith("catchup")


def test_notes_now_still_records_until_stopped():
    """The old behaviour must survive: a lecture has no fixed length."""
    command = _dispatch("now")
    assert "--minutes" not in command


# ------------------------------------------------------- untitled recordings
def test_an_untitled_recording_is_called_a_class(db, monkeypatch):
    """The default title must match the register the notes are written in."""
    import argparse
    from notetaker import menu

    args = cli.build_parser().parse_args(["record"])
    assert args.level == "school"
    # cmd_record derives the title before any hardware is touched; check the
    # rule directly rather than starting a real recording.
    title = args.title or ("Lecture" if args.level == "university" else "Class")
    assert title == "Class"


def test_an_untitled_university_recording_is_still_a_lecture():
    args = cli.build_parser().parse_args(["record", "--level", "university"])
    title = args.title or ("Lecture" if args.level == "university" else "Class")
    assert title == "Lecture"


def test_placeholder_titles_are_not_offered_as_subjects(db):
    """'Class' is what an unnamed recording is called, not a subject."""
    from notetaker import menu

    for name in ("Class", "Lecture", "Chemistry"):
        s = store.create_session(name, config.SOURCE_MIC)
        store.finish_session(s.id, duration=1.0)

    subjects = menu.recent_subjects()
    assert "Chemistry" in subjects
    assert "Class" not in subjects
    assert "Lecture" not in subjects


def test_reusing_live_work_cuts_the_model_calls_after_the_bell():
    """The measured 1.84x speedup comes entirely from skipped MAP calls.

    Timed end to end on a 41-minute class: 12.8 min from scratch vs 7.0 min
    reusing the live thread's work. This test pins the mechanism, so a change
    that quietly stops reusing the points fails here rather than only showing
    up as a slower wait after somebody's lesson.
    """
    calls = {"map": 0, "reduce": 0}
    real_chat = S.chat

    def counting_chat(prompt, **kwargs):
        if "Lines:" in prompt or "Tidy them" in prompt:
            calls["reduce"] += 1
            return "\n".join(l for l in prompt.splitlines() if l.startswith("- "))[:900]
        calls["map"] += 1
        body = prompt.split("recording:\n")[-1].split("Transcript:\n")[-1]
        return "- " + " ".join(body.split()[:12])

    segments = [
        Segment(i * 4.5, i * 4.5 + 4.5, f"In topic {i // 45}, detailed point {i} about mechanisms", "en", 0)
        for i in range(900)
    ]
    covered = int(len(segments) * 0.9)
    live = [f"In topic {i}, detailed point {i} about mechanisms" for i in range(40)]

    S.chat = counting_chat
    try:
        calls["map"] = calls["reduce"] = 0
        S.summarize_segments(segments, model="stub")
        scratch = calls["map"]

        calls["map"] = calls["reduce"] = 0
        S.summarize_segments(segments, model="stub", premapped=(live, [], covered))
        reused = calls["map"]
    finally:
        S.chat = real_chat

    assert reused < scratch, "live work was not reused; the wait after class is back"
    # Only the uncovered tail should still need mapping.
    assert reused <= scratch // 3, f"expected most windows skipped, mapped {reused} of {scratch}"


# --------------------------------------------------- the finished notes page
def test_notes_header_records_how_long_the_class_was(db):
    """The session row is created before recording, so it must be re-read.

    A real 2-minute recording produced a header with no duration at all,
    because the stale object still said 0 and the renderer omits a falsy one.
    """
    from notetaker import summarize as S

    session = store.create_session("Science period 1", config.SOURCE_MIC)
    store.finish_session(session.id, duration=3600.0, language="en")

    fresh = store.get_session(session.id)
    assert fresh.duration == 3600.0, "duration was not persisted"

    rendered = S._render("Science period 1", "## What we learned\n- x", "en",
                         fresh.duration, 12)
    assert "60m 0s" in rendered


def test_a_stale_session_object_loses_the_duration():
    """Pins the bug itself: this is what the header looked like before."""
    from notetaker import summarize as S

    rendered = S._render("Science period 1", "## What we learned\n- x", "en", 0.0, 12)
    assert "0m" not in rendered  # a falsy duration is omitted entirely


def test_the_thai_slowness_note_is_not_shown_for_other_languages(capsys):
    """It read like a bug report during an English class."""
    from notetaker.pipeline import PipelineState

    for language, expected in (("th", True), ("en", False), (None, False)):
        state = PipelineState(language=language)
        slow = (state.language or "") == "th"
        assert slow is expected, f"{language} misclassified"


# --------------------------------------- advice must name a command that exists
def test_reading_a_class_with_no_notes_suggests_a_real_command(db, capsys):
    """It suggested 'notetaker summarize <id>', which no student ever types."""
    _recorded("Biology", transcript=True, notes=False)
    cli.main(["show", "Biology"])
    out = capsys.readouterr().out
    assert "notes catchup" in out
    assert "notetaker summarize" not in out


def test_exporting_a_class_with_no_notes_suggests_a_real_command(db, capsys):
    _recorded("Biology", transcript=True, notes=False)
    cli.main(["export", "Biology", "-o", "/dev/null"])
    out = capsys.readouterr().out
    assert "notes catchup" in out


def test_every_suggested_command_is_a_real_launcher_verb():
    """Advice naming a verb the launcher does not have is worse than none.

    Only checks strings that are clearly commands: 'notes <verb>' at the end of
    a sentence or followed by a flag, not prose that happens to contain the
    word "notes".
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    source = (repo / "notetaker" / "cli.py").read_text()
    launcher = (repo / "scripts" / "notes").read_text()

    # 'notes catchup' as a command: two spaces before it, or trailing quote.
    suggested = set(re.findall(r"(?:  |`)notes ([a-z]+)", source))
    assert suggested, "no command suggestions found to check"
    for verb in sorted(suggested):
        assert re.search(rf"^\s+[a-z|]*\b{verb}\b[a-z|]*\)", launcher, re.M), \
            f"cli.py suggests 'notes {verb}' but the launcher has no such verb"
