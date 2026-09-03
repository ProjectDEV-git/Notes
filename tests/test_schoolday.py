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

    def fake_map(window, language, model, grounding_source=None):
        mapped.append(window.text)
        return ([f"mapped {window.text[:12]}"], [])

    monkeypatch.setattr(S, "map_window", fake_map)
    monkeypatch.setattr(S, "reduce_points", lambda pts, lang, model: "\n".join(f"- {p}" for p in pts))

    segments = _segments(30)
    S.summarize_segments(
        segments, model="stub",
        premapped=(["already known point about photosynthesis"], [], 30),
    )
    assert mapped == [], "re-mapped work the live pass had already done"


def test_the_tail_after_the_last_live_pass_is_still_mapped(monkeypatch):
    """Whatever the live thread never reached must not be silently dropped."""
    mapped: list[str] = []

    def fake_map(window, language, model, grounding_source=None):
        mapped.append(window.text)
        return (["tail point"], [])

    monkeypatch.setattr(S, "map_window", fake_map)
    monkeypatch.setattr(S, "reduce_points", lambda pts, lang, model: "\n".join(f"- {p}" for p in pts))

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
    monkeypatch.setattr(S, "reduce_points", lambda pts, lang, model: "- x")

    notes = S.summarize_segments(
        _segments(5), model="stub", premapped=(["known"], [], 9999),
    )
    assert notes.key_points  # did not crash, did not lose the notes


def test_behaviour_is_unchanged_without_premapped_work(monkeypatch):
    """The plain path (no live notes) must map every window as before."""
    mapped: list[str] = []

    def fake_map(window, language, model, grounding_source=None):
        mapped.append(window.text)
        return (["p"], [])

    monkeypatch.setattr(S, "map_window", fake_map)
    monkeypatch.setattr(S, "reduce_points", lambda pts, lang, model: "- p")

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
