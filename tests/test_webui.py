"""The NoteTaker window (`notes ui`).

The window runs the ordinary CLI in a child process, so these tests cover
what is new: turning a click into the right command, the guards that keep
other web pages from starting a recording, what the page is told about each
class, and stopping a recording from the window the same safe way as Ctrl-C.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
from urllib.parse import quote

import pytest

from notetaker import cli, config, store, webui
from notetaker.asr import Segment, TranscriptWriter


@pytest.fixture()
def db(tmp_path, monkeypatch):
    # Where a child process looks too: $XDG_DATA_HOME/notetaker
    data = tmp_path / "notetaker"
    sessions = data / "sessions"
    sessions.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")
    # Child processes started by a job must use this library too, never the
    # real one.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(webui, "ollama_ok", lambda: False)
    monkeypatch.setattr(webui, "ollama_models", lambda: [])
    # Settings, for this process and for the CLI jobs it starts.
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    monkeypatch.setenv("NOTETAKER_SETTINGS_FILE", str(settings))
    for var in config.SETTINGS_ENV.values():
        monkeypatch.delenv(var, raising=False)
    # Nothing listens here, so a job never reaches a real notes writer.
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    return data


def _class(title="Physics", notes=True):
    session = store.create_session(title, config.SOURCE_MIC, "test")
    with TranscriptWriter(session.transcript_path) as writer:
        writer.write([Segment(0.0, 3.0, "energy is conserved", "en", 0)])
    if notes:
        store.write_notes(session, f"# {title}\n\n*today*\n\n## Key ideas\n- energy is conserved\n")
    store.finish_session(session.id, duration=3000.0, language="en")
    return store.get_session(session.id)


# ---------------------------------------------------------- click -> command
def test_a_class_becomes_the_same_command_as_notes_class():
    kind, argv, label = webui.job_argv(
        {"kind": "record", "source": "mic", "title": "  Physics  ", "minutes": 45}
    )
    assert kind == "record" and label == "Physics"
    assert argv[:3] == ["record", "--source", "mic"]
    assert ["--title", "Physics"] == argv[argv.index("--title"):argv.index("--title") + 2]
    assert ["--minutes", "45"] == argv[argv.index("--minutes"):argv.index("--minutes") + 2]
    assert "--live-notes" in argv
    # The child has no terminal, so the live panel must be off.
    assert "--plain" in argv
    cli.build_parser().parse_args(argv)  # the CLI accepts it


def test_write_later_records_sound_only():
    _, argv, _ = webui.job_argv({"kind": "record", "source": "system", "later": True})
    assert "--later" in argv and "--live-notes" not in argv
    assert "--minutes" not in argv, "no length means record until stopped"


@pytest.mark.parametrize("request_, message", [
    ({"kind": "record", "source": "speaker"}, "in class or online"),
    ({"kind": "record", "source": "mic", "minutes": -5}, "negative"),
    ({"kind": "record", "source": "mic", "minutes": "soon"}, "whole number"),
    ({"kind": "record", "source": "mic", "lang": "xx-nope"}, "language"),
    ({"kind": "video", "source": "   "}, "paste a link"),
    ({"kind": "video", "source": "https://youtu.be/x", "start": "banana"}, "start time"),
    ({"kind": "summarize", "session": "no-such-class"}, "no longer exists"),
    ({"kind": "rm -rf"}, "unknown"),
])
def test_bad_requests_are_explained_not_run(db, request_, message):
    with pytest.raises(webui.JobError, match=message):
        webui.job_argv(request_)


def test_a_video_keeps_its_trim_times():
    _, argv, _ = webui.job_argv(
        {"kind": "video", "source": "https://youtu.be/abc", "start": "2:30", "end": "45:00"}
    )
    parsed = cli.build_parser().parse_args(argv)
    assert parsed.start == 150 and parsed.end == 2700


def test_only_one_job_runs_at_a_time(db):
    jobs = webui.Jobs()

    class Running:
        status = "running"

    jobs.current = Running()  # type: ignore[assignment]
    with pytest.raises(webui.JobError, match="already running"):
        jobs.start({"kind": "catchup"})


# ------------------------------------------------------------ a real job
def test_a_job_runs_the_cli_and_reports_back(db):
    job = webui.Job("catchup", ["catchup"], "Write up", {})
    job.start()
    deadline = time.time() + 60
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.1)
    assert job.status == "done", list(job.log)
    assert any("Nothing to catch up on" in line for line in job.log)
    assert job.to_json()["command"] == "notes catchup"


def test_a_finished_recording_is_found_from_its_output(db):
    session = _class("Biology")
    job = webui.Job("record", ["record"], "Biology", {"source": "mic"})
    job.started = time.time() + 3600  # nothing counts as new, so the log must
    job.log.append(f"saved 12 segments · 1:02 · {session.id}")
    job._find_session()
    assert job.session_id == session.id


# ------------------------------------------------- stopping from the window
def test_the_window_stops_a_recording_the_same_way_as_ctrl_c(db, monkeypatch, tmp_path):
    from notetaker.audio import AudioSource
    from notetaker.pipeline import PipelineState

    monkeypatch.setattr(cli, "resolve_source",
                        lambda _: AudioSource("stub", "Test microphone", config.SOURCE_MIC))
    monkeypatch.setattr(cli, "console", None)
    monkeypatch.setattr(cli, "HAVE_RICH", False)
    stopped = {}

    class Pipeline:
        def __init__(self, **kwargs):
            self.state = PipelineState()
            stopped["pipe"] = self

        def start(self):
            pass

        def stop(self):
            stopped["clean"] = True  # the careful path: ffmpeg is asked to finish
            return True

        def finish(self):
            return 5.0

    monkeypatch.setattr(cli, "RecordingPipeline", Pipeline)
    stop_file = tmp_path / "stop"
    monkeypatch.setenv("NOTETAKER_STOP_FILE", str(stop_file))

    args = cli.build_parser().parse_args(["record", "--later", "--plain", "--title", "Maths"])
    # The recording runs on the main thread, as it does for real (it installs
    # a Ctrl-C handler). The window's click arrives from elsewhere.
    threading.Timer(0.3, stop_file.touch).start()
    # If the stop file is ignored, end the test anyway instead of hanging CI.
    safety = threading.Timer(5, lambda: setattr(stopped["pipe"].state, "capture_stopped", True))
    safety.start()
    started = time.monotonic()
    code = cli.cmd_record(args)
    safety.cancel()
    assert time.monotonic() - started < 5, "the stop file did not stop the recording"
    assert code == 0 and stopped.get("clean")
    assert [s.title for s in store.list_sessions()] == ["Maths"]


# ---------------------------------------------- the window going away
@pytest.fixture()
def link(tmp_path, monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock["now"])
    alive = tmp_path / "alive"
    alive.touch()
    monkeypatch.setenv("NOTETAKER_STOP_FILE", str(tmp_path / "stop"))
    monkeypatch.setenv("NOTETAKER_ALIVE_FILE", str(alive))
    return cli._WindowLink(), clock, alive, tmp_path / "stop"


def test_a_recording_keeps_going_while_the_window_is_there(link):
    window, clock, alive, _ = link
    for beat in range(30):
        clock["now"] += 2
        os.utime(alive, (beat + 1, beat + 1))  # the window's heartbeat
        assert not window.should_stop()


def test_a_recording_stops_itself_when_the_window_goes_quiet(link):
    """Closing the terminal must not leave a class recording forever."""
    window, clock, _, _ = link
    assert not window.should_stop()
    clock["now"] += cli._WindowLink.GRACE_SECONDS - 1
    assert not window.should_stop(), "a short pause is not the window closing"
    clock["now"] += 2
    assert window.should_stop()


def test_a_missing_heartbeat_file_also_stops_it(link):
    window, clock, alive, _ = link
    window.should_stop()
    alive.unlink()
    assert not window.should_stop()
    clock["now"] += cli._WindowLink.GRACE_SECONDS + 1
    assert window.should_stop()


def test_pressing_stop_is_immediate(link):
    window, _, _, stop = link
    stop.touch()
    assert window.should_stop()


def test_a_recording_from_the_terminal_never_stops_by_itself(monkeypatch):
    monkeypatch.delenv("NOTETAKER_STOP_FILE", raising=False)
    monkeypatch.delenv("NOTETAKER_ALIVE_FILE", raising=False)
    monkeypatch.setattr(cli.time, "monotonic", lambda: 1e9)
    assert not cli._WindowLink().should_stop()


def test_the_window_keeps_its_heartbeat_going_while_a_job_runs(db, monkeypatch):
    monkeypatch.setattr(webui, "HEARTBEAT_SECONDS", 0.05)
    job = webui.Job("catchup", ["catchup"], "Write up", {})
    job.start()
    assert job._alive_file.exists()
    deadline = time.time() + 60
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert not job._alive_file.exists(), "cleaned up when the job ends"


# ----------------------------------------------------------------- HTTP
@pytest.fixture()
def server(db):
    srv = webui.make_server(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _request(server, method, path, body=None, headers=None, host=None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    hdrs = {"Host": host or f"127.0.0.1:{port}"}
    hdrs.update(headers or {})
    payload = json.dumps(body).encode() if body is not None else None
    if payload is not None:
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp, data


def test_the_page_is_served_with_a_strict_policy(server):
    resp, data = _request(server, "GET", "/")
    assert resp.status == 200
    assert b"<title>NoteTaker</title>" in data
    assert "default-src 'none'" in resp.getheader("Content-Security-Policy")


def test_requests_for_another_host_name_are_refused(server):
    """A site that points its own name at 127.0.0.1 must not get in."""
    resp, _ = _request(server, "GET", "/api/state", host="attacker.example")
    assert resp.status == 403


def test_another_web_page_cannot_start_a_recording(server):
    resp, _ = _request(server, "POST", "/api/jobs", {"kind": "record", "source": "mic"})
    assert resp.status == 403
    assert server.RequestHandlerClass.jobs.current is None


def test_a_bad_request_from_the_window_gets_a_sentence(server):
    resp, data = _request(server, "POST", "/api/jobs", {"kind": "record", "source": "x"},
                          headers={"X-NoteTaker": "1"})
    assert resp.status == 400
    assert "in class or online" in json.loads(data)["error"]


def test_the_window_sees_every_class_and_what_is_waiting(server):
    done = _class("Physics", notes=True)
    waiting = _class("Chemistry", notes=False)
    resp, data = _request(server, "GET", "/api/state")
    state = json.loads(data)
    by_id = {s["id"]: s for s in state["sessions"]}
    assert by_id[done.id]["hasNotes"] and not by_id[done.id]["pending"]
    assert by_id[waiting.id]["pending"]
    assert state["pendingCount"] == 1
    assert state["job"] is None


def test_a_class_can_be_read_downloaded_and_its_transcript_shown(server):
    session = _class("ภาษาไทย")
    quoted = quote(f"/api/sessions/{session.id}")
    resp, data = _request(server, "GET", quoted)
    detail = json.loads(data)
    assert resp.status == 200
    assert "energy is conserved" in detail["notes"]
    assert detail["hasTranscript"] and not detail["partial"]

    resp, data = _request(server, "GET", quoted + "/transcript")
    assert json.loads(data) == [{"start": 0.0, "text": "energy is conserved"}]

    resp, data = _request(server, "GET", quoted + "/download?kind=md")
    assert resp.status == 200
    assert "attachment" in resp.getheader("Content-Disposition")
    assert "## Key ideas" in data.decode("utf-8")
    assert "filename*=UTF-8''" in resp.getheader("Content-Disposition")


def test_a_class_made_from_a_video_links_back_to_it(server):
    session = store.create_session("Photosynthesis", config.SOURCE_VIDEO, "https://youtu.be/x")
    (session.directory / "video.json").write_text(
        json.dumps({"url": "https://www.youtube.com/watch?v=abc", "offset": 0})
    )
    store.finish_session(session.id, duration=60.0, language="en")
    resp, data = _request(server, "GET", quote(f"/api/sessions/{session.id}"))
    assert resp.status == 200
    assert json.loads(data)["link"] == "https://www.youtube.com/watch?v=abc"


def test_an_unknown_class_is_a_clean_404(server):
    resp, data = _request(server, "GET", "/api/sessions/nope")
    assert resp.status == 404
    assert "no longer exists" in json.loads(data)["error"]


# --------------------------------------------------------------- launcher
def test_the_window_is_one_word_away():
    from notetaker import shortcuts

    for word in ("ui", "app", "window"):
        assert shortcuts.translate([word]) == ["ui"]


def test_ui_command_parses():
    args = cli.build_parser().parse_args(["ui", "--no-browser", "--port", "0"])
    assert args.no_browser and args.port == 0


def test_readme_walkthrough_names_buttons_the_window_really_has():
    """The first-time steps send a new user to buttons by name.

    If a button is renamed, the walkthrough must change with it, or a new
    user is told to click something that is not there.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    steps = readme.split("### First time using NoteTaker?")[1].split("###")[0]
    page = (root / "notetaker" / "webui" / "index.html").read_text(encoding="utf-8")
    for button in ("Settings", "Check this computer", "New notes", "Start recording"):
        assert f"**{button}**" in steps
        assert button in page


def test_readme_images_exist():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    images = re.findall(r"(docs/images/[\w.-]+\.png)", readme)
    assert images, "the README should show the window"
    for image in images:
        assert (root / image).is_file(), image


# ------------------------------------------------------------------ search
def test_search_finds_what_was_said_in_every_class(server):
    _class("Physics")
    _class("Chemistry", notes=False)
    resp, data = _request(server, "GET", "/api/search?q=" + quote("ENERGY conserved"))
    found = json.loads(data)
    assert resp.status == 200
    assert found["terms"] == ["energy", "conserved"]
    assert {r["title"] for r in found["results"]} == {"Physics", "Chemistry"}
    hit = found["results"][0]["hits"][0]
    assert hit["text"] == "energy is conserved" and hit["label"] == "0:00"
    assert hit["link"] is None, "only a video has somewhere to play from"


def test_a_search_hit_in_a_video_links_to_that_moment(server):
    session = store.create_session("Lesson", config.SOURCE_VIDEO, "https://youtu.be/x")
    (session.directory / "video.json").write_text(
        json.dumps({"url": "https://www.youtube.com/watch?v=abc", "offset": 60}))
    with TranscriptWriter(session.transcript_path) as writer:
        writer.write([Segment(30.0, 33.0, "photosynthesis needs light", "en", 0)])
    store.finish_session(session.id, duration=60.0, language="en")
    _, data = _request(server, "GET", "/api/search?q=photosynthesis")
    hit = json.loads(data)["results"][0]["hits"][0]
    assert hit["start"] == 30.0, "where it is in the transcript"
    assert hit["label"] == "1:30", "the video's own clock, after the skipped intro"
    assert "t=87" in hit["link"], "a moment before it is said"


def test_an_empty_search_finds_nothing(server):
    _class("Physics")
    _, data = _request(server, "GET", "/api/search?q=%20%20")
    assert json.loads(data)["results"] == []


# ------------------------------------------------------------------ practise
def test_practice_before_questions_are_written_uses_the_words_to_know(server):
    session = _class("Biology", notes=False)
    store.write_notes(session, "# Biology\n\n## Key ideas\n- cells divide\n\n"
                               "## Terms & definitions\n- **Mitosis**: a cell splitting in two\n")
    _, data = _request(server, "GET", quote(f"/api/sessions/{session.id}/quiz"))
    view = json.loads(data)
    assert view["written"] is False
    assert [(c["question"], c["answer"]) for c in view["cards"]] == [("Mitosis", "a cell splitting in two")]


def test_written_questions_are_shown_and_go_to_anki(server):
    from notetaker import quiz

    session = _class("Physics")
    quiz.save(session.directory, [quiz.Card("Is energy conserved?", "energy is conserved", "idea")])
    _, data = _request(server, "GET", quote(f"/api/sessions/{session.id}/quiz"))
    view = json.loads(data)
    assert view["written"] is True and view["cards"][0]["question"] == "Is energy conserved?"

    resp, data = _request(server, "GET", quote(f"/api/sessions/{session.id}/quiz/anki"))
    assert resp.status == 200 and "attachment" in resp.getheader("Content-Disposition")
    assert "Is energy conserved?\tenergy is conserved" in data.decode("utf-8")


def test_no_anki_file_without_any_cards(server):
    session = _class("Physics", notes=False)
    resp, _ = _request(server, "GET", quote(f"/api/sessions/{session.id}/quiz/anki"))
    assert resp.status == 404


def test_writing_questions_is_the_same_command_as_notes_quiz(db):
    session = _class("Physics")
    kind, argv, label = webui.job_argv({"kind": "quiz", "session": session.id})
    assert (kind, argv, label) == ("quiz", ["quiz", session.id, "--list"], "Physics")
    _, argv, _ = webui.job_argv({"kind": "quiz", "session": session.id, "fresh": True})
    assert argv[-1] == "--new"
    cli.build_parser().parse_args(argv)
    with pytest.raises(webui.JobError, match="no notes"):
        webui.job_argv({"kind": "quiz", "session": _class("Maths", notes=False).id})


def test_a_quiz_job_runs_without_a_notes_writer(db):
    """With Ollama off it still finishes, with the words-to-know cards."""
    session = _class("Physics", notes=False)
    store.write_notes(session, "## Terms & definitions\n- **Energy**: the ability to do work\n")
    job = webui.Job("quiz", ["quiz", session.id, "--list"], "Physics", {"session": session.id})
    job.start()
    deadline = time.time() + 60
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.1)
    assert job.status == "done", list(job.log)
    assert job.session_id == session.id
    assert any("Energy" in line for line in job.log)


# ------------------------------------------------------------------ settings
def test_settings_page_shows_values_and_defaults(server):
    resp, data = _request(server, "GET", "/api/settings")
    view = json.loads(data)
    assert resp.status == 200
    assert view["values"] == config.SETTINGS_DEFAULTS
    assert view["overridden"] == {}
    assert {"code": "th", "name": "ภาษาไทย"} in view["languages"] or any(
        l["code"] == "th" for l in view["languages"])


def test_a_setting_saved_in_the_window_reaches_the_file(server):
    resp, data = _request(server, "POST", "/api/settings",
                          {"changes": {"class_minutes": 45, "notes_language": "th"}},
                          headers={"X-NoteTaker": "1"})
    assert resp.status == 200
    assert json.loads(data)["values"]["class_minutes"] == 45
    assert config.load_settings(config.SETTINGS_PATH) == {"class_minutes": 45, "notes_language": "th"}
    # The New notes page picks it up straight away, without a restart.
    _, data = _request(server, "GET", "/api/state")
    assert json.loads(data)["defaultMinutes"] == 45


@pytest.mark.parametrize("changes, message", [
    ({"class_minutes": 9000}, "between 1 and 600"),
    ({"notes_language": "zz"}, "no notes language"),
    ({"sneaky": 1}, "no setting"),
    ({}, "nothing to change"),
])
def test_a_bad_setting_is_refused_with_a_sentence(server, changes, message):
    resp, data = _request(server, "POST", "/api/settings", {"changes": changes},
                          headers={"X-NoteTaker": "1"})
    assert resp.status == 400
    assert message in json.loads(data)["error"]
    assert not config.SETTINGS_PATH.exists()


def test_settings_cannot_be_changed_by_another_web_page(server):
    resp, _ = _request(server, "POST", "/api/settings", {"changes": {"class_minutes": 45}})
    assert resp.status == 403
    assert not config.SETTINGS_PATH.exists()


def test_settings_say_when_a_terminal_variable_wins(server, monkeypatch):
    monkeypatch.setenv("NOTETAKER_NOTES_LEVEL", "university")
    _, data = _request(server, "GET", "/api/settings")
    view = json.loads(data)
    assert view["overridden"] == {"notes_level": "NOTETAKER_NOTES_LEVEL=university"}
    assert view["values"]["notes_level"] == "university"
