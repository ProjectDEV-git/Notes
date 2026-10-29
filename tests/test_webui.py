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
    data = tmp_path / "data"
    sessions = data / "sessions"
    sessions.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")
    # Child processes started by a job must use this library too, never the
    # real one.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(webui, "ollama_ok", lambda: False)
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
