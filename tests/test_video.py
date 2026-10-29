"""Notes from a video: YouTube (or other) links, and video/audio files.

Nothing here touches the internet. yt-dlp is exercised for real against a
web server on this machine, and the network-facing parts are faked.
"""

from __future__ import annotations

import functools
import http.server
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from notetaker import cli, config, shortcuts, store, video
from notetaker.asr import Segment

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture()
def db(tmp_path, monkeypatch):
    data = tmp_path / "data"
    sessions = data / "sessions"
    sessions.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")
    monkeypatch.delenv("NOTES_CALLER_DIR", raising=False)
    return data / "notetaker.db"


@pytest.fixture()
def lesson_mp4(tmp_path):
    """A 30-second 'video' whose sound is the English test lecture."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "week_4-energy.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=30",
         "-i", str(FIXTURES / "lecture_en_30s.wav"), "-shortest",
         "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True,
    )
    return path


# -------------------------------------------------------- what was pasted?
@pytest.mark.parametrize("text", [
    "https://www.youtube.com/watch?v=abc",
    "http://example.com/lesson.mp4",
    "youtu.be/abc",
    "www.youtube.com/watch?v=abc",
    "m.youtube.com/watch?v=abc",
    "vimeo.com/12345",
    "  https://youtu.be/abc  ",
])
def test_links_are_recognised(text):
    assert video.looks_like_url(text)


@pytest.mark.parametrize("text", ["lesson.mp4", "physics", "/home/me/a.mp4", "ftp://x/y", ""])
def test_non_links_are_not(text):
    assert not video.looks_like_url(text)


def test_a_link_without_https_still_works():
    assert video.normalize_url("youtu.be/abc") == "https://youtu.be/abc"
    assert video.normalize_url("https://youtu.be/abc") == "https://youtu.be/abc"


@pytest.mark.parametrize("pasted, expected", [
    ("'/My Videos/a.mp4'", "/My Videos/a.mp4"),
    ('"/My Videos/a.mp4"', "/My Videos/a.mp4"),
    ("file:///My%20Videos/a.mp4", "/My Videos/a.mp4"),
    ("  https://youtu.be/abc\n", "https://youtu.be/abc"),
])
def test_a_dragged_in_file_is_cleaned_up(pasted, expected):
    assert video.clean_pasted(pasted) == expected


@pytest.mark.skipif(os.name == "nt", reason="backslash is a separator on Windows")
def test_a_backslash_escaped_path_is_unescaped():
    assert video.clean_pasted(r"/My\ Videos/week\ 4.mp4") == "/My Videos/week 4.mp4"


def test_a_file_title_is_readable():
    assert video.title_from_filename(Path("week_4-energy.mp4")) == "week 4 energy"
    assert video.title_from_filename(Path("___.mp4")) == "Video"


# ------------------------------------------------------- reading the info
def test_creator_subtitles_in_the_spoken_language_are_picked():
    subs = {"live_chat": [], "fr": [], "en-GB": [], "th": []}
    assert video.pick_caption(subs, "th") == "th"
    assert video.pick_caption(subs, "en") == "en-GB"
    assert video.pick_caption({"en": [], "en-US": []}, "en") == "en"


def test_subtitles_are_not_used_when_the_language_is_unknown():
    """A track in another language is a translation; notes would change language."""
    assert video.pick_caption({"en": []}, None) is None
    assert video.pick_caption({"fr": []}, "en") is None


def test_video_info_is_read():
    info = video.parse_info({
        "webpage_url": "https://www.youtube.com/watch?v=abc",
        "title": "Photosynthesis explained",
        "duration": 612,
        "language": "en-US",
        "subtitles": {"en": [{"ext": "vtt"}]},
    })
    assert info.url == "https://www.youtube.com/watch?v=abc"
    assert info.title == "Photosynthesis explained"
    assert info.duration == 612
    assert info.language == "en"
    assert info.caption_key == "en"


def test_the_chosen_language_overrides_the_site():
    info = video.parse_info({"language": "en", "subtitles": {"en": [], "th": []}}, "th")
    assert info.language == "th" and info.caption_key == "th"


@pytest.mark.parametrize("data, says", [
    ({"_type": "playlist", "entries": []}, "playlist"),
    ({"is_live": True}, "live right now"),
    ({"live_status": "is_live"}, "live right now"),
    ({"live_status": "is_upcoming"}, "not started"),
    ({"live_status": "post_live"}, "just finished"),
])
def test_links_that_are_not_one_finished_video_are_explained(data, says):
    with pytest.raises(video.VideoError, match=says):
        video.parse_info(data)


@pytest.mark.parametrize("reason, says", [
    ("Unsupported URL: https://example.com", "cannot get videos from that website"),
    ("Private video. Sign in if you've been granted access", "private"),
    ("Sign in to confirm your age", "age-restricted"),
    ("Sign in to confirm you're not a bot", "not a bot"),
    ("Video unavailable", "unavailable"),
    ("Unable to download webpage: <urlopen error [Errno -2]>", "internet connection"),
])
def test_permanent_failures_are_explained_and_not_retried(reason, says):
    message, fixable = video.explain(reason)
    assert says in message
    assert not fixable


def test_unknown_failures_may_be_fixed_by_an_upgrade():
    message, fixable = video.explain("Requested format is not available")
    assert fixable and "Requested format" in message


def test_the_last_error_line_is_reported_without_prefixes():
    stderr = "[youtube] abc: Downloading webpage\nERROR: [youtube] abc: Video unavailable\n"
    assert video._last_error(stderr) == "Video unavailable"


# ------------------------------------------------ upgrade when a site changes
def _completed(code, stderr="", stdout=""):
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr=stderr)


def test_a_stale_downloader_is_upgraded_once_and_retried(monkeypatch):
    results = iter([_completed(1, "ERROR: [youtube] x: Requested format is not available"),
                    _completed(0, stdout="{}")])
    monkeypatch.setattr(video.subprocess, "run", lambda *a, **k: next(results))
    upgrades = []
    monkeypatch.setattr(video, "upgrade_downloader", lambda echo: upgrades.append(1))

    assert video._run_downloader(["x"], print, True, 5).returncode == 0
    assert upgrades == [1]


def test_the_downloader_is_not_upgraded_for_a_private_video(monkeypatch):
    monkeypatch.setattr(video.subprocess, "run",
                        lambda *a, **k: _completed(1, "ERROR: Private video"))
    monkeypatch.setattr(video, "upgrade_downloader",
                        lambda echo: pytest.fail("upgrading cannot fix a private video"))
    with pytest.raises(video.VideoError, match="private"):
        video._run_downloader(["x"], print, True, 5)


def test_it_gives_up_after_one_upgrade(monkeypatch):
    monkeypatch.setattr(video.subprocess, "run",
                        lambda *a, **k: _completed(1, "ERROR: HTTP Error 403: Forbidden"))
    upgrades = []
    monkeypatch.setattr(video, "upgrade_downloader", lambda echo: upgrades.append(1))
    with pytest.raises(video.VideoError, match="403"):
        video._run_downloader(["x"], print, True, 5)
    assert upgrades == [1]


def test_the_helper_installs_into_this_python(monkeypatch):
    calls = []
    monkeypatch.setattr(video.subprocess, "run",
                        lambda cmd, **k: calls.append(cmd) or _completed(0))
    video.install_downloader(lambda m: None)
    assert calls[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert str(video.REQUIREMENTS) in calls[0]


def test_a_failed_install_says_why(monkeypatch):
    monkeypatch.setattr(video.subprocess, "run",
                        lambda cmd, **k: _completed(1, "ERROR: No matching distribution"))
    with pytest.raises(video.VideoError, match="No matching distribution"):
        video.install_downloader(lambda m: None)


def test_the_video_helper_requirements_bring_youtubes_javascript_runtime():
    text = video.REQUIREMENTS.read_text()
    assert "yt-dlp[" in text and "deno" in text
    # Deno is large: it must not be in what the startup updater installs.
    assert "yt-dlp" not in (REPO / "requirements.txt").read_text()


# --------------------------------------------------------------- subtitles
VTT = """WEBVTT
Kind: captions

NOTE this is a comment

00:00:00.000 --> 00:00:03.000
[Music]

00:00:03.000 --> 00:00:08.500 align:start position:0%
Today we look at <c>energy</c> &amp; how
it is conserved.

00:00:08.500 --> 00:00:09.000
Today we look at energy & how it is conserved.

01:02:03,250 --> 01:02:05,000
Potential turns into kinetic.

00:10.000 --> 00:12.000
♪ ♪
"""


def test_subtitles_become_a_transcript():
    segments = video.parse_vtt(VTT, "en")
    assert [s.text for s in segments] == [
        "Today we look at energy & how it is conserved.",
        "Potential turns into kinetic.",
    ]
    assert segments[0].start == 3.0 and segments[0].end == 8.5
    assert segments[1].start == pytest.approx(3723.25)
    assert all(s.lang == "en" for s in segments)


def test_empty_subtitles_give_an_empty_transcript():
    assert video.parse_vtt("WEBVTT\n\n", "en") == []


# --------------------------------------------------------------- the sound
@needs_ffmpeg
def test_a_video_files_sound_becomes_a_recording(lesson_mp4, tmp_path):
    from notetaker.audio import wav_duration

    out = tmp_path / "audio.wav"
    video.to_wav(lesson_mp4, out)
    assert wav_duration(out) == pytest.approx(30, abs=1)
    assert not (tmp_path / "audio.partial.wav").exists()


@needs_ffmpeg
def test_a_file_with_no_sound_is_explained(tmp_path):
    silent_film = tmp_path / "film.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=64x64:d=1", "-c:v", "libx264", str(silent_film)], check=True)
    with pytest.raises(video.VideoError, match="no sound"):
        video.to_wav(silent_film, tmp_path / "audio.wav")
    assert not (tmp_path / "audio.wav").exists()


@needs_ffmpeg
def test_a_file_that_is_not_media_is_explained(tmp_path):
    bogus = tmp_path / "notes.mp4"
    bogus.write_text("not a video")
    with pytest.raises(video.VideoError, match="could not read the sound"):
        video.to_wav(bogus, tmp_path / "audio.wav")


# --------------------------------------------------------- the whole command
class FakeTranscriber:
    def __init__(self, model, language=None):
        self.model, self.language = model, language or "en"

    def transcribe_file(self, path, progress=None):
        for done in (10.0, 20.0, 30.0):
            if progress:
                progress(done, 30.0)
        return [Segment(0.0, 5.0, "energy is conserved", "en", 0)]


def _no_summary():
    return patch.object(cli, "_summarize_session", return_value=0)


@needs_ffmpeg
def test_a_video_file_becomes_a_class(db, lesson_mp4, capsys):
    with patch.object(cli, "Transcriber", FakeTranscriber), _no_summary() as summarize:
        assert cli.main(["video", str(lesson_mp4)]) == 0

    [session] = store.list_sessions()
    assert session.title == "week 4 energy"
    assert session.source_kind == config.SOURCE_VIDEO
    assert session.source_name == str(lesson_mp4.resolve())
    assert session.duration == pytest.approx(30, abs=1)
    assert session.language == "en"
    assert session.audio_path.exists()
    assert [s.text for s in store.load_transcript(session)] == ["energy is conserved"]
    summarize.assert_called_once()
    out = capsys.readouterr().out
    assert "transcribed 100%" in out


@needs_ffmpeg
def test_a_relative_path_means_where_the_student_ran_notes(db, lesson_mp4, monkeypatch):
    """The Mac/Linux launcher runs Python from the app folder."""
    monkeypatch.setenv("NOTES_CALLER_DIR", str(lesson_mp4.parent))
    monkeypatch.chdir(REPO)
    assert cli.main(["video", lesson_mp4.name, "--later"]) == 0
    assert store.list_sessions()[0].audio_path.exists()


@needs_ffmpeg
def test_later_only_saves_the_sound_for_catchup(db, lesson_mp4):
    with patch.object(cli, "Transcriber", side_effect=AssertionError("nothing to transcribe yet")):
        assert cli.main(["video", str(lesson_mp4), "--later", "--title", "Physics"]) == 0
    [session] = store.pending_sessions()
    assert session.title == "Physics"
    assert not session.transcript_path.exists()


@needs_ffmpeg
def test_a_failed_transcription_keeps_the_sound(db, lesson_mp4, capsys):
    class Broken(FakeTranscriber):
        def transcribe_file(self, path, progress=None):
            raise RuntimeError("out of memory")

    with patch.object(cli, "Transcriber", Broken):
        assert cli.main(["video", str(lesson_mp4)]) == 1
    assert "notes catchup" in capsys.readouterr().out
    assert store.pending_sessions()[0].audio_path.exists()


def test_something_that_is_neither_a_link_nor_a_file(db, capsys):
    assert cli.main(["video", "physics-lesson"]) == 1
    assert "not a link" in capsys.readouterr().out
    assert store.list_sessions() == []


def _fake_link(monkeypatch, tmp_path, caption_key="en", info_language="en", fail=None):
    """Pretend yt-dlp fetched a YouTube video; the media is the test lecture."""
    info = video.VideoInfo(
        url="https://www.youtube.com/watch?v=abc", title="Energy | Physics",
        duration=30.0, language=info_language, caption_key=caption_key,
    )
    asked = {}

    def fake_download(url, folder, echo, caption_key=None):
        asked["caption_key"] = caption_key
        if fail:
            raise video.VideoError(fail)
        folder.mkdir(parents=True, exist_ok=True)
        media = folder / "download.wav"
        shutil.copy(FIXTURES / "lecture_en_30s.wav", media)
        vtt = None
        if caption_key:
            vtt = folder / f"download.{caption_key}.vtt"
            vtt.write_text(VTT)
        return media, vtt

    monkeypatch.setattr(video, "have_downloader", lambda: True)
    monkeypatch.setattr(video, "fetch_info", lambda url, echo, language=None: info)
    monkeypatch.setattr(video, "download", fake_download)
    return asked


@needs_ffmpeg
def test_a_videos_own_subtitles_skip_transcription(db, monkeypatch, tmp_path, capsys):
    """Creator subtitles are accurate and free; Thai on CPU is ~5x real time."""
    _fake_link(monkeypatch, tmp_path)
    with patch.object(cli, "Transcriber", side_effect=AssertionError("should not transcribe")), \
            _no_summary() as summarize:
        assert cli.main(["video", "https://youtu.be/abc"]) == 0

    [session] = store.list_sessions()
    assert session.title == "Energy | Physics"
    assert session.source_name == "https://www.youtube.com/watch?v=abc"
    assert session.language == "en"
    assert len(store.load_transcript(session)) == 2
    assert session.audio_path.exists(), "kept so --hq can still re-transcribe"
    assert not (session.directory / "download").exists()
    summarize.assert_called_once()
    assert "own subtitles" in capsys.readouterr().out


@needs_ffmpeg
def test_whisper_can_be_chosen_over_the_subtitles(db, monkeypatch, tmp_path):
    asked = _fake_link(monkeypatch, tmp_path)
    with patch.object(cli, "Transcriber", FakeTranscriber), _no_summary():
        assert cli.main(["video", "https://youtu.be/abc", "--whisper"]) == 0
    assert asked["caption_key"] is None
    assert store.load_transcript(store.list_sessions()[0])[0].text == "energy is conserved"


@needs_ffmpeg
def test_hq_uses_the_accurate_model(db, monkeypatch, tmp_path):
    _fake_link(monkeypatch, tmp_path, caption_key=None)
    used = []

    class Recording(FakeTranscriber):
        def __init__(self, model, language=None):
            used.append(model)
            super().__init__(model, language)

    with patch.object(cli, "Transcriber", Recording), _no_summary():
        assert cli.main(["video", "https://youtu.be/abc", "--hq"]) == 0
    assert used == [config.ASR_MODEL_HQ]


@needs_ffmpeg
def test_the_same_video_is_not_done_twice(db, monkeypatch, tmp_path, capsys):
    _fake_link(monkeypatch, tmp_path)
    with _no_summary():
        assert cli.main(["video", "https://youtu.be/abc"]) == 0
        assert cli.main(["video", "https://www.youtube.com/watch?v=abc"]) == 0
    assert len(store.list_sessions()) == 1
    assert "already have this video" in capsys.readouterr().out

    with _no_summary():
        assert cli.main(["video", "https://youtu.be/abc", "--again"]) == 0
    assert len(store.list_sessions()) == 2


def test_a_failed_download_leaves_no_empty_class(db, monkeypatch, tmp_path, capsys):
    _fake_link(monkeypatch, tmp_path, fail="That video is private, so it cannot be downloaded.")
    assert cli.main(["video", "https://youtu.be/abc"]) == 1
    assert "private" in capsys.readouterr().out
    assert store.list_sessions() == []


def test_the_helper_is_installed_on_first_use(db, monkeypatch, tmp_path):
    _fake_link(monkeypatch, tmp_path, fail="stop here")
    monkeypatch.setattr(video, "have_downloader", lambda: False)
    installs = []
    monkeypatch.setattr(video, "install_downloader", lambda echo: installs.append(1))
    cli.main(["video", "https://youtu.be/abc"])
    assert installs == [1]


def test_a_local_file_never_needs_the_helper(db, monkeypatch, lesson_mp4):
    monkeypatch.setattr(video, "have_downloader",
                        lambda: pytest.fail("a file does not need yt-dlp"))
    assert cli.main(["video", str(lesson_mp4), "--later"]) == 0


# ------------------------------------- real yt-dlp against a local web server
@pytest.fixture()
def local_site(tmp_path, lesson_mp4):
    site = tmp_path / "site"
    site.mkdir()
    shutil.copy(lesson_mp4, site / "lesson.mp4")
    (site / "lesson.vtt").write_text(VTT)
    shutil.copy(lesson_mp4, site / "lesson2.mp4")
    (site / "course.html").write_text(
        "<html><head><title>Course</title></head><body>"
        '<video src="lesson.mp4"></video><video src="lesson2.mp4"></video></body></html>'
    )
    (site / "page.html").write_text(
        "<html><head><title>Physics week 4</title></head><body>"
        '<video src="lesson.mp4"><track kind="subtitles" srclang="en" src="lesson.vtt">'
        "</video></body></html>"
    )
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    old = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.skipif(not video.have_downloader(), reason="yt-dlp not installed")
def test_a_real_download_with_subtitles(db, local_site, monkeypatch):
    with _no_summary():
        assert cli.main(["video", f"{local_site}/page.html", "--lang", "en"]) == 0
    [session] = store.list_sessions()
    assert session.title.startswith("Physics week 4")
    assert session.duration == pytest.approx(30, abs=1)
    assert len(store.load_transcript(session)) == 2


@pytest.mark.skipif(not video.have_downloader(), reason="yt-dlp not installed")
def test_a_real_download_of_a_media_link(db, local_site):
    assert cli.main(["video", f"{local_site}/lesson.mp4", "--later"]) == 0
    [session] = store.pending_sessions()
    assert session.audio_path.exists()


@pytest.mark.skipif(not video.have_downloader(), reason="yt-dlp not installed")
def test_a_real_missing_video_is_explained(db, local_site, capsys):
    with patch.object(video, "upgrade_downloader", lambda echo: None):
        assert cli.main(["video", f"{local_site}/gone.mp4"]) == 1
    assert "unavailable" in capsys.readouterr().out
    assert store.list_sessions() == []


# ------------------------------------------------------------ getting there
def test_notes_with_a_link_makes_notes_from_it():
    assert shortcuts.translate(["https://youtu.be/abc"]) == ["video", "https://youtu.be/abc"]
    assert shortcuts.translate(["youtu.be/abc", "--later"]) == ["video", "youtu.be/abc", "--later"]


def test_notes_youtube_records_unless_given_a_link():
    assert shortcuts.translate(["youtube"]) == shortcuts.translate(["online"])
    assert shortcuts.translate(["youtube", "--title", "X"])[0] == "record"
    assert shortcuts.translate(["youtube", "https://youtu.be/abc"]) == [
        "video", "https://youtu.be/abc"]


def test_notes_video_and_a_dropped_file(tmp_path, monkeypatch):
    clip = tmp_path / "lesson.mp4"
    clip.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert shortcuts.translate(["video", "x.mp4"]) == ["video", "x.mp4"]
    assert shortcuts.translate(["lesson.mp4"]) == ["video", "lesson.mp4"]
    (tmp_path / "class").write_bytes(b"")
    assert shortcuts.translate(["class"])[0] == "record", "a verb beats a file of that name"


def _launch(*argv: str, cwd: Path) -> str:
    """What CLI command, from which folder, does the bash launcher run?"""
    launcher = (REPO / "scripts" / "notes").read_text().replace(
        'exec "$PY" -m notetaker.cli', 'echo "$NOTES_CALLER_DIR" CLI'
    )
    with tempfile.TemporaryDirectory() as tmp:
        app_bin = Path(tmp) / "app" / ".venv" / "bin"
        app_bin.mkdir(parents=True)
        (app_bin / "python").write_text("#!/bin/sh\nexit 0\n")
        (app_bin / "python").chmod(0o755)
        script = Path(tmp) / "notes"
        script.write_text(launcher)
        result = subprocess.run(
            ["bash", str(script), *argv], capture_output=True, text=True, cwd=cwd,
            env={**os.environ, "NOTETAKER_APP_DIR": str(Path(tmp) / "app"),
                 "NOTES_AUTO_UPDATE": "0"},
        )
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("argv, expected", [
    (["https://youtu.be/abc"], "CLI video https://youtu.be/abc"),
    (["youtu.be/abc"], "CLI video youtu.be/abc"),
    (["video", "x.mp4"], "CLI video x.mp4"),
    (["youtube", "https://youtu.be/abc"], "CLI video https://youtu.be/abc"),
    (["youtube"], "CLI record --source system --live-notes"),
    (["youtube", "--title", "X"], "CLI record --source system --live-notes --title X"),
    (["my lesson.mp4"], "CLI video my lesson.mp4"),
])
def test_the_bash_launcher_routes_videos(tmp_path, argv, expected):
    (tmp_path / "my lesson.mp4").write_bytes(b"")
    line = _launch(*argv, cwd=tmp_path)
    caller, _, command = line.partition(" ")
    assert command == expected
    assert Path(caller).resolve() == tmp_path.resolve(), "Python must know where the student was"


def test_export_writes_where_the_student_ran_notes(db, tmp_path, monkeypatch):
    """Not the app folder: a file there is lost, and it switches off updates."""
    session = store.create_session("Physics", config.SOURCE_MIC)
    store.write_notes(session, "## Key ideas\n- energy\n")
    here = tmp_path / "desk"
    here.mkdir()
    monkeypatch.setenv("NOTES_CALLER_DIR", str(here))
    monkeypatch.chdir(REPO)

    assert cli.main(["export", "Physics", "-o", "physics.md"]) == 0
    assert (here / "physics.md").read_text().startswith("## Key ideas")
    assert cli.main(["export", "Physics"]) == 0
    assert (here / f"{session.id}.md").exists()


def test_the_menu_takes_a_pasted_link(monkeypatch):
    from notetaker import menu

    with patch("builtins.input", side_effect=["'/My Videos/lesson.mp4'"]), \
            patch("notetaker.cli.main", return_value=0) as run:
        assert menu._from_video() == 0
    run.assert_called_once_with(["video", "/My Videos/lesson.mp4"])


def test_the_menu_does_nothing_when_nothing_is_pasted():
    from notetaker import menu

    with patch("builtins.input", side_effect=[""]), \
            patch("notetaker.cli.main", side_effect=AssertionError("must not run")):
        assert menu._from_video() == 0


# ------------------------------------------------------------ transcription
def test_whole_file_transcription_reports_progress():
    from notetaker.asr import Transcriber

    transcriber = Transcriber.__new__(Transcriber)
    transcriber.forced_language = None
    parts = [SimpleNamespace(start=0.0, end=10.0, text=" one "),
             SimpleNamespace(start=10.0, end=20.0, text="  "),
             SimpleNamespace(start=20.0, end=40.0, text="two")]
    transcriber._model = SimpleNamespace(
        transcribe=lambda *a, **k: (iter(parts), SimpleNamespace(language="en", duration=40.0))
    )
    seen = []
    segments = transcriber.transcribe_file(Path("x.wav"), progress=lambda d, t: seen.append((d, t)))
    assert [s.text for s in segments] == ["one", "two"]
    assert seen == [(10.0, 40.0), (20.0, 40.0), (40.0, 40.0)]


# ---------------------------------------------------------------- playlists
def test_a_youtube_playlist_lists_its_videos():
    playlist = video.parse_playlist({
        "_type": "playlist", "title": "Physics course",
        "entries": [
            {"_type": "url", "ie_key": "Youtube", "title": "Lesson 1",
             "url": "https://www.youtube.com/watch?v=a"},
            {"_type": "url", "ie_key": "Youtube", "title": "[Private video]",
             "url": "https://www.youtube.com/watch?v=b"},
            None,
            {"title": "no address"},
        ],
    })
    assert playlist.title == "Physics course"
    assert [(e.url, e.title) for e in playlist.entries] == [
        ("https://www.youtube.com/watch?v=a", "Lesson 1"),
        ("https://www.youtube.com/watch?v=b", "[Private video]"),
    ]


def test_a_whole_channel_is_explained_not_downloaded():
    with pytest.raises(video.VideoError, match="whole channel"):
        video.parse_playlist({"_type": "playlist", "entries": [
            {"_type": "url", "ie_key": "YoutubeTab", "url": "https://www.youtube.com/@x/videos"},
        ]})


def test_a_page_with_several_videos_lists_the_files():
    playlist = video.parse_playlist({"_type": "playlist", "entries": [
        {"title": "Course (1)", "formats": [{"url": "http://x/1.mp4"}]},
    ]})
    assert playlist.entries[0].url == "http://x/1.mp4"


def test_looking_up_a_playlist_does_not_open_every_video(monkeypatch):
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        return _completed(0, stdout='{"_type": "playlist", "title": "C", "entries": []}')

    monkeypatch.setattr(video.subprocess, "run", run)
    assert isinstance(video.fetch_info("https://youtube.com/playlist?list=x", print), video.Playlist)
    assert "--flat-playlist" in calls[0]


def _fake_playlist(monkeypatch, titles, broken=()):
    entries = [video.PlaylistEntry(f"https://youtu.be/{i}", t) for i, t in enumerate(titles)]

    def fetch(url, echo, language=None):
        if url == "https://youtube.com/playlist?list=x":
            return video.Playlist("Physics course", entries)
        index = int(url.rsplit("/", 1)[1])
        if titles[index] in broken:
            raise video.VideoError("That video is private, so it cannot be downloaded.")
        return video.VideoInfo(url=f"https://www.youtube.com/watch?v={index}",
                               title=titles[index], duration=30, language="en",
                               caption_key=None)

    monkeypatch.setattr(video, "have_downloader", lambda: True)
    monkeypatch.setattr(video, "fetch_info", fetch)
    imported = []
    monkeypatch.setattr(cli, "_import_one",
                        lambda args, **k: imported.append(k["title"]) or 0)
    return imported


def test_every_video_in_a_playlist_becomes_a_class(db, monkeypatch, capsys):
    imported = _fake_playlist(monkeypatch, ["Lesson 1", "Lesson 2", "Lesson 3"])
    assert cli.main(["video", "https://youtube.com/playlist?list=x", "--later"]) == 0
    assert imported == ["Lesson 1", "Lesson 2", "Lesson 3"]
    assert "finished 3 of 3" in capsys.readouterr().out


def test_a_playlist_can_be_limited_and_titled(db, monkeypatch):
    imported = _fake_playlist(monkeypatch, ["Lesson 1", "Lesson 2", "Lesson 3"])
    assert cli.main(["video", "https://youtube.com/playlist?list=x",
                     "--limit", "2", "--title", "Physics"]) == 0
    assert imported == ["Physics: Lesson 1", "Physics: Lesson 2"]


def test_one_bad_video_does_not_stop_the_playlist(db, monkeypatch, capsys):
    imported = _fake_playlist(monkeypatch, ["Lesson 1", "Gone", "Lesson 3"], broken={"Gone"})
    assert cli.main(["video", "https://youtube.com/playlist?list=x"]) == 1
    assert imported == ["Lesson 1", "Lesson 3"]
    out = capsys.readouterr().out
    assert "skipped" in out and "Not finished: Gone" in out


@pytest.mark.skipif(not video.have_downloader(), reason="yt-dlp not installed")
def test_a_real_playlist_downloads_every_video_and_resumes(db, local_site, capsys):
    url = f"{local_site}/course.html"
    assert cli.main(["video", url, "--later"]) == 0
    assert len(store.pending_sessions()) == 2
    assert cli.main(["video", url, "--later"]) == 0
    assert len(store.list_sessions()) == 2, "finished videos are skipped on a re-run"
    assert "already done" in capsys.readouterr().out
