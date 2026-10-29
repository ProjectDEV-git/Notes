"""A small window for NoteTaker, served to the browser from this computer.

    notes ui

Nothing here records or summarizes by itself. Every action runs the same
CLI command the menu would, in a child process, so the window can never
behave differently from `notes class` or `notes video`. The server only
listens on 127.0.0.1 and nothing leaves the machine.

Only one job runs at a time: two recordings, or a recording and a write-up,
would fight over the CPU and the microphone.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .. import config, store

APP_DIR = Path(__file__).resolve().parent.parent.parent
PAGE = Path(__file__).resolve().parent / "index.html"
DEFAULT_PORT = 8765
IS_WINDOWS = sys.platform == "win32"

# How often the window tells a recording it is still there. A recording that
# hears nothing for HEARTBEAT_GRACE seconds (see cli._WindowLink) stops and
# saves itself, so closing the terminal can never leave one running unseen.
HEARTBEAT_SECONDS = 2.0

# How many lines of a job's output the window keeps. Enough to show an error
# in full, small enough that polling every second stays cheap.
LOG_LINES = 200


class JobError(ValueError):
    """A request the window can explain to the student in one sentence."""


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def job_argv(request: dict) -> tuple[str, list[str], str]:
    """Turn a request from the window into (kind, CLI arguments, label).

    Validated here rather than left to argparse, so a bad value becomes a
    sentence in the window instead of a usage dump in the job log.
    """
    kind = request.get("kind")
    if kind == "record":
        source = request.get("source")
        if source not in (config.SOURCE_MIC, config.SOURCE_SYSTEM):
            raise JobError("choose in class or online")
        argv = ["record", "--source", source, "--plain"]
        title = _clean(request.get("title"))
        if title:
            argv += ["--title", title]
        try:
            minutes = int(request.get("minutes") or 0)
        except (TypeError, ValueError):
            raise JobError("the class length must be a whole number of minutes")
        if minutes < 0:
            raise JobError("the class length cannot be negative")
        if minutes:
            argv += ["--minutes", str(minutes)]
        argv += ["--later"] if request.get("later") else ["--live-notes"]
        lang = request.get("lang")
        if lang and lang != "auto":
            if lang not in config.supported_languages():
                raise JobError(f"no notes language called {lang!r}")
            argv += ["--lang", lang]
        return kind, argv, title or "Class"

    if kind == "video":
        from ..video import clean_pasted, parse_time

        source = clean_pasted(str(request.get("source") or ""))
        if not source:
            raise JobError("paste a link or the path to a video file")
        argv = ["video", source]
        title = _clean(request.get("title"))
        if title:
            argv += ["--title", title]
        for flag, key in (("--from", "start"), ("--to", "end")):
            value = _clean(request.get(key))
            if value:
                try:
                    parse_time(value)
                except ValueError as exc:
                    raise JobError(f"{key} time: {exc}")
                argv += [flag, value]
        if request.get("later"):
            argv.append("--later")
        return kind, argv, title or "Video"

    if kind == "summarize":
        session = store.get_session(str(request.get("session") or ""))
        if session is None:
            raise JobError("that class no longer exists")
        argv = ["summarize", session.id]
        if request.get("rerun"):
            argv.append("--rerun")
        return kind, argv, session.title

    if kind == "catchup":
        return kind, ["catchup"], "Write up waiting classes"

    if kind == "check":
        return kind, ["check"], "Check this computer"

    raise JobError("unknown action")


class Job:
    """One CLI command running in the background, and what it has printed."""

    def __init__(self, kind: str, argv: list[str], label: str, request: dict) -> None:
        self.id = f"{int(time.time() * 1000):x}"
        self.kind = kind
        self.argv = argv
        self.label = label
        self.request = request
        self.started = time.time()
        self.log: deque[str] = deque(maxlen=LOG_LINES)
        self.status = "running"
        self.returncode: int | None = None
        self.stop_requested = False
        self.session_id: str | None = (
            request.get("session") if kind == "summarize" else None
        )
        tmp = Path(tempfile.gettempdir())
        self._stop_file = tmp / f"notetaker-stop-{self.id}"
        self._alive_file = tmp / f"notetaker-alive-{self.id}"
        self._proc: subprocess.Popen | None = None

    # ---------------------------------------------------------------- run
    def start(self) -> None:
        env = dict(os.environ)
        env.update(
            PYTHONUNBUFFERED="1",
            PYTHONIOENCODING="utf-8",
            # Rich wraps at 80 columns when it is not in a terminal, which
            # breaks paths and links in the log. A wide page reads better.
            COLUMNS="160",
            NO_COLOR="1",
            NOTETAKER_STOP_FILE=str(self._stop_file),
            NOTETAKER_ALIVE_FILE=str(self._alive_file),
            # A job is not the place to update the app under itself.
            NOTES_AUTO_UPDATE="0",
        )
        # The child is kept apart from the terminal, so closing that window
        # (or pressing Ctrl-C in it) cannot kill a recording mid-write. The
        # window stops it instead, the careful way, or the heartbeat does.
        kwargs: dict = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            )
        else:
            kwargs["start_new_session"] = True
        self._alive_file.touch()
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "notetaker.cli", *self.argv],
            cwd=APP_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **kwargs,
        )
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def _heartbeat(self) -> None:
        while self.status == "running":
            try:
                self._alive_file.touch()
            except OSError:
                pass
            time.sleep(HEARTBEAT_SECONDS)

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.rstrip()
            if line:
                self.log.append(line)
        self.returncode = self._proc.wait()
        self._stop_file.unlink(missing_ok=True)
        self._alive_file.unlink(missing_ok=True)
        # Before the status changes: the window opens this class the moment
        # it sees the job finish.
        self._find_session()
        if self.stop_requested and self.kind != "record":
            self.status = "cancelled"
        else:
            # A recording stopped early is still a success: its notes follow.
            self.status = "done" if self.returncode == 0 else "failed"

    def stop(self) -> None:
        """Stop gently. A recording finishes and writes its notes."""
        if self.status != "running" or not self._proc:
            return
        self.stop_requested = True
        if self.kind == "record":
            # The same path as Ctrl-C: ffmpeg closes the file properly, the
            # last chunk is transcribed and the notes are still written.
            self._stop_file.touch()
            return
        if IS_WINDOWS:
            self._proc.terminate()
        else:
            self._proc.send_signal(signal.SIGINT)

    # ------------------------------------------------------------ status
    def _find_session(self) -> None:
        if self.session_id:
            return
        if self.kind not in ("record", "video"):
            return
        # The CLI names the class it made (or already had) in its output.
        for line in reversed(self.log):
            match = re.search(r"notes show (\S+)|· (\d{4}-\d{2}-\d{2}_\S+)$", line)
            if match:
                found = match.group(1) or match.group(2)
                if store.get_session(found):
                    self.session_id = found
                    return
        # Otherwise, the class created after this job started.
        since = datetime.fromtimestamp(self.started - 2).isoformat(timespec="seconds")
        for session in store.list_sessions(limit=5):
            if session.started_at >= since:
                self.session_id = session.id
                return

    def to_json(self) -> dict:
        if self.status == "running" and self.kind in ("record", "video"):
            self._find_session()
        heard: list[str] = []
        if self.session_id and self.kind == "record" and self.status == "running":
            session = store.get_session(self.session_id)
            if session:
                heard = [s.text for s in store.load_transcript(session)[-4:]]
        minutes = int(self.request.get("minutes") or 0) if self.kind == "record" else 0
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "stopping": self.stop_requested and self.status == "running",
            "started": self.started,
            "elapsed": time.time() - self.started,
            "limit": minutes * 60 + config.CLASS_OVERRUN_SECONDS if minutes else None,
            "minutes": minutes,
            "later": bool(self.request.get("later")),
            "source": self.request.get("source"),
            "returncode": self.returncode,
            "session": self.session_id,
            "heard": heard,
            "log": list(self.log)[-60:],
            "command": "notes " + " ".join(a for a in self.argv if a != "--plain"),
        }


class Jobs:
    def __init__(self) -> None:
        self.current: Job | None = None
        self._lock = threading.Lock()

    def start(self, request: dict) -> Job:
        kind, argv, label = job_argv(request)
        with self._lock:
            if self.current and self.current.status == "running":
                raise JobError("something is already running. Stop it first.")
            job = Job(kind, argv, label, request)
            job.start()
            self.current = job
            return job

    def stop_all(self) -> None:
        job = self.current
        if job and job.status == "running":
            job.stop()


# --------------------------------------------------------------------------
# Sessions as the window sees them
# --------------------------------------------------------------------------
_ollama_cache = {"at": 0.0, "ok": False}


def ollama_ok() -> bool:
    """Whether notes can be written. Cached: the window asks every few seconds."""
    if time.time() - _ollama_cache["at"] > 15:
        from .. import summarize

        try:
            _ollama_cache["ok"] = summarize.ollama_available()
        except Exception:
            _ollama_cache["ok"] = False
        _ollama_cache["at"] = time.time()
    return bool(_ollama_cache["ok"])


def session_summary(session: store.Session, pending_ids: set[str]) -> dict:
    return {
        "id": session.id,
        "title": session.title,
        "started": session.started_at,
        "duration": session.duration,
        "source": session.source_kind,
        "language": session.language,
        "hasNotes": session.has_notes and session.notes_path.exists(),
        "pending": session.id in pending_ids,
        "complete": session.is_complete,
    }


def session_detail(session: store.Session, pending_ids: set[str]) -> dict:
    data = session_summary(session, pending_ids)
    notes = None
    if session.notes_path.exists():
        notes = session.notes_path.read_text(encoding="utf-8")
    data.update(
        notes=notes,
        hasAudio=session.audio_path.exists(),
        hasTranscript=session.transcript_path.exists()
        and session.transcript_path.stat().st_size > 0,
        partial=session.transcript_is_partial,
        link=None,
    )
    if session.source_kind == config.SOURCE_VIDEO:
        try:
            from .. import video

            data["link"] = video.link_for(video.load_meta(session.directory))
        except Exception:
            pass
    return data


def languages_list() -> list[dict]:
    from .. import languages

    out = []
    for code in config.supported_languages():
        try:
            out.append({"code": code, "name": languages.get(code).name})
        except Exception:
            out.append({"code": code, "name": code})
    return out


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "NoteTaker"
    jobs: Jobs  # set by serve()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # the terminal is for the student, not for request logs

    # ------------------------------------------------------------ guards
    def _host_ok(self) -> bool:
        """Refuse requests addressed to any other name (DNS rebinding)."""
        host = (self.headers.get("Host") or "").lower()
        port = self.server.server_address[1]
        return host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    # ---------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error("forbidden", 403)
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]

        if not parts:
            page = PAGE.read_bytes()
            return self._send(200, page, "text/html; charset=utf-8", {
                "Content-Security-Policy": (
                    "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
                ),
            })
        if parts[0] != "api":
            return self._error("not found", 404)

        route = parts[1:]
        if route == ["ping"]:
            return self._json({"app": "notetaker"})
        if route == ["state"]:
            sessions = store.list_sessions()
            pending = {s.id for s in store.pending_sessions()}
            job = self.jobs.current
            return self._json({
                "sessions": [session_summary(s, pending) for s in sessions],
                "pendingCount": len(pending),
                "ollama": ollama_ok(),
                "job": job.to_json() if job else None,
                "languages": languages_list(),
                "classMinutes": list(config.CLASS_LENGTH_CHOICES),
                "defaultMinutes": config.DEFAULT_CLASS_MINUTES,
            })
        if route == ["job"]:
            job = self.jobs.current
            return self._json({"job": job.to_json() if job else None})
        if len(route) >= 2 and route[0] == "sessions":
            session = store.get_session(route[1])
            if session is None:
                return self._error("that class no longer exists", 404)
            if len(route) == 2:
                pending = {s.id for s in store.pending_sessions()}
                return self._json(session_detail(session, pending))
            if route[2:] == ["transcript"]:
                return self._json([
                    {"start": s.start, "text": s.text} for s in store.load_transcript(session)
                ])
            if route[2:] == ["download"]:
                kind = parse_qs(url.query).get("kind", ["md"])[0]
                if kind == "txt":
                    segments = store.load_transcript(session)
                    if not segments:
                        return self._error("no transcript yet", 404)
                    body, suffix = "\n".join(s.text for s in segments) + "\n", ".txt"
                else:
                    if not session.notes_path.exists():
                        return self._error("no notes yet", 404)
                    body, suffix = session.notes_path.read_text(encoding="utf-8"), ".md"
                # HTTP headers are Latin-1, so a Thai title goes in the UTF-8
                # form; the plain one is only for very old browsers.
                name = f"{store.slugify(session.title)}{suffix}"
                plain = name.encode("ascii", "ignore").decode()
                if not plain.strip("-")[: -len(suffix)]:
                    plain = f"notes{suffix}"
                return self._send(200, body.encode("utf-8"), "text/plain; charset=utf-8", {
                    "Content-Disposition": (
                        f"attachment; filename=\"{plain}\"; filename*=UTF-8''{quote(name)}"
                    ),
                })
        return self._error("not found", 404)

    # --------------------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error("forbidden", 403)
        # A web page on another site cannot set this header without a CORS
        # preflight, which this server never approves. So only this window
        # can start a recording.
        if self.headers.get("X-NoteTaker") != "1":
            return self._error("forbidden", 403)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(length, 65536)) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._error("bad request")

        if self.path == "/api/jobs":
            try:
                job = self.jobs.start(body)
            except JobError as exc:
                return self._error(str(exc), 409 if "already" in str(exc) else 400)
            except OSError as exc:
                return self._error(f"could not start: {exc}", 500)
            return self._json({"job": job.to_json()})
        if self.path == "/api/job/stop":
            job = self.jobs.current
            if job:
                job.stop()
            return self._json({"job": job.to_json() if job else None})
        return self._error("not found", 404)


def _already_running(port: int) -> bool:
    """True when this port already serves a NoteTaker window."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1) as resp:
            return json.loads(resp.read()).get("app") == "notetaker"
    except Exception:
        return False


def make_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("NoteTakerHandler", (Handler,), {"jobs": Jobs()})
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        # The usual port is taken by something else: let the system pick.
        return ThreadingHTTPServer(("127.0.0.1", 0), handler)


def _quietly(echo):
    """An echo that survives its terminal being closed."""
    def say(message: str) -> None:
        try:
            echo(message)
        except Exception:
            pass
    return say


def _close_on(signum, frame) -> None:
    raise KeyboardInterrupt


def serve(port: int = DEFAULT_PORT, open_browser: bool = True, echo=print) -> int:
    echo = _quietly(echo)
    config.ensure_dirs()
    if port and _already_running(port):
        url = f"http://127.0.0.1:{port}/"
        echo(f"NoteTaker is already open at {url}")
        if open_browser:
            webbrowser.open(url)
        return 0

    server = make_server(port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    echo(f"NoteTaker is open at {url}")
    echo("Keep this window open while you use it. Press Ctrl-C to close NoteTaker.")
    echo("Prefer typing? Every notes command still works in a terminal.")
    # Closing the terminal window is the same as Ctrl-C: finish safely.
    for name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), _close_on)
            except ValueError:  # not the main thread (tests)
                pass
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        echo("\nclosing NoteTaker")
    finally:
        job = server.RequestHandlerClass.jobs.current  # type: ignore[attr-defined]
        if job and job.status == "running":
            # Never abandon a recording: stop it the safe way and wait for it
            # to save, exactly as Ctrl-C in `notes class` would.
            echo("finishing the current job first, so nothing is lost...")
            job.stop()
            if job._proc:
                try:
                    job._proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    pass
        server.server_close()
    return 0
