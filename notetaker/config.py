"""Central configuration for NoteTaker.

Every tunable lives here. Do not scatter magic numbers through other modules.
See docs/BUILD_PLAN.md for the reasoning behind these choices.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
# The choices a student makes in the NoteTaker window's Settings page. They
# live in one small JSON file beside the language packs, and the terminal
# reads the same file, so `notes class` and the window always agree.
#
# An environment variable always wins over the file, so every setup that
# already uses one keeps working exactly as before. A missing, broken or
# hand-edited file never stops a class being recorded: anything unusable is
# ignored and the built-in default is used instead.
SETTINGS_PATH = Path(
    os.environ.get("NOTETAKER_SETTINGS_FILE")
    or Path.home() / ".config" / "notetaker" / "settings.json"
)

# setting -> the environment variable that overrides it
SETTINGS_ENV = {
    "class_minutes": "NOTES_CLASS_MINUTES",
    "notes_level": "NOTETAKER_NOTES_LEVEL",
    "notes_language": "NOTETAKER_NOTES_LANGUAGE",
    "summary_model": "NOTETAKER_SUMMARY_MODEL",
    "device": "NOTETAKER_DEVICE",
    "auto_update": "NOTES_AUTO_UPDATE",
}

SETTINGS_DEFAULTS: dict = {
    "class_minutes": 60,
    "notes_level": "school",
    "notes_language": "",  # empty = the language spoken in the class
    "summary_model": "llama3.2:3b",
    "device": "auto",
    "auto_update": True,
}

DEVICES = ("auto", "cpu", "cuda")
_MODEL_NAME = re.compile(r"^[\w.:/+-]{1,100}$")
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{1,8})?$")


def check_setting(key: str, value):
    """The value to store for a setting, or ValueError saying what is wrong.

    Used both for what the window sends and for what is read back from the
    file, so a hand-edited typo is caught the same way.
    """
    if key == "class_minutes":
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError("the class length must be a number of minutes")
        try:
            minutes = int(value)
        except ValueError:
            raise ValueError("the class length must be a number of minutes") from None
        if not 1 <= minutes <= 600:
            raise ValueError("the class length must be between 1 and 600 minutes")
        return minutes
    if key == "notes_level":
        if value not in ("school", "university"):
            raise ValueError("notes are written for school or university")
        return value
    if key == "notes_language":
        if value in (None, ""):
            return ""
        if not isinstance(value, str) or not _LANGUAGE_CODE.match(value):
            raise ValueError("that is not a language code")
        return value
    if key == "summary_model":
        if not isinstance(value, str) or not _MODEL_NAME.match(value.strip()):
            raise ValueError("that is not an Ollama model name, e.g. llama3.2:3b")
        return value.strip()
    if key == "device":
        if value not in DEVICES:
            raise ValueError("choose auto, cpu or cuda")
        return value
    if key == "auto_update":
        if not isinstance(value, bool):
            raise ValueError("updates are either on or off")
        return value
    raise ValueError(f"there is no setting called {key!r}")


def load_settings(path: Path | None = None) -> dict:
    """Saved settings that are still valid. Never raises."""
    try:
        data = json.loads(Path(path or SETTINGS_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    good = {}
    for key, value in data.items():
        try:
            good[key] = check_setting(key, value)
        except ValueError:
            continue
    return good


def save_settings(changes: dict, path: Path | None = None) -> dict:
    """Validate and store changes. None puts a setting back to its default."""
    from .artifacts import write_text_atomic

    path = Path(path or SETTINGS_PATH)
    current = load_settings(path)
    for key, value in changes.items():
        if value is None:
            check_setting(key, SETTINGS_DEFAULTS.get(key))  # rejects unknown keys
            current.pop(key, None)
        else:
            current[key] = check_setting(key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(current, ensure_ascii=False, indent=2) + "\n")
    return current


def effective_settings(path: Path | None = None) -> dict:
    """What each setting is right now: the variable, else the file, else default."""
    saved = load_settings(path)
    result = {}
    for key, default in SETTINGS_DEFAULTS.items():
        value = saved.get(key, default)
        raw = os.environ.get(SETTINGS_ENV[key])
        if raw not in (None, ""):
            if key == "auto_update":
                value = raw != "0"
            else:
                try:
                    value = check_setting(key, raw)
                except ValueError:
                    pass  # an unusable variable is ignored, like the file
        result[key] = value
    return result


_SETTINGS = effective_settings()

# --------------------------------------------------------------------------
# Paths (XDG-aware). Recordings live OUTSIDE the repo and are never committed.
# --------------------------------------------------------------------------
_xdg_data = os.environ.get("XDG_DATA_HOME")
if _xdg_data:
    DATA_DIR = Path(_xdg_data).expanduser() / "notetaker"
elif sys.platform == "win32":
    # %LOCALAPPDATA% stays on this PC: OneDrive does not sync it, so an hour
    # of audio is not uploaded behind the student's back.
    DATA_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "NoteTaker"
else:
    DATA_DIR = Path.home() / ".local" / "share" / "notetaker"
SESSIONS_DIR = DATA_DIR / "sessions"
DB_PATH = DATA_DIR / "notetaker.db"


def ensure_dirs() -> None:
    """Create the data directories if they do not exist."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def caller_dir() -> Path:
    """The folder the student ran `notes` from.

    The Mac/Linux launcher changes into the app folder before starting Python,
    so the current directory is not where they were. It passes the original
    one along in NOTES_CALLER_DIR.
    """
    return Path(os.environ.get("NOTES_CALLER_DIR") or Path.cwd())


def user_path(value: str | os.PathLike[str]) -> Path:
    """A path the student typed, resolved against where they typed it."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else caller_dir() / path


# --------------------------------------------------------------------------
# Audio capture
# --------------------------------------------------------------------------
SAMPLE_RATE = 16_000  # Whisper's native rate. Do not change.
CHANNELS = 1
CHUNK_SECONDS = 30  # length of each segment handed to the ASR worker
OVERLAP_SECONDS = 2  # nominal seam allowance; dedupe happens at text level in asr.py

# Friendly aliases so users never type a raw PulseAudio device name.
SOURCE_MIC = "mic"  # in-person lecture
SOURCE_SYSTEM = "system"  # online lecture (Zoom/Teams/YouTube) via .monitor source
SOURCE_VIDEO = "video"  # a YouTube link or a video/audio file, not recorded live

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

# --------------------------------------------------------------------------
# Class length
# --------------------------------------------------------------------------
# A school period is a known length, so recording can stop on its own. The
# student presses one key at the start and does not have to remember to stop
# at the bell. 0 means "record until Ctrl-C", which is what a lecture wants.
# Change it in Settings, or with NOTES_CLASS_MINUTES.
DEFAULT_CLASS_MINUTES = _SETTINGS["class_minutes"]
CLASS_LENGTH_CHOICES = (30, 45, 50, 60, 90)

# Keep recording a little past the scheduled end, because classes overrun and
# the last thing said is usually the homework.
CLASS_OVERRUN_SECONDS = 120

# Grace period (seconds) to let ffmpeg finalize WAV headers after SIGINT
# before escalating to kill. Never SIGKILL first: it corrupts the header.
FFMPEG_STOP_TIMEOUT = 5.0

# --------------------------------------------------------------------------
# ASR (faster-whisper via ctranslate2 — CPU only, no CUDA on this machine)
# --------------------------------------------------------------------------
# MUST be a multilingual model: lectures may be English OR Thai.
# Never use a ".en" variant here, it cannot transcribe Thai.
ASR_MODEL = "small"
ASR_MODEL_HQ = "large-v3-turbo"  # optional post-lecture high-accuracy re-run
# "auto" uses an NVIDIA graphics card when one is present AND its libraries
# pass a test run, otherwise the processor. See notetaker/gpu.py.
# Override with NOTETAKER_DEVICE=cpu (never use the card) or cuda.
ASR_DEVICE = _SETTINGS["device"]
COMPUTE_TYPE = "int8"  # fast CPU path for ctranslate2
GPU_COMPUTE_TYPE = os.environ.get("NOTETAKER_GPU_COMPUTE_TYPE", "float16")
# Leave two cores for capture and the display, so a 4-core school laptop does
# not starve ffmpeg. Override with NOTETAKER_CPU_THREADS.
CPU_THREADS = int(os.environ.get("NOTETAKER_CPU_THREADS") or max(1, (os.cpu_count() or 4) - 2))

LANGUAGE = None  # None = autodetect. Override with --lang.


def supported_languages() -> tuple[str, ...]:
    """Language codes NoteTaker can write notes in.

    Built-in packs plus any the user has added under
    ~/.config/notetaker/languages/. See docs/LANGUAGES.md.
    """
    from . import languages

    return tuple(languages.codes())

# After this many chunks, lock to the majority detected language so a
# mid-lecture misdetection cannot flip the transcript back and forth.
LANGUAGE_LOCK_AFTER_CHUNKS = 3

VAD_FILTER = True
VAD_MIN_SILENCE_MS = 500

# --------------------------------------------------------------------------
# Summarization (local Ollama)
# --------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Model choice is the single biggest quality/speed lever, and this machine has
# no GPU. Measured on the Intel Core 5 120U, summarizing one 3-minute window:
#
#   llama3.2:3b   45s, clean bullets first try, correct Thai      <- default
#   qwen3:4b      >300s timeout; reasoning model, narrates instead
#                 of answering and replies in English to Thai input
#   qwen3:8b      1.6 tok/s, ~25 min for a 50-minute lecture
#   qwen3:0.6b    fast but echoes the transcript instead of summarizing
#
# The qwen3 family is reasoning-first: with think=false it dumps its chain of
# thought into `content`, so it cannot reliably honour an output format.
# Prefer a non-reasoning instruct model here.
# Override with --model or NOTETAKER_SUMMARY_MODEL.
SUMMARY_MODEL = _SETTINGS["summary_model"]
TEST_MODEL = os.environ.get("NOTETAKER_TEST_MODEL", "llama3.2:3b")

# Hard cap on generated tokens per call. Without it a model that starts
# rambling runs until the request times out.
MAP_MAX_TOKENS = 300
REDUCE_MAX_TOKENS = 700

# Above this many mapped points, REDUCE runs in batches and the batch outputs
# are reduced again. A 60-minute class produces roughly 40-60 points, which
# does not fit in REDUCE_MAX_TOKENS: asking for it in one go truncates the
# notes mid-sentence and the end of the class disappears.
REDUCE_BATCH_POINTS = 20

# Fraction of a bullet's distinctive words that must appear in the source text
# for it to be kept. Small models will happily invent a whole lecture from a
# single vague sentence, so every point must be traceable to what was said.
# Lower = more permissive. 0.4 rejects invention while allowing paraphrase.
GROUNDING_THRESHOLD = 0.4

MAP_WINDOW_SECONDS = 180  # group transcript into ~3 min windows for the MAP stage
LIVE_NOTES_INTERVAL_SECONDS = 180  # how often live incremental notes refresh
OLLAMA_TIMEOUT = 300  # seconds; an 8B model on CPU is not fast

# qwen3 is a reasoning model and may emit <think>...</think>. Always strip it.
STRIP_THINK_BLOCKS = True


# --------------------------------------------------------------------------
# Prompts live in language packs, so users can add languages without touching
# code. Built-ins are in notetaker/languages.py; user packs go in
# ~/.config/notetaker/languages/*.json. See docs/LANGUAGES.md.
# --------------------------------------------------------------------------
# Who the notes are written for. A 14-year-old and a postgraduate need the same
# facts written very differently: the school register defines every new term,
# keeps sentences short, and says "homework" rather than "action items".
# Override with --level or NOTETAKER_NOTES_LEVEL.
NOTES_LEVEL = _SETTINGS["notes_level"]
NOTES_LEVELS = ("school", "university")

# The language the notes are written in. Empty = the language spoken in the
# class. Set it to read every class in one language, e.g. Thai notes from
# English lessons. Override with --notes-lang or NOTETAKER_NOTES_LANGUAGE.
NOTES_LANGUAGE = _SETTINGS["notes_language"] or None

# Whether `notes` checks for an update when it starts. Off in Settings, or
# NOTES_AUTO_UPDATE=0. `notes update` always works.
AUTO_UPDATE = _SETTINGS["auto_update"]
# Lines per translation call: enough to keep context, small enough that
# llama3.2:3b answers every line within the token budget.
TRANSLATE_BATCH_LINES = 12
TRANSLATE_MAX_TOKENS = 900

# Practice questions: facts per model call, and its token budget.
QUIZ_BATCH_FACTS = 10
QUIZ_MAX_TOKENS = 700


def prompts_for(language: str | None, level: str | None = None) -> dict[str, str]:
    """Return the map and reduce prompts for a language and reading level.

    A language pack that has no school prompts falls back to its normal ones,
    so every pack a user has already written keeps working untouched.
    """
    from . import languages

    lang = languages.get(language)
    return lang.prompts(level or NOTES_LEVEL)
