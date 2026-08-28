"""Central configuration for NoteTaker.

Every tunable lives here. Do not scatter magic numbers through other modules.
See docs/BUILD_PLAN.md for the reasoning behind these choices.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (XDG-aware). Recordings live OUTSIDE the repo and are never committed.
# --------------------------------------------------------------------------
_xdg_data = os.environ.get("XDG_DATA_HOME")
DATA_DIR = Path(_xdg_data).expanduser() / "notetaker" if _xdg_data else Path.home() / ".local" / "share" / "notetaker"
SESSIONS_DIR = DATA_DIR / "sessions"
DB_PATH = DATA_DIR / "notetaker.db"


def ensure_dirs() -> None:
    """Create the data directories if they do not exist."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


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

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

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
ASR_DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # fast CPU path for ctranslate2
CPU_THREADS = 8  # of 12 available; leave headroom for capture + UI

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
SUMMARY_MODEL = os.environ.get("NOTETAKER_SUMMARY_MODEL", "llama3.2:3b")
TEST_MODEL = os.environ.get("NOTETAKER_TEST_MODEL", "llama3.2:3b")

# Hard cap on generated tokens per call. Without it a model that starts
# rambling runs until the request times out.
MAP_MAX_TOKENS = 300
REDUCE_MAX_TOKENS = 700

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
def prompts_for(language: str | None) -> dict[str, str]:
    """Return the map and reduce prompts for a language, falling back to English."""
    from . import languages

    lang = languages.get(language)
    return {"map": lang.map_prompt, "reduce": lang.reduce_prompt}
