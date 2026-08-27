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

LANGUAGE = None  # None = autodetect (en/th). Override with --lang.
SUPPORTED_LANGUAGES = ("en", "th")

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

MAP_WINDOW_SECONDS = 180  # group transcript into ~3 min windows for the MAP stage
LIVE_NOTES_INTERVAL_SECONDS = 180  # how often live incremental notes refresh
OLLAMA_TIMEOUT = 300  # seconds; an 8B model on CPU is not fast

# qwen3 is a reasoning model and may emit <think>...</think>. Always strip it.
STRIP_THINK_BLOCKS = True

# --------------------------------------------------------------------------
# Prompts, keyed by language. A Thai lecture must produce Thai notes.
# --------------------------------------------------------------------------
PROMPTS: dict[str, dict[str, str]] = {
    "en": {
        "map": (
            "You are taking notes on a university lecture.\n"
            "Below is one segment of the transcript.\n\n"
            "Extract only the 2-4 most important ideas from this segment.\n"
            "Rules:\n"
            "- Output plain bullet points starting with '- '. No preamble, no headings.\n"
            "- Each bullet must be a complete, self-contained idea.\n"
            "- Ignore filler, greetings, tangents, and repetition.\n"
            "- If the segment contains administrative info (deadlines, exam dates, "
            "office hours), prefix that bullet with 'ADMIN: '.\n"
            "- If the segment contains no substantive content, output nothing.\n\n"
            "Transcript segment:\n{text}"
        ),
        "reduce": (
            "You are consolidating notes from a university lecture.\n"
            "Below are key points extracted from consecutive segments of the lecture.\n\n"
            "Produce the final notes in Markdown with exactly these sections:\n"
            "## Key ideas\n"
            "## Terms & definitions\n"
            "## Action items\n\n"
            "Rules:\n"
            "- 'Key ideas' must be tight bullets covering the substance of the lecture. "
            "Merge duplicates and near-duplicates. Order them logically.\n"
            "- 'Terms & definitions' lists any technical term that was defined, as "
            "'**term** — definition'. Omit the section if there are none.\n"
            "- 'Action items' collects everything marked ADMIN (deadlines, exams, "
            "assignments). Omit the section if there are none.\n"
            "- Be concise. This is a summary, not a retelling. No filler.\n\n"
            "Extracted points:\n{text}"
        ),
    },
    "th": {
        "map": (
            "คุณกำลังจดโน้ตจากการบรรยายในมหาวิทยาลัย\n"
            "ด้านล่างนี้คือบทถอดเสียงหนึ่งช่วง\n\n"
            "สรุปเฉพาะแนวคิดสำคัญที่สุด 2-4 ข้อจากช่วงนี้\n"
            "กฎ:\n"
            "- ตอบเป็นบูลเล็ตขึ้นต้นด้วย '- ' เท่านั้น ห้ามมีคำนำหรือหัวข้อ\n"
            "- แต่ละข้อต้องเป็นใจความสมบูรณ์ในตัวเอง\n"
            "- ข้ามคำพูดเยิ่นเย้อ การทักทาย เรื่องนอกประเด็น และการพูดซ้ำ\n"
            "- ถ้าเป็นข้อมูลด้านธุรการ (กำหนดส่งงาน วันสอบ เวลาเข้าพบอาจารย์) "
            "ให้ขึ้นต้นข้อนั้นด้วย 'ADMIN: '\n"
            "- ถ้าช่วงนี้ไม่มีเนื้อหาสาระ ไม่ต้องตอบอะไรเลย\n\n"
            "บทถอดเสียง:\n{text}"
        ),
        "reduce": (
            "คุณกำลังรวบรวมโน้ตจากการบรรยายในมหาวิทยาลัย\n"
            "ด้านล่างนี้คือประเด็นสำคัญที่สกัดมาจากแต่ละช่วงของการบรรยาย\n\n"
            "จัดทำโน้ตฉบับสมบูรณ์เป็น Markdown โดยมีหัวข้อเหล่านี้:\n"
            "## แนวคิดสำคัญ\n"
            "## คำศัพท์และนิยาม\n"
            "## สิ่งที่ต้องทำ\n\n"
            "กฎ:\n"
            "- 'แนวคิดสำคัญ' ต้องกระชับ ครอบคลุมสาระของการบรรยาย "
            "รวมข้อที่ซ้ำหรือใกล้เคียงกันเข้าด้วยกัน และเรียงลำดับให้สมเหตุสมผล\n"
            "- 'คำศัพท์และนิยาม' ระบุศัพท์เทคนิคที่มีการให้นิยาม ในรูปแบบ "
            "'**คำศัพท์** — นิยาม' ถ้าไม่มีให้ตัดหัวข้อนี้ออก\n"
            "- 'สิ่งที่ต้องทำ' รวบรวมทุกข้อที่ทำเครื่องหมาย ADMIN ไว้ "
            "ถ้าไม่มีให้ตัดหัวข้อนี้ออก\n"
            "- ต้องกระชับ นี่คือบทสรุป ไม่ใช่การเล่าซ้ำ\n\n"
            "ประเด็นที่สกัดได้:\n{text}"
        ),
    },
}


def prompts_for(language: str | None) -> dict[str, str]:
    """Return the prompt pair for a language, falling back to English."""
    return PROMPTS.get(language or "en", PROMPTS["en"])
