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
# Prompts, keyed by language. A Thai lecture must produce Thai notes.
# --------------------------------------------------------------------------
PROMPTS: dict[str, dict[str, str]] = {
    "en": {
        "map": (
            "You are a top student taking notes in a university lecture.\n"
            "Below is part of the lecture transcript (speech-to-text, so it may "
            "contain small errors).\n\n"
            "Write the 2-4 most important points from this part.\n\n"
            "Rules:\n"
            "- Output ONLY bullet lines starting with '- '. No preamble, no headings.\n"
            "- Each bullet must make sense on its own, without the transcript.\n"
            "- KEEP specifics: numbers, units, formulas, names, dates, definitions. "
            "'15 kg lifted 1 m gives about 150 J' is useful; 'the lecturer discussed "
            "energy' is not.\n"
            "- Write what was TAUGHT, not what the lecturer did. Never start a bullet "
            "with 'The lecturer', 'The speaker', or 'The professor'.\n"
            "- Skip greetings, jokes, tangents, repetition, and technical difficulties.\n"
            "- If a term is defined, write it as '**term** — definition'.\n"
            "- Prefix deadlines, exam dates, homework, and office hours with 'ADMIN: '.\n"
            "- If this part has no real content, output nothing at all.\n\n"
            "Transcript:\n{text}"
        ),
        "reduce": (
            "Below are notes taken during one university lecture.\n"
            "Tidy them into final revision notes. This is an EDITING task: you are "
            "reorganising existing lines, not writing new ones.\n\n"
            "Use ONLY these headings:\n"
            "## Key ideas\n"
            "## Terms & definitions\n"
            "## Action items\n\n"
            "Rules:\n"
            "- Every bullet you output must come from the lines below. Copy their "
            "wording, including all numbers, units and formulas, exactly.\n"
            "- NEVER invent a point. If something is not in the lines below, it must "
            "not appear. Adding 'review the slides' or similar is a serious error.\n"
            "- Drop vague filler that says nothing specific, for example 'physics is "
            "important' or 'this is a fundamental concept'.\n"
            "- Where two lines say the same thing, keep the more specific one.\n"
            "- Lines marked ADMIN go under 'Action items' with the ADMIN prefix removed.\n"
            "- Lines shaped like '**term** — definition' go under 'Terms & definitions'.\n"
            "- Everything else goes under 'Key ideas', ordered so the lecture flows.\n"
            "- OMIT any heading with nothing to put under it. Never write 'None'.\n\n"
            "Lines:\n{text}"
        ),
    },
    "th": {
        "map": (
            "คุณเป็นนักศึกษาที่จดโน้ตเก่งที่สุดในห้องเรียนมหาวิทยาลัย\n"
            "ด้านล่างคือบทถอดเสียงบางส่วนของการบรรยาย "
            "(ถอดด้วยระบบอัตโนมัติ จึงอาจมีคำผิดบ้าง)\n\n"
            "เขียนประเด็นสำคัญที่สุด 2-4 ข้อจากส่วนนี้\n\n"
            "กฎ:\n"
            "- ตอบเป็นบรรทัดบูลเล็ตขึ้นต้นด้วย '- ' เท่านั้น ห้ามมีคำนำหรือหัวข้อ\n"
            "- แต่ละข้อต้องเข้าใจได้ด้วยตัวเอง โดยไม่ต้องอ่านบทถอดเสียง\n"
            "- คงรายละเอียดสำคัญไว้: ตัวเลข หน่วย สูตร ชื่อ วันที่ และคำนิยาม\n"
            "- เขียนสิ่งที่ 'สอน' ไม่ใช่สิ่งที่ผู้สอน 'ทำ' "
            "ห้ามขึ้นต้นข้อด้วย 'ผู้สอน' หรือ 'อาจารย์'\n"
            "- ข้ามคำทักทาย มุกตลก เรื่องนอกประเด็น และการพูดซ้ำ\n"
            "- ถ้ามีการให้นิยามศัพท์ ให้เขียนว่า '**ศัพท์** — นิยาม'\n"
            "- ถ้าเป็นกำหนดส่งงาน วันสอบ การบ้าน หรือเวลาเข้าพบอาจารย์ "
            "ให้ขึ้นต้นข้อนั้นด้วย 'ADMIN: '\n"
            "- ถ้าส่วนนี้ไม่มีเนื้อหาสาระ ไม่ต้องตอบอะไรเลย\n\n"
            "บทถอดเสียง:\n{text}"
        ),
        "reduce": (
            "ด้านล่างคือโน้ตที่จดไว้ระหว่างการบรรยายหนึ่งครั้ง\n"
            "จัดระเบียบให้เป็นโน้ตสรุปฉบับสมบูรณ์ "
            "งานนี้คือการ 'จัดเรียง' บรรทัดที่มีอยู่ ไม่ใช่การเขียนขึ้นใหม่\n\n"
            "ใช้หัวข้อเหล่านี้เท่านั้น:\n"
            "## แนวคิดสำคัญ\n"
            "## คำศัพท์และนิยาม\n"
            "## สิ่งที่ต้องทำ\n\n"
            "กฎ:\n"
            "- ทุกข้อที่ตอบต้องมาจากบรรทัดด้านล่างเท่านั้น "
            "คัดลอกข้อความเดิม รวมทั้งตัวเลข หน่วย และสูตร ให้ตรงตามเดิม\n"
            "- ห้ามแต่งข้อใหม่เด็ดขาด ถ้าไม่มีอยู่ในบรรทัดด้านล่าง ห้ามใส่\n"
            "- ตัดข้อความกว้างๆ ที่ไม่ได้บอกอะไรเจาะจงทิ้ง เช่น 'ฟิสิกส์สำคัญมาก'\n"
            "- ถ้าสองบรรทัดมีใจความเดียวกัน ให้เก็บบรรทัดที่เจาะจงกว่า\n"
            "- บรรทัดที่มี ADMIN ให้อยู่ใต้ 'สิ่งที่ต้องทำ' โดยตัดคำว่า ADMIN ออก\n"
            "- บรรทัดรูปแบบ '**ศัพท์** — นิยาม' ให้อยู่ใต้ 'คำศัพท์และนิยาม'\n"
            "- ที่เหลือให้อยู่ใต้ 'แนวคิดสำคัญ' เรียงให้เนื้อหาต่อเนื่อง\n"
            "- หัวข้อใดไม่มีเนื้อหา ให้ตัดทิ้ง ห้ามเขียนว่า 'ไม่มี'\n\n"
            "บรรทัด:\n{text}"
        ),
    },
}


def prompts_for(language: str | None) -> dict[str, str]:
    """Return the prompt pair for a language, falling back to English."""
    return PROMPTS.get(language or "en", PROMPTS["en"])
