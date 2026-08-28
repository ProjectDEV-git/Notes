"""Key-idea summarization for NoteTaker.

A lecture transcript is far too long to summarize in one shot with a local
model, so this is a map-reduce:

    MAP     each ~3-minute window -> 2-4 key points
    REDUCE  all points -> deduped, ordered notes.md

Administrative asides (deadlines, exam dates) are tagged ADMIN during MAP and
split into their own section, keeping "Key ideas" purely about content.

Notes are produced in the language of the lecture: a Thai lecture yields Thai
notes. See docs/BUILD_PLAN.md phase 7.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime

from . import config
from .asr import Segment, normalize

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK = re.compile(r"</?think>", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
# Internal tag used to route admin asides; must never appear in final notes.
ADMIN_LINE = re.compile(r"ADMIN\s*:\s*", re.IGNORECASE)


class SummarizerError(RuntimeError):
    """Raised when the local model is unreachable or returns nothing usable."""


@dataclass
class Window:
    """A slice of transcript handed to the MAP stage."""

    start: float
    end: float
    text: str

    @property
    def timestamp(self) -> str:
        minutes, seconds = divmod(int(self.start), 60)
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class Notes:
    """Result of a summarization run."""

    markdown: str
    key_points: list[str] = field(default_factory=list)
    admin_points: list[str] = field(default_factory=list)
    language: str = "en"


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def strip_think(text: str) -> str:
    """Remove reasoning blocks that qwen3-style models may emit.

    Requests set think=false, but a stray tag must never reach notes.md.
    """
    return _ORPHAN_THINK.sub("", _THINK_BLOCK.sub("", text)).strip()


_NONE_MARKERS = {
    "none", "n/a", "na", "nothing", "no terms", "none.", "(none)", "-", "—",
    "ไม่มี", "ไม่มี.", "none identified", "not applicable",
}


def _is_empty_marker(line: str) -> bool:
    """True for placeholder text a model writes instead of omitting a section."""
    cleaned = _BULLET.sub("", line.strip()).strip().strip("*_`").lower()
    return cleaned in _NONE_MARKERS


def strip_empty_sections(markdown: str) -> str:
    """Remove headings with no real content, and tidy model artefacts.

    Handles three things models do despite instructions:
      * emitting a heading with nothing under it
      * writing "None" instead of omitting the section
      * leaking the internal 'ADMIN:' tag into the final notes
    """
    lines = []
    for raw in markdown.splitlines():
        line = ADMIN_LINE.sub("", raw)  # drop the internal tag
        if line.strip() and _is_empty_marker(line):
            continue  # placeholder such as "None"
        lines.append(line)

    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("##"):
            # Look ahead for content before the next heading.
            has_content = False
            probe = index + 1
            while probe < len(lines) and not lines[probe].lstrip().startswith("##"):
                if lines[probe].strip():
                    has_content = True
                    break
                probe += 1
            if not has_content:
                index += 1
                continue
        out.append(line)
        index += 1

    # Collapse the blank runs left behind by removed headings.
    cleaned: list[str] = []
    for line in out:
        if not line.strip() and cleaned and not cleaned[-1].strip():
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def parse_bullets(text: str) -> list[str]:
    """Extract bullet lines, ignoring any preamble the model adds."""
    bullets: list[str] = []
    for line in strip_think(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if _BULLET.match(line):
            cleaned = _BULLET.sub("", line).strip()
            # Models sometimes emit "- - point"; strip any repeated marker so
            # the rendered notes do not show a doubled bullet.
            while _BULLET.match(cleaned):
                cleaned = _BULLET.sub("", cleaned).strip()
            if cleaned:
                bullets.append(cleaned)
    return bullets


def dedupe_points(points: list[str], threshold: float = 0.85) -> list[str]:
    """Drop repeated points, keeping the first (usually best-phrased) wording.

    Comparison is character-based so it works for Thai as well as English.
    """
    kept: list[str] = []
    seen: list[str] = []
    for point in points:
        norm = normalize(point)
        if not norm:
            continue
        duplicate = False
        for existing in seen:
            if norm == existing or norm in existing or existing in norm:
                duplicate = True
                break
            shorter, longer = sorted((norm, existing), key=len)
            if len(shorter) / len(longer) >= threshold:
                matches = sum(1 for a, b in zip(shorter, longer) if a == b)
                if matches / len(shorter) >= threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(point.strip())
            seen.append(norm)
    return kept


def build_windows(
    segments: list[Segment],
    window_seconds: int = config.MAP_WINDOW_SECONDS,
) -> list[Window]:
    """Group segments into fixed-duration windows for the MAP stage."""
    if not segments:
        return []

    windows: list[Window] = []
    current: list[Segment] = []
    window_start = segments[0].start

    for seg in segments:
        if current and seg.end - window_start > window_seconds:
            windows.append(
                Window(window_start, current[-1].end, " ".join(s.text for s in current))
            )
            current = []
            window_start = seg.start
        current.append(seg)

    if current:
        windows.append(Window(window_start, current[-1].end, " ".join(s.text for s in current)))
    return windows


# --------------------------------------------------------------------------
# Ollama client
# --------------------------------------------------------------------------
def ollama_available(url: str = config.OLLAMA_URL, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def chat(
    prompt: str,
    model: str = config.SUMMARY_MODEL,
    url: str = config.OLLAMA_URL,
    timeout: int = config.OLLAMA_TIMEOUT,
    max_tokens: int = config.MAP_MAX_TOKENS,
) -> str:
    """Single-turn completion against a local Ollama model.

    num_predict is capped: without it a model that starts rambling generates
    until the HTTP request times out and the whole window is lost.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # Suppress chain-of-thought where the model supports it.
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": max_tokens,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{url}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
    except urllib.error.URLError as exc:
        raise SummarizerError(
            f"cannot reach Ollama at {url} ({exc}). "
            "Start it with 'ollama serve'. Your transcript has been saved."
        ) from exc
    except TimeoutError as exc:
        raise SummarizerError(
            f"Ollama timed out after {timeout}s using model {model!r}. "
            "Reasoning models are slow on CPU; try --model llama3.2:3b. "
            "Your transcript has been saved."
        ) from exc

    if "error" in data:
        raise SummarizerError(f"Ollama error: {data['error']}")

    message = data.get("message", {})
    content = message.get("content", "")
    if not content.strip() and message.get("thinking"):
        # Reasoning models can spend the entire budget thinking and return an
        # empty content field. Surface that clearly instead of silently
        # producing empty notes.
        raise SummarizerError(
            f"model {model!r} returned only reasoning and no answer. "
            "Use a non-reasoning instruct model such as llama3.2:3b."
        )
    return strip_think(content)


# --------------------------------------------------------------------------
# Map-reduce
# --------------------------------------------------------------------------
# Anchored variant: classifies a bullet as administrative during MAP.
ADMIN_PREFIX = re.compile(r"^ADMIN\s*:\s*", re.IGNORECASE)


def map_window(window: Window, language: str, model: str) -> tuple[list[str], list[str]]:
    """Extract key points from one window. Returns (key_points, admin_points)."""
    prompt = config.prompts_for(language)["map"].format(text=window.text)
    bullets = parse_bullets(chat(prompt, model=model))

    key_points: list[str] = []
    admin_points: list[str] = []
    for bullet in bullets:
        if ADMIN_PREFIX.match(bullet):
            admin_points.append(ADMIN_PREFIX.sub("", bullet).strip())
        else:
            key_points.append(bullet)
    return key_points, admin_points


def reduce_points(points: list[str], language: str, model: str) -> str:
    """Consolidate mapped points into the final markdown body."""
    joined = "\n".join(f"- {p}" for p in points)
    prompt = config.prompts_for(language)["reduce"].format(text=joined)
    return strip_think(chat(prompt, model=model, max_tokens=config.REDUCE_MAX_TOKENS))


def summarize_segments(
    segments: list[Segment],
    title: str = "Lecture",
    language: str | None = None,
    model: str = config.SUMMARY_MODEL,
    duration: float | None = None,
    window_seconds: int = config.MAP_WINDOW_SECONDS,
    progress=None,
) -> Notes:
    """Run the full map-reduce over a transcript."""
    if not segments:
        raise SummarizerError("transcript is empty, nothing to summarize")

    language = language or segments[0].lang or "en"
    windows = build_windows(segments, window_seconds)

    key_points: list[str] = []
    admin_points: list[str] = []
    for index, window in enumerate(windows, start=1):
        if progress:
            progress(index, len(windows))
        keys, admins = map_window(window, language, model)
        key_points.extend(keys)
        admin_points.extend(admins)

    if not key_points and not admin_points:
        raise SummarizerError("model produced no key points from this transcript")

    key_points = dedupe_points(key_points)
    admin_points = dedupe_points(admin_points)

    # Short lectures need no second pass: one window is already consolidated.
    if len(windows) > 1:
        body = reduce_points(key_points + [f"ADMIN: {p}" for p in admin_points], language, model)
    else:
        body = _fallback_body(key_points, admin_points, language)

    if not body.strip():
        body = _fallback_body(key_points, admin_points, language)

    body = strip_empty_sections(body)
    markdown = _render(title, body, language, duration, len(segments))
    return Notes(markdown=markdown, key_points=key_points, admin_points=admin_points, language=language)


def _fallback_body(key_points: list[str], admin_points: list[str], language: str) -> str:
    """Deterministic rendering used when the reduce stage adds no value."""
    headings = {
        "th": ("## แนวคิดสำคัญ", "## สิ่งที่ต้องทำ"),
        "en": ("## Key ideas", "## Action items"),
    }
    key_heading, admin_heading = headings.get(language, headings["en"])

    parts = [key_heading]
    parts.extend(f"- {p}" for p in key_points)
    if admin_points:
        parts.append("")
        parts.append(admin_heading)
        parts.extend(f"- {p}" for p in admin_points)
    return "\n".join(parts)


def _render(title: str, body: str, language: str, duration: float | None, segment_count: int) -> str:
    meta = [datetime.now().strftime("%Y-%m-%d")]
    if duration:
        minutes, seconds = divmod(int(duration), 60)
        meta.append(f"{minutes}m {seconds}s")
    meta.append({"th": "ภาษาไทย", "en": "English"}.get(language, language))
    meta.append(f"{segment_count} segments")

    return f"# {title}\n\n*{' · '.join(meta)}*\n\n{body.strip()}\n"


def compression_ratio(transcript: str, notes: str) -> float:
    """Notes length as a fraction of transcript length. Lower is tighter."""
    if not transcript:
        return 0.0
    return len(notes) / len(transcript)
