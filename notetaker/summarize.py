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

from . import config, languages
from .asr import Segment, normalize

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK = re.compile(r"</?think>", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
# Internal tag used to route admin asides; must never appear in final notes.
ADMIN_LINE = re.compile(r"ADMIN\s*:\s*", re.IGNORECASE)

# Bullets that narrate the lesson instead of recording what was taught.
# Every prompt forbids these, and small models produce them anyway: "the
# teacher talked about X" is worthless in revision notes, because it says a
# topic came up without saying anything about it. Enforced here rather than
# trusted to the prompt.
NARRATION = re.compile(
    r"^\s*(?:the\s+)?"
    r"(?:teacher|lecturer|professor|speaker|instructor|tutor|class|lesson)\b"
    r"\s*(?:also\s+)?"
    r"(?:talked|spoke|discussed|mentioned|explained|said|told|covered|went|"
    r"introduced|described|noted|referred|reminded)\b",
    re.IGNORECASE,
)

# The same failure in Thai, which the Thai prompts forbid in the same words.
# Thai is written without spaces, so this cannot be word-anchored: it matches a
# speaker noun at the very start of the bullet followed closely by a speech
# verb. Requiring both, adjacent, keeps it from eating a sentence that merely
# mentions a teacher.
NARRATION_TH = re.compile(
    r"^\s*(?:ผู้สอน|อาจารย์|ครู|วิทยากร|ผู้บรรยาย)"
    r".{0,12}?"
    r"(?:พูดถึง|กล่าวถึง|อธิบาย|บอกว่า|เล่าถึง|บรรยาย|กล่าว|บอก|สอนว่า)"
)


def is_narration(text: str) -> bool:
    """True for a bullet that reports the lesson happening, not its content.

    Checked for every language, because the transcript language is not always
    known at the point a bullet is parsed, and a false positive costs one
    bullet while a false negative costs the reader's trust.
    """
    return bool(NARRATION.match(text) or NARRATION_TH.match(text))
# Punctuation stripped for grounding checks, but digits and units are kept so a
# bullet citing "150 J" stays traceable to its source line.
_PUNCT_KEEP = re.compile(r"[^\w\s]", re.UNICODE)


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


def _is_empty_marker(line: str) -> bool:
    """True for placeholder text a model writes instead of omitting a section."""
    cleaned = _BULLET.sub("", line.strip()).strip().strip("*_`").lower()
    return cleaned in languages.none_markers()


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
            if cleaned and not is_narration(cleaned):
                bullets.append(cleaned)
    return bullets


def _stem(word: str) -> str:
    """Crude suffix stripping so 'lifted'/'lift' and 'converts'/'convert' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    """Distinctive words of a line, for grounding checks.

    Short words are dropped because they are shared by almost any sentence.
    Words are stemmed so paraphrase is not punished. Scripts written without
    word spaces (Thai, Japanese, Chinese...) fall back to character n-grams,
    using the ranges declared by the installed language packs.
    """
    lowered = _PUNCT_KEEP.sub(" ", text.lower())
    words = {_stem(w) for w in lowered.split() if len(w) > 3}
    for low, high in languages.unspaced_ranges():
        script = "".join(ch for ch in lowered if low <= ch <= high)
        if script:
            words |= {script[i:i + 4] for i in range(len(script) - 3)}
    return words


def _numbers(text: str) -> set[str]:
    """Digit sequences, which are strong evidence a bullet came from the source."""
    return set(re.findall(r"\d+", text))


def drop_ungrounded(
    bullets: list[str],
    sources: list[str],
    threshold: float = config.GROUNDING_THRESHOLD,
) -> list[str]:
    """Remove bullets that the source text does not support.

    A small model handed a short or vague excerpt will confidently invent a
    whole lecture around the topic it thinks it heard. Every bullet must be
    traceable to what was actually said.

    Two signals are used:
      * shared stemmed content words, which tolerates paraphrase
      * shared numbers, which are near-impossible to produce by chance and so
        immediately accept a bullet such as "15 kg lifted 1 m gives 150 J"
    """
    if not sources:
        return bullets

    vocabulary: set[str] = set()
    source_numbers: set[str] = set()
    for source in sources:
        vocabulary |= _content_words(source)
        source_numbers |= _numbers(source)

    kept: list[str] = []
    for bullet in bullets:
        words = _content_words(bullet)
        if not words:
            continue

        digits = _numbers(bullet)
        shared_digits = digits & source_numbers
        if shared_digits:
            # Quotes a figure that really was said. Distinctive enough to trust,
            # even if the bullet also writes "1 m" where the speaker said
            # "one meter".
            kept.append(bullet)
            continue

        if digits and not shared_digits and len(digits) > 1:
            # Several numbers, none of which appear in the source: fabricated.
            continue

        if len(words & vocabulary) / len(words) >= threshold:
            kept.append(bullet)
    return kept


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


def installed_models(url: str = config.OLLAMA_URL, timeout: float = 3.0) -> list[str]:
    """Names of models Ollama has locally.

    Used by the setup check: an unreachable Ollama and a missing model look the
    same at record time (no notes), but the fixes are different.
    """
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    return [entry.get("name", "") for entry in data.get("models", [])]


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


def map_window(
    window: Window,
    language: str,
    model: str,
    grounding_source: str | None = None,
    level: str | None = None,
) -> tuple[list[str], list[str]]:
    """Extract key points from one window. Returns (key_points, admin_points).

    Points are grounded against the transcript, because a small model handed a
    short or vague excerpt will confidently invent a whole lecture around the
    topic it thinks it heard. `grounding_source` defaults to this window, but
    callers pass the whole transcript so a point that legitimately draws on
    nearby context is not discarded.
    """
    prompt = config.prompts_for(language, level)["map"].format(text=window.text)
    bullets = parse_bullets(chat(prompt, model=model))

    key_points: list[str] = []
    admin_points: list[str] = []
    for bullet in bullets:
        if ADMIN_PREFIX.match(bullet):
            admin_points.append(ADMIN_PREFIX.sub("", bullet).strip())
        else:
            key_points.append(bullet)

    source = [grounding_source or window.text]
    return (
        drop_ungrounded(key_points, source),
        drop_ungrounded(admin_points, source),
    )


def reduce_points(points: list[str], language: str, model: str, level: str | None = None) -> str:
    """Consolidate mapped points into the final markdown body.

    A 40-minute lecture yields perhaps 20 points and reduces in one pass. A full
    school day period yields far more, and asking for all of them back at once
    silently truncates at REDUCE_MAX_TOKENS: the model runs out of budget
    mid-sentence and the end of the class is simply missing from the notes.

    Past a threshold the reduce is therefore done in batches, each small enough
    to answer in full, and the batch outputs are reduced once more. Every input
    point passes through exactly one batch, so nothing is dropped by the split
    itself.
    """
    if len(points) <= config.REDUCE_BATCH_POINTS:
        return _reduce_once(points, language, model, level)

    batches = [
        points[i:i + config.REDUCE_BATCH_POINTS]
        for i in range(0, len(points), config.REDUCE_BATCH_POINTS)
    ]

    # Each batch comes back as markdown; strip it to bullets so the final pass
    # sees the same shape of input as a single-pass reduce would.
    partial: list[str] = []
    for batch in batches:
        bullets = parse_bullets(_reduce_once(batch, language, model, level))
        # A batch that returns nothing usable must not silently delete its
        # points; fall back to the originals so content survives.
        partial.extend(bullets or batch)

    partial = dedupe_points(partial)
    if len(partial) <= config.REDUCE_BATCH_POINTS:
        return _reduce_once(partial, language, model, level)

    # Still too many after one pass (a very long class). Recurse only while the
    # list is actually shrinking; a model that echoes its input back unchanged
    # must not spin forever, so fall through to a single pass instead.
    if len(partial) < len(points):
        return reduce_points(partial, language, model, level)
    return _reduce_once(partial[: config.REDUCE_BATCH_POINTS], language, model, level)


def _reduce_once(points: list[str], language: str, model: str, level: str | None = None) -> str:
    """One REDUCE call over a list of points that fits in the token budget."""
    joined = "\n".join(f"- {p}" for p in points)
    prompt = config.prompts_for(language, level)["reduce"].format(text=joined)
    return strip_think(chat(prompt, model=model, max_tokens=config.REDUCE_MAX_TOKENS))


def apply_grounding(markdown: str, sources: list[str]) -> str:
    """Clean the reduced notes: drop invented bullets and repeated ones.

    Two failure modes are handled:
      * hallucination - a bullet no mapped point supports
      * restatement   - the same idea emitted twice in different words, which
                        dedupe_points cannot catch because it runs before reduce
    """
    out: list[str] = []
    seen: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if _BULLET.match(stripped):
            text = _BULLET.sub("", stripped).strip()
            if is_narration(text):
                continue  # says a topic came up, not what was taught about it
            if not drop_ungrounded([text], sources):
                continue  # nobody said this
            if len(dedupe_points(seen + [text])) == len(seen):
                continue  # already said, in other words
            seen.append(text)
        out.append(line)
    return "\n".join(out)


def drop_unbacked_actions(markdown: str, admin_points: list[str]) -> str:
    """Remove an Action items section when no ADMIN point was ever extracted.

    Deadlines and exam dates are only trustworthy if the lecturer actually said
    them. If the map stage found none, anything the model puts under Action
    items is invented, and an invented deadline is worse than no deadline.
    """
    if admin_points:
        return markdown

    action_headings = languages.action_headings()
    out: list[str] = []
    skipping = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            heading = stripped.lstrip("#").strip().lower()
            skipping = any(name in heading for name in action_headings)
            if skipping:
                continue
        if skipping:
            continue
        out.append(line)
    return "\n".join(out)


def summarize_segments(
    segments: list[Segment],
    title: str = "Lecture",
    language: str | None = None,
    model: str = config.SUMMARY_MODEL,
    duration: float | None = None,
    window_seconds: int = config.MAP_WINDOW_SECONDS,
    progress=None,
    premapped: tuple[list[str], list[str], int] | None = None,
    level: str | None = None,
) -> Notes:
    """Run the map-reduce over a transcript.

    `premapped` is `(key_points, admin_points, segments_covered)` from the
    live-notes thread, which already MAPped most of the class while it was
    being taught. Reusing it means only the uncovered tail is mapped after the
    bell, so an hour-long class produces notes in a couple of minutes instead
    of re-doing twenty model calls the student already paid for.
    """
    level = level or config.NOTES_LEVEL
    if not segments:
        raise SummarizerError("transcript is empty, nothing to summarize")

    language = language or segments[0].lang or "en"

    key_points: list[str] = []
    admin_points: list[str] = []
    covered = 0
    if premapped:
        live_keys, live_admins, covered = premapped
        # Guard against a stale count: never skip more than we actually have.
        covered = max(0, min(covered, len(segments)))
        key_points.extend(live_keys)
        admin_points.extend(live_admins)

    # Only the segments the live pass never saw still need mapping.
    windows = build_windows(segments[covered:], window_seconds)

    # Ground against the whole transcript: a point made in one window often
    # draws on wording from an adjacent one, and should not be discarded.
    full_transcript = " ".join(s.text for s in segments)
    for index, window in enumerate(windows, start=1):
        if progress:
            progress(index, len(windows))
        keys, admins = map_window(
            window, language, model, grounding_source=full_transcript, level=level
        )
        key_points.extend(keys)
        admin_points.extend(admins)

    if not key_points and not admin_points:
        # Usually means the recording caught noise, music or silence rather
        # than teaching, or every point was rejected as ungrounded. The
        # transcript is intact either way, so point at it.
        raise SummarizerError(
            "no key ideas could be found in this recording. "
            "That usually means it captured background noise rather than a "
            "lesson. Your transcript is saved: check it with "
            "'notes show <id> --transcript'."
        )

    key_points = dedupe_points(key_points)
    admin_points = dedupe_points(admin_points)

    # One window and nothing reused is already consolidated; a second pass adds
    # nothing. Anything larger, including reused live points, needs reducing.
    if len(windows) + (1 if covered else 0) > 1:
        body = reduce_points(
            key_points + [f"ADMIN: {p}" for p in admin_points], language, model, level
        )
        # The reduce stage may invent plausible-sounding points that nobody
        # said. Keep only what the mapped points actually support.
        body = apply_grounding(body, key_points + admin_points)
        body = drop_unbacked_actions(body, admin_points)
    else:
        body = _fallback_body(key_points, admin_points, language, level)

    if not body.strip():
        body = _fallback_body(key_points, admin_points, language, level)

    body = strip_empty_sections(body)
    markdown = _render(title, body, language, duration, len(segments))
    return Notes(markdown=markdown, key_points=key_points, admin_points=admin_points, language=language)


def _fallback_body(
    key_points: list[str],
    admin_points: list[str],
    language: str,
    level: str | None = None,
) -> str:
    """Deterministic rendering used when the reduce stage adds no value."""
    lang = languages.get(language)

    parts = [lang.heading_for_key_ideas(level or config.NOTES_LEVEL)]
    parts.extend(f"- {p}" for p in key_points)
    if admin_points:
        parts.append("")
        parts.append(lang.heading_for_actions(level or config.NOTES_LEVEL))
        parts.extend(f"- {p}" for p in admin_points)
    return "\n".join(parts)


def _render(title: str, body: str, language: str, duration: float | None, segment_count: int) -> str:
    meta = [datetime.now().strftime("%Y-%m-%d")]
    if duration:
        minutes, seconds = divmod(int(duration), 60)
        meta.append(f"{minutes}m {seconds}s")
    meta.append(languages.get(language).name)
    meta.append(f"{segment_count} segments")

    return f"# {title}\n\n*{' · '.join(meta)}*\n\n{body.strip()}\n"


def compression_ratio(transcript: str, notes: str) -> float:
    """Notes length as a fraction of transcript length. Lower is tighter."""
    if not transcript:
        return 0.0
    return len(notes) / len(transcript)
