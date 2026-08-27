"""Speech recognition for NoteTaker.

Wraps faster-whisper (ctranslate2 backend, CPU int8) and turns a stream of
audio chunks into absolute-timestamped transcript segments.

Two things here are easy to get wrong and are handled explicitly:

1. Overlap dedupe. Consecutive chunks can repeat text at the seam. The
   comparison must be CHARACTER based, because Thai is written without
   spaces between words and whitespace tokenization collapses to one token.

2. Timestamps. Whisper reports times relative to the chunk it was given.
   They are shifted by the chunk's offset to get absolute session time.

See docs/BUILD_PLAN.md phase 3.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from . import config


@dataclass
class Segment:
    """One transcribed span, in absolute session time."""

    start: float
    end: float
    text: str
    lang: str
    chunk: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# --------------------------------------------------------------------------
# Text normalisation and overlap dedupe
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold text to a comparable form: NFC, lowercase, no punctuation/space.

    Whitespace is stripped entirely rather than used as a separator, so the
    same routine works for spaced scripts (English) and unspaced ones (Thai).
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT.sub("", text)
    return _SPACE.sub("", text)


def _overlap_length(tail: str, head: str, min_chars: int = 12) -> int:
    """Longest suffix of `tail` that is also a prefix of `head`."""
    limit = min(len(tail), len(head))
    for size in range(limit, min_chars - 1, -1):
        if tail[-size:] == head[:size]:
            return size
    return 0


def dedupe_overlap(
    previous: list[Segment],
    incoming: list[Segment],
    min_chars: int = 12,
) -> list[Segment]:
    """Drop text at the head of `incoming` already present at the tail of `previous`.

    Works character-wise so Thai (no word spaces) is handled correctly.
    Segments fully contained in the overlap are dropped; a segment that is only
    partially duplicated is kept, since dropping it would lose real content.
    """
    if not previous or not incoming:
        return incoming

    # Compare against a bounded window; a seam never spans the whole lecture.
    tail = normalize(" ".join(s.text for s in previous[-6:]))
    if not tail:
        return incoming

    kept: list[Segment] = []
    consumed = 0  # normalised chars of `incoming` matched so far
    for index, seg in enumerate(incoming):
        norm = normalize(seg.text)
        if not norm:
            continue

        if kept:
            kept.append(seg)
            continue

        # Exact repeat of something recently said.
        if norm in tail:
            consumed += len(norm)
            continue

        # Partial seam: the chunk starts mid-way through the previous text.
        if index == 0 or consumed:
            matched = _overlap_length(tail, norm, min_chars=min_chars)
            if matched and matched >= len(norm) * 0.8:
                consumed += len(norm)
                continue

        kept.append(seg)

    return kept


# --------------------------------------------------------------------------
# Transcriber
# --------------------------------------------------------------------------
class Transcriber:
    """faster-whisper wrapper with language locking and overlap dedupe."""

    def __init__(
        self,
        model_name: str = config.ASR_MODEL,
        language: str | None = config.LANGUAGE,
        device: str = config.ASR_DEVICE,
        compute_type: str = config.COMPUTE_TYPE,
        cpu_threads: int = config.CPU_THREADS,
    ) -> None:
        if model_name.endswith(".en"):
            raise ValueError(
                f"{model_name!r} is English-only and cannot transcribe Thai. "
                "Use a multilingual model such as 'small'."
            )
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.forced_language = language
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self._detected: Counter[str] = Counter()
        self._locked_language: str | None = language
        self._previous: list[Segment] = []

    # -- language handling ----------------------------------------------
    @property
    def language(self) -> str | None:
        """Best current guess: forced, locked-in, or the running majority.

        Never returns None once any audio has been transcribed, so a short
        lecture that ends before the lock threshold still has a language for
        prompt selection.
        """
        if self.forced_language:
            return self.forced_language
        if self._locked_language:
            return self._locked_language
        if self._detected:
            return self._detected.most_common(1)[0][0]
        return None

    def _update_language(self, detected: str) -> str:
        if self.forced_language:
            return self.forced_language
        self._detected[detected] += 1
        if self._locked_language is None and sum(self._detected.values()) >= config.LANGUAGE_LOCK_AFTER_CHUNKS:
            # Lock to the majority so a single bad chunk cannot flip the
            # transcript language halfway through a lecture.
            self._locked_language = self._detected.most_common(1)[0][0]
        return self._locked_language or detected

    # -- transcription ---------------------------------------------------
    def transcribe_chunk(
        self,
        path: Path,
        offset_seconds: float = 0.0,
        chunk_index: int = 0,
    ) -> list[Segment]:
        """Transcribe one chunk, returning deduped absolute-time segments."""
        segments, info = self._model.transcribe(
            str(path),
            # Once locked, pin the language so a noisy chunk cannot flip it.
            language=self._locked_language,
            vad_filter=config.VAD_FILTER,
            vad_parameters=dict(min_silence_duration_ms=config.VAD_MIN_SILENCE_MS),
            condition_on_previous_text=False,  # avoids repetition loops across chunks
        )

        lang = self._update_language(info.language)

        produced = [
            Segment(
                start=round(seg.start + offset_seconds, 3),
                end=round(seg.end + offset_seconds, 3),
                text=seg.text.strip(),
                lang=lang,
                chunk=chunk_index,
            )
            for seg in segments
            if seg.text.strip()
        ]

        deduped = dedupe_overlap(self._previous, produced)
        if deduped:
            self._previous = (self._previous + deduped)[-12:]
        return deduped

    def transcribe_file(self, path: Path) -> list[Segment]:
        """Transcribe a complete recording in one pass (used by --hq re-runs)."""
        segments, info = self._model.transcribe(
            str(path),
            language=self.forced_language,
            vad_filter=config.VAD_FILTER,
            vad_parameters=dict(min_silence_duration_ms=config.VAD_MIN_SILENCE_MS),
        )
        lang = self.forced_language or info.language
        self._locked_language = lang
        return [
            Segment(round(s.start, 3), round(s.end, 3), s.text.strip(), lang, 0)
            for s in segments
            if s.text.strip()
        ]


# --------------------------------------------------------------------------
# Transcript persistence (append + flush: crash safety)
# --------------------------------------------------------------------------
class TranscriptWriter:
    """Appends segments to transcript.jsonl, flushing after each one.

    A dropped laptop mid-lecture must lose at most the current chunk, so the
    whole transcript is never held in memory.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self) -> "TranscriptWriter":
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, segments: Iterable[Segment]) -> int:
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")
        count = 0
        for seg in segments:
            self._fh.write(seg.to_json() + "\n")
            count += 1
        if count:
            self._fh.flush()
        return count

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_transcript(path: Path) -> list[Segment]:
    """Load transcript.jsonl, skipping any torn final line from a hard crash."""
    segments: list[Segment] = []
    if not Path(path).exists():
        return segments
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial write at the moment of the crash
        segments.append(
            Segment(
                start=data["start"],
                end=data["end"],
                text=data["text"],
                lang=data.get("lang", "en"),
                chunk=data.get("chunk", 0),
            )
        )
    return segments
