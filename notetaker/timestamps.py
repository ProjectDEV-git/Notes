"""Point each note back to the moment in the video it came from.

The notes are reduced and reworded by the model, so no bullet carries its
source time. It is recovered afterwards by matching each bullet against the
transcript with the same word overlap the grounding check uses: the stretch
of transcript sharing the most distinctive words is where it was said.

A bullet with no clear match gets no timestamp. A wrong link is worse than
none, because the student would trust it.
"""

from __future__ import annotations

import re
from typing import Callable

from .asr import Segment

_BULLET = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)(.*\S)\s*$")
# Already annotated (a re-run, or the model copied one): leave it alone.
_HAS_TIME = re.compile(r"\(\[?\d{1,2}:\d{2}(?::\d{2})?\]?(?:\([^)]*\))?\)\s*$")

# Share of a bullet's distinctive words that must appear in one stretch of
# transcript. Lower than this and the "match" is usually common vocabulary.
MATCH_THRESHOLD = 0.5
# Segments considered together, so a point spanning a sentence break matches.
SPAN = 3
# Start the link slightly early: landing mid-word loses the sentence.
LEAD_IN_SECONDS = 3.0


def format_time(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _words(text: str) -> set[str]:
    from .summarize import _content_words

    return _content_words(text)


def locate(bullet: str, segments: list[Segment], indexed=None) -> float | None:
    """Start time of the transcript stretch a bullet came from, or None."""
    wanted = _words(bullet)
    if not wanted or not segments:
        return None
    indexed = indexed if indexed is not None else [_words(s.text) for s in segments]

    best_score, best_index = 0.0, -1
    for start in range(len(segments)):
        pool: set[str] = set()
        for words in indexed[start:start + SPAN]:
            pool |= words
        score = len(wanted & pool) / len(wanted)
        if score > best_score:  # strict: ties keep the earliest mention
            best_score, best_index = score, start
    if best_index < 0 or best_score < MATCH_THRESHOLD:
        return None

    # Inside the winning stretch, start at the segment that carries the most.
    stretch = range(best_index, min(best_index + SPAN, len(segments)))
    first = max(stretch, key=lambda i: (len(wanted & indexed[i]), -i))
    return segments[first].start


def add_timestamps(
    markdown: str,
    segments: list[Segment],
    link: Callable[[float], str] | None = None,
    offset: float = 0.0,
) -> str:
    """Append '(12:34)' to each bullet, as a link when `link` is given.

    `offset` is added to transcript times: a trimmed video's transcript starts
    at zero, but the link must point into the full video.
    """
    indexed = [_words(s.text) for s in segments]
    out: list[str] = []
    for line in markdown.splitlines():
        match = _BULLET.match(line)
        if not match or _HAS_TIME.search(line):
            out.append(line)
            continue
        found = locate(match.group(2), segments, indexed)
        if found is None:
            out.append(line)
            continue
        at = found + offset
        label = format_time(at)
        stamp = f"[{label}]({link(max(at - LEAD_IN_SECONDS, 0))})" if link else label
        out.append(f"{line.rstrip()} ({stamp})")
    return "\n".join(out) + ("\n" if markdown.endswith("\n") else "")
