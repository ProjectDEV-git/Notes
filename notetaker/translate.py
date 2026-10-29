"""Notes in a different language from the one spoken.

A Thai student learning from an English YouTube lesson wants Thai notes.

The notes cannot simply be written in Thai from the English transcript: every
point is checked against the transcript by shared words, and a Thai point
shares none with English, so all of them would be thrown away as invented.
Instead the notes are written and checked in the spoken language as usual,
and the finished notes are translated as the very last step.

Translation is cautious. A line whose numbers do not survive, or that the
model failed to return, stays in the original language: a mistranslated
"150 J" is worse than an English line in Thai notes.
"""

from __future__ import annotations

import re
from typing import Callable

from . import config, languages

_BULLET = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)(.*?)\s*$")
# The "(12:34)" or "([12:34](link))" a video note ends with.
_STAMP = re.compile(r"\s*\((?:\[\d{1,2}:\d{2}(?::\d{2})?\]\([^)]*\)|\d{1,2}:\d{2}(?::\d{2})?)\)\s*$")
_REPLY = re.compile(r"^\s*(\d+)\s*[|.:)]\s*(.*?)\s*$")
_DIGITS = re.compile(r"\d+")

PROMPT = """Translate each numbered line of these study notes into {language}.
Keep every number, unit, formula, name and symbol exactly as written.
Write simply, for a student. Answer with the same numbered lines, one per
line, in the form "3| translated line", and nothing else.

{lines}"""


def _heading_map(source: str, target: str) -> dict[str, str]:
    """Section headings are fixed per language pack, so swap them directly."""
    src, dst = languages.get(source), languages.get(target)
    mapping: dict[str, str] = {}
    for level in config.NOTES_LEVELS:
        for getter in ("heading_for_key_ideas", "heading_for_terms", "heading_for_actions"):
            before = getattr(src, getter)(level).strip()
            after = getattr(dst, getter)(level).strip()
            mapping.setdefault(before, after)
    return mapping


def _prompt_name(code: str) -> str:
    """'Thai (ภาษาไทย)': small models follow an English language name best."""
    from .cli import WHISPER_LANGUAGES

    own = languages.get(code).name
    english = WHISPER_LANGUAGES.get(code)
    return f"{english} ({own})" if english and english != own else own


def _keeps_numbers(original: str, translated: str) -> bool:
    return set(_DIGITS.findall(original)) <= set(_DIGITS.findall(translated))


def translate_notes(
    markdown: str,
    source: str,
    target: str,
    model: str = config.SUMMARY_MODEL,
    chat: Callable[..., str] | None = None,
) -> str:
    """Translate finished notes, keeping layout, timestamps and links.

    Only words go to the model: bullet markers, heading marks and the
    timestamp links are taken off first and put back afterwards, so a small
    model cannot break a link or a heading.
    """
    if not markdown.strip() or source == target:
        return markdown
    if chat is None:
        from . import summarize

        chat = summarize.chat

    headings = _heading_map(source, target)
    lines = markdown.splitlines()
    # (line index, text before, words to translate, text after)
    jobs: list[tuple[int, str, str, str]] = []
    src_name, dst_name = languages.get(source).name, languages.get(target).name

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or (index == 0 and stripped.startswith("# ")):
            continue  # the title is the class's name, not note text
        if stripped.startswith("*") and stripped.endswith("*") and f"· {src_name}" in stripped:
            lines[index] = line.replace(f"· {src_name}", f"· {src_name} → {dst_name}", 1)
            continue
        if stripped in headings:
            lines[index] = headings[stripped]
            continue
        heading = re.match(r"^(#{2,6}\s+)(.*)$", line)
        if heading:
            jobs.append((index, heading.group(1), heading.group(2), ""))
            continue
        bullet = _BULLET.match(line)
        prefix, body = (bullet.group(1), bullet.group(2)) if bullet else ("", line.strip())
        stamp = _STAMP.search(body)
        suffix = stamp.group(0) if stamp else ""
        words = body[: stamp.start()] if stamp else body
        if words.strip():
            jobs.append((index, prefix, words.strip(), suffix))

    batch = config.TRANSLATE_BATCH_LINES
    for start in range(0, len(jobs), batch):
        chunk = jobs[start:start + batch]
        numbered = "\n".join(f"{n}| {job[2]}" for n, job in enumerate(chunk, start=1))
        reply = chat(PROMPT.format(language=_prompt_name(target), lines=numbered),
                     model=model, max_tokens=config.TRANSLATE_MAX_TOKENS)
        answers: dict[int, str] = {}
        for raw in reply.splitlines():
            match = _REPLY.match(raw)
            if match and match.group(2):
                answers.setdefault(int(match.group(1)), match.group(2))
        for n, (index, prefix, words, suffix) in enumerate(chunk, start=1):
            translated = answers.get(n)
            if translated and _keeps_numbers(words, translated):
                lines[index] = f"{prefix}{translated}{suffix}"

    return "\n".join(lines) + ("\n" if markdown.endswith("\n") else "")
