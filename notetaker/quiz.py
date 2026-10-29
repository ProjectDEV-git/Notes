"""Practice questions and flashcards from a class's notes.

Two kinds of card:

  * words   "Words to know" lines ('**osmosis** — water moving through a
            membrane') become flashcards directly. No model is involved.
  * ideas   each key idea gets a question written by the model.

The model only ever writes the QUESTION. The answer is always the student's
own note, word for word, so a quiz can never teach something the class did
not say. A question that does not relate to its note, or that simply gives
the answer away, is dropped.

Homework and reminders are left out: "what is the homework?" is not revision.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import config, languages

QUIZ_NAME = "quiz.json"

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*?)\s*$")
_STAMP = re.compile(r"\s*\((?:\[(\d{1,2}:\d{2}(?::\d{2})?)\]\([^)]*\)|(\d{1,2}:\d{2}(?::\d{2})?))\)\s*$")
_TERM = re.compile(r"^\*\*(.+?)\*\*\s*(?:[—–:]|-{1,2})\s*(.+)$|^([^:—–]{1,40})\s*[—–:]\s*(.+)$")
_REPLY = re.compile(r"^\s*(\d+)\s*[|.:)]\s*(.*?)\s*$")

PROMPT = """Here are numbered facts from a student's class notes, in {language}.
For each fact, write ONE short question in {language} that the fact answers.
Do not put the answer in the question. Do not add anything the fact does not say.
Answer with one line per fact, in the form "3| question", and nothing else.

{facts}"""


@dataclass
class Card:
    question: str
    answer: str
    kind: str  # "idea" or "word"
    time: str = ""  # when it was said, for videos and recordings with times


# --------------------------------------------------------------------------
# Reading the notes
# --------------------------------------------------------------------------
def _heading_names(attr_pairs: tuple[str, str]) -> set[str]:
    names: set[str] = set()
    for lang in languages.load_all().values():
        for attr in attr_pairs:
            heading = getattr(lang, attr)
            if heading:
                names.add(heading.lstrip("#").strip().casefold())
    return names


def sections(markdown: str) -> dict[str, list[tuple[str, str]]]:
    """Bullets of the notes by kind: 'ideas', 'words', 'admin'.

    Each bullet is (text, time), with its timestamp link taken off.
    """
    words = _heading_names(("terms_heading", "terms_heading_school"))
    admin = set(languages.action_headings())
    found: dict[str, list[tuple[str, str]]] = {"ideas": [], "words": [], "admin": []}
    current = "ideas"
    for line in markdown.splitlines():
        if line.startswith("## "):
            name = line.lstrip("#").strip().casefold()
            current = "words" if name in words else "admin" if name in admin else "ideas"
            continue
        match = _BULLET.match(line)
        if not match:
            continue
        text = match.group(1)
        stamp = _STAMP.search(text)
        time = (stamp.group(1) or stamp.group(2)) if stamp else ""
        text = text[: stamp.start()].strip() if stamp else text.strip()
        if text:
            found[current].append((text, time))
    return found


def word_cards(bullets: list[tuple[str, str]]) -> list[Card]:
    cards = []
    for text, time in bullets:
        match = _TERM.match(text)
        if not match:
            continue
        term = (match.group(1) or match.group(3) or "").strip(" *")
        meaning = (match.group(2) or match.group(4) or "").strip()
        if term and meaning:
            cards.append(Card(term, meaning, "word", time))
    return cards


# --------------------------------------------------------------------------
# Questions from the model
# --------------------------------------------------------------------------
def _fold(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def good_question(question: str, answer: str) -> bool:
    """About the note, and not just the note with a question mark."""
    from .summarize import _content_words

    q, a = _fold(question), _fold(answer)
    if len(q) < 6 or not a or q == a or a in q:
        return False
    return bool(_content_words(question) & _content_words(answer))


def idea_cards(
    bullets: list[tuple[str, str]],
    language: str,
    model: str,
    chat: Callable[..., str],
) -> list[Card]:
    from .translate import _prompt_name

    cards: list[Card] = []
    batch = config.QUIZ_BATCH_FACTS
    for start in range(0, len(bullets), batch):
        chunk = bullets[start:start + batch]
        facts = "\n".join(f"{n}| {text}" for n, (text, _) in enumerate(chunk, start=1))
        reply = chat(PROMPT.format(language=_prompt_name(language), facts=facts),
                     model=model, max_tokens=config.QUIZ_MAX_TOKENS)
        questions: dict[int, str] = {}
        for raw in reply.splitlines():
            match = _REPLY.match(raw)
            if match and match.group(2):
                questions.setdefault(int(match.group(1)), match.group(2).strip())
        for n, (answer, time) in enumerate(chunk, start=1):
            question = questions.get(n, "")
            if good_question(question, answer):
                cards.append(Card(question, answer, "idea", time))
    return cards


def notes_language(markdown: str, fallback: str | None) -> str:
    """The language the notes are written in, which may not be the one spoken."""
    for code, lang in languages.load_all().items():
        if lang.script_range and lang.contains_script(markdown):
            return code
    return fallback or "en"


def build(
    markdown: str,
    language: str | None,
    model: str = config.SUMMARY_MODEL,
    chat: Callable[..., str] | None = None,
    use_model: bool = True,
) -> list[Card]:
    found = sections(markdown)
    cards = word_cards(found["words"])
    if use_model and found["ideas"]:
        if chat is None:
            from . import summarize

            chat = summarize.chat
        cards += idea_cards(found["ideas"], notes_language(markdown, language), model, chat)
    return cards


# --------------------------------------------------------------------------
# Saving, and Anki
# --------------------------------------------------------------------------
def save(folder: Path, cards: list[Card]) -> None:
    from .artifacts import write_text_atomic

    write_text_atomic(folder / QUIZ_NAME,
                      json.dumps([asdict(c) for c in cards], ensure_ascii=False, indent=1))


def load(folder: Path) -> list[Card] | None:
    try:
        data = json.loads((folder / QUIZ_NAME).read_text(encoding="utf-8"))
        return [Card(**item) for item in data]
    except (OSError, ValueError, TypeError):
        return None


def to_anki(cards: list[Card], deck: str) -> str:
    """Tab-separated text that Anki imports as-is (File > Import).

    The header lines tell Anki the separator and deck, so the student only
    has to pick the file.
    """
    def clean(text: str) -> str:
        return " ".join(text.replace("\t", " ").split())

    lines = ["#separator:tab", "#html:false", f"#deck:{clean(deck)}", "#tags column:3"]
    for card in cards:
        tag = "notetaker_" + card.kind
        lines.append(f"{clean(card.question)}\t{clean(card.answer)}\t{tag}")
    return "\n".join(lines) + "\n"
