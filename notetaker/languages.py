"""Language pack loading.

Built-in languages live in this module. Users add their own by dropping a JSON
file into ~/.config/notetaker/languages/ (or by running
`notetaker lang add <code>`), with no code changes and no reinstall.

A language pack tells NoteTaker four things:

    name        what to call the language in the notes header
    script      optional Unicode range, for languages without word spaces
    headings    the three section titles used in the notes
    prompts     the map and reduce prompts, in that language

See docs/LANGUAGES.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".config" / "notetaker"
LANGUAGE_DIR = CONFIG_DIR / "languages"

# Placeholders that a model may write instead of omitting an empty section.
BASE_NONE_MARKERS = {
    "none", "n/a", "na", "nothing", "none.", "(none)", "-", "—",
    "none identified", "not applicable",
}


@dataclass
class Language:
    """Everything NoteTaker needs to work in one language."""

    code: str
    name: str
    map_prompt: str
    reduce_prompt: str
    key_heading: str = "## Key ideas"
    terms_heading: str = "## Terms & definitions"
    action_heading: str = "## Action items"
    # Unicode range for scripts written without spaces between words, so the
    # dedupe and grounding checks can fall back to character n-grams.
    script_range: tuple[str, str] | None = None
    none_markers: set[str] = field(default_factory=set)

    @property
    def is_unspaced(self) -> bool:
        return self.script_range is not None

    def contains_script(self, text: str) -> bool:
        if not self.script_range:
            return False
        low, high = self.script_range
        return any(low <= ch <= high for ch in text)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "headings": {
                "key_ideas": self.key_heading,
                "terms": self.terms_heading,
                "action_items": self.action_heading,
            },
            "prompts": {"map": self.map_prompt, "reduce": self.reduce_prompt},
        }
        if self.script_range:
            data["script_range"] = list(self.script_range)
        if self.none_markers:
            data["none_markers"] = sorted(self.none_markers)
        return data

    @classmethod
    def from_dict(cls, code: str, data: dict[str, Any]) -> "Language":
        prompts = data.get("prompts") or {}
        missing = [key for key in ("map", "reduce") if not prompts.get(key)]
        if missing:
            raise ValueError(f"language {code!r} is missing prompts: {', '.join(missing)}")
        for key in ("map", "reduce"):
            if "{text}" not in prompts[key]:
                raise ValueError(
                    f"language {code!r}: the {key} prompt must contain "
                    "{text}, which is where the transcript is inserted"
                )

        headings = data.get("headings") or {}
        script = data.get("script_range")
        return cls(
            code=code,
            name=data.get("name", code),
            map_prompt=prompts["map"],
            reduce_prompt=prompts["reduce"],
            key_heading=headings.get("key_ideas", "## Key ideas"),
            terms_heading=headings.get("terms", "## Terms & definitions"),
            action_heading=headings.get("action_items", "## Action items"),
            script_range=(script[0], script[1]) if script else None,
            none_markers=set(data.get("none_markers", [])),
        )


# --------------------------------------------------------------------------
# Built-in languages
# --------------------------------------------------------------------------
_ENGLISH_MAP = (
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
)

_ENGLISH_REDUCE = (
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
)

_THAI_MAP = (
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
)

_THAI_REDUCE = (
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
)

BUILTIN: dict[str, Language] = {
    "en": Language(
        code="en",
        name="English",
        map_prompt=_ENGLISH_MAP,
        reduce_prompt=_ENGLISH_REDUCE,
    ),
    "th": Language(
        code="th",
        name="ภาษาไทย",
        map_prompt=_THAI_MAP,
        reduce_prompt=_THAI_REDUCE,
        key_heading="## แนวคิดสำคัญ",
        terms_heading="## คำศัพท์และนิยาม",
        action_heading="## สิ่งที่ต้องทำ",
        script_range=("\u0e00", "\u0e7f"),  # Thai has no spaces between words
        none_markers={"ไม่มี", "ไม่มี."},
    ),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
_cache: dict[str, Language] | None = None


def load_all(refresh: bool = False) -> dict[str, Language]:
    """Built-in languages plus any the user has added.

    A user file with the same code as a built-in one replaces it, so prompts
    can be customised without editing the source.
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache

    languages = dict(BUILTIN)
    if LANGUAGE_DIR.is_dir():
        for path in sorted(LANGUAGE_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                languages[path.stem] = Language.from_dict(path.stem, data)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                # A broken user file must not stop a lecture being recorded.
                import sys

                print(f"warning: ignoring language pack {path.name}: {exc}", file=sys.stderr)

    _cache = languages
    return languages


def get(code: str | None) -> Language:
    """Look up a language, falling back to English."""
    languages = load_all()
    if code and code in languages:
        return languages[code]
    return languages["en"]


def codes() -> list[str]:
    return sorted(load_all())


def unspaced_ranges() -> list[tuple[str, str]]:
    """Script ranges of all languages written without word spaces."""
    return [lang.script_range for lang in load_all().values() if lang.script_range]


def none_markers() -> set[str]:
    markers = set(BASE_NONE_MARKERS)
    for lang in load_all().values():
        markers |= lang.none_markers
    return markers


def action_headings() -> list[str]:
    """Lowercased action-item headings across all languages."""
    return [lang.action_heading.lstrip("#").strip().lower() for lang in load_all().values()]


def template(code: str, name: str | None = None) -> dict[str, Any]:
    """A starter pack, in English, for the user to translate."""
    english = BUILTIN["en"]
    return Language(
        code=code,
        name=name or code,
        map_prompt=english.map_prompt,
        reduce_prompt=english.reduce_prompt,
    ).to_dict()


def save(code: str, data: dict[str, Any]) -> Path:
    """Write a language pack and refresh the cache."""
    Language.from_dict(code, data)  # validate before writing
    LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = LANGUAGE_DIR / f"{code}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    load_all(refresh=True)
    return path
