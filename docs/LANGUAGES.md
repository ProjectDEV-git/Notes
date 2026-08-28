# Adding a language

NoteTaker ships with **English** and **Thai**. Adding another takes about a
minute and needs no code changes.

## The two halves

There are two separate things going on, and it helps to keep them apart:

**Transcription** (speech → text) already works for roughly 100 languages. That
is Whisper, and you do not have to configure anything. Record a German lecture
and you get a German transcript today.

**Notes** (text → key ideas) is what a language pack controls. Without a pack,
your German lecture gets German transcription but the notes come out in English.
Adding a pack makes the notes German too.

## Add one

```bash
notetaker lang add ja        # or de, es, zh, ko, fr, ...
notetaker lang list          # see what you have
```

That writes `~/.config/notetaker/languages/ja.json`. It works immediately, but
the prompts start out in English. Record with:

```bash
notetaker record --lang ja
```

## Make it good

A model follows instructions best when they are written in the language you
want back. Translate the two prompts:

```bash
notetaker lang edit ja       # opens $EDITOR
```

Also translate the three `headings`, so your notes say `## 要点` rather than
`## Key ideas`.

**Keep `{text}` in both prompts.** That is where the transcript gets inserted.
NoteTaker refuses to load a pack without it, rather than silently producing
empty notes.

## The file

```json
{
  "name": "日本語",
  "script_range": ["\u3040", "\u30ff"],
  "headings": {
    "key_ideas": "## 要点",
    "terms": "## 用語と定義",
    "action_items": "## やるべきこと"
  },
  "prompts": {
    "map": "... {text}",
    "reduce": "... {text}"
  },
  "none_markers": ["なし"]
}
```

| field | what it does |
|---|---|
| `name` | shown in the notes header and `lang list` |
| `script_range` | **only for languages written without spaces between words** (Thai, Japanese, Chinese, Khmer, Lao, Burmese). Lets duplicate-detection and the anti-hallucination check work on character runs instead of words. Omit it for European languages. |
| `headings` | your three section titles |
| `prompts.map` | turns a ~3 minute chunk into 2-4 bullets |
| `prompts.reduce` | merges all the bullets into the final notes |
| `none_markers` | words your model writes when a section is empty, such as "なし". These get stripped so you do not get a heading with "None" under it. |

`notetaker lang add` fills in `script_range` automatically for the unspaced
languages it knows about.

## Writing prompts that work

The built-in English prompts were tuned against real lectures. Keep these
properties when you translate:

- **Demand bullets only.** "Output ONLY lines starting with '- '". Small models
  love to add a preamble.
- **Demand specifics.** Ask for numbers, units, formulas and dates. Without
  this you get "the lecturer discussed energy" instead of "15 kg lifted 1 m
  gives about 150 J".
- **Ban narration.** Explicitly forbid "The lecturer said...". You want what was
  taught, not a description of the lecture.
- **Frame reduce as editing, not writing.** This is the single most effective
  line against invented content: "you are reorganising existing lines, not
  writing new ones".
- **Keep the `ADMIN: ` marker in Latin letters.** The code looks for that exact
  string to route deadlines into Action items. Translate the surrounding
  instruction, not the marker.

## Customising a built-in language

Same command. It copies the built-in pack out for you to edit, and your version
then takes priority:

```bash
notetaker lang edit en
```

Useful for tailoring prompts to your subject, for instance asking a maths
lecture to preserve every formula.

## Removing

```bash
notetaker lang remove ja
```

Deletes only your custom pack. Built-in `en` and `th` always survive.

## If something breaks

A malformed pack is **skipped with a warning**, never a crash, so a typo cannot
stop you recording a lecture. Check it with `notetaker lang list`: if your
language is missing, the warning on stderr says why.

Common causes:

- missing `{text}` in a prompt
- trailing comma or unquoted string (it must be valid JSON)
- `script_range` given as one string instead of a two-item list
