# NoteTaker

Records a class, transcribes it, and writes down **the key ideas only**.

Built for a normal school day: a period has a known length, so recording
**stops by itself at the end**, and the notes are written for a secondary
school student, defining new words instead of assuming them.

Works for both kinds of class:

- **In person** — captures your microphone
- **Online** (Zoom, Teams, YouTube) — captures your system audio

Handles **English and Thai** out of the box, and a Thai class produces Thai
notes. [Adding another language](docs/LANGUAGES.md) takes about a minute:

```bash
notes lang add ja        # then: notes record --lang ja
```

Everything runs **offline on your machine**. No API keys, no subscription, and
no audio ever leaves the laptop.

---

## Quick start

One word, and pick from a list. Nothing to remember:

```bash
notes            # opens a menu: pick a number, press Enter
```

```
  1. Record the class I am in — uses the microphone
  2. Record an online class — Zoom, Teams, YouTube
  3. Read my notes — from a past lecture
  4. Write notes for a past lecture — if they are missing
  5. Save notes to a file — to share or print
  6. Write up everything I have not done — after school
  7. Record now, write notes later — saves battery in class
  8. Record with more options — language, title, source
  9. Check that everything works — microphone, notes writer
```

It asks which subject, and how long the class is. Enter is a valid answer to
both. Every action prints the command it ran, so you can skip the menu later:

```bash
notes class      # record a class; stops by itself after 60 minutes
notes later      # record the sound only; write the notes after school
notes catchup    # write notes for every class that does not have them yet
notes now        # record until you press Ctrl-C (a long lecture)
notes online     # record an online class (Zoom/Teams/YouTube)
notes last       # show the notes from your last class
notes all        # list every class
notes check      # confirm your microphone and notes writer work
notes update     # update NoteTaker and Python dependencies safely
```

The `notes` launcher checks for updates automatically when it starts. It skips
the check if the checkout has local changes, and continues opening the app if
the network is unavailable. Set `NOTES_AUTO_UPDATE=0` to disable startup
checks.

`notes class` **stops on its own** two minutes after the period ends, because
classes overrun and the homework is usually the last thing said. Ctrl-C always
stops sooner. Key ideas also appear live while you record.

If your periods are not 60 minutes:

```bash
NOTES_CLASS_MINUTES=45 notes class
```

### A whole day of classes

Transcribing while you record makes the laptop work hard. For back-to-back
periods, record the sound only and write everything up once you are home:

```bash
notes later      # in each class: no model runs, battery lasts
notes catchup    # after school: transcribes and writes up everything
```

`notes catchup` is also the recovery path. If Ollama was not running, or
transcription could not keep up, the class is picked up here rather than lost.
The menu tells you when classes are waiting.

Anything else is passed to the full CLI:

```bash
notes record --title "Physics week 4" --lang th --minutes 50
notes summarize physics --hq
notes export physics -o notes.md
notes summarize physics --rerun --level university   # denser wording
```

`<id>` can be part of the title, so `notes show thermo` works.

<details>
<summary>Full command list</summary>

```bash
notetaker devices                      # which mic / system-audio sources exist
notetaker menu                         # the numbered menu
notetaker check                        # verify audio devices and Ollama
notetaker update                       # update code and Python dependencies
notetaker record [--source mic|system] [--title T] [--live-notes] [--lang auto|en|th]
               [--minutes N]           # stop by itself after N minutes
               [--later]               # record sound only, write notes later
               [--level school|university]
notetaker catchup [--limit N]          # write up every class still missing notes
notetaker list                         # past recordings
notetaker show <id> [--transcript]
notetaker summarize <id> [--rerun] [--hq] [--level school|university]
notetaker export <id> [--md|--txt] [-o FILE]
notetaker lang list|add|edit|remove    # languages your notes are written in
```

</details>

## Who the notes are written for

By default the notes are written for a **secondary school student**: short
sentences, plain words, and every new term defined where the teacher
introduced it. The admin section is called *Homework & reminders*.

The facts do not change with the level. Numbers, units and formulas are copied
exactly at both settings, and inventing homework that was never set is
forbidden at both.

```bash
notes summarize physics --rerun --level university   # denser, assumes more
export NOTETAKER_NOTES_LEVEL=university              # make it the default
```

Adding a language? A pack can carry its own school prompts (`map_school`,
`reduce_school`). A pack without them simply uses its normal prompts, so
nothing you have already written needs changing.

## Languages

Transcription already works for ~100 languages with no setup, because that is
Whisper. A **language pack** decides what language your *notes* are written in.

```bash
notes lang list          # what you have
notes lang add ja        # add Japanese
notes lang edit ja       # translate the prompts (recommended)
```

Packs are JSON files in `~/.config/notetaker/languages/`. You can also override
the built-in English or Thai prompts to suit your subject. A malformed pack is
skipped with a warning rather than crashing, so a typo cannot cost you a
lecture. See **[docs/LANGUAGES.md](docs/LANGUAGES.md)**.

---

## Install

Runs on **Linux** (PipeWire/PulseAudio) and **macOS** (AVFoundation).

One command sets up everything, including ffmpeg, Ollama and the summary model:

```bash
git clone https://github.com/ProjectDEV-git/Notes.git NoteTaker && cd NoteTaker
./install.sh
```

It asks before installing anything and prints every command it runs, including
each `sudo`. It detects apt, dnf, pacman, zypper, apk and Homebrew, adds the
`notes` command to your PATH, and finishes by checking that recording and
note-writing actually work.

```bash
./install.sh --yes          # install everything without asking
./install.sh --no-install   # only check, install nothing
```

### Update NoteTaker

After the initial install, update from the configured Git repository with:

```bash
notes update
```

Updates use a fast-forward-only Git pull, so local commits and uncommitted
changes are never overwritten. Commit or stash local changes first if the
command reports a dirty checkout. The update also refreshes Python packages;
system tools such as ffmpeg and Ollama are left unchanged.

Already have your own setup? `notes check` tells you what is missing and the
exact command to fix it.

The Whisper model downloads itself on first run (~500 MB).

**On macOS**, recording an *online* lecture needs a loopback driver, because
CoreAudio has no way to capture what the speakers are playing. `install.sh`
offers to install BlackHole for you; otherwise:

```bash
brew install --cask blackhole-2ch
```

Either way, in **Audio MIDI Setup** create a Multi-Output Device combining BlackHole
with your speakers, and select it as the output, so you still hear the lecture
while it is recorded. The first recording asks for Microphone permission for
your terminal.

---

## How it works

```
microphone ─┐
            ├─> ffmpeg ─> 30s chunks ─> faster-whisper ─> transcript.jsonl
system audio┘                                                    │
                                                                 v
                                              map-reduce over ~3 min windows
                                                    (local Ollama)
                                                                 v
                                                            notes.md
```

A 50-minute lecture is far too long to summarize in one pass, so it is split
into windows, each reduced to a few points, then consolidated and deduplicated.
Administrative asides ("the exam is on Friday") are separated from the actual
content.

Recordings live in `~/.local/share/notetaker/sessions/<id>/`:

| file | what it is |
|---|---|
| `audio.wav` | the recording, kept so you can re-run `--hq` |
| `transcript.jsonl` | timestamped transcript, written as the lecture happens |
| `notes.md` | the key ideas |

---

## Performance on this machine

Measured on an Intel Core 5 120U (12 threads, **no GPU**).

**Transcription** — real-time factor measured with the `small` model:

| language | RTF | meaning |
|---|---|---|
| English | **0.62 – 0.82** | keeps up with a live lecture, with headroom |
| Thai | **~4.9** | roughly 5x slower than real time |

Thai is much harder for Whisper on CPU. Recording still works and **nothing is
lost** — chunks queue on disk and are transcribed after you stop — but a
50-minute Thai lecture needs a long catch-up once you press stop. The app warns
you when it starts falling behind. For Thai, consider recording and then
walking away while it finishes.

`large-v3-turbo` (via `--hq`) is *not* slower than `small` on Thai here (RTF
4.94 vs 4.88) and is meaningfully more accurate, correctly recovering words like
นักศึกษา and พืช that `small` mangles. Character accuracy against a known
reference: 89.1% for `small`, 90.8% for `large-v3-turbo`.

**Summarization** — model choice matters enormously on CPU:

| model | result |
|---|---|
| **llama3.2:3b** | **63s for a 3-minute lecture. Clean bullets, correct Thai. Default.** |
| qwen3:4b | timed out at 300s. Narrates instead of answering, replies in English to Thai input |
| qwen3:8b | 1.6 tok/s, roughly 25 minutes for a 50-minute lecture |
| qwen3:0.6b | fast, but echoes the transcript instead of summarizing it |

The qwen3 family is reasoning-first. Even with `think=false` it spends its
token budget on chain-of-thought instead of answering, so a **non-reasoning
instruct model is the right choice here**. Change it with `--model` or
`NOTETAKER_SUMMARY_MODEL`.

### Verified on a real 10-minute lecture

Tested end to end on a genuine 10-minute university lecture (Aristotle's logic),
captured through system audio exactly as an online lecture would be:

| stage | result |
|---|---|
| recording | 10:48 captured, 123 segments, duration reported correctly |
| transcription | kept up live; ~90 s to drain the backlog after stopping |
| summarizing | 3 m 31 s across 4 map-reduce windows |
| **output** | **1130 words → 171 words (17.3% of the transcript)** |

Every claim in the notes was checked against the transcript and none was
fabricated: the first-cause discussion, the physics/metaphysics split, and the
exoteric works "lacking literary value" were all genuinely said.

Extrapolating, a 50-minute lecture takes roughly 15-20 minutes to summarize
after class. Transcription itself happens live.

---

## Known limitations

- **Thai transcription runs ~5x slower than real time** on this CPU, so it
  cannot keep up live. Nothing is lost: the chunks queue on disk, and if the
  wait after stopping is still not enough they are kept for `notes catchup`.
  For a full Thai class, `notes later` is the better choice. English is fine
  live.
- **For Thai, prefer `--hq`.** `large-v3-turbo` costs no extra time over `small`
  on Thai and is noticeably more accurate.
- **Thai summaries occasionally invent a term.** Observed "Kleorophil pars" in
  place of a real word. Check anything that matters against the transcript.
- **System audio captures everything you can hear.** Mute unrelated tabs before
  recording an online class, or you will get their content in your notes.
- Summarizing is CPU-bound and will make the laptop warm. For back-to-back
  periods use `notes later` and run `notes catchup` once, after school.
- **Recordings are never deleted.** A one-hour class keeps about 115 MB of
  audio so `--hq` re-runs stay possible. Delete old sessions yourself from
  `~/.local/share/notetaker/sessions/` if space runs short.
- **macOS needs a loopback driver for online lectures.** BlackHole or similar;
  see Install. In-person recording works with no extra setup.
- Linux and macOS only. Windows is not supported.

## If something goes wrong

**"no 'system' source available"** — run `notes check`. On Linux your machine
exposes no `.monitor` device; on macOS you need a loopback driver
(`brew install blackhole-2ch`).

**Recording is silent** — for online lectures, make sure the audio really is
playing through the sink you selected. `notetaker devices` marks the default.

**Not sure what is wrong** — run `notes check`. It tests every part and prints
the exact command to fix whatever is missing.

**"cannot reach Ollama"** — run `ollama serve`. Your transcript is already
saved; run `notetaker summarize <id>` once Ollama is up.

**Notes look wrong** — check the transcript first with
`notetaker show <id> --transcript`. The summary can only be as good as the
transcription.

---

## Development

```bash
.venv/bin/python -m pytest tests/ -q      # full suite
```

Tests that need audio hardware or Ollama skip themselves when unavailable.
`docs/BUILD_PLAN.md` documents the design decisions and the traps involved.
