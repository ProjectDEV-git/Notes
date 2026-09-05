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
  3. Read my notes — from a past class
  4. Write notes for a past class — if they are missing
  5. Save notes to a file — to share or print
  6. Write up everything I have not done — after school
  7. Record now, write notes later — saves battery in class
  8. Record with more options — language, length, reading level
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
transcription could not keep up and stopped part-way, the class is picked up
here rather than lost: a transcript that was cut short is redone from the full
recording, so you get the whole class and not just the start. The menu tells
you when classes are waiting.

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

Pick **8. Record with more options** in the menu to change it for one class,
or use the command line:

```bash
notes summarize physics --rerun --level university   # denser, assumes more
export NOTETAKER_NOTES_LEVEL=university              # make it the default
```

Adding a language? A pack can carry its own school prompts and headings
(`map_school`, `reduce_school`, `key_ideas_school`, ...). A pack without them
simply uses its normal ones, so nothing you have already written needs
changing. See **[docs/LANGUAGES.md](docs/LANGUAGES.md)**.

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

### Staying up to date

NoteTaker updates itself when you start it. There is nothing to run.

```bash
notes update              # update right now
NOTES_AUTO_UPDATE=0 notes # skip the check this once
```

Updates are fast-forward only, so local commits and uncommitted changes are
never overwritten: a modified checkout is left completely alone. System tools
such as ffmpeg and Ollama are never touched.

Every step is time-limited, so a captive portal or dead wifi costs a few
seconds and then gets out of the way. **An update can never stop you
recording a class.**

Already have your own setup? `notes check` tells you what is missing and the
exact command to fix it.

The Whisper model downloads itself on first run (~500 MB).

### On a Mac

Run the same `./install.sh`. It installs Homebrew for you if the Mac does not
have it, and finds an existing one whether it lives in `/opt/homebrew` (Apple
Silicon) or `/usr/local` (Intel).

Two Mac-only things are worth knowing:

**Microphone permission.** The first recording asks for it. You have to say
yes, or every class records silence. If you have already said no, macOS will
not ask again: turn it on under
*System Settings > Privacy & Security > Microphone* and tick your terminal.

**Online classes need a loopback driver.** Recording a class you are sitting
in works straight away. Only *online* classes need this, because CoreAudio
cannot capture what the speakers are playing. The installer offers it;
otherwise:

```bash
brew install --cask blackhole-2ch
```

Then in **Audio MIDI Setup** create a Multi-Output Device combining BlackHole
with your speakers and select it as the output, so you still hear the class
while it records.

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
| `transcript.jsonl` | timestamped transcript, written as the class happens |
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

### Verified on a real recording

Tested end to end on a genuine 10-minute university lecture (Aristotle's logic),
captured through system audio exactly as an online class would be, then
re-measured under the current school-level defaults:

| stage | result |
|---|---|
| recording | 10:48 captured, 123 segments, duration reported correctly |
| transcription | kept up live; ~90 s to drain the backlog after stopping |
| summarizing | 3-6 min across 4 map-reduce windows |
| **output** | **1130 words -> ~130 words (~12% of the transcript)** |

The notes correctly picked up the Physics/Metaphysics split, the discussion of
first causes, and the exoteric/esoteric distinction, all genuinely said.

This recording is also what exposed two real defects, both since fixed: the
model narrated the lesson ("the teacher talked about...") despite every prompt
forbidding it, and it turned words the transcriber mis-heard into confident
vocabulary definitions.

The first is now filtered in code. The second is only discouraged by the
prompt, so it can still happen: a transcription error that looks like a term
may reach "Words to know" with an invented meaning. **Check anything that
matters against the transcript** with `notetaker show <id> --transcript`.

### Measured on a real class

Timed end to end on this CPU with `llama3.2:3b`, on a 41-minute class
(14 map-reduce windows), summarizing the whole transcript for real:

| after the bell | time | speedup |
|---|---|---|
| without live notes | 12.8 min | — |
| **with live notes on** | **7.0 min** | **1.84x** |

Both produced equivalent notes (38 vs 37 points). Live notes do the per-window
MAP work *during* the lesson, so only the final minutes and the merge are left
once the class ends.

Scaling by window count, a full 60-minute class is roughly **18 minutes**
without live notes and **10 minutes** with. The remaining time is the merge
step, which cannot start until the class is over.

Component costs, for reference:

| step | cost |
|---|---|
| one MAP window | 13 s |
| one REDUCE (20 points) | 79 s |

If that wait matters, record with `notes later` and run `notes catchup` once
for the whole day rather than waiting after each period.

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
- **"Words to know" can contain an invented definition.** School notes are
  asked to define new words, so a word the transcriber mis-heard can arrive
  looking like real vocabulary and be given a confident meaning. The prompts
  discourage this, but a small model still does it occasionally. Glance over
  that section before revising from it.
- **System audio captures everything you can hear.** Mute unrelated tabs before
  recording an online class, or you will get their content in your notes.
- **Notes do not appear the moment class ends.** Merging a full class takes
  roughly 10 minutes on this CPU with live notes on, or 18 without. That is
  longer than a break, so do not stand waiting for it. Either walk away and
  read the notes later, or use `notes later` and write up the whole day at
  once with `notes catchup`.
- Summarizing is CPU-bound and will make the laptop warm. A faster machine, or
  a smaller model via `--summary-model`, cuts the wait proportionally.
- **Recordings are never deleted.** A one-hour class keeps about 115 MB of
  audio so `--hq` re-runs stay possible. Delete old sessions yourself from
  `~/.local/share/notetaker/sessions/` if space runs short.
- **macOS needs a loopback driver for online classes.** BlackHole or similar;
  see Install. In-person recording works with no extra setup.
- Linux and macOS only. Windows is not supported.

## If something goes wrong

**"no 'system' source available"** — run `notes check`. On Linux your machine
exposes no `.monitor` device; on macOS you need a loopback driver
(`brew install blackhole-2ch`).

**Recording is silent** — for online classes, make sure the audio really is
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
