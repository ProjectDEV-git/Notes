# NoteTaker

Records a lecture, transcribes it, and writes down **the key ideas only**.

Works for both kinds of lecture:

- **In person** — captures your microphone
- **Online** (Zoom, Teams, YouTube) — captures your system audio

Handles **English and Thai**, and a Thai lecture produces Thai notes.

Everything runs **offline on your machine**. No API keys, no subscription, and
no audio ever leaves the laptop.

---

## Quick start

One word to start recording:

```bash
notes            # record this lecture now (microphone)
notes online     # record an online lecture (Zoom/Teams/YouTube)
notes last       # show the notes from your last lecture
notes all        # list every lecture
```

Press **Ctrl-C** to stop. It transcribes the last chunk, then prints and saves
the key ideas. Key ideas also appear live while you record.

Anything else is passed to the full CLI:

```bash
notes record --title "Physics week 4" --lang th
notes summarize physics --hq
notes export physics -o notes.md
```

`<id>` can be part of the title, so `notes show thermo` works.

<details>
<summary>Full command list</summary>

```bash
notetaker devices                      # which mic / system-audio sources exist
notetaker record [--source mic|system] [--title T] [--live-notes] [--lang auto|en|th]
notetaker list                         # past recordings
notetaker show <id> [--transcript]
notetaker summarize <id> [--rerun] [--hq]
notetaker export <id> [--md|--txt] [-o FILE]
```

</details>

---

## Install

Requires Linux with PipeWire or PulseAudio, `ffmpeg`, and
[Ollama](https://ollama.com).

```bash
git clone <this repo> && cd NoteTaker
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt

ollama pull llama3.2:3b     # writes the notes
./run.sh devices            # check your audio devices
```

The Whisper model downloads itself on first run (~500 MB).

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

---

## Known limitations

- **Thai transcription runs ~5x slower than real time** on this CPU, so it
  cannot keep up live. Nothing is lost (chunks queue on disk), but expect a wait
  after stopping. English is fine live.
- **For Thai, prefer `--hq`.** `large-v3-turbo` costs no extra time over `small`
  on Thai and is noticeably more accurate.
- **Thai summaries occasionally invent a term.** Observed "Kleorophil pars" in
  place of a real word. Check anything that matters against the transcript.
- **System audio captures everything you can hear.** Mute unrelated tabs before
  recording an online lecture, or you will get their content in your notes.
- Summarizing is CPU-bound and will make the laptop warm.
- Linux only. The capture layer is built on PulseAudio/PipeWire.

## If something goes wrong

**"no 'system' source available"** — your machine exposes no `.monitor` device.
Check `notetaker devices`.

**Recording is silent** — for online lectures, make sure the audio really is
playing through the sink you selected. `notetaker devices` marks the default.

**"cannot reach Ollama"** — run `ollama serve`. Your transcript is already
saved; run `notetaker summarize <id>` once Ollama is up.

**Notes look wrong** — check the transcript first with
`notetaker show <id> --transcript`. The summary can only be as good as the
transcription.

---

## Development

```bash
.venv/bin/python -m pytest tests/ -q      # 123 tests
```

Tests that need audio hardware or Ollama skip themselves when unavailable.
`docs/BUILD_PLAN.md` documents the design decisions and the traps involved.
