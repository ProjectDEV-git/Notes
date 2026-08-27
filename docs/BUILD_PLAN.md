# NoteTaker — Build Plan (step-by-step execution guide)

This document is written so that **any agent or developer can pick it up cold** and
continue the build without re-deriving decisions. Follow phases in order. Do not skip
the "Verify" step at the end of each phase.

---

## 0. What we are building (read this first)

A **local-first** desktop app that:

1. Listens to a lecture — either the **microphone** (in-person class) or the
   **system audio monitor** (online lecture: Zoom / Teams / YouTube). User can switch.
2. **Transcribes it live** with faster-whisper. Lectures may be **English or Thai**.
3. **Summarizes into key ideas only** using a local LLM via Ollama.
4. Optionally shows **live incremental notes while recording** (toggleable on/off).

**Everything runs offline on this laptop.** No API keys. No audio leaves the machine.
This is a hard requirement, not a preference — do not add a cloud API dependency.

Build **CLI first**. GUI is Phase 9 and is explicitly deferred.

---

## 1. Environment facts (already verified — trust these, don't re-check)

| Thing | Value |
|---|---|
| OS | Zorin OS 18 (Ubuntu-based), X11, GNOME |
| CPU | Intel Core 5 120U, **12 threads** |
| RAM | 23 GB total, ~10 GB free |
| GPU | Intel integrated only — **no CUDA**. CPU inference only. |
| Disk free | 620 GB |
| Python | 3.12.3 |
| Audio server | PipeWire (with PulseAudio compat via `pactl`) |
| ffmpeg | 6.1.1, **has `pulse` input support** |
| Ollama | installed, API up at `http://localhost:11434` |

### Already installed (inherited by venv via `--system-site-packages`)
`faster_whisper 1.2.1`, `ctranslate2 4.8.0`, `numpy`, `av`, `onnxruntime`,
`tokenizers`, `huggingface_hub`, `rich`, `tqdm`, `flask`, `PyQt6`, `gi`, `sqlite3`

### Deliberately NOT used
- **torch** — not installed and not needed. faster-whisper uses ctranslate2. Do not add torch.
- **openai-whisper** — slower. Use faster-whisper.
- **Any cloud API** — all provider key files in `~/.config/jcode/*.env` are empty.

### Audio sources on this machine (exact names — use these strings)
```
alsa_input.pci-0000_00_1f.3.analog-stereo           -> Microphone   (in-person lecture)
alsa_output.pci-0000_00_1f.3.analog-stereo.monitor  -> System audio (online lecture)
```
Rule of thumb: a source whose name ends in `.monitor` is **system audio playback**;
everything else is a real input device. Enumerate dynamically, never hardcode — but these
are the expected values on this box.

---

## 2. Key design decisions (and why — do not silently reverse these)

| Decision | Reason |
|---|---|
| Whisper model `small` (**not** `small.en`) | Must handle **Thai**. `.en` models are English-only and would fail requirement 5. |
| `compute_type="int8"` | No GPU. int8 is the fast CPU path for ctranslate2. |
| Chunked capture (~30 s) with ~2 s overlap | Enables live transcription. Overlap prevents words being cut in half at boundaries. |
| Overlap **must be deduped** | Otherwise the same phrase appears twice in the transcript. This is the #1 correctness bug in this design. |
| Append to `transcript.jsonl` immediately | Crash-safety. A dropped laptop mid-lecture must not lose an hour of notes. |
| Map-reduce summarization | A 50-min lecture is ~8k+ words and overflows a small model's context. Cannot one-shot it. |
| Ollama `qwen3:8b` | ~5 GB, multilingual (handles Thai), fits in free RAM. `qwen2.5-coder:7b` is a **code** model — wrong tool. `qwen3:0.6b` is for fast tests only. |
| CLI before GUI | Prove the audio→ASR→summary engine works before spending effort on UI. |

---

## 3. Target layout

```
NoteTaker/
├── .venv/                     created with --system-site-packages
├── notetaker/
│   ├── __init__.py
│   ├── config.py              paths, model names, chunk sizes, PROMPTS
│   ├── audio.py               Phase 2 — device enum + ffmpeg capture
│   ├── asr.py                 Phase 3 — faster-whisper worker + dedupe
│   ├── store.py               Phase 5 — SQLite index + session dirs
│   ├── summarize.py           Phase 7 — map-reduce key ideas
│   └── cli.py                 Phase 8 — command surface
├── tests/
│   ├── fixtures/              sample audio + reference transcript
│   └── test_*.py
├── docs/BUILD_PLAN.md         this file
├── requirements.txt
└── README.md
```

**Data lives outside the repo** (never commit recordings):
```
~/.local/share/notetaker/
├── notetaker.db                       SQLite session index
└── sessions/<session_id>/
    ├── audio.wav                      full recording (for re-runs)
    ├── chunks/                        transient, deleted after processing
    ├── transcript.jsonl               one JSON object per segment
    └── notes.md                       final key-idea summary
```

---

## 4. Phase 1 — Scaffold  *(status: DONE)*

Already completed:
- `git init` + identity configured
- `.venv` created with `--system-site-packages` (inherits faster_whisper, no re-download)
- `notetaker/`, `tests/fixtures/`, `docs/` created
- stray empty sibling dir `../taker` removed (user-approved)

Remaining in this phase:
1. Write `.gitignore` — must include `.venv/`, `__pycache__/`, `*.wav`, `*.jsonl`, `data/`.
2. Write `requirements.txt` pinning **the versions already proven working**:
   `faster-whisper==1.2.1`, `ctranslate2==4.8.0`, plus `pytest`.
3. Write `notetaker/config.py` (see spec below).
4. Commit: `chore: scaffold project structure`.

### `config.py` must expose
```python
DATA_DIR      = ~/.local/share/notetaker        # respect XDG_DATA_HOME if set
DB_PATH       = DATA_DIR / "notetaker.db"
SAMPLE_RATE   = 16000      # whisper's native rate — do not change
CHANNELS      = 1
CHUNK_SECONDS = 30
OVERLAP_SECONDS = 2
ASR_MODEL     = "small"            # multilingual; NOT small.en
ASR_MODEL_HQ  = "large-v3-turbo"   # optional higher-accuracy re-run
COMPUTE_TYPE  = "int8"
CPU_THREADS   = 8                  # of 12; leave headroom for capture + UI
LANGUAGE      = None               # None = autodetect en/th; overridable
SUMMARY_MODEL = "qwen3:8b"
TEST_MODEL    = "qwen3:0.6b"
OLLAMA_URL    = "http://localhost:11434"
PROMPTS       = {...}              # map + reduce prompts, see Phase 7
```
Every tunable lives here. No magic numbers scattered through other modules.

---

## 5. Phase 2 — Audio capture (`audio.py`)

### Implement
```python
list_sources() -> list[AudioSource]
```
Shell out to `pactl list short sources`, parse, and return objects with
`name`, `description`, `kind` where `kind` is `"mic"` or `"system"`
(`"system"` iff the name ends with `.monitor`). Also expose a `is_default` flag
using `pactl info`.

```python
record_to_chunks(source_name, out_dir, stop_event) -> Iterator[Path]
```
Launch ffmpeg and yield chunk paths as they appear on disk.

**Working ffmpeg invocation** (verified device syntax for this box):
```
ffmpeg -hide_banner -loglevel error \
  -f pulse -i <SOURCE_NAME> \
  -ac 1 -ar 16000 \
  -f segment -segment_time 30 -reset_timestamps 1 \
  <out_dir>/chunk_%05d.wav
```
Also write the continuous full-session `audio.wav` (use ffmpeg `-map` to a second
output, or concatenate chunks at stop — either is acceptable, full file is for re-runs).

### Requirements
- Stop cleanly on `stop_event`: send `SIGINT` to ffmpeg (**not** `SIGKILL`) so it
  finalizes WAV headers. Then wait with a timeout, escalating to kill only if it hangs.
- A chunk is only yielded once it is **fully written** — watch for the *next* chunk file
  appearing, which proves the previous one is closed. Reading a half-written WAV
  produces garbage transcripts.
- Expose `rms_level(path) -> float` for the CLI level meter and silence detection.
- Handle: source disappears mid-recording, disk full, ffmpeg not found — raise clear errors.

> **On overlap:** ffmpeg's `-f segment` produces *non-overlapping* chunks. Achieve the
> 2 s overlap on the **read** side in `asr.py` by prepending the tail of the previous
> chunk's audio, or by carrying transcript-level context. Simplest correct approach:
> feed whisper `chunk[i]` with `condition_on_previous_text` and dedupe at the text level
> (Phase 3). Do not fake overlap by re-cutting files.

### Verify
- `list_sources()` returns ≥2 entries and correctly tags the `.monitor` one as `"system"`.
- Record 10 s from the mic → WAV is **16 kHz, mono**, duration 10 s ±0.5 s, RMS > 0
  (i.e. not silence). Confirm with `ffprobe`.
- Repeat against the `.monitor` source **while audio is playing** — must also be non-silent.
- Ctrl-C mid-recording leaves a **playable, non-truncated** WAV.

---

## 6. Phase 3 — ASR worker (`asr.py`)

### Implement
```python
class Transcriber:
    def __init__(self, model=ASR_MODEL, language=None): ...
    def transcribe_chunk(self, path, offset_seconds) -> list[Segment]
```
Using:
```python
WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=8)
model.transcribe(path, language=None, vad_filter=True,
                 vad_parameters=dict(min_silence_duration_ms=500))
```
`language=None` autodetects — required since lectures may be English **or** Thai.
Record the detected language per chunk into the transcript; a whole session is
normally one language, so consider locking to the majority detection after ~3 chunks
to stop mid-lecture flip-flopping.

Timestamps from whisper are **chunk-relative**. Add `offset_seconds` to convert to
absolute session time before writing.

### The dedupe problem (most important part of this phase)
Consecutive chunks can repeat text at the seam. Implement:
```python
def dedupe_overlap(prev_segments, new_segments) -> list[Segment]
```
Compare the tail of the previous chunk against the head of the new one; if the
normalized token sequences overlap, drop the duplicated leading segments from the new
chunk. Normalize by lowercasing and stripping punctuation.

⚠️ **Thai has no spaces between words.** Whitespace tokenization will not work for Thai.
Use character-level comparison (or n-gram similarity over characters) for the overlap
check so it works for both scripts.

### Output format — `transcript.jsonl`, one object per line
```json
{"start": 12.4, "end": 18.9, "text": "...", "lang": "en", "chunk": 3}
```
Append and `flush()` after every segment. Never buffer a whole lecture in memory.

### Verify
- Transcribe `tests/fixtures/` sample audio; word-overlap vs reference above threshold.
- **Real-time factor < 1.0** — log it. If `small` cannot keep up on 12 CPU threads,
  fall back to `base`; record the measured number in the README.
- Dedupe unit test: craft two segment lists with a known duplicated seam, assert the
  phrase appears **exactly once**. Include a **Thai** case with no spaces.
- Kill the process mid-run → `transcript.jsonl` is still valid line-delimited JSON.

---

## 7. Phase 4 — Model benchmark

Measure real-time factor for `small` vs `large-v3-turbo` on a ~2 min sample, English
and Thai. Record results in README. Wire `--model` so the user can choose, and add a
`--hq` re-run path that re-transcribes the saved `audio.wav` with the bigger model for
better Thai accuracy after the lecture ends. Only keep `large-v3-turbo` as the live
default if RTF < 1.0, which is unlikely on this CPU.

---

## 8. Phase 5 — Storage (`store.py`)

SQLite table `sessions`:
`id TEXT PK, title TEXT, started_at TS, ended_at TS, source_kind TEXT, language TEXT, duration REAL, has_notes INT`

Functions: `create_session`, `finish_session`, `list_sessions`, `get_session`,
`session_dir(id)`, `read_transcript(id)`, `write_notes(id, md)`.

Session id: timestamp-based and sortable, e.g. `2026-08-27_0930_lecture-title-slug`.

### Verify
- Create → list → get roundtrip.
- Simulated mid-recording kill: session row still exists, `transcript.jsonl` replays
  cleanly, and `summarize` can run on the partial transcript.

---

## 9. Phase 6 — Pull the summarizer model

`ollama pull qwen3:8b` (~5 GB). **Already started in the background** during scaffolding;
check `/tmp/ollama_pull_qwen3.log` and confirm with `ollama list` before Phase 7.

---

## 10. Phase 7 — Summarization (`summarize.py`)

### Why map-reduce
A 50-minute lecture is far past what an 8B model handles well in one shot. Two stages:

**MAP** — group segments into ~3-minute windows (with timestamps). For each window ask
for 2–4 key points. Return a compact list, no prose.

**REDUCE** — feed all map outputs; merge, drop near-duplicates, order logically, and
split out anything that is administrative ("exam is Friday", "office hours moved")
into a separate **Action items** list rather than mixing it into key ideas.

### Output `notes.md` structure
```markdown
# <title>            <date> · <duration> · <language>
## Key ideas         ← the point of the app. Tight bullets, no filler.
## Terms & definitions
## Action items      ← admin/deadlines, kept separate
```

### Rules
- Call Ollama over HTTP at `OLLAMA_URL` (`/api/chat`), `stream=false` for map/reduce.
- **Prompt in the lecture's language** — Thai lecture must produce Thai notes.
  Prompts live in `config.PROMPTS`, keyed by language.
- Ollama unreachable → clear actionable error, and the **transcript must still be saved**.
  Losing the summary is recoverable; losing the transcript is not.
- `qwen3` is a reasoning model and may emit `<think>` blocks — **strip them** before
  writing `notes.md`.

### Verify
- Run on `tests/fixtures/` transcript: output is markdown bullets, is **< ~15% of input
  length** (it must actually condense), and mentions the fixture's known key terms.
- Use `qwen3:0.6b` in tests so they stay fast and fully offline.
- Assert no `<think>` tags leak into output.

---

## 11. Phase 8 — Live incremental notes (toggleable)

User asked for live notes "if possible, toggleable". Implement as `--live-notes/--no-live-notes`.

When on: every N minutes (default 3, from config) run the MAP stage over new segments
only and append to a running key-ideas pane. On stop, still run the **full** map-reduce
so the final `notes.md` is coherent rather than a pile of incremental fragments.

Must run on a **background thread** so summarization never stalls audio capture or ASR.
If live summarization falls behind, skip a cycle rather than queueing up backlog.

---

## 12. Phase 8b — CLI (`cli.py`)

```
notetaker devices                                 list + label capture sources
notetaker record [--source mic|system] [--title T]
                 [--live-notes] [--lang en|th|auto] [--model small]
notetaker list                                    past sessions
notetaker show <id>                               transcript and/or notes
notetaker summarize <id> [--rerun] [--hq]         re-summarize, no re-record
notetaker export <id> --md|--txt
```
Live display via `rich`: elapsed timer, input level meter, last few transcript lines,
and (if enabled) the live key-ideas pane. Ctrl-C stops cleanly and runs the final summary.

`--source` accepts the friendly `mic`/`system` alias — the user should never have to
type `alsa_output.pci-0000_00_1f.3.analog-stereo.monitor`.

---

## 13. Phase 9 — GUI (deferred, PyQt6)

Only after the CLI is proven end-to-end. Single window: source dropdown, Record/Stop,
timer + level meter, live transcript (left), key ideas (right), session list, Export.
Reuse the exact same modules — the GUI must be a thin layer over the CLI's engine, with
no duplicated capture/ASR logic. Threaded so the UI never blocks.

---

## 14. Definition of done

- [ ] Records from **both** mic and system audio; user can switch via `--source`.
- [ ] Live transcription keeps up on CPU (**measured RTF < 1.0**, number in README).
- [ ] Handles **English and Thai**, including Thai's no-space dedupe edge case.
- [ ] Live notes toggle works and never stalls capture.
- [ ] `notes.md` is genuinely **key ideas only** — dramatically shorter than the transcript.
- [ ] Crash mid-lecture loses at most the current chunk.
- [ ] Test suite passes offline with no API keys.
- [ ] **Acceptance:** a real ~10-minute lecture recorded in English *and* one in Thai,
      with a human confirming the key ideas are accurate and non-redundant.

---

## 15. Traps — read before coding

1. **Do not use `small.en`.** It cannot do Thai. Requirement 5 says English *and* Thai.
2. **Do not install torch.** faster-whisper runs on ctranslate2. Adding torch wastes
   ~2 GB and buys nothing.
3. **Do not transcribe a chunk that is still being written.** Wait until the next chunk
   file appears.
4. **Do not split Thai on whitespace.** Thai text has no word spaces; use character-level
   comparison in dedupe.
5. **Do not buffer the whole lecture in RAM.** Append to `transcript.jsonl` and flush.
6. **Do not one-shot summarize a 50-minute lecture.** Use map-reduce.
7. **Do not `SIGKILL` ffmpeg.** It corrupts the WAV header. Send `SIGINT`.
8. **Do not commit recordings.** `.gitignore` covers `*.wav` / `*.jsonl`; data lives in
   `~/.local/share/notetaker/`.
9. **Strip `<think>` blocks** from qwen3 output before writing notes.
10. **Never add a cloud API fallback.** Offline-only is a hard requirement.
