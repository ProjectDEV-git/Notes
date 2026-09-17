# NoteTaker upgrade plan

## Direction

Evolve the existing offline Python recorder into a reliable, searchable study
workspace. Preserve English/Thai, CPU-friendly defaults, audio-only recording,
CLI commands, existing sessions, and language packs. No cloud fallback, inference
rewrite, or GUI dependency in the first milestone.

## Current implementation slice

Atomic artifact publication is implemented and regression-validated:
- Same-directory temporary writes, file flush/fsync, atomic replacement, and
  best-effort directory fsync.
- Publish notes before updating the database notes flag.
- HQ and catch-up replace transcripts without truncating the previous file first.
- Failed publication reports disk/permission guidance; catch-up retains chunks.
- Tests cover injected failures, cleanup, notes flags, and multilingual roundtrip.

Durable MAP checkpoints are **not implemented** yet. Preliminary scaffolding was
removed rather than shipping an unused cache. Model defaults and UI are unchanged.

## Tasks, dependencies, and acceptance gates

| ID | Work | Depends on | Acceptance criteria |
|---|---|---|---|
| B1 | Reproducible tests | None | Isolated data/config; explicitly separate deterministic, model, network, and hardware tests. |
| B2 | Benchmark harness | B1 | Consented English/Thai fixtures; hardware/settings/model identity; capture latency, ASR RTF, memory, MAP/REDUCE timings, and human-reviewed coverage. |
| R1 | Atomic artifacts | Baseline | Failures retain old files; notes flag changes only after publication; HQ/catch-up never truncate first. Current slice. |
| R2 | Durable MAP checkpoints | R1 | Restart reuses compatible contiguous windows; tails never skipped; corrupt/stale records safely recomputed. |
| R3 | Processing states and migrations | R1, R2 | Distinct capture/transcription/summary states; migrations preserve sessions; retry/cancellation and ownership prevent duplicate processing. |
| R4 | Recording lifecycle | R3 | Capture starts independently of model loading; audio survives model failure; device loss, silence, FFmpeg exit, and disk pressure visible. |
| N1 | Evidence-linked extraction | R2, B2 | Stable segment IDs/revisions; validate schema and source references; unsupported claims omitted or flagged. |
| N2 | Quick/polished notes | N1 | MAP results immediately usable; reduction failure preserves content; final minutes/homework survive tests and human review. |
| P1 | Performance profiles | B2, N2 | Compare existing models/settings under concurrent load; publish separate English/Thai speed-memory-quality tradeoffs. |
| S1 | Searchable library | R3, N1 | Subjects/tags/date filters; SQLite search with Thai/short-query tests; timestamped excerpts; edits invalidate derived indexes. |
| S2 | Import/export/backup | R3, S1 | FFmpeg import, SRT/VTT export, tested restores, reversible deletion; retain sole recoverable artifacts. |
| D1 | Shared services | R3, N2 | CLI and GUI share workflows; immutable progress snapshots. |
| D2 | Desktop/distribution | D1, S2 | Explicit GUI/license decision; responsive views, Thai typography, keyboard access, install tests and rollback-aware updates. |

## Next milestone: R2 design

Persist successful MAP ranges with schema version, segment boundaries/hashes,
grounding-context hash, extracted points, language, reading level, immutable model
identity where available, and prompt/configuration fingerprint.

- Never treat a segment count alone as evidence of compatibility.
- Reuse only contiguous validated coverage; include successful empty MAP results.
- If model identity is unavailable, recompute instead of trusting a mutable tag.
- Distinguish live window-local grounding from whole-transcript grounding.
- Independent atomic records must prevent late live work overwriting newer work.
- Test restart, corruption, transcript edits/appended tails, changed model/prompts/
  language/level, and failures mid-MAP.
- Integrate and test live recording and catch-up before declaring completion.

## Validation and boundaries

Initial targeted baseline: **219 passed, 3 skipped**, Ollama disabled.

Atomic-write implementation validation:
- Complete suite in a temporary checkout: **393 passed, 20 skipped**, exit 0,
  117.10 seconds. Audio/Ollama access disabled; cached Whisper test ran offline.
- Two additional mocked CLI publication-failure cases added afterward: the
  complete atomic-write test file passed **13 tests in 0.44 seconds**.
- Full-run log: `/tmp/notetaker-atomic-verify.t7LX6Y/pytest.log`.
- Exit status: `/tmp/notetaker-atomic-verify.t7LX6Y/exit-status`.
- Live capture and Ollama integration remain unverified; skips are not passes.

Filesystem publication is not a transaction with SQLite: publication followed by
a database failure can leave stale metadata. Reconciliation belongs in R3.
Directory fsync is best-effort. Abrupt termination may leave sibling temporary
files but must not truncate the prior published artifact. Live JSONL append
behavior is unchanged. Transcript replacement currently materializes text in
memory; streaming replacement and writer ownership remain future work.

No fresh inference benchmarks or universal power-loss guarantees are claimed.

## Research

- https://github.com/SYSTRAN/faster-whisper — INT8, VAD, batching and benchmark parity.
- https://docs.ollama.com/capabilities/structured-outputs — schema, not factual guarantees.
- https://www.sqlite.org/fts5.html — full-text/trigram search and query limitations.

FTS5/trigram availability was verified locally. VAD is already enabled. README
performance figures are historical, not newly reproduced. Windows, diarization,
cloud sync, and study chat are outside the first release.
