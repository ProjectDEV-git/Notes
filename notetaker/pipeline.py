"""Recording pipeline: capture -> transcribe -> (optional live notes).

Runs the audio recorder, ASR worker, and optional incremental summarizer on
separate threads so that summarization can never stall audio capture. If the
live summarizer falls behind it skips a cycle rather than queueing a backlog.

See docs/BUILD_PLAN.md phases 8 and 8b.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import config, store, summarize
from .asr import Segment, Transcriber, TranscriptWriter
from .audio import AudioSource, Recorder, rms_level


@dataclass
class PipelineState:
    """Snapshot of a running session, for the live display."""

    elapsed: float = 0.0
    level: float = 0.0
    chunks_done: int = 0
    chunks_pending: int = 0
    segments: list[Segment] = field(default_factory=list)
    live_points: list[str] = field(default_factory=list)
    language: str | None = None
    error: str | None = None
    warning: str | None = None
    # Set when stop() gave up before the ASR worker had drained its backlog.
    # The chunks are still on disk and can be finished later.
    undrained_chunks: int = 0
    # Key/admin points already extracted by the live-notes thread, and how many
    # segments they cover. Reused at finish so an hour of MAP work is not
    # repeated after the bell.
    live_key_points: list[str] = field(default_factory=list)
    live_admin_points: list[str] = field(default_factory=list)
    live_consumed: int = 0

    @property
    def recent_text(self) -> list[str]:
        return [s.text for s in self.segments[-6:]]

    @property
    def is_falling_behind(self) -> bool:
        """True when ASR cannot keep up and a backlog is accumulating.

        Chunks queue safely on disk, so nothing is lost, but the user should
        know that stopping will not end the work immediately. This happens with
        Thai, which runs several times slower than real time on CPU.
        """
        return self.chunks_pending >= 3

    @property
    def finished_cleanly(self) -> bool:
        """True when every recorded chunk made it into the transcript."""
        return self.undrained_chunks == 0


class RecordingPipeline:
    """Owns the threads for one recording session."""

    def __init__(
        self,
        source: AudioSource,
        session: store.Session,
        model: str = config.ASR_MODEL,
        language: str | None = None,
        live_notes: bool = False,
        summary_model: str = config.SUMMARY_MODEL,
        chunk_seconds: int = config.CHUNK_SECONDS,
        live_interval: int = config.LIVE_NOTES_INTERVAL_SECONDS,
        on_update: Callable[[PipelineState], None] | None = None,
    ) -> None:
        self.source = source
        self.session = session
        self.model = model
        self.language = language
        self.live_notes = live_notes
        self.summary_model = summary_model
        self.live_interval = live_interval
        self.on_update = on_update

        self.state = PipelineState(language=language)
        self.recorder = Recorder(source, session.directory, chunk_seconds=chunk_seconds)
        self._transcriber: Transcriber | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._chunk_seconds = chunk_seconds
        # Rolling measurement of how long one chunk takes to transcribe, used to
        # size the drain timeout. Thai on CPU runs several times slower than
        # real time, so a fixed timeout truncates the end of a long class.
        self._chunk_cost: float = 0.0
        self._asr_thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        # Load the model before recording so startup latency does not eat
        # the first minute of the lecture.
        self._transcriber = Transcriber(self.model, language=self.language)
        self.recorder.start()

        asr = threading.Thread(target=self._transcribe_loop, name="asr", daemon=True)
        self._asr_thread = asr
        self._threads = [
            asr,
            threading.Thread(target=self._tick_loop, name="tick", daemon=True),
        ]
        if self.live_notes:
            self._threads.append(
                threading.Thread(target=self._live_notes_loop, name="live", daemon=True)
            )
        for thread in self._threads:
            thread.start()

    def drain_timeout(self, floor: float = 120.0) -> float:
        """How long to wait for the ASR worker, based on the real backlog.

        A flat timeout silently truncates a long class: an hour of Thai can
        leave a 20-chunk backlog that needs far longer than two minutes to
        finish. Budget the measured per-chunk cost for every pending chunk,
        with generous headroom, and never wait less than `floor`.
        """
        with self._lock:
            pending = self.state.chunks_pending
        cost = self._chunk_cost or float(self._chunk_seconds)
        return max(floor, pending * cost * 1.5 + floor)

    def stop(self, timeout: float | None = None) -> bool:
        """Stop capture and wait for the ASR worker to drain remaining chunks.

        Returns True when the worker drained. When it did not, the leftover
        chunks stay on disk and `state.undrained_chunks` says how many, so the
        caller can tell the user how to finish the job rather than deleting
        their audio.
        """
        self.recorder.stop()
        budget = self.drain_timeout() if timeout is None else timeout
        self._stop.set()

        # The ASR worker is the only thread whose unfinished work loses data,
        # so it gets the whole budget; the display threads exit promptly.
        drained = True
        if self._asr_thread is not None:
            self._asr_thread.join(timeout=budget)
            drained = not self._asr_thread.is_alive()
        for thread in self._threads:
            if thread is not self._asr_thread:
                thread.join(timeout=10.0)

        with self._lock:
            self.state.undrained_chunks = 0 if drained else self._pending_on_disk()
        return drained

    # -- worker threads --------------------------------------------------
    def _pending_on_disk(self) -> int:
        """Chunks written but not yet transcribed. Caller holds the lock."""
        try:
            on_disk = len(list(self.recorder.chunks_dir.glob("chunk_*.wav")))
        except Exception:
            return 0
        return max(on_disk - self.state.chunks_done, 0)

    def _transcribe_loop(self) -> None:
        assert self._transcriber is not None
        try:
            with TranscriptWriter(self.session.transcript_path) as writer:
                for index, chunk in enumerate(self.recorder.chunks()):
                    try:
                        level = rms_level(chunk)
                    except Exception:
                        level = 0.0

                    started = time.monotonic()
                    segments = self._transcriber.transcribe_chunk(
                        chunk,
                        offset_seconds=index * self._chunk_seconds,
                        chunk_index=index,
                    )
                    writer.write(segments)
                    elapsed = time.monotonic() - started
                    # Exponential moving average: recent chunks predict the
                    # remaining backlog better than a lifetime mean.
                    self._chunk_cost = (
                        elapsed if not self._chunk_cost
                        else self._chunk_cost * 0.7 + elapsed * 0.3
                    )

                    with self._lock:
                        self.state.segments.extend(segments)
                        self.state.chunks_done = index + 1
                        self.state.level = level
                        self.state.language = self._transcriber.language
                    self._notify()
        except Exception as exc:  # keep the recording alive; surface the problem
            with self._lock:
                self.state.error = f"transcription stopped: {exc}"
            self._notify()

    def _tick_loop(self) -> None:
        """Keeps the elapsed clock moving and tracks the ASR backlog."""
        while not self._stop.is_set():
            try:
                on_disk = len(list(self.recorder.chunks_dir.glob("chunk_*.wav")))
            except Exception:
                on_disk = 0
            with self._lock:
                self.state.elapsed = self.recorder.elapsed
                # The newest chunk is still being written, so it is not backlog.
                self.state.chunks_pending = max(on_disk - self.state.chunks_done - 1, 0)
                if self.state.is_falling_behind:
                    self.state.warning = (
                        f"transcription is {self.state.chunks_pending} chunks behind; "
                        "it will finish after you stop"
                    )
                else:
                    self.state.warning = None
            self._notify()
            time.sleep(0.5)

    def _live_notes_loop(self) -> None:
        """Periodically summarize new segments only.

        Deliberately does not queue: if a cycle is still running when the next
        is due, the next is skipped. Falling behind must not snowball.
        """
        consumed = 0
        while not self._stop.is_set():
            if self._stop.wait(self.live_interval):
                break

            with self._lock:
                pending = self.state.segments[consumed:]
                language = self.state.language or "en"
            if not pending:
                continue

            window_text = " ".join(s.text for s in pending)
            try:
                window = summarize.Window(pending[0].start, pending[-1].end, window_text)
                keys, admins = summarize.map_window(window, language, self.summary_model)
            except summarize.SummarizerError:
                continue  # live notes are best-effort; the transcript is safe

            consumed += len(pending)
            with self._lock:
                # Kept separately from the display list so the finish step can
                # reuse the MAP work instead of paying for it a second time.
                self.state.live_key_points.extend(keys)
                self.state.live_admin_points.extend(admins)
                self.state.live_consumed = consumed
                self.state.live_points = summarize.dedupe_points(
                    self.state.live_points + keys + admins
                )
            self._notify()

    def _notify(self) -> None:
        if self.on_update:
            try:
                self.on_update(self.state)
            except Exception:
                pass  # a broken display must never kill the recording

    # -- results ---------------------------------------------------------
    def finish(self, cleanup: bool = True) -> float:
        """Finalize the session row and remove transient chunks.

        Chunks are only deleted once the ASR worker has drained them. If it was
        cut short, they stay on disk: an untranscribed chunk is the only copy of
        that part of the class that can still be recovered cheaply.
        """
        wall_clock = self.recorder.elapsed
        duration = wall_clock
        try:
            if self.recorder.audio_path.exists():
                from .audio import wav_duration

                measured = wav_duration(self.recorder.audio_path)
                # Wall clock is the ground truth for how long we recorded.
                # Only trust the file if it agrees, so a malformed WAV header
                # can never store a 65-second lecture as 37 hours.
                if wall_clock <= 0 or abs(measured - wall_clock) <= max(30.0, wall_clock * 0.25):
                    duration = measured
        except Exception:
            pass

        store.finish_session(
            self.session.id,
            duration=duration,
            language=self.state.language,
        )
        if cleanup and self.state.finished_cleanly:
            self.recorder.cleanup_chunks()
        return duration
