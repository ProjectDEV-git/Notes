"""A normal ~1-hour class, not a short demo recording.

Three things break at that length and are verified here:

1. The ASR worker can still be draining a backlog when the user stops. A fixed
   two-minute wait truncates an hour of Thai, so the timeout must scale with the
   measured backlog.
2. If the worker is cut short anyway, the untranscribed chunks are the only
   recoverable copy of that part of the class. They must survive.
3. Sixty minutes of transcript produces far more mapped points than one REDUCE
   call can answer, so the reduce must batch instead of truncating.
"""

from __future__ import annotations

import threading

import pytest

from notetaker import config, store, summarize as S
from notetaker.asr import Segment
from notetaker.audio import AudioSource
from notetaker.pipeline import PipelineState, RecordingPipeline


# --------------------------------------------------------------- drain budget
class _StubRecorder:
    """Stands in for the ffmpeg recorder: no audio hardware involved."""

    def __init__(self, session_dir, pending_chunks: int = 0):
        self.session_dir = session_dir
        self.chunks_dir = session_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.audio_path = session_dir / "audio.wav"
        self.elapsed = 3600.0
        self.cleaned = False
        for index in range(pending_chunks):
            (self.chunks_dir / f"chunk_{index:05d}.wav").write_bytes(b"")

    def start(self):
        pass

    def stop(self):
        pass

    def chunks(self):
        return iter(())

    def cleanup_chunks(self):
        self.cleaned = True


def _pipeline(tmp_path, pending_chunks=0, chunks_done=0):
    session = store.create_session(
        "Biology period 3", config.SOURCE_MIC, db_path=tmp_path / "db.sqlite"
    )
    source = AudioSource(name="stub", description="stub", kind=config.SOURCE_MIC)
    pipe = RecordingPipeline(source=source, session=session)
    pipe.recorder = _StubRecorder(tmp_path / "session", pending_chunks)
    pipe.state.chunks_done = chunks_done
    pipe.state.chunks_pending = max(pending_chunks - chunks_done, 0)
    return pipe


def test_drain_budget_is_at_least_the_floor(tmp_path):
    pipe = _pipeline(tmp_path)
    assert pipe.drain_timeout(floor=120.0) >= 120.0


def test_drain_budget_grows_with_the_backlog(tmp_path):
    """An hour of Thai leaves a large backlog and must not be cut off at 2 min."""
    small = _pipeline(tmp_path / "a", pending_chunks=1)
    large = _pipeline(tmp_path / "b", pending_chunks=40)
    large._chunk_cost = 60.0  # measured: slower than real time
    assert large.drain_timeout() > small.drain_timeout()
    # 40 chunks at 60 s each cannot possibly finish inside the old flat 120 s.
    assert large.drain_timeout() > 120.0


def test_drain_budget_uses_measured_cost_not_a_guess(tmp_path):
    fast = _pipeline(tmp_path / "a", pending_chunks=10)
    fast._chunk_cost = 2.0
    slow = _pipeline(tmp_path / "b", pending_chunks=10)
    slow._chunk_cost = 45.0
    assert slow.drain_timeout() > fast.drain_timeout()


# ------------------------------------------------------- chunks are not lost
def test_undrained_chunks_are_kept_for_recovery(tmp_path):
    """Cut short: the chunks are the only copy of the end of the class."""
    pipe = _pipeline(tmp_path, pending_chunks=5, chunks_done=2)
    pipe.state.undrained_chunks = 3

    pipe.finish(cleanup=True)

    assert not pipe.recorder.cleaned, "deleted audio the user could still recover"
    assert list(pipe.recorder.chunks_dir.glob("chunk_*.wav"))


def test_a_clean_finish_still_tidies_up(tmp_path):
    """The normal path must not start leaking chunk files."""
    pipe = _pipeline(tmp_path, pending_chunks=3, chunks_done=3)
    pipe.state.undrained_chunks = 0

    pipe.finish(cleanup=True)

    assert pipe.recorder.cleaned


def test_state_reports_whether_the_class_finished_cleanly():
    assert PipelineState().finished_cleanly
    assert not PipelineState(undrained_chunks=2).finished_cleanly


def test_stop_reports_success_when_the_worker_is_done(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe._asr_thread = None  # nothing outstanding
    assert pipe.stop(timeout=0.1) is True
    assert pipe.state.undrained_chunks == 0


def test_stop_reports_failure_and_counts_what_is_left(tmp_path):
    """A worker still busy at the deadline must be reported, not ignored."""
    pipe = _pipeline(tmp_path, pending_chunks=4, chunks_done=1)
    release = threading.Event()
    stuck = threading.Thread(target=release.wait, daemon=True)
    stuck.start()
    pipe._asr_thread = stuck
    pipe._threads = [stuck]
    try:
        assert pipe.stop(timeout=0.2) is False
        assert pipe.state.undrained_chunks == 3
    finally:
        release.set()


# ------------------------------------------------------------ batched reduce
def _points(count: int) -> list[str]:
    """Distinctive points, so nothing is lost to duplicate detection."""
    return [f"Topic {i} explains mechanism {i} with value {i * 7} joules" for i in range(count)]


def test_a_short_class_still_reduces_in_one_call(monkeypatch):
    calls = []

    def fake_chat(prompt, **kwargs):
        calls.append(prompt)
        return "- kept point"

    monkeypatch.setattr(S, "chat", fake_chat)
    S.reduce_points(_points(5), "en", "stub")
    assert len(calls) == 1, "short input must not pay for batching"


def test_a_full_hour_is_reduced_in_batches(monkeypatch):
    """60 points in one call would truncate; it must be split."""
    calls = []

    def fake_chat(prompt, **kwargs):
        calls.append(prompt)
        return "\n".join(f"- {line}" for line in prompt.splitlines() if line.startswith("- "))[:2000]

    monkeypatch.setattr(S, "chat", fake_chat)
    S.reduce_points(_points(60), "en", "stub")
    assert len(calls) > 1


def test_no_point_is_dropped_by_the_split(monkeypatch):
    """Every input point must reach some batch, or content is lost silently."""
    seen: list[str] = []

    def fake_chat(prompt, **kwargs):
        seen.append(prompt)
        return "- placeholder"

    monkeypatch.setattr(S, "chat", fake_chat)
    points = _points(45)
    S.reduce_points(points, "en", "stub")

    combined = "\n".join(seen)
    missing = [p for p in points if p not in combined]
    assert not missing, f"{len(missing)} points never reached a reduce call"


def test_a_batch_returning_nothing_does_not_delete_its_points(monkeypatch):
    """A model that answers with prose must not cost the user that batch."""
    seen: list[str] = []

    def fake_chat(prompt, **kwargs):
        seen.append(prompt)
        # No bullets at all, which parse_bullets yields nothing from.
        return "Here are the notes you asked for."

    monkeypatch.setattr(S, "chat", fake_chat)
    points = _points(40)
    S.reduce_points(points, "en", "stub")

    final = seen[-1]
    assert any(p in final for p in points), "content vanished when a batch failed"


def test_reduce_terminates_when_the_model_echoes_its_input(monkeypatch):
    """A model that never condenses must not spin forever."""
    count = {"n": 0}

    def fake_chat(prompt, **kwargs):
        count["n"] += 1
        if count["n"] > 40:
            raise AssertionError("reduce did not terminate")
        return "\n".join(line for line in prompt.splitlines() if line.startswith("- "))

    monkeypatch.setattr(S, "chat", fake_chat)
    S.reduce_points(_points(80), "en", "stub")
