"""Phase 2 verification: audio device enumeration and capture.

Tests that touch real hardware are marked and skipped when unavailable, so the
suite still runs on a machine without PulseAudio.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

import pytest

from notetaker import audio, config

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _have_pulse() -> bool:
    if shutil.which("pactl") is None or shutil.which("ffmpeg") is None:
        return False
    try:
        return subprocess.run(["pactl", "info"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


def _have_avfoundation() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        return bool(audio._list_sources_avfoundation())
    except Exception:
        return False


def _have_capture() -> bool:
    return _have_avfoundation() if audio.IS_MACOS else _have_pulse()


needs_audio = pytest.mark.skipif(not _have_capture(), reason="no capture backend available")
needs_pulse = pytest.mark.skipif(not _have_pulse(), reason="PulseAudio/ffmpeg unavailable")


# ---------------------------------------------------------------- enumeration
@needs_audio
def test_lists_at_least_one_source():
    assert audio.list_sources()


@needs_pulse
def test_monitor_sources_are_tagged_as_system_audio():
    """A '.monitor' source captures playback: that is the online-lecture path."""
    for src in audio.list_sources():
        expected = config.SOURCE_SYSTEM if src.name.endswith(".monitor") else config.SOURCE_MIC
        assert src.kind == expected, f"{src.name} mis-tagged as {src.kind}"


@needs_audio
def test_resolve_accepts_friendly_aliases():
    """Users type 'mic'/'system', never a raw device name."""
    for alias in (config.SOURCE_MIC, config.SOURCE_SYSTEM):
        if any(s.kind == alias for s in audio.list_sources()):
            assert audio.resolve_source(alias).kind == alias


@needs_audio
def test_resolve_none_returns_a_usable_default():
    assert audio.resolve_source(None).name


@needs_audio
def test_resolve_rejects_unknown_source():
    with pytest.raises(audio.AudioError):
        audio.resolve_source("no-such-device")


# ------------------------------------------------------------------- capture
def _record(source, tmp_path, seconds: float, chunk_seconds: int = 3):
    rec = audio.Recorder(source, tmp_path, chunk_seconds=chunk_seconds)
    rec.start()
    collected: list = []
    worker = threading.Thread(
        target=lambda: collected.extend(rec.chunks(poll_interval=0.2)), daemon=True
    )
    worker.start()
    time.sleep(seconds)
    rec.stop()
    worker.join(timeout=10)
    return rec, collected


@needs_audio
def test_chunks_have_whisper_native_format(tmp_path):
    """16 kHz mono PCM is what faster-whisper expects; wrong rate degrades ASR."""
    src = audio.resolve_source(config.SOURCE_MIC)
    _, chunks = _record(src, tmp_path, seconds=8)

    assert chunks, "no chunks produced"
    for path in chunks:
        with wave.open(str(path), "rb") as wf:
            assert wf.getframerate() == config.SAMPLE_RATE
            assert wf.getnchannels() == config.CHANNELS
            assert wf.getsampwidth() == 2  # 16-bit


@needs_audio
def test_full_chunks_have_expected_duration(tmp_path):
    src = audio.resolve_source(config.SOURCE_MIC)
    _, chunks = _record(src, tmp_path, seconds=8, chunk_seconds=3)

    # The last chunk is a partial tail; the rest should be full length.
    for path in chunks[:-1]:
        assert abs(audio.wav_duration(path) - 3.0) < 0.5


@needs_audio
def test_session_wav_is_written_and_playable(tmp_path):
    """The continuous recording is what enables a high-accuracy re-run later."""
    src = audio.resolve_source(config.SOURCE_MIC)
    rec, _ = _record(src, tmp_path, seconds=6)

    assert rec.audio_path.exists()
    assert audio.wav_duration(rec.audio_path) > 3.0


@needs_audio
def test_stop_finalizes_wav_header(tmp_path):
    """Regression guard for trap #7: SIGKILL corrupts the WAV, SIGINT does not.

    Verified empirically that killing ffmpeg outright leaves a 0-byte
    unreadable file. Recorder.stop() must send SIGINT first.
    """
    src = audio.resolve_source(config.SOURCE_MIC)
    rec, _ = _record(src, tmp_path, seconds=5)

    assert rec.audio_path.stat().st_size > 0
    # Readable header == ffmpeg finalized cleanly.
    with wave.open(str(rec.audio_path), "rb") as wf:
        assert wf.getnframes() > 0
    assert not rec.is_running


@needs_audio
def test_no_chunk_is_yielded_twice(tmp_path):
    """Duplicate chunks would duplicate transcript text."""
    src = audio.resolve_source(config.SOURCE_MIC)
    _, chunks = _record(src, tmp_path, seconds=9)

    assert len(chunks) == len(set(chunks))


@needs_audio
def test_final_partial_chunk_is_drained(tmp_path):
    """Stopping mid-chunk must not silently discard the last seconds of a lecture."""
    src = audio.resolve_source(config.SOURCE_MIC)
    rec, chunks = _record(src, tmp_path, seconds=8, chunk_seconds=3)

    on_disk = sorted(rec.chunks_dir.glob("chunk_*.wav"))
    assert len(chunks) == len(on_disk), "a chunk was left unprocessed"


@needs_pulse
def test_captures_real_signal_from_system_audio(tmp_path):
    """The online-lecture path: play a tone, confirm the monitor source hears it.

    Regression guard: 'system' must resolve to the monitor of the sink that is
    actually playing. Resolving the built-in card's monitor while audio goes to
    Bluetooth records an hour of silence.
    """
    sources = audio.list_sources()
    if not any(s.kind == config.SOURCE_SYSTEM for s in sources):
        pytest.skip("no monitor source on this machine")

    src = audio.resolve_source(config.SOURCE_SYSTEM)
    # Play into the very sink whose monitor we are about to capture.
    sink = src.name.removesuffix(".monitor")

    tone = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
         "-f", "pulse", "-device", sink, "tone"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.5)
        _, chunks = _record(src, tmp_path, seconds=6)
        assert chunks
        assert max(audio.rms_level(c) for c in chunks) > 0.01, "monitor captured silence"
    finally:
        tone.terminate()
        tone.wait(timeout=5)


@needs_audio
def test_cleanup_removes_chunks_but_keeps_audio(tmp_path):
    src = audio.resolve_source(config.SOURCE_MIC)
    rec, _ = _record(src, tmp_path, seconds=5)

    rec.cleanup_chunks()
    assert not rec.chunks_dir.exists()
    assert rec.audio_path.exists()


# ------------------------------------------------------------------ duration
def test_duration_of_a_clean_wav():
    fixture = Path("tests/fixtures/lecture_en_30s.wav")
    if not fixture.exists():
        pytest.skip("fixture missing")
    assert audio.wav_duration(fixture) == pytest.approx(30.0, abs=0.1)


def test_duration_ignores_placeholder_frame_count(tmp_path):
    """Regression: a SIGINT-stopped recording has a bogus RIFF frame count.

    ffmpeg cannot always seek back to patch the header, leaving INT32_MAX.
    Trusting it reported a 65-second lecture as 37 hours.
    """
    path = tmp_path / "truncated.wav"
    seconds, rate = 2, config.SAMPLE_RATE
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * rate * seconds)

    # Corrupt the header the way an interrupted ffmpeg does.
    raw = bytearray(path.read_bytes())
    raw[40:44] = (2147483647).to_bytes(4, "little")  # data chunk size
    path.write_bytes(raw)

    assert audio.wav_duration(path) == pytest.approx(seconds, abs=0.1)


@needs_audio
def test_recorded_duration_is_realistic(tmp_path):
    """End-to-end guard: a 6 second recording must not report hours."""
    src = audio.resolve_source(config.SOURCE_MIC)
    rec, _ = _record(src, tmp_path, seconds=6)
    assert 3.0 < audio.wav_duration(rec.audio_path) < 30.0


def test_bad_device_raises_clear_error(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    bogus = audio.AudioSource(name="definitely-not-a-device", description="bogus",
                              kind=config.SOURCE_MIC)
    with pytest.raises(audio.AudioError):
        audio.Recorder(bogus, tmp_path).start()


# ------------------------------------------------------- macOS backend
# These run everywhere: they parse a captured ffmpeg listing rather than
# touching hardware, so the macOS path stays covered from a Linux machine.
_AVF_LISTING = """\
[AVFoundation indev @ 0x7f8] AVFoundation video devices:
[AVFoundation indev @ 0x7f8] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f8] [1] Capture screen 0
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x7f8] [1] BlackHole 2ch
"""


def _parse_avf(monkeypatch, listing=_AVF_LISTING):
    monkeypatch.setattr(audio, "_avfoundation_output", lambda: listing)
    return audio._list_sources_avfoundation()


def test_macos_listing_ignores_video_devices(monkeypatch):
    """Capturing 'FaceTime HD Camera' as audio device 0 would record nothing."""
    sources = _parse_avf(monkeypatch)
    assert [s.description for s in sources] == ["MacBook Pro Microphone", "BlackHole 2ch"]


def test_macos_device_names_are_avfoundation_indices(monkeypatch):
    """ffmpeg takes '<video>:<audio>', so an audio-only input must start with ':'."""
    assert [s.name for s in _parse_avf(monkeypatch)] == [":0", ":1"]


def test_macos_loopback_is_the_system_audio_source(monkeypatch):
    """macOS has no monitor source: only a loopback driver can capture playback."""
    sources = _parse_avf(monkeypatch)
    kinds = {s.description: s.kind for s in sources}
    assert kinds["BlackHole 2ch"] == config.SOURCE_SYSTEM
    assert kinds["MacBook Pro Microphone"] == config.SOURCE_MIC


def test_macos_first_audio_device_is_default(monkeypatch):
    sources = _parse_avf(monkeypatch)
    assert sources[0].is_default and not sources[1].is_default


def test_macos_system_without_loopback_explains_the_fix(monkeypatch):
    """The failure mode users will hit: no BlackHole installed."""
    listing = """\
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] MacBook Pro Microphone
"""
    monkeypatch.setattr(audio, "IS_MACOS", True)
    monkeypatch.setattr(audio, "_avfoundation_output", lambda: listing)
    monkeypatch.setattr(audio, "list_sources", audio._list_sources_avfoundation)
    with pytest.raises(audio.AudioError, match="blackhole"):
        audio.resolve_source(config.SOURCE_SYSTEM)


def test_macos_source_can_be_named_by_description(monkeypatch):
    """':1' is meaningless to a user; 'BlackHole 2ch' is what they see."""
    monkeypatch.setattr(audio, "IS_MACOS", True)
    monkeypatch.setattr(audio, "_avfoundation_output", lambda: _AVF_LISTING)
    monkeypatch.setattr(audio, "list_sources", audio._list_sources_avfoundation)
    assert audio.resolve_source("BlackHole 2ch").name == ":1"


def test_macos_recorder_uses_avfoundation_input(monkeypatch, tmp_path):
    """'-f pulse' does not exist on macOS; the input must be avfoundation."""
    monkeypatch.setattr(audio, "IS_MACOS", True)
    src = audio.AudioSource(name=":1", description="BlackHole 2ch",
                            kind=config.SOURCE_SYSTEM)
    cmd = audio.Recorder(src, tmp_path)._build_command()
    assert "pulse" not in cmd
    assert cmd[cmd.index("-f") + 1] == "avfoundation"
    assert cmd[cmd.index("-i") + 1] == ":1"


def test_linux_recorder_still_uses_pulse_input(tmp_path):
    src = audio.AudioSource(name="mic0", description="Mic", kind=config.SOURCE_MIC)
    cmd = audio.Recorder(src, tmp_path)._build_command()
    expected = "avfoundation" if audio.IS_MACOS else "pulse"
    assert cmd[cmd.index("-f") + 1] == expected
