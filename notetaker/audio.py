"""Audio capture for NoteTaker.

Captures lecture audio via ffmpeg, writing both:
  * rolling fixed-length chunks, handed to the ASR worker as they complete
  * one continuous session WAV, kept for high-accuracy re-runs

Two source kinds matter:
  mic     - a real input device (in-person lecture)
  system  - what the speakers are playing (online lecture: Zoom / Teams /
            YouTube). On Linux this is a PulseAudio ".monitor" source; on
            macOS it requires a loopback device such as BlackHole because
            CoreAudio exposes no monitor of the output by default.

Platform support:
  Linux  - ffmpeg '-f pulse', devices enumerated with pactl
  macOS  - ffmpeg '-f avfoundation', devices enumerated from ffmpeg itself

See docs/BUILD_PLAN.md phase 2.
"""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import config

IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

# Device names that are loopback drivers: on macOS these are the only way to
# capture system audio, since CoreAudio has no monitor source of its own.
_MACOS_LOOPBACK_HINTS = (
    "blackhole",
    "soundflower",
    "loopback",
    "ishowu",
    "aggregate",
    "multi-output",
    "vb-cable",
    "existential audio",
)


class AudioError(RuntimeError):
    """Raised when capture cannot start or a device is unusable."""


# Canonical PCM WAV header length, used to recover the true sample count when
# ffmpeg leaves a placeholder frame count in the header.
_WAV_HEADER_BYTES = 44


@dataclass(frozen=True)
class AudioSource:
    """A capture source: a PulseAudio device on Linux, an AVFoundation one on macOS."""

    name: str  # raw device spec passed to ffmpeg (":1" on macOS)
    description: str  # human-readable label
    kind: str  # config.SOURCE_MIC or config.SOURCE_SYSTEM
    is_default: bool = False

    @property
    def label(self) -> str:
        tag = "system audio" if self.kind == config.SOURCE_SYSTEM else "microphone"
        star = " (default)" if self.is_default else ""
        return f"{self.description} [{tag}]{star}"


# --------------------------------------------------------------------------
# Device enumeration
# --------------------------------------------------------------------------
def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise AudioError(f"required command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"{cmd[0]} timed out") from exc
    if proc.returncode != 0:
        raise AudioError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _default_source_name() -> str | None:
    try:
        for line in _run(["pactl", "info"]).splitlines():
            if line.startswith("Default Source:"):
                return line.split(":", 1)[1].strip()
    except AudioError:
        pass
    return None


def _default_sink_monitor() -> str | None:
    """Monitor source of the current default sink.

    This is the correct 'system audio' target: if the user switches output to
    Bluetooth headphones mid-lecture, the built-in card's monitor would capture
    silence. Always follow the sink that is actually playing.
    """
    try:
        for line in _run(["pactl", "info"]).splitlines():
            if line.startswith("Default Sink:"):
                return line.split(":", 1)[1].strip() + ".monitor"
    except AudioError:
        pass
    return None


def _list_sources_pulse() -> list[AudioSource]:
    """Enumerate PulseAudio/PipeWire capture sources (Linux).

    A source whose name ends in '.monitor' captures playback, which is what we
    want for online lectures. Everything else is treated as a microphone.
    """
    default = _default_source_name()
    descriptions: dict[str, str] = {}

    # `pactl list sources` gives descriptions; the short form gives stable names.
    current: str | None = None
    for raw in _run(["pactl", "list", "sources"]).splitlines():
        line = raw.strip()
        if line.startswith("Name:"):
            current = line.split(":", 1)[1].strip()
        elif line.startswith("Description:") and current:
            descriptions[current] = line.split(":", 1)[1].strip()
            current = None

    sources: list[AudioSource] = []
    for raw in _run(["pactl", "list", "short", "sources"]).splitlines():
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1]
        kind = config.SOURCE_SYSTEM if name.endswith(".monitor") else config.SOURCE_MIC
        sources.append(
            AudioSource(
                name=name,
                description=descriptions.get(name, name),
                kind=kind,
                is_default=(name == default),
            )
        )
    return sources


# ---------------------------------------------------------------- macOS
# ffmpeg prints AVFoundation devices to stderr and exits non-zero, e.g.
#   [AVFoundation indev @ 0x...] AVFoundation audio devices:
#   [AVFoundation indev @ 0x...] [0] MacBook Pro Microphone
#   [AVFoundation indev @ 0x...] [1] BlackHole 2ch
_AVF_AUDIO_HEADER = re.compile(r"AVFoundation (audio|video) devices")
_AVF_DEVICE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


def _macos_device_kind(description: str) -> str:
    """Classify an AVFoundation input as microphone or system audio.

    macOS has no monitor source: capturing what the speakers play requires a
    virtual loopback driver (BlackHole, Soundflower, Loopback). Those are the
    only devices that can serve the online-lecture path.
    """
    lowered = description.lower()
    if any(hint in lowered for hint in _MACOS_LOOPBACK_HINTS):
        return config.SOURCE_SYSTEM
    return config.SOURCE_MIC


def _avfoundation_output() -> str:
    """Raw ffmpeg device listing. ffmpeg exits 1 here by design, so do not use _run."""
    if shutil.which(config.FFMPEG_BIN) is None:
        raise AudioError("ffmpeg not found on PATH (install it with: brew install ffmpeg)")
    try:
        proc = subprocess.run(
            [config.FFMPEG_BIN, "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("ffmpeg timed out listing avfoundation devices") from exc
    return proc.stderr


def _list_sources_avfoundation() -> list[AudioSource]:
    """Enumerate macOS capture devices via ffmpeg's AVFoundation listing.

    Device *indices* are what ffmpeg accepts as input (':1' means audio device
    1), and they are not stable across reboots or device plugging, so they are
    always re-resolved rather than stored.
    """
    sources: list[AudioSource] = []
    in_audio = False
    for raw in _avfoundation_output().splitlines():
        header = _AVF_AUDIO_HEADER.search(raw)
        if header:
            in_audio = header.group(1) == "audio"
            continue
        if not in_audio:
            continue
        match = _AVF_DEVICE.search(raw)
        if not match:
            continue
        index, description = match.group(1), match.group(2).strip()
        sources.append(
            AudioSource(
                # ffmpeg avfoundation input is "<video>:<audio>"; audio only.
                name=f":{index}",
                description=description,
                kind=_macos_device_kind(description),
                # AVFoundation lists the system default input first.
                is_default=(not sources),
            )
        )
    return sources


def list_sources() -> list[AudioSource]:
    """Enumerate capture sources for the current platform."""
    if IS_MACOS:
        return _list_sources_avfoundation()
    return _list_sources_pulse()


def resolve_source(selector: str | None) -> AudioSource:
    """Resolve 'mic', 'system', a raw device name, or None (default) to a source.

    Users should never have to type a raw name like
    'alsa_output.pci-0000_00_1f.3.analog-stereo.monitor'.
    """
    sources = list_sources()
    if not sources:
        raise AudioError(
            "no audio capture sources found "
            + (
                "(grant microphone permission to your terminal in "
                "System Settings > Privacy & Security > Microphone)"
                if IS_MACOS
                else "(is PipeWire/PulseAudio running?)"
            )
        )

    if selector is None:
        for src in sources:
            if src.is_default:
                return src
        return sources[0]

    sel = selector.strip()
    if sel in (config.SOURCE_MIC, config.SOURCE_SYSTEM):
        matches = [s for s in sources if s.kind == sel]
        if not matches:
            if IS_MACOS and sel == config.SOURCE_SYSTEM:
                raise AudioError(
                    "no system-audio source on this Mac. macOS cannot capture playback "
                    "without a loopback driver. Install one, e.g.:\n"
                    "  brew install blackhole-2ch\n"
                    "then send lecture audio to it with a Multi-Output Device "
                    "(Audio MIDI Setup) so you can still hear it."
                )
            raise AudioError(f"no '{sel}' source available on this machine")

        if sel == config.SOURCE_SYSTEM and not IS_MACOS:
            # Follow the sink that is actually playing. Picking the wrong
            # card's monitor (e.g. built-in while audio goes to Bluetooth)
            # silently records silence for the whole lecture.
            active = _default_sink_monitor()
            for src in matches:
                if src.name == active:
                    return src

        # Prefer the default device when several match.
        for src in matches:
            if src.is_default:
                return src
        return matches[0]

    for src in sources:
        if src.name == sel:
            return src
    # On macOS the raw name is an opaque index like ':1', so let users name the
    # device they actually see, e.g. --source "BlackHole 2ch".
    for src in sources:
        if src.description.lower() == sel.lower():
            return src
    raise AudioError(
        f"unknown audio source: {selector!r}. "
        f"Use 'mic', 'system', or one of: "
        f"{', '.join(s.description if IS_MACOS else s.name for s in sources)}"
    )


# --------------------------------------------------------------------------
# Inspection helpers
# --------------------------------------------------------------------------
def wav_duration(path: Path) -> float:
    """Duration in seconds of a PCM WAV file.

    ffmpeg cannot always seek back to patch the RIFF header when it is stopped
    mid-write, leaving a placeholder frame count (INT32_MAX). Trusting the
    header then reports a 65-second recording as 37 hours, so the length is
    derived from the real data size and only falls back to the header.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate() or config.SAMPLE_RATE
        block = max(wf.getnchannels() * wf.getsampwidth(), 1)
        header_frames = wf.getnframes()

    actual_frames = max(path.stat().st_size - _WAV_HEADER_BYTES, 0) // block
    if header_frames <= 0 or header_frames > actual_frames + 1:
        # Header is a placeholder or otherwise inconsistent with the file.
        return actual_frames / float(rate)
    return header_frames / float(rate)


def rms_level(path: Path) -> float:
    """Normalised RMS (0.0-1.0) of a 16-bit mono WAV, for level meters."""
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    if not frames:
        return 0.0
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)) / 32768.0)


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------
class Recorder:
    """Runs ffmpeg, yielding chunk paths as they become safe to read.

    A chunk is only yielded once the *next* chunk file has appeared, which
    proves ffmpeg has closed and finalized the previous one. Reading a
    half-written WAV yields garbage transcripts.
    """

    def __init__(
        self,
        source: AudioSource,
        session_dir: Path,
        chunk_seconds: int = config.CHUNK_SECONDS,
    ) -> None:
        self.source = source
        self.session_dir = Path(session_dir)
        self.chunks_dir = self.session_dir / "chunks"
        self.audio_path = self.session_dir / "audio.wav"
        self.chunk_seconds = chunk_seconds
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._started_at: float | None = None

    # -- lifecycle ------------------------------------------------------
    def _input_args(self) -> list[str]:
        """ffmpeg input flags for the current platform.

        avfoundation needs the sample rate declared on the *input*, because the
        device negotiates its own format and ffmpeg otherwise refuses to open
        it ("Selected audio format ... not supported").
        """
        if IS_MACOS:
            return [
                "-f", "avfoundation",
                "-ar", str(config.SAMPLE_RATE),
                "-i", self.source.name,
            ]
        return ["-f", "pulse", "-i", self.source.name]

    def _build_command(self) -> list[str]:
        return [
            config.FFMPEG_BIN,
            "-hide_banner",
            "-loglevel", "error",
            *self._input_args(),
            # segmented chunks for streaming ASR
            "-ac", str(config.CHANNELS),
            "-ar", str(config.SAMPLE_RATE),
            "-f", "segment",
            "-segment_time", str(self.chunk_seconds),
            "-reset_timestamps", "1",
            str(self.chunks_dir / "chunk_%05d.wav"),
            # continuous session recording for re-runs
            "-ac", str(config.CHANNELS),
            "-ar", str(config.SAMPLE_RATE),
            str(self.audio_path),
        ]

    def start(self) -> None:
        if shutil.which(config.FFMPEG_BIN) is None:
            hint = " (install it with: brew install ffmpeg)" if IS_MACOS else ""
            raise AudioError(f"ffmpeg not found on PATH{hint}")

        # PulseAudio silently falls back to the default source when given an
        # unknown device name, so ffmpeg would "succeed" while recording the
        # wrong input. Validate the name ourselves before starting.
        try:
            known = {s.name for s in list_sources()}
        except AudioError:
            known = set()
        if known and self.source.name not in known:
            raise AudioError(
                f"audio source not available: {self.source.name!r}. "
                f"Available: {', '.join(sorted(known))}"
            )

        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            self._build_command(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._started_at = time.monotonic()
        # Fail fast if the device is bad: ffmpeg exits almost immediately.
        time.sleep(0.4)
        if self._proc.poll() is not None:
            err = (self._proc.stderr.read() if self._proc.stderr else "") or ""
            hint = ""
            if IS_MACOS:
                # The usual cause on a Mac is a missing TCC grant: ffmpeg gets
                # an empty device list or fails to open the input.
                hint = (
                    "\nIf this is the first recording, allow your terminal under "
                    "System Settings > Privacy & Security > Microphone, then retry."
                )
            raise AudioError(
                f"ffmpeg failed to start on {self.source.description!r}: {err.strip()}{hint}"
            )

    def stop(self) -> None:
        """Signal capture to end. Safe to call more than once."""
        self._stop.set()
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        # SIGINT lets ffmpeg finalize WAV headers. SIGKILL corrupts the file
        # (verified: produces a 0-byte unreadable WAV). Never kill first.
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=config.FFMPEG_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started_at is None else time.monotonic() - self._started_at

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- chunk streaming -------------------------------------------------
    def _sorted_chunks(self) -> list[Path]:
        return sorted(self.chunks_dir.glob("chunk_*.wav"))

    def chunks(self, poll_interval: float = 0.5) -> Iterator[Path]:
        """Yield completed chunk paths until stopped, then drain the final one."""
        yielded: set[Path] = set()

        while not self._stop.is_set():
            if self._proc is not None and self._proc.poll() is not None:
                break  # ffmpeg died on its own; drain below
            available = self._sorted_chunks()
            # All but the newest are complete, because a later file exists.
            for path in available[:-1]:
                if path not in yielded:
                    yielded.add(path)
                    yield path
            time.sleep(poll_interval)

        # Capture has ended: ffmpeg closed the last chunk, so it is safe now.
        for path in self._sorted_chunks():
            if path not in yielded:
                yielded.add(path)
                yield path

    def cleanup_chunks(self) -> None:
        """Delete transient chunk files. The full audio.wav is kept."""
        if self.chunks_dir.exists():
            shutil.rmtree(self.chunks_dir, ignore_errors=True)
