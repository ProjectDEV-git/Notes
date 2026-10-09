"""Guided, beginner-proof setup for the parts each OS makes hard.

macOS and Windows both block the microphone until the user agrees, and
neither records what the speakers play out of the box. Each of those fails
silently: the class records an hour of nothing. This module turns them into
short guided steps that open the right settings page, wait for the user, and
then *measure* that sound is really arriving.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from . import config
from .audio import (
    IS_MACOS,
    IS_WINDOWS,
    AudioError,
    AudioSource,
    Recorder,
    list_sources,
    open_settings,
    resolve_source,
    rms_level,
    silent_input_help,
)

# Speech in a room is usually well above this; digital silence is ~0.
HEARD_RMS = 0.002
SILENT_RMS = 1e-4


def capture_level(source: AudioSource, seconds: int = 4) -> float:
    """Record a few seconds from `source` and return its RMS level.

    Recording is also what makes macOS show its microphone permission prompt,
    so running this during setup gets that question out of the way before
    the first real class.
    """
    if shutil.which(config.FFMPEG_BIN) is None:
        raise AudioError("ffmpeg is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sample.wav"
        recorder = Recorder(source, Path(tmp))
        command = [
            config.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            *recorder._input_args(),
            "-t", str(seconds),
            "-ac", str(config.CHANNELS), "-ar", str(config.SAMPLE_RATE),
            str(out),
        ]
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=seconds + 30,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioError("the test recording did not finish") from exc
        if proc.returncode != 0 or not out.exists():
            raise AudioError(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip()
                             else "the test recording failed")
        return rms_level(out)


def _verdict(level: float) -> str:
    if level < SILENT_RMS:
        return "silent"
    if level < HEARD_RMS:
        return "quiet"
    return "ok"


def listen_test(
    say: Callable[[str], None],
    ask: Callable[[str], str],
    kind: str = config.SOURCE_MIC,
) -> bool:
    """Record a short sample and say plainly whether sound arrived."""
    try:
        source = resolve_source(kind)
    except AudioError as exc:
        say(f"[red]✗[/red] {exc}")
        return False

    if kind == config.SOURCE_MIC:
        say(f"\nTesting the microphone: [bold]{source.description}[/bold]")
        if IS_MACOS:
            say("If macOS asks to allow the microphone, click [bold]Allow[/bold].")
        say("Say a few words out loud now...")
    else:
        say(f"\nTesting online-class sound: [bold]{source.description}[/bold]")
        ask("Start playing any video with sound (YouTube is fine), then press Enter")

    try:
        level = capture_level(source)
    except AudioError as exc:
        say(f"[red]✗ could not record:[/red] {exc}")
        if kind == config.SOURCE_MIC and open_settings("microphone"):
            say(silent_input_help(kind))
        return False

    verdict = _verdict(level)
    if verdict == "ok":
        say("[green]✓ sound is coming through. You are ready.[/green]")
        return True
    if verdict == "quiet":
        say("[yellow]! sound is arriving, but it is very quiet.[/yellow] "
            "Move closer, or turn the input volume up in Sound settings.")
        return True

    say(f"[red]✗ the recording was completely silent.[/red] {silent_input_help(kind)}")
    if kind == config.SOURCE_MIC:
        open_settings("microphone")
    return False


# --------------------------------------------------------------------------
# Online classes
# --------------------------------------------------------------------------
def _has_system_source() -> bool:
    try:
        return any(s.kind == config.SOURCE_SYSTEM for s in list_sources())
    except AudioError:
        return False


def _setup_online_mac(say, ask, confirm) -> bool:
    if not _has_system_source():
        say("macOS cannot record what your speakers play by itself. A free, "
            "open-source helper called [bold]BlackHole[/bold] fixes that.")
        brew = shutil.which("brew") or next(
            (p for p in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew") if Path(p).exists()),
            None,
        )
        if brew and confirm("Install BlackHole now? (your Mac password may be asked)"):
            say("  $ brew install --cask blackhole-2ch")
            subprocess.run([brew, "install", "--cask", "blackhole-2ch"])
        else:
            say("Download it from https://existential.audio/blackhole/ "
                "(choose 2ch), install it, then run [cyan]notes setup-online[/cyan] again.")
            return False
        if not _has_system_source():
            say("[yellow]BlackHole is installed but macOS has not loaded it yet.[/yellow] "
                "Restart the Mac, then run [cyan]notes setup-online[/cyan] again.")
            return False
    say("[green]✓ BlackHole is installed.[/green]")

    say("\nNow one step so you can [bold]hear[/bold] the class and record it at the same time.")
    say("Audio MIDI Setup is opening. In it:")
    open_settings("audio-midi")
    steps = [
        "Click the [bold]+[/bold] at the bottom left, then [bold]Create Multi-Output Device[/bold]",
        "Tick [bold]your speakers or headphones[/bold] AND [bold]BlackHole 2ch[/bold]",
        "Double-click the new device's name and call it [bold]NoteTaker Output[/bold]",
    ]
    for number, text in enumerate(steps, start=1):
        ask(f"  {number}. {text}   (press Enter when done)")

    say("\nLast step, before every online class:")
    say("  Click the sound icon in the menu bar (or Control Centre > Sound)")
    say("  and choose [bold]NoteTaker Output[/bold]. Switch back afterwards.")
    open_settings("sound")
    ask("Choose NoteTaker Output now, then press Enter")
    return listen_test(say, ask, config.SOURCE_SYSTEM)


def _setup_online_windows(say, ask, confirm) -> bool:
    if not _has_system_source():
        say("Windows can record what your speakers play, but hides the option "
            "([bold]Stereo Mix[/bold]). The Sound window is opening on the Recording tab. In it:")
        open_settings("recording-devices")
        steps = [
            "Right-click an empty area of the list, tick [bold]Show Disabled Devices[/bold]",
            "Right-click [bold]Stereo Mix[/bold] and choose [bold]Enable[/bold]",
            "Click OK",
        ]
        for number, text in enumerate(steps, start=1):
            ask(f"  {number}. {text}   (press Enter when done)")
        if not _has_system_source():
            say("\n[yellow]Stereo Mix did not appear.[/yellow] Some laptops do not have it.")
            say("Install the free VB-CABLE instead: https://vb-audio.com/Cable/")
            say("Then set your Sound output to [bold]CABLE Input[/bold] during online "
                "classes, and run [cyan]notes setup-online[/cyan] again.")
            return False
    say("[green]✓ a system-audio recorder is available.[/green]")
    return listen_test(say, ask, config.SOURCE_SYSTEM)


def setup_online(say, ask, confirm) -> bool:
    """Walk a beginner through making online classes recordable."""
    if IS_MACOS:
        return _setup_online_mac(say, ask, confirm)
    if IS_WINDOWS:
        return _setup_online_windows(say, ask, confirm)
    if _has_system_source():
        say("[green]✓ system audio is available; nothing to set up.[/green]")
        return listen_test(say, ask, config.SOURCE_SYSTEM)
    say("No .monitor source was found. Make sure PipeWire or PulseAudio is running.")
    return False
