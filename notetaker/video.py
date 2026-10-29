"""Make notes from a video: a YouTube (or other site) link, or a file.

A lesson does not have to be live to be worth notes. Teachers post lessons on
YouTube, and a student may already have the video file. Both become a normal
session: the sound is saved as audio.wav, so `catchup`, `summarize --hq` and
everything else work on it exactly as they do on a recording.

Links are fetched with yt-dlp, which supports YouTube and over a thousand other
sites. It is installed the first time a link is used rather than with the rest
of NoteTaker: it brings its own JavaScript runtime (Deno, about 40 MB) that
YouTube needs, and the startup updater gives pip too little time for that on
school wifi. Local files only need ffmpeg, which NoteTaker already has.
"""

from __future__ import annotations

import html
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

from . import config
from .asr import Segment
from .timestamps import clean_chapters

APP_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS = APP_DIR / "requirements-video.txt"

# Installing yt-dlp happens once, while the student is waiting for it, so it
# gets far longer than the startup updater's budget.
INSTALL_TIMEOUT = 900.0
INFO_TIMEOUT = 180.0
SOCKET_TIMEOUT = 30

Echo = Callable[[str], None]


class VideoError(RuntimeError):
    """A video could not be fetched or read. The message is for the student."""


@dataclass
class VideoInfo:
    url: str  # the page's canonical address, used to spot a repeat import
    title: str
    duration: float  # seconds; 0 when the site does not say
    language: str | None  # the spoken language, when the site knows it
    caption_key: str | None  # creator-written subtitles in that language
    chapters: list[dict] = field(default_factory=list)  # {start, title}, full-video time


@dataclass
class PlaylistEntry:
    url: str
    title: str


@dataclass
class Playlist:
    title: str
    entries: list[PlaylistEntry]


# --------------------------------------------------------------------------
# What did the student paste?
# --------------------------------------------------------------------------
_BARE_VIDEO_HOST = re.compile(
    r"^(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be|vimeo\.com)/", re.I
)


def looks_like_url(text: str) -> bool:
    """True for a web link, including a YouTube link pasted without https://."""
    text = text.strip()
    parsed = urlparse(text)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return True
    return bool(_BARE_VIDEO_HOST.match(text))


def normalize_url(text: str) -> str:
    text = text.strip()
    return text if urlparse(text).scheme in ("http", "https") else f"https://{text}"


def clean_pasted(text: str) -> str:
    """Undo what a terminal does to a file dragged into it.

    Dragging a file in gives a quoted path ('/My Videos/a.mp4'), a
    backslash-escaped one (/My\\ Videos/a.mp4), or a file:// address,
    depending on the terminal. None of those exist as typed.
    """
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    if text.startswith("file://"):
        return unquote(urlparse(text).path)
    # Backslash is a path separator on Windows, so only unescape elsewhere,
    # and only when the escaped form is not itself a real file.
    if os.name != "nt" and "\\" in text and not Path(text).exists():
        text = re.sub(r"\\(.)", r"\1", text)
    return text


def title_from_filename(path: Path) -> str:
    """'week_4-photosynthesis.mp4' -> 'week 4 photosynthesis'."""
    return re.sub(r"[_\-.]+", " ", path.stem).strip() or "Video"


# --------------------------------------------------------------------------
# Remembering where a class came from
# --------------------------------------------------------------------------
META_NAME = "video.json"


def save_meta(folder: Path, **fields) -> None:
    """What the notes need later: the link, chapters, and any trim offset.

    Kept beside audio.wav rather than in the database, so `--rerun` and
    `catchup` can rebuild links long after the import.
    """
    from .artifacts import write_text_atomic

    write_text_atomic(folder / META_NAME, json.dumps(fields, ensure_ascii=False, indent=1))


def load_meta(folder: Path) -> dict:
    try:
        data = json.loads((folder / META_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def timestamp_url(url: str, seconds: float) -> str:
    """A link that opens the video at `seconds`.

    YouTube reads `t=` from the query; Vimeo and plain media files read a
    `#t=` fragment. On any other site the fragment is harmless.
    """
    at = max(int(seconds), 0)
    parts = urlparse(url)
    host = parts.netloc.lower()
    if host.endswith("youtube.com") or host.endswith("youtu.be"):
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "t"]
        query.append(("t", f"{at}s"))
        return urlunparse(parts._replace(query=urlencode(query), fragment=""))
    fragment = f"t={at}s" if host.endswith("vimeo.com") else f"t={at}"
    return urlunparse(parts._replace(fragment=fragment))


def link_for(meta: dict) -> Callable[[float], str] | None:
    url = meta.get("url")
    if isinstance(url, str) and urlparse(url).scheme in ("http", "https"):
        return lambda seconds: timestamp_url(url, seconds)
    return None


# --------------------------------------------------------------------------
# yt-dlp: installed on first use, upgraded when a site changes
# --------------------------------------------------------------------------
def have_downloader() -> bool:
    return importlib.util.find_spec("yt_dlp") is not None


def _pip(args: list[str], what: str) -> None:
    command = [sys.executable, "-m", "pip", "install", "-q",
               "--disable-pip-version-check", *args, "-r", str(REQUIREMENTS)]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=INSTALL_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise VideoError(
            f"{what} took too long and was stopped. Check the internet "
            "connection and try again."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise VideoError(f"{what} failed: {detail[-1] if detail else 'pip error'}")
    importlib.invalidate_caches()


def install_downloader(echo: Echo) -> None:
    echo(
        "[bold]Video links need one extra helper.[/bold] It downloads once "
        "(about 60 MB) and is then ready for every video.\n"
        "[dim]installing the video helper...[/dim]"
    )
    _pip([], "installing the video helper")
    echo("[green]video helper installed[/green]")


def upgrade_downloader(echo: Echo) -> None:
    echo("[dim]the website may have changed; updating the video helper and "
         "trying again...[/dim]")
    _pip(["--upgrade"], "updating the video helper")


def _base_command() -> list[str]:
    command = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist", "--no-warnings",
        "--socket-timeout", str(SOCKET_TIMEOUT),
    ]
    # Deno ships with the helper and is preferred. Node is a fallback that
    # yt-dlp only uses when it is enabled explicitly.
    if shutil.which("node"):
        command += ["--js-runtimes", "node"]
    return command


def _last_error(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("ERROR")]
    line = (errors or lines or ["unknown error"])[-1]
    return re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?([\w-]+:\s*)?", "", line)


# Failures no newer yt-dlp can fix. Upgrading for these only wastes a minute.
_PERMANENT = [
    (r"Unsupported URL|is not a valid URL",
     "NoteTaker cannot get videos from that website. If you can download the "
     "video yourself, give NoteTaker the file instead."),
    (r"Private video|members-only|Join this channel",
     "That video is private, so it cannot be downloaded."),
    (r"confirm your age|age-restricted|inappropriate for some users",
     "That video is age-restricted and needs a signed-in account, so it "
     "cannot be downloaded. Download it another way and give NoteTaker the file."),
    (r"not a bot",
     "YouTube is asking NoteTaker to prove it is not a bot. Wait an hour and "
     "try again, or download the video another way and give NoteTaker the file."),
    (r"Video unavailable|has been removed|does not exist|HTTP Error 404",
     "That video is unavailable. It may have been removed, or the link is wrong."),
    (r"Unable to download webpage|getaddrinfo|Name or service not known|"
     r"timed out|Temporary failure in name resolution|Network is unreachable|"
     r"Connection refused|nodename nor servname",
     "Could not reach the website. Check the internet connection and try again."),
]


def explain(reason: str) -> tuple[str, bool]:
    """A plain-language message, and whether an upgrade might fix it."""
    for pattern, message in _PERMANENT:
        if re.search(pattern, reason, re.I):
            return message, False
    return f"could not get that video: {reason}", True


def _run_downloader(
    args: list[str], echo: Echo, capture_stdout: bool, timeout: float | None
) -> subprocess.CompletedProcess[str]:
    """Run yt-dlp; on a fixable failure, upgrade it once and try again.

    YouTube changes often and an old yt-dlp is the usual reason a link that
    worked last month stops working. Upgrading on the spot means the student
    never has to know that.
    """
    for attempt in range(2):
        try:
            result = subprocess.run(
                _base_command() + args,
                stdout=subprocess.PIPE if capture_stdout else None,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise VideoError(
                "the website took too long to answer. Try again in a minute."
            ) from exc
        if result.returncode == 0:
            return result
        message, fixable = explain(_last_error(result.stderr or ""))
        if not fixable or attempt:
            raise VideoError(message)
        upgrade_downloader(echo)
    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------
# Looking a video up, and downloading its sound
# --------------------------------------------------------------------------
def _base_language(code: object) -> str | None:
    if not isinstance(code, str) or not code.strip():
        return None
    return re.split(r"[-_]", code.strip().lower())[0] or None


def pick_caption(subtitles: dict, language: str | None) -> str | None:
    """The creator-written subtitle track in the spoken language, if any.

    Only uploaded subtitles count. YouTube's automatic captions are no better
    than Whisper, and a track in another language is a translation: notes
    written from it would quietly change the class's language.
    """
    if not language or not isinstance(subtitles, dict):
        return None
    keys = [k for k in subtitles if isinstance(k, str) and k != "live_chat"]
    for key in keys:
        if key.lower() == language:
            return key
    for key in keys:
        if _base_language(key) == language:
            return key
    return None


def parse_info(data: dict, language: str | None = None) -> VideoInfo:
    """Turn yt-dlp's description of a link into what NoteTaker needs."""
    if data.get("_type") == "playlist" or "entries" in data:
        raise VideoError(
            "That link is a whole playlist. Open the one video you want and "
            "copy its link instead."
        )
    status = data.get("live_status")
    if data.get("is_live") or status == "is_live":
        raise VideoError(
            "That video is live right now. Record it as an online class while "
            "it plays (notes online), or come back once it has finished."
        )
    if status == "is_upcoming":
        raise VideoError("That video has not started yet. Come back once it has finished.")
    if status == "post_live":
        raise VideoError(
            "That live stream has only just finished and is still being "
            "processed. Try again in a few hours."
        )

    spoken = language or _base_language(data.get("language"))
    return VideoInfo(
        url=str(data.get("webpage_url") or data.get("original_url") or ""),
        title=str(data.get("title") or "Video").strip() or "Video",
        duration=float(data.get("duration") or 0),
        language=spoken,
        caption_key=pick_caption(data.get("subtitles") or {}, spoken),
        chapters=clean_chapters(data.get("chapters")),
    )


def parse_playlist(data: dict) -> Playlist:
    """The videos in a playlist, as listed without opening each one.

    A channel page lists its tabs (Videos, Shorts, Live) as playlists of their
    own. Downloading a whole channel is rarely what a student means, so that
    is explained rather than attempted.
    """
    entries: list[PlaylistEntry] = []
    for item in data.get("entries") or []:
        if not isinstance(item, dict):
            continue
        nested = item.get("_type") == "playlist" or str(item.get("ie_key") or "").endswith("Tab")
        if nested:
            raise VideoError(
                "That link is a whole channel. Open one of its playlists, or one "
                "video, and copy that link instead."
            )
        url = item.get("url") or item.get("webpage_url")
        formats = item.get("formats") or []
        if not url and len(formats) == 1 and isinstance(formats[0], dict):
            # A plain web page with several <video> tags lists the media files
            # themselves, which can be fetched directly.
            url = formats[0].get("url")
        if isinstance(url, str) and url:
            title = str(item.get("title") or url).strip()
            entries.append(PlaylistEntry(normalize_url(url), title))
    return Playlist(str(data.get("title") or "Playlist").strip() or "Playlist", entries)


def fetch_info(url: str, echo: Echo, language: str | None = None) -> VideoInfo | Playlist:
    """Look a link up. A playlist comes back as its list of videos.

    --flat-playlist lists a playlist without opening every video in it, so a
    50-video course is looked up in seconds. A single video is unaffected,
    and --no-playlist keeps a video opened from inside a playlist to itself.
    """
    result = _run_downloader(["--dump-single-json", "--flat-playlist", url], echo,
                             capture_stdout=True, timeout=INFO_TIMEOUT)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoError("the website sent something NoteTaker could not read") from exc
    if data.get("_type") == "playlist" or "entries" in data:
        return parse_playlist(data)
    info = parse_info(data, language)
    info.url = info.url or url
    return info


def download(
    url: str, folder: Path, echo: Echo, caption_key: str | None = None
) -> tuple[Path, Path | None]:
    """Download the sound (and subtitles, if wanted) into `folder`.

    Returns the media file and the subtitle file, if one arrived. The best
    audio-only stream is far smaller than the video, so this is quick even for
    an hour-long lesson.
    """
    folder.mkdir(parents=True, exist_ok=True)
    args = [
        "-f", "bestaudio/best",
        "-o", str(folder / "download.%(ext)s"),
        "--quiet", "--progress",
    ]
    if caption_key:
        args += [
            "--write-subs", "--sub-langs", re.escape(caption_key),
            "--sub-format", "vtt/best", "--convert-subs", "vtt",
        ]
    # No timeout: an hour of audio on slow wifi legitimately takes a while,
    # and yt-dlp's own socket timeout catches a dead connection.
    _run_downloader(args + [url], echo, capture_stdout=False, timeout=None)

    files = [p for p in folder.iterdir() if p.is_file()]
    captions = next((p for p in files if p.suffix == ".vtt"), None)
    media = [p for p in files
             if p.suffix not in (".vtt", ".part", ".ytdl", ".json", ".srt")]
    if not media:
        raise VideoError("the download finished but no sound file arrived")
    return max(media, key=lambda p: p.stat().st_size), captions


# --------------------------------------------------------------------------
# Converting to the format the rest of NoteTaker reads
# --------------------------------------------------------------------------
def to_wav(source: Path, destination: Path) -> None:
    """Extract the sound as 16 kHz mono WAV, which is what a recording is.

    Written to a temporary name first, so a failed conversion never leaves a
    half-written audio.wav that `catchup` would then try to transcribe.
    """
    ffmpeg = shutil.which(config.FFMPEG_BIN)
    if not ffmpeg:
        raise VideoError("ffmpeg is not installed. Run:  notes check")
    partial = destination.with_name(destination.stem + ".partial.wav")
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-vn", "-ac", str(config.CHANNELS), "-ar", str(config.SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(partial),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        detail = result.stderr.strip()
        if re.search(r"does not contain any stream|matches no streams|no audio", detail, re.I):
            raise VideoError(f"{source.name} has no sound in it")
        last = detail.splitlines()[-1] if detail else "unknown error"
        raise VideoError(f"could not read the sound from {source.name}: {last}")
    os.replace(partial, destination)


# --------------------------------------------------------------------------
# Subtitles -> transcript
# --------------------------------------------------------------------------
_CUE_TIME = re.compile(
    r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)
_TAG = re.compile(r"<[^>]*>")
# "[Music]", "(applause)", "♪ ♪": sound descriptions, not anything said.
_SOUND_ONLY = re.compile(r"^(?:\[[^\]]*\]|\([^)]*\)|[♪♫\s])+$")


def _seconds(hours, minutes, secs, millis) -> float:
    return (int(hours or 0) * 3600 + int(minutes) * 60 + int(secs)
            + int(millis.ljust(3, "0")) / 1000)


def parse_vtt(text: str, language: str) -> list[Segment]:
    """WebVTT subtitles as transcript segments.

    Creator-written subtitles are what was said, checked by a person, and
    they cost nothing to use. For a Thai lesson that is hours of CPU saved.
    """
    segments: list[Segment] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0
    while index < len(lines):
        match = _CUE_TIME.match(lines[index])
        index += 1
        if not match:
            continue
        cue: list[str] = []
        while index < len(lines) and lines[index].strip():
            cue.append(lines[index].strip())
            index += 1
        spoken = html.unescape(_TAG.sub("", " ".join(cue)))
        spoken = re.sub(r"\s+", " ", spoken).strip()
        if not spoken or _SOUND_ONLY.match(spoken):
            continue
        # Some subtitle files repeat a line across cues as it scrolls.
        if segments and segments[-1].text == spoken:
            continue
        start = _seconds(*match.group(1, 2, 3, 4))
        end = _seconds(*match.group(5, 6, 7, 8))
        segments.append(Segment(round(start, 3), round(max(end, start), 3), spoken, language, 0))
    return segments
