"""Notes as a page to print, share, or save as PDF.

PDF libraries for Python render Thai badly: its vowels and tone marks stack
above and below letters and need a shaping engine plus the right font. Every
web browser already does that correctly, so the notes become a small,
self-contained HTML page, and a browser that is already on the computer
prints it to PDF (Edge is on every Windows PC; Chrome or Chromium elsewhere).

With no browser to drive, the page is still saved, and opening it and
pressing Print gives the same PDF.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])")
_BULLET = re.compile(r"^\s*([-*•])\s+(.*)$")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

STYLE = """
:root { color-scheme: light; }
body { font-family: "Sarabun", "Leelawadee UI", "Thonburi", "Noto Sans Thai",
       system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 46rem; margin: 2.5rem auto; padding: 0 1.25rem;
       line-height: 1.6; color: #1b1b1b; background: #fff; font-size: 12pt; }
h1 { font-size: 1.7rem; margin: 0 0 .25rem; }
h2 { font-size: 1.2rem; margin: 1.6rem 0 .4rem; padding-bottom: .2rem;
     border-bottom: 1px solid #ddd; }
h3 { font-size: 1.02rem; margin: 1.1rem 0 .3rem; color: #333; }
p.meta { color: #666; margin: 0 0 1rem; font-size: .92rem; }
ul, ol { padding-left: 1.3rem; margin: .3rem 0; }
li { margin: .25rem 0; }
a { color: #1a5fb4; text-decoration: none; }
a.time { font-size: .85em; color: #666; white-space: nowrap; }
footer { margin-top: 2.5rem; color: #999; font-size: .8rem; }
@media print {
  body { margin: 0 auto; }
  h2, h3 { break-after: avoid; }
  li { break-inside: avoid; }
}
"""


def _inline(text: str) -> str:
    """Bold, italic and links, with everything else escaped."""
    out, last = [], 0
    for match in _LINK.finditer(text):
        out.append(_plain(text[last:match.start()]))
        label, url = match.group(1), match.group(2)
        css = ' class="time"' if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", label) else ""
        out.append(f'<a{css} href="{html.escape(url, quote=True)}">{_plain(label)}</a>')
        last = match.end()
    out.append(_plain(text[last:]))
    return "".join(out)


def _plain(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    return _ITALIC.sub(r"<em>\1</em>", escaped)


def to_html(markdown: str, title: str, lang: str = "en") -> str:
    """The subset of markdown the notes use, as one standalone page."""
    body: list[str] = []
    open_list: str | None = None

    def close() -> None:
        nonlocal open_list
        if open_list:
            body.append(f"</{open_list}>")
            open_list = None

    for line in markdown.splitlines():
        stripped = line.strip()
        heading = _HEADING.match(stripped)
        bullet = _BULLET.match(line)
        numbered = _NUMBERED.match(line)
        if not stripped:
            close()
        elif heading:
            close()
            level = len(heading.group(1))
            body.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif bullet or numbered:
            kind = "ul" if bullet else "ol"
            if open_list != kind:
                close()
                body.append(f"<{kind}>")
                open_list = kind
            body.append(f"<li>{_inline((bullet or numbered).group(2))}</li>")
        elif stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            close()
            body.append(f'<p class="meta">{_plain(stripped.strip("*"))}</p>')
        else:
            close()
            body.append(f"<p>{_inline(stripped)}</p>")
    close()

    return (
        f'<!doctype html>\n<html lang="{html.escape(lang)}">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n<footer>Made with NoteTaker</footer>\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------
# PDF through a browser that is already installed
# --------------------------------------------------------------------------
def _candidates() -> list[str]:
    names = ["msedge", "google-chrome", "google-chrome-stable", "chromium",
             "chromium-browser", "chrome", "brave-browser", "microsoft-edge"]
    found = [path for path in (shutil.which(n) for n in names) if path]
    if sys.platform == "darwin":
        found += [f"/Applications/{app}.app/Contents/MacOS/{app}" for app in
                  ("Google Chrome", "Microsoft Edge", "Chromium", "Brave Browser")]
    if sys.platform == "win32":
        for root in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES"),
                     os.environ.get("LOCALAPPDATA")):
            if root:
                found += [str(Path(root) / "Microsoft/Edge/Application/msedge.exe"),
                          str(Path(root) / "Google/Chrome/Application/chrome.exe")]
    extra = os.environ.get("NOTETAKER_BROWSER")
    return ([extra] if extra else []) + found


def find_browser() -> str | None:
    for path in _candidates():
        if path and Path(path).is_file():
            return path
    return None


def html_to_pdf(page: Path, pdf: Path, browser: str, timeout: float = 60.0) -> bool:
    """Print `page` to `pdf` with a headless browser. True when it worked."""
    with tempfile.TemporaryDirectory() as profile:
        command = [
            browser, "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", f"--user-data-dir={profile}",
            # Printing needs no internet. Without these the browser may sit
            # waiting on its update servers behind a school firewall.
            "--disable-background-networking", "--disable-component-update",
            "--disable-sync", "--disable-extensions",
            "--no-pdf-header-footer", f"--print-to-pdf={pdf}", page.resolve().as_uri(),
        ]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            command.insert(1, "--no-sandbox")  # Chromium refuses to run as root otherwise
        try:
            subprocess.run(command, capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return False
    return pdf.is_file() and pdf.stat().st_size > 0
