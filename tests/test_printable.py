"""Notes as a page to print or share, and as a PDF."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from notetaker import cli, config, printable, store

NOTES = """# Physics <week 4>

*2026-10-29 · 3m 23s · English*

## What we learned

### Lifting
- Lifting 15 kg stores **150 J** ([0:10](https://youtu.be/a?t=7s&x=1))
- Energy & momentum are *both* conserved (0:47)

## สิ่งที่เรียนวันนี้
1. พลังงานไม่สูญหาย
2. second
"""

# Playwright's headless Chromium, where installed (CI images, this repo's sandbox).
CHROME = next(Path("/opt/pw-browsers").glob(
    "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"),
    Path("/nonexistent"))


def test_the_page_has_the_notes_structure():
    page = printable.to_html(NOTES, "Physics <week 4>", "en")
    assert page.startswith("<!doctype html>") and 'charset="utf-8"' in page
    assert "<title>Physics &lt;week 4&gt;</title>" in page
    assert "<h1>Physics &lt;week 4&gt;</h1>" in page, "text is escaped"
    assert '<p class="meta">2026-10-29 · 3m 23s · English</p>' in page
    assert "<h3>Lifting</h3>" in page
    assert "<strong>150 J</strong>" in page and "<em>both</em>" in page
    assert '<a class="time" href="https://youtu.be/a?t=7s&amp;x=1">0:10</a>' in page
    assert "Energy &amp; momentum" in page
    assert "<ol>\n<li>พลังงานไม่สูญหาย</li>" in page


def test_only_web_links_become_links():
    page = printable.to_html("- [click](javascript:alert(1))\n", "x")
    assert "<a" not in page


def test_thai_notes_are_marked_as_thai_for_the_browser():
    assert '<html lang="th">' in printable.to_html(NOTES, "x", "th")


@pytest.fixture()
def db(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "sessions").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SESSIONS_DIR", data / "sessions")
    monkeypatch.setattr(config, "DB_PATH", data / "notetaker.db")
    monkeypatch.setenv("NOTES_CALLER_DIR", str(tmp_path))
    session = store.create_session("Physics", config.SOURCE_MIC)
    store.write_notes(session, NOTES)
    return session


def test_html_export(db, tmp_path):
    assert cli.main(["export", "Physics", "--html"]) == 0
    assert "<h2>What we learned</h2>" in (tmp_path / f"{db.id}.html").read_text()


def test_without_a_browser_the_page_is_saved_with_print_steps(db, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(printable, "find_browser", lambda: None)
    assert cli.main(["export", "Physics", "--pdf", "-o", "physics.pdf"]) == 0
    assert (tmp_path / "physics.html").exists() and not (tmp_path / "physics.pdf").exists()
    assert "Save as PDF" in capsys.readouterr().out


def test_a_browser_can_be_named(monkeypatch, tmp_path):
    fake = tmp_path / "mybrowser"
    fake.write_text("")
    monkeypatch.setenv("NOTETAKER_BROWSER", str(fake))
    assert printable.find_browser() == str(fake)


@pytest.mark.skipif(not CHROME.exists() and not printable.find_browser(),
                    reason="no Chromium-family browser here")
def test_a_real_pdf_with_thai_text(db, tmp_path, monkeypatch):
    if CHROME.exists():
        monkeypatch.setenv("NOTETAKER_BROWSER", str(CHROME))
    assert cli.main(["export", "Physics", "--pdf", "-o", "notes.pdf"]) == 0
    pdf = tmp_path / "notes.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert not (tmp_path / "notes.html").exists(), "the page was only a step"
