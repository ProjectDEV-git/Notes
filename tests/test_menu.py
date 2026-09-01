"""The interactive menu: what a non-technical user actually touches.

Every test here is about the promise that nothing must be remembered or typed:
Enter is always valid, a wrong answer never destroys anything, and no action
requires knowing a session id or a flag.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from notetaker import config, menu


class _Session:
    """Minimal stand-in for store.Session, so no database is needed."""

    def __init__(self, sid: str, title: str, has_notes: bool = False):
        self.id = sid
        self.title = title
        self.has_notes = has_notes
        self.started_at = "2026-01-01T09:00:00"


def _sessions(*items: _Session):
    return patch("notetaker.store.list_sessions", lambda limit=None: list(items))


def _answers(*replies: str):
    """Feed the menu a fixed sequence of keystrokes."""
    return patch("builtins.input", side_effect=list(replies))


def _tty():
    return patch.object(sys.stdin, "isatty", lambda: True)


# ------------------------------------------------------------------ prompts
def test_enter_accepts_the_default():
    """Enter must always be a valid answer, or the menu is a dead end."""
    with _answers(""):
        assert menu.ask("pick: ", "fallback") == "fallback"


def test_eof_does_not_crash():
    """Piped input previously raised EOFError, which reads as a crash."""
    with patch("builtins.input", side_effect=EOFError):
        assert menu.ask("pick: ", "fallback") == "fallback"


def test_ctrl_c_at_a_prompt_returns_the_default():
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        assert menu.ask("pick: ", "fallback") == "fallback"


def test_confirm_defaults_to_no_for_destructive_answers():
    with _answers(""):
        assert menu.confirm("regenerate?") is False
    with _answers(""):
        assert menu.confirm("continue?", default=True) is True


# ------------------------------------------------------------------ pickers
def test_picking_a_lecture_never_requires_typing_an_id():
    with _sessions(_Session("a", "Physics"), _Session("b", "Chemistry")), _answers("2"):
        assert menu.pick_session().id == "b"


def test_enter_picks_the_most_recent_lecture():
    """The overwhelmingly common intent: 'the one I just recorded'."""
    with _sessions(_Session("a", "Physics"), _Session("b", "Chemistry")), _answers(""):
        assert menu.pick_session().id == "a"


def test_a_single_lecture_is_chosen_without_asking():
    with _sessions(_Session("only", "Physics")):
        assert menu.pick_session().id == "only"


def test_nonsense_answer_falls_back_instead_of_failing():
    with _sessions(_Session("a", "Physics"), _Session("b", "Chem")), _answers("banana"):
        assert menu.pick_session().id == "a"


def test_out_of_range_answer_falls_back():
    with _sessions(_Session("a", "Physics")), _answers("99"):
        assert menu.pick_session().id == "a"


def test_zero_cancels():
    """There must always be a way out that changes nothing."""
    with _sessions(_Session("a", "Physics"), _Session("b", "Chem")), _answers("0"):
        assert menu.pick_session() is None


def test_no_lectures_yet_is_explained_not_an_error():
    with _sessions():
        assert menu.pick_session() is None


def test_language_defaults_to_autodetect():
    with _answers(""):
        assert menu.pick_language() == "auto"
    with _answers("nonsense"):
        assert menu.pick_language() == "auto"


# --------------------------------------------------------------------- menu
def test_bare_enter_records_a_lecture():
    """The single most likely action must need zero knowledge."""
    with _tty(), _answers("", ""), patch("notetaker.cli.main", return_value=0) as cli:
        assert menu.main() == 0
    argv = cli.call_args[0][0]
    assert argv[0] == "record"
    assert "--source" in argv and config.SOURCE_MIC in argv


def test_online_lecture_option_uses_system_audio():
    with _tty(), _answers("2", ""), patch("notetaker.cli.main", return_value=0) as cli:
        menu.main()
    assert config.SOURCE_SYSTEM in cli.call_args[0][0]


def test_a_typed_title_is_passed_through():
    with _tty(), _answers("1", "Physics week 4"), patch("notetaker.cli.main", return_value=0) as cli:
        menu.main()
    argv = cli.call_args[0][0]
    assert argv[argv.index("--title") + 1] == "Physics week 4"


def test_skipping_the_title_does_not_pass_an_empty_one():
    with _tty(), _answers("1", ""), patch("notetaker.cli.main", return_value=0) as cli:
        menu.main()
    assert "--title" not in cli.call_args[0][0]


def test_quit_changes_nothing():
    with _tty(), _answers("q"), patch("notetaker.cli.main") as cli:
        assert menu.main() == 0
    cli.assert_not_called()


def test_unrecognised_menu_answer_runs_nothing():
    """A wrong keypress must never start a recording by accident."""
    with _tty(), _answers("banana"), patch("notetaker.cli.main") as cli:
        assert menu.main() == 1
    cli.assert_not_called()


def test_reading_notes_needs_no_session_id():
    with _tty(), _answers("3"), _sessions(_Session("a", "Physics", has_notes=True)), \
            patch("notetaker.cli.main", return_value=0) as cli:
        menu.main()
    assert cli.call_args[0][0] == ["show", "a"]


def test_export_offers_a_ready_made_filename():
    with _tty(), _answers("5", ""), _sessions(_Session("a", "Physics", has_notes=True)), \
            patch("notetaker.cli.main", return_value=0) as cli:
        menu.main()
    argv = cli.call_args[0][0]
    assert argv[:2] == ["export", "a"]
    assert argv[argv.index("-o") + 1] == "a.md"


def test_existing_notes_are_not_regenerated_without_consent():
    """Re-summarizing costs minutes of CPU, so it must be opt-in."""
    # One lecture, so it is chosen without a prompt: keystrokes are menu, confirm.
    with _tty(), _answers("4", ""), _sessions(_Session("a", "Physics", has_notes=True)), \
            patch("notetaker.cli.main", return_value=0) as cli:
        menu.main()
    assert "--rerun" not in cli.call_args[0][0]


def test_rerun_happens_when_the_user_agrees():
    with _tty(), _answers("4", "y"), _sessions(_Session("a", "Physics", has_notes=True)), \
            patch("notetaker.cli.main", return_value=0) as cli:
        menu.main()
    assert "--rerun" in cli.call_args[0][0]


def test_non_interactive_use_does_not_open_a_menu():
    """A menu on a pipe would hang forever; record instead."""
    with patch.object(sys.stdin, "isatty", lambda: False), \
            patch("notetaker.cli.main", return_value=0) as cli:
        menu.main()
    assert cli.call_args[0][0][0] == "record"


# -------------------------------------------------------------------- check
def test_check_reports_a_missing_loopback_with_a_fix():
    """The commonest macOS failure must come with the command that fixes it."""
    from notetaker import audio

    with patch.object(audio, "IS_MACOS", True), \
            patch.object(audio, "list_sources", lambda: [
                audio.AudioSource(":0", "MacBook Pro Microphone", config.SOURCE_MIC, True)
            ]), \
            patch("notetaker.summarize.ollama_available", lambda: False):
        checks = menu.run_checks()

    by_name = {c.name: c for c in checks}
    system = by_name["online lectures (system audio)"]
    assert not system.ok
    assert "blackhole" in system.fix.lower()


def test_check_distinguishes_stopped_ollama_from_missing_model():
    from notetaker import audio

    sources = [audio.AudioSource("mic0", "Mic", config.SOURCE_MIC, True)]
    with patch.object(audio, "list_sources", lambda: sources), \
            patch("notetaker.summarize.ollama_available", lambda: True), \
            patch("notetaker.summarize.installed_models", lambda: []):
        checks = menu.run_checks()

    names = [c.name for c in checks]
    model_check = next(c for c in checks if c.name.startswith("summary model"))
    assert "Ollama (writes the notes)" in names
    assert not model_check.ok
    assert "ollama pull" in model_check.fix


def test_check_passes_when_everything_is_present():
    from notetaker import audio

    sources = [
        audio.AudioSource("mic0", "Mic", config.SOURCE_MIC, True),
        audio.AudioSource("mon0", "Monitor", config.SOURCE_SYSTEM),
    ]
    with patch.object(audio, "list_sources", lambda: sources), \
            patch("notetaker.summarize.ollama_available", lambda: True), \
            patch("notetaker.summarize.installed_models", lambda: [config.SUMMARY_MODEL]):
        assert menu.print_checks(menu.run_checks()) is True


def test_check_survives_a_machine_with_no_audio_stack():
    from notetaker import audio

    def boom():
        raise audio.AudioError("no audio capture sources found")

    with patch.object(audio, "list_sources", boom), \
            patch("notetaker.summarize.ollama_available", lambda: False):
        checks = menu.run_checks()
    assert any(not c.ok for c in checks)  # reported, not raised


# ---------------------------------------------------------------------- cli
def test_bare_cli_opens_the_menu():
    from notetaker import cli

    with patch("notetaker.menu.main", return_value=0) as opened:
        assert cli.main([]) == 0
    opened.assert_called_once()


def test_check_is_available_as_a_command():
    from notetaker import cli

    with patch("notetaker.menu.run_checks", return_value=[menu.Check("x", True, "ok")]):
        assert cli.main(["check"]) == 0


def test_check_exits_nonzero_when_something_is_missing():
    from notetaker import cli

    broken = [menu.Check("ffmpeg", False, "missing", "install it")]
    with patch("notetaker.menu.run_checks", return_value=broken):
        assert cli.main(["check"]) == 1
