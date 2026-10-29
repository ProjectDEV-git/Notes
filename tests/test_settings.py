"""Settings chosen in the window, and the terminal honouring them.

The promise: a choice made in Settings changes `notes class` too, an
environment variable still wins, and a broken settings file can never stop a
class being recorded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from notetaker import cli, config, shortcuts

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    for var in config.SETTINGS_ENV.values():
        monkeypatch.delenv(var, raising=False)
    return path


# --------------------------------------------------------------- the file
def test_a_missing_file_means_the_defaults(settings):
    assert config.load_settings(settings) == {}
    assert config.effective_settings(settings) == config.SETTINGS_DEFAULTS


def test_settings_are_saved_and_read_back(settings):
    config.save_settings({"class_minutes": 45, "notes_level": "university"}, settings)
    assert config.load_settings(settings) == {"class_minutes": 45, "notes_level": "university"}
    assert config.effective_settings(settings)["class_minutes"] == 45


def test_none_puts_a_setting_back_to_its_default(settings):
    config.save_settings({"class_minutes": 45}, settings)
    config.save_settings({"class_minutes": None}, settings)
    assert config.effective_settings(settings)["class_minutes"] == 60


@pytest.mark.parametrize("key, value", [
    ("class_minutes", 0),
    ("class_minutes", "soon"),
    ("class_minutes", True),
    ("notes_level", "professor"),
    ("notes_language", "Thai please"),
    ("summary_model", "rm -rf /"),
    ("device", "tpu"),
    ("auto_update", "yes"),
    ("favourite_colour", "blue"),
])
def test_bad_values_are_refused_with_a_reason(settings, key, value):
    with pytest.raises(ValueError):
        config.save_settings({key: value}, settings)
    assert not settings.exists() or config.load_settings(settings) == {}


def test_a_broken_file_never_stops_a_class(settings):
    """A typo in a hand-edited file falls back to the defaults, quietly."""
    for text in ("{not json", "[1, 2]", '"just a string"', ""):
        settings.write_text(text, encoding="utf-8")
        assert config.effective_settings(settings) == config.SETTINGS_DEFAULTS


def test_one_bad_value_does_not_lose_the_good_ones(settings):
    settings.write_text(json.dumps({"class_minutes": 45, "notes_level": "nonsense"}))
    assert config.load_settings(settings) == {"class_minutes": 45}


def test_an_environment_variable_wins_over_the_file(settings, monkeypatch):
    config.save_settings({"class_minutes": 45, "auto_update": True}, settings)
    monkeypatch.setenv("NOTES_CLASS_MINUTES", "30")
    monkeypatch.setenv("NOTES_AUTO_UPDATE", "0")
    current = config.effective_settings(settings)
    assert current["class_minutes"] == 30
    assert current["auto_update"] is False


def test_an_unusable_environment_variable_is_ignored(settings, monkeypatch):
    config.save_settings({"class_minutes": 45}, settings)
    monkeypatch.setenv("NOTES_CLASS_MINUTES", "lots")
    assert config.effective_settings(settings)["class_minutes"] == 45


# ------------------------------------------------------ the terminal obeys
def _in_fresh_python(code: str, settings: Path, **env) -> str:
    """Run code in a new process, as `notes class` would start."""
    environ = {k: v for k, v in os.environ.items() if k not in config.SETTINGS_ENV.values()}
    environ.update(NOTETAKER_SETTINGS_FILE=str(settings), **env)
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=environ,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_notes_class_uses_the_class_length_from_settings(settings):
    config.save_settings({"class_minutes": 45}, settings)
    code = ("from notetaker import cli\n"
            "print(cli.build_parser().parse_args(['record', '--minutes', 'usual']).minutes)")
    assert _in_fresh_python(code, settings) == "45"
    # A variable for one class still wins.
    assert _in_fresh_python(code, settings, NOTES_CLASS_MINUTES="30") == "30"


def test_notes_are_written_the_way_settings_say(settings):
    config.save_settings({"notes_level": "university", "notes_language": "th",
                          "summary_model": "qwen2.5:7b"}, settings)
    code = ("from notetaker import cli\n"
            "a = cli.build_parser().parse_args(['summarize', 'x'])\n"
            "print(a.level, a.model)\n"
            "from notetaker import config\n"
            "print(config.NOTES_LANGUAGE)")
    assert _in_fresh_python(code, settings).split() == ["university", "qwen2.5:7b", "th"]


def test_a_number_still_works_for_one_class():
    assert cli.build_parser().parse_args(["record", "--minutes", "45"]).minutes == 45


def test_turning_updates_off_skips_the_startup_check(monkeypatch):
    monkeypatch.setattr(config, "AUTO_UPDATE", False)

    def must_not_run(*args, **kwargs):
        raise AssertionError("updated although Settings turned it off")

    monkeypatch.setattr(cli, "update_checkout", must_not_run)
    assert cli.cmd_update(cli.build_parser().parse_args(["update", "--quiet"])) == 0
    monkeypatch.setattr(shortcuts, "subprocess", None)  # would crash if reached
    shortcuts._auto_update("class")


def test_notes_update_by_name_still_updates_with_the_check_off(monkeypatch):
    monkeypatch.setattr(config, "AUTO_UPDATE", False)
    called = []
    monkeypatch.setattr(cli, "update_checkout", lambda *a: called.append(1) or (False, "ok"))
    cli.cmd_update(cli.build_parser().parse_args(["update"]))
    assert called
