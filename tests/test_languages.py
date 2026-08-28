"""Language pack loading and the `lang` command.

Users must be able to add a language by dropping in a JSON file, with no code
changes. These tests also guard the failure mode that matters most: a broken
pack must never stop a lecture being recorded.
"""

from __future__ import annotations

import json

import pytest

from notetaker import cli, config, languages


@pytest.fixture()
def lang_dir(tmp_path, monkeypatch):
    """Isolate the user language directory."""
    directory = tmp_path / "languages"
    directory.mkdir()
    monkeypatch.setattr(languages, "LANGUAGE_DIR", directory)
    languages.load_all(refresh=True)
    yield directory
    languages.load_all(refresh=True)


def write_pack(directory, code, **overrides):
    data = {
        "name": overrides.get("name", code.upper()),
        "headings": overrides.get("headings", {"key_ideas": "## Points"}),
        "prompts": overrides.get("prompts", {"map": "map {text}", "reduce": "reduce {text}"}),
    }
    if "script_range" in overrides:
        data["script_range"] = overrides["script_range"]
    if "none_markers" in overrides:
        data["none_markers"] = overrides["none_markers"]
    (directory / f"{code}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return languages.load_all(refresh=True)


# ------------------------------------------------------------------ built-ins
def test_english_and_thai_are_built_in(lang_dir):
    installed = languages.load_all(refresh=True)
    assert "en" in installed and "th" in installed


def test_thai_declares_its_script_range():
    """Thai has no word spaces, so dedupe needs the range."""
    assert languages.get("th").is_unspaced
    assert languages.get("th").contains_script("สวัสดี")


def test_english_is_not_unspaced():
    assert not languages.get("en").is_unspaced


def test_unknown_language_falls_back_to_english():
    assert languages.get("qq").code == "en"
    assert languages.get(None).code == "en"


# --------------------------------------------------------------- user packs
def test_user_pack_is_discovered(lang_dir):
    installed = write_pack(lang_dir, "de", name="Deutsch")
    assert "de" in installed
    assert installed["de"].name == "Deutsch"


def test_user_pack_supplies_its_prompts(lang_dir):
    write_pack(lang_dir, "de", prompts={"map": "Fasse zusammen {text}", "reduce": "Ordne {text}"})
    prompts = config.prompts_for("de")
    assert "Fasse zusammen" in prompts["map"]


def test_user_pack_can_override_a_builtin(lang_dir):
    """Users must be able to retune the shipped English prompts."""
    write_pack(lang_dir, "en", prompts={"map": "MY OWN PROMPT {text}", "reduce": "r {text}"})
    assert "MY OWN PROMPT" in config.prompts_for("en")["map"]


def test_new_language_appears_in_supported_list(lang_dir):
    write_pack(lang_dir, "ko")
    assert "ko" in config.supported_languages()


def test_unspaced_script_is_registered(lang_dir):
    """A new unspaced language must join Thai in the n-gram path."""
    write_pack(lang_dir, "ja", script_range=["\u3040", "\u30ff"])
    assert ("\u3040", "\u30ff") in languages.unspaced_ranges()


def test_custom_none_markers_are_collected(lang_dir):
    write_pack(lang_dir, "ja", none_markers=["なし"])
    assert "なし" in languages.none_markers()


def test_custom_action_heading_is_collected(lang_dir):
    write_pack(lang_dir, "ja", headings={"action_items": "## やるべきこと"})
    assert "## やるべきこと".lstrip("#").strip().lower() in languages.action_headings()


# ------------------------------------------------------------------ validation
def test_prompt_without_text_placeholder_is_rejected():
    """Without {text} the transcript would never reach the model."""
    with pytest.raises(ValueError, match=r"\{text\}"):
        languages.Language.from_dict("xx", {"prompts": {"map": "no placeholder", "reduce": "{text}"}})


def test_missing_prompts_are_rejected():
    with pytest.raises(ValueError, match="missing prompts"):
        languages.Language.from_dict("xx", {"name": "X"})


def test_broken_pack_is_skipped_not_fatal(lang_dir, capsys):
    """A typo in a language file must never stop a lecture being recorded."""
    (lang_dir / "bad.json").write_text("{ not valid json", encoding="utf-8")
    write_pack(lang_dir, "de")

    installed = languages.load_all(refresh=True)
    assert "bad" not in installed
    assert "de" in installed  # the good one still loads
    assert "en" in installed
    assert "bad" in capsys.readouterr().err


def test_pack_missing_text_placeholder_is_skipped(lang_dir, capsys):
    (lang_dir / "oops.json").write_text(
        json.dumps({"prompts": {"map": "nope", "reduce": "nope"}}), encoding="utf-8"
    )
    assert "oops" not in languages.load_all(refresh=True)


# --------------------------------------------------------------- round trips
def test_template_is_valid_and_loadable(lang_dir):
    data = languages.template("ja", "日本語")
    lang = languages.Language.from_dict("ja", data)
    assert lang.name == "日本語"
    assert "{text}" in lang.map_prompt


def test_save_then_load(lang_dir):
    languages.save("ja", languages.template("ja", "日本語"))
    assert languages.load_all(refresh=True)["ja"].name == "日本語"


def test_save_rejects_an_invalid_pack(lang_dir):
    with pytest.raises(ValueError):
        languages.save("bad", {"prompts": {"map": "no placeholder", "reduce": "x {text}"}})


def test_builtin_roundtrips_through_dict():
    """`lang edit` copies a built-in out to JSON; it must survive the trip."""
    thai = languages.BUILTIN["th"]
    restored = languages.Language.from_dict("th", thai.to_dict())
    assert restored.name == thai.name
    assert restored.script_range == thai.script_range
    assert restored.key_heading == thai.key_heading


# --------------------------------------------------------------------- CLI
def test_lang_list_runs(lang_dir, capsys):
    assert cli.cmd_lang(cli.build_parser().parse_args(["lang", "list"])) == 0
    assert "en" in capsys.readouterr().out


def test_lang_add_creates_a_pack(lang_dir, capsys):
    assert cli.cmd_lang(cli.build_parser().parse_args(["lang", "add", "ja"])) == 0
    assert (lang_dir / "ja.json").exists()
    assert "ja" in config.supported_languages()


def test_lang_add_sets_script_range_for_unspaced_languages(lang_dir):
    cli.cmd_lang(cli.build_parser().parse_args(["lang", "add", "ja"]))
    data = json.loads((lang_dir / "ja.json").read_text(encoding="utf-8"))
    assert "script_range" in data, "Japanese needs character n-grams like Thai"


def test_lang_add_omits_script_range_for_spaced_languages(lang_dir):
    cli.cmd_lang(cli.build_parser().parse_args(["lang", "add", "de"]))
    data = json.loads((lang_dir / "de.json").read_text(encoding="utf-8"))
    assert "script_range" not in data


def test_lang_add_rejects_a_code_whisper_cannot_transcribe(lang_dir, capsys):
    assert cli.cmd_lang(cli.build_parser().parse_args(["lang", "add", "zzz"])) == 1
    assert "not a Whisper language" in capsys.readouterr().out


def test_lang_add_refuses_to_clobber_without_force(lang_dir, capsys):
    cli.cmd_lang(cli.build_parser().parse_args(["lang", "add", "ja"]))
    assert cli.cmd_lang(cli.build_parser().parse_args(["lang", "add", "ja"])) == 1


def test_lang_remove_deletes_a_custom_pack(lang_dir):
    cli.cmd_lang(cli.build_parser().parse_args(["lang", "add", "ja"]))
    assert cli.cmd_lang(cli.build_parser().parse_args(["lang", "remove", "ja"])) == 0
    assert not (lang_dir / "ja.json").exists()


def test_lang_remove_of_a_builtin_leaves_it_usable(lang_dir, capsys):
    """Removing a customised 'en' must fall back to the built-in, not break."""
    write_pack(lang_dir, "en")
    cli.cmd_lang(cli.build_parser().parse_args(["lang", "remove", "en"]))
    assert "en" in languages.load_all(refresh=True)


def test_lang_remove_unknown_fails_cleanly(lang_dir, capsys):
    assert cli.cmd_lang(cli.build_parser().parse_args(["lang", "remove", "ja"])) == 1
