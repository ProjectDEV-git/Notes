"""Using an NVIDIA graphics card only when it really works."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from notetaker import asr, config, gpu, menu


def test_no_card_means_the_processor_exactly_as_before(monkeypatch):
    monkeypatch.setattr(gpu, "cuda_devices", lambda: 0)
    assert gpu.candidates("auto", "int8", "float16") == [("cpu", "int8")]


def test_a_card_is_tried_first_with_the_processor_behind_it(monkeypatch):
    monkeypatch.setattr(gpu, "cuda_devices", lambda: 1)
    assert gpu.candidates("auto", "int8", "float16") == [("cuda", "float16"), ("cpu", "int8")]


def test_the_card_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(gpu, "cuda_devices", lambda: 1)
    assert gpu.candidates("cpu", "int8", "float16") == [("cpu", "int8")]


class FakeModel:
    """Stands in for faster-whisper. `broken` decides which devices fail, and how."""

    broken: dict[str, str] = {}
    loaded: list[tuple[str, str]] = []

    def __init__(self, name, device, compute_type, cpu_threads):
        FakeModel.loaded.append((device, compute_type))
        if FakeModel.broken.get(device) == "load":
            raise RuntimeError("CUDA driver version is insufficient")
        self.device = device

    def transcribe(self, audio, **kwargs):
        if FakeModel.broken.get(self.device) == "run":
            raise RuntimeError("Library cudnn_ops64_9.dll is not found or cannot be loaded")
        return iter([]), None


@pytest.fixture()
def fake_whisper(monkeypatch):
    import faster_whisper

    FakeModel.broken, FakeModel.loaded = {}, []
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr(gpu, "cuda_devices", lambda: 1)
    monkeypatch.setattr(gpu, "expose_pip_libraries", lambda: None)
    return FakeModel


def test_a_working_card_is_used(fake_whisper):
    transcriber = asr.Transcriber("small", device="auto")
    assert transcriber.device == "cuda" and transcriber.gpu_problem is None
    assert fake_whisper.loaded == [("cuda", config.GPU_COMPUTE_TYPE)]


def test_a_card_that_will_not_load_falls_back(fake_whisper):
    fake_whisper.broken = {"cuda": "load"}
    transcriber = asr.Transcriber("small", device="auto")
    assert transcriber.device == "cpu"
    assert "driver" in transcriber.gpu_problem


def test_missing_libraries_are_caught_by_the_test_run(fake_whisper):
    """cuDNN being absent only shows up when the model first runs."""
    fake_whisper.broken = {"cuda": "run"}
    transcriber = asr.Transcriber("small", device="auto")
    assert transcriber.device == "cpu"
    assert "cudnn" in transcriber.gpu_problem
    assert fake_whisper.loaded == [("cuda", config.GPU_COMPUTE_TYPE), ("cpu", config.COMPUTE_TYPE)]


def test_a_broken_processor_is_still_reported(fake_whisper):
    fake_whisper.broken = {"cuda": "load", "cpu": "load"}
    with pytest.raises(RuntimeError):
        asr.Transcriber("small", device="auto")


def test_the_check_is_quiet_without_a_card(monkeypatch):
    monkeypatch.setattr(gpu, "cuda_devices", lambda: 0)
    with patch("notetaker.audio.list_sources", return_value=[]), \
            patch("notetaker.summarize.ollama_available", return_value=False):
        names = [c.name for c in menu.run_checks()]
    assert not any("graphics" in n for n in names)


def test_missing_card_libraries_are_optional_with_the_command(monkeypatch):
    monkeypatch.setattr(gpu, "cuda_devices", lambda: 1)
    monkeypatch.setattr(gpu, "libraries_ok", lambda: False)
    with patch("notetaker.audio.list_sources", return_value=[]), \
            patch("notetaker.summarize.ollama_available", return_value=False):
        row = next(c for c in menu.run_checks() if "graphics" in c.name)
    assert row.optional and not row.ok
    assert "requirements-gpu.txt" in row.fix


def test_an_optional_row_never_fails_the_check():
    rows = [menu.Check("ffmpeg", True, "found"),
            menu.Check("graphics card", False, "libraries missing", "pip ...", optional=True)]
    assert menu.print_checks(rows) is True
    assert menu.print_checks(rows + [menu.Check("Ollama", False, "not running")]) is False


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_macs_never_try_cuda_libraries():
    assert gpu.libraries_ok() is False


def test_gpu_requirements_name_cuda_12_and_cudnn_9():
    text = gpu.REQUIREMENTS.read_text()
    assert "nvidia-cublas-cu12" in text and "nvidia-cudnn-cu12==9" in text
