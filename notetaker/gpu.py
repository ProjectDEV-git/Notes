"""Transcribing on an NVIDIA graphics card, when there is one.

A graphics card transcribes several times faster than the processor, which
matters most for Thai (about 5x slower than real time on a laptop CPU).

Three things have to be true, and only the first is common:
  1. an NVIDIA card with a working driver
  2. NVIDIA's maths libraries (cuBLAS, cuDNN), which are a separate ~1 GB
     download that most computers do not have
  3. they load and actually run

So nothing is assumed. The card is used only after a one-second test run
succeeds, and on any failure NoteTaker quietly uses the processor exactly as
before. A missing library must never be the reason a class is not recorded.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS = APP_DIR / "requirements-gpu.txt"

# Names the libraries are loaded by on each system (CUDA 12, cuDNN 9: what
# ctranslate2 4.x is built against).
_LIBS = {
    "win32": ["cublas64_12.dll", "cudnn64_9.dll"],
    "linux": ["libcublas.so.12", "libcudnn.so.9"],
}


def cuda_devices() -> int:
    """How many NVIDIA cards ctranslate2 can see (0 without a driver)."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def expose_pip_libraries() -> None:
    """Make NVIDIA's pip-installed libraries findable by ctranslate2.

    `pip install -r requirements-gpu.txt` puts them inside the venv, where
    the system loader does not look. Windows is told about the folders;
    on Linux they are loaded up front, so later lookups by name succeed.
    """
    try:
        import nvidia  # namespace package from the nvidia-* wheels
    except ImportError:
        return
    for root in map(Path, getattr(nvidia, "__path__", [])):
        for package in ("cublas", "cudnn"):
            if sys.platform == "win32":
                folder = root / package / "bin"
                if folder.is_dir():
                    try:
                        os.add_dll_directory(str(folder))
                    except OSError:
                        pass
                    os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"
            else:
                folder = root / package / "lib"
                # cublasLt before cublas, which needs it.
                for library in sorted(folder.glob("lib*.so*"), key=lambda p: "Lt" not in p.name):
                    try:
                        ctypes.CDLL(str(library), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                    except OSError:
                        pass


def libraries_ok() -> bool:
    """True when cuBLAS and cuDNN can be loaded on this system."""
    names = _LIBS.get("win32" if sys.platform == "win32" else "linux")
    if not names or sys.platform == "darwin":
        return False
    expose_pip_libraries()
    for name in names:
        try:
            ctypes.CDLL(name)
        except OSError:
            return False
    return True


def candidates(requested: str, cpu_compute: str, gpu_compute: str) -> list[tuple[str, str]]:
    """Devices to try in order, as (device, compute_type)."""
    if requested == "cpu":
        return [("cpu", cpu_compute)]
    if requested in ("auto", "cuda") and cuda_devices() > 0:
        return [("cuda", gpu_compute), ("cpu", cpu_compute)]
    return [("cpu", cpu_compute)]


def install_hint() -> str:
    return f'"{sys.executable}" -m pip install -r "{REQUIREMENTS}"'
