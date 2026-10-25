"""Loader for the compiled C++/CUDA extension.

The engine itself lives in ``wddm_bypass._wddm_bypass`` (pybind11). Importing it
is the only step that needs a working build, so it is isolated here: everything
else in this package is pure Python and importable on any machine.

Nothing here ever silently degrades to a Python fallback. A graph replay engine
that pretends to work would be worse than an import error, so the failure is a
single, precise exception carrying the build command.
"""

from __future__ import annotations

import contextlib
import glob
import importlib
import os
import sys
from types import ModuleType

__all__ = [
    "BUILD_HINT",
    "EXTENSION_NAME",
    "BackendUnavailable",
    "backend_available",
    "load_backend",
]

#: Importable name of the pybind11 module.
EXTENSION_NAME = "wddm_bypass._wddm_bypass"

BUILD_HINT = (
    "Build the extension from the WDDM-Bypass folder:\n"
    "    python setup.py build_ext --inplace\n"
    "Requirements: the Windows SDK (corecrt.h, kernel32.lib), nvcc, and a\n"
    "PyTorch build whose CUDA major.minor matches the installed toolkit.\n"
    "The built module links cudart64_*.dll, so the CUDA toolkit's bin folder\n"
    "must be on PATH or named by CUDA_HOME / CUDA_PATH."
)


def _cuda_dll_directories():
    """Directories that may hold ``cudart64_*.dll``, most specific first."""
    candidates = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(variable)
        if home:
            candidates.append(os.path.join(home, "bin"))

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and glob.glob(os.path.join(entry, "cudart64_*.dll")):
            candidates.append(entry)

    seen = set()
    for directory in candidates:
        key = os.path.normcase(os.path.abspath(directory))
        if key not in seen and os.path.isdir(directory):
            seen.add(key)
            yield directory


@contextlib.contextmanager
def _dll_directories():
    """Register the CUDA runtime directory for the duration of the import.

    Since Python 3.8 a Windows extension module is loaded with the restricted
    search flags, so ``PATH`` alone does not resolve its dependencies: a module
    that links ``cudart64_12.dll`` fails with "DLL load failed" even though
    ``where cudart64_12.dll`` finds it. Registering the directory is what the
    loader actually honours.
    """
    if not (sys.platform == "win32" and hasattr(os, "add_dll_directory")):
        yield
        return

    handles = []
    try:
        for directory in _cuda_dll_directories():
            handles.append(os.add_dll_directory(directory))
        yield
    finally:
        for handle in handles:
            handle.close()


class BackendUnavailable(RuntimeError):
    """Raised when the compiled extension cannot be imported."""


def load_backend(module_name: str = EXTENSION_NAME) -> ModuleType:
    """Import and return the extension module.

    Args:
        module_name: Importable module name; override only for tests.

    Raises:
        BackendUnavailable: The extension is missing or failed to load.
    """
    try:
        with _dll_directories():
            return importlib.import_module(module_name)
    except ImportError as error:
        message = (
            f"WDDM-Bypass: the compiled backend '{module_name}' is not available "
            f"({error}).\n{BUILD_HINT}"
        )
        raise BackendUnavailable(message) from error



def backend_available(module_name: str = EXTENSION_NAME) -> bool:
    """True when :func:`load_backend` would succeed."""
    try:
        load_backend(module_name)
    except BackendUnavailable:
        return False
    return True
