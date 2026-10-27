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
    "heal_windows_dll_environment",
    "load_backend",
]

#: Importable name of the pybind11 module.
EXTENSION_NAME = "wddm_bypass._wddm_bypass"

#: Set once SetErrorMode() has been applied. The heal function is idempotent.
_DLL_HEALED = False
#: Handles returned by ``os.add_dll_directory``. They have to outlive the import
#: of any CUDA-linking extension, so they are kept here instead of being closed.
_DLL_HANDLES: list = []
#: Directories already registered by :func:`heal_windows_dll_environment`.
_HEALED_DIRS: set = set()


def heal_windows_dll_environment() -> list:
    """Repair the Windows DLL search path and suppress modal loader dialogs.

    Two problems show up when a plain ``python.exe`` imports a CUDA-linking
    extension out of a Conda environment:

    * A missing dependency raises a *modal* "DLL load failed" dialog from the
      loader, which blocks an unattended process forever. ``SetErrorMode`` with
      ``SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX``
      (``0x8003``) turns it into a normal exception instead.
    * Since Python 3.8 the extension is loaded with the restricted search flags,
      so ``PATH`` alone does not resolve ``cudart64_*.dll`` or the Conda
      ``Library\\bin`` dependencies: the directories have to be registered with
      ``os.add_dll_directory``.

    Idempotent: the second and every later call registers nothing and returns an
    empty list. On any platform other than Windows it does nothing and returns
    an empty list.

    Returns:
        The directories registered by *this* call, in registration order.
    """
    global _DLL_HEALED

    added_dirs: list = []
    if sys.platform != "win32":
        return added_dirs

    if not _DLL_HEALED:
        try:
            import ctypes

            ctypes.windll.kernel32.SetErrorMode(0x8003)
        except Exception:  # pragma: no cover - always available on Windows
            pass
        _DLL_HEALED = True

    prefixes = {sys.prefix, sys.exec_prefix}
    for variable in ("CONDA_PREFIX", "CUDA_HOME"):
        value = os.environ.get(variable)
        if value:
            prefixes.add(value)

    candidate_subdirs = [
        "",
        os.path.join("Library", "bin"),
        os.path.join("Library", "mingw-w64", "bin"),
        "DLLs",
        "Scripts",
    ]

    candidates = []
    for prefix in sorted(prefixes):
        for sub in candidate_subdirs:
            directory = os.path.join(prefix, sub) if sub else prefix
            if os.path.isdir(directory):
                candidates.append(directory)

    for variable in ("CUDA_PATH", "CUDA_HOME"):
        home = os.environ.get(variable)
        if home:
            cuda_bin = os.path.join(home, "bin")
            if os.path.isdir(cuda_bin):
                candidates.append(cuda_bin)

    current_path = os.environ.get("PATH", "")
    path_parts = {
        os.path.normcase(os.path.normpath(part))
        for part in current_path.split(os.pathsep)
        if part
    }

    for directory in candidates:
        key = os.path.normcase(os.path.abspath(directory))
        if key in _HEALED_DIRS:
            continue
        _HEALED_DIRS.add(key)

        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(directory))
            except OSError:  # pragma: no cover - non-directory or already gone
                pass
        added_dirs.append(directory)

        if key not in path_parts:
            current_path = directory + os.pathsep + current_path
            path_parts.add(key)

    os.environ["PATH"] = current_path
    return added_dirs

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
    # Register Conda/CUDA DLL directories and switch the loader out of its modal
    # error mode before the import, so a missing dependency raises an exception
    # instead of opening a dialog that blocks the process.
    heal_windows_dll_environment()
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
