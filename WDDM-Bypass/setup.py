"""WDDM-Bypass v1.0.0 build script.

Builds the pybind11 extension ``wddm_bypass._wddm_bypass`` from ``csrc/``.

Packaging stays fail-safe in the same sense as the repository root ``setup.py``:
a machine without a CUDA toolchain still gets the pure-Python layer
(``wddm_bypass.bucketing`` and the ``Engine`` facade), and the missing extension
surfaces as ``wddm_bypass.BackendUnavailable`` with the exact build command.
Nothing silently pretends the engine exists.

Three Windows-specific problems are handled here, because they are what makes a
plain ``CUDAExtension`` fail on a modern box:

* ``CUDA_HOME`` must be set *before* torch is imported - torch's extension
  helper resolves ``nvcc`` at import time.
* MSVC 19.40+ is newer than the host-compiler whitelist of CUDA 12.1, so nvcc
  refuses with ``unsupported Microsoft Visual Studio version``. ``-allow-
  unsupported-compiler`` (plus ``_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`` for
  the STL) is added for exactly that reason.
* ``vcvars64.bat`` locates the Windows SDK through ``KitsRoot10`` in the
  registry. Once the kit is moved or re-installed on another drive that key goes
  stale, ``vcvars`` reports a valid CL but an ``INCLUDE`` without ``ucrt``, and
  every ``<corecrt.h>`` include fails. The SDK root is therefore resolved here
  (environment, registry, drive scan) and spliced into ``INCLUDE``/``LIB``.


Set ``WDDM_BYPASS_WITH_TORCH=1`` to additionally link c10 so ``replay()``
follows the current PyTorch stream without a Python round trip.
"""

import os
import re
import subprocess
import sys

from setuptools import setup

CUDA_FALLBACK_WARNING = (
    "Warning: the WDDM-Bypass CUDA extension cannot be built. "
    "Installing the pure-Python layer only."
)

SOURCES = [
    os.path.join("csrc", "graph_engine.cu"),
    os.path.join("csrc", "bypass_core.cpp"),
    os.path.join("csrc", "torch_interop.cpp"),
]

INCLUDE_DIRS = [os.path.join("csrc", "include")]

#: Libraries ``torch.utils.cpp_extension.CUDAExtension`` appends to every
#: extension it describes. They are stripped unless libtorch is linked on
#: purpose, because a .pyd that links them can only be imported by a process
#: that can also find that exact torch build's DLLs.
TORCH_LIBRARIES = (
    "c10",
    "c10_cuda",
    "sleef",
    "torch",
    "torch_cpu",
    "torch_cuda",
    "torch_python",
)

#: Toolkit this project is validated against on Windows. It is tried before the
#: process environment so a stale ``CUDA_HOME`` left behind by an older install
#: cannot silently pick the wrong nvcc. The environment still wins when this
#: directory is absent (Linux, CI, a different box).
DEFAULT_CUDA_HOME = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

CXX_FLAGS = [
    "/std:c++17",
    "/utf-8",
    "/O2",
    "/EHsc",
    # MSVC 14.4x <yvals_core.h> hard-errors when the compiler identity and the
    # STL it ships with disagree, which is exactly what nvcc's host pass trips
    # over. The macro is what the STL itself documents for that case.
    "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
]


def _is_windows():
    return sys.platform == "win32"


def _find_nvcc():
    """Return ``(cuda_home, nvcc_path)`` or ``(None, None)``."""
    exe = "nvcc.exe" if _is_windows() else "nvcc"

    candidate = os.path.join(DEFAULT_CUDA_HOME, "bin", exe)
    if os.path.isfile(candidate):
        return DEFAULT_CUDA_HOME, candidate

    for variable in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(variable)
        if not cuda_home:
            continue
        candidate = os.path.join(cuda_home, "bin", exe)
        if os.path.isfile(candidate):
            return cuda_home, candidate

    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = os.path.join(directory, exe)
        if os.path.isfile(candidate):
            # <cuda_home>/bin/nvcc -> <cuda_home>
            return os.path.dirname(os.path.dirname(candidate)), candidate
    return None, None


def _nvcc_release(nvcc_path):
    """Parse ``nvcc --version`` into ``"12.1"`` (or ``None``)."""
    try:
        output = subprocess.check_output(
            [nvcc_path, "--version"], stderr=subprocess.STDOUT, text=True
        )
    except Exception:
        return None
    match = re.search(r"release (\d+)\.(\d+)", output)
    if match is None:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def _activate_msvc():
    """Import the MSVC environment into ``os.environ`` via vcvars64.bat.

    Returns a human readable status. Every failure is swallowed: the caller ends
    up with the same "cannot build" message it would have printed anyway.
    """
    if os.environ.get("VSCMD_ARG_TGT_ARCH"):
        return "MSVC environment already active"

    program_files = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")
    if not program_files:
        return "no Program Files directory in the environment"

    vswhere = os.path.join(program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return "vswhere.exe not found (Visual Studio Build Tools missing?)"

    try:
        location = subprocess.check_output(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as error:
        return f"vswhere failed ({error})"

    if not location:
        return "no Visual Studio installation with the C++ toolchain"

    vcvars = os.path.join(location, "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not os.path.isfile(vcvars):
        return f"vcvars64.bat missing under {location}"

    try:
        # `cmd /c "vcvars64.bat && set"` prints the environment it produced.
        dumped = subprocess.check_output(
            f'"{vcvars}" && set', shell=True, text=True, stderr=subprocess.DEVNULL
        )
    except Exception as error:
        return f"vcvars64.bat failed ({error})"

    for line in dumped.splitlines():
        name, separator, value = line.partition("=")
        if separator and name and not name.startswith("="):
            os.environ[name] = value

    os.environ.setdefault("DISTUTILS_USE_SDK", "1")
    os.environ.setdefault("MSSdk", "1")
    return f"MSVC environment loaded from {location}"


SDK_INCLUDE_LEAVES = ("ucrt", "um", "shared", "winrt", "cppwinrt")

SDK_LIB_LEAVES = (("ucrt", "x64"), ("um", "x64"))


def _sdk_version_from_root(root):
    """Newest SDK version under ``<root>/Include`` that really ships ``corecrt.h``."""
    include_root = os.path.join(root, "Include")
    if not os.path.isdir(include_root):
        return None

    versions = [
        name
        for name in os.listdir(include_root)
        if os.path.isfile(os.path.join(include_root, name, "ucrt", "corecrt.h"))
    ]
    if not versions:
        return None

    def _as_tuple(name):
        try:
            return tuple(int(part) for part in name.split("."))
        except ValueError:
            return (0,)

    return max(versions, key=_as_tuple)


def _registry_kits_root():
    """``KitsRoot10`` from the registry - only if that directory still exists."""
    try:
        import winreg
    except ImportError:
        return None

    views = (
        winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        winreg.KEY_READ | winreg.KEY_WOW64_32KEY,
    )
    for view in views:
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows Kits\Installed Roots",
                0,
                view,
            ) as key:
                root = winreg.QueryValueEx(key, "KitsRoot10")[0]
        except OSError:
            continue
        if root and os.path.isdir(root):
            return root
    return None


def _find_windows_sdk():
    """Locate the Windows SDK as ``(root, version, source)``.

    ``vcvars64.bat`` resolves the kit through ``KitsRoot10``, so a kit that was
    installed on another drive leaves a stale key behind: ``cl`` is found, the
    SDK is not, and the first ``<corecrt.h>`` include fails. The environment
    variable, the registry and a drive scan are therefore all consulted.
    """
    declared_root = os.environ.get("WindowsSdkDir")
    declared_version = (os.environ.get("WindowsSDKVersion") or "").strip("\\")
    if declared_root and declared_version:
        header = os.path.join(declared_root, "Include", declared_version, "ucrt", "corecrt.h")
        if os.path.isfile(header):
            return declared_root, declared_version, "WindowsSdkDir environment"

    root = _registry_kits_root()
    if root is not None:
        version = _sdk_version_from_root(root)
        if version is not None:
            return root, version, "KitsRoot10 registry"

    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZAB":
        candidate = os.path.join(f"{letter}:\\", "Windows Kits", "10")
        version = _sdk_version_from_root(candidate)
        if version is not None:
            return candidate, version, f"drive scan ({letter}:)"

    return None, None, "no Windows SDK with UCRT headers found"


def _activate_windows_sdk():
    """Splice the SDK's headers and x64 import libraries into INCLUDE/LIB.

    Returns a one-line status. Missing SDK stays a warning here: the compile
    itself will fail with the real diagnostic, which is more useful than a
    fabricated one.
    """
    root, version, source = _find_windows_sdk()
    if root is None:
        return f"windows sdk not found ({source})"

    os.environ["WindowsSdkDir"] = root
    os.environ["WindowsSDKVersion"] = version + "\\"

    include_dirs = [
        os.path.join(root, "Include", version, leaf)
        for leaf in SDK_INCLUDE_LEAVES
        if os.path.isdir(os.path.join(root, "Include", version, leaf))
    ]
    lib_dirs = [
        os.path.join(root, "Lib", version, *leaves)
        for leaves in SDK_LIB_LEAVES
        if os.path.isdir(os.path.join(root, "Lib", version, *leaves))
    ]

    for variable, additions in (("INCLUDE", include_dirs), ("LIB", lib_dirs)):
        current = [entry for entry in os.environ.get(variable, "").split(os.pathsep) if entry]
        missing = [entry for entry in additions if entry not in current]
        if missing:
            os.environ[variable] = os.pathsep.join(missing + current)

    return f"windows sdk {version} from {source}"


def _configure_environment():
    """Locate CUDA and export ``CUDA_HOME`` before torch is ever imported.

    Returns ``(cuda_home, nvcc_path, nvcc_release)``; all ``None`` when nvcc is
    absent. This must run at import time, because importing torch freezes the
    toolkit location it will use.
    """
    cuda_home, nvcc_path = _find_nvcc()
    if cuda_home is None:
        return None, None, None

    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CUDA_PATH"] = cuda_home
    for directory in (os.path.join(cuda_home, "bin"), os.path.join(cuda_home, "libnvvp")):
        current = os.environ.get("PATH", "")
        if os.path.isdir(directory) and directory not in current.split(os.pathsep):
            os.environ["PATH"] = directory + os.pathsep + current

    return cuda_home, nvcc_path, _nvcc_release(nvcc_path)


CUDA_HOME, NVCC_PATH, NVCC_RELEASE = _configure_environment()

# Only touch the Visual Studio environment when there is something to build
# with; a pure-Python install on a machine without CUDA stays silent.
MSVC_STATUS = _activate_msvc() if (_is_windows() and NVCC_PATH is not None) else "skipped"
SDK_STATUS = _activate_windows_sdk() if (_is_windows() and NVCC_PATH is not None) else "skipped"

if _is_windows() and NVCC_PATH is not None:
    # distutils must not re-probe the (possibly stale) SDK on its own once the
    # environment above has been curated.
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")
    os.environ.setdefault("MSSdk", "1")


def _cuda_status():
    """Explain, in one sentence, whether the extension can be built."""
    if NVCC_PATH is None:
        return False, "nvcc not found in CUDA_HOME, CUDA_PATH or PATH"

    try:
        import torch
    except Exception as error:
        return False, f"torch is not importable ({error})"

    torch_cuda = getattr(torch.version, "cuda", None)
    if torch_cuda is None:
        return False, "this PyTorch build has no CUDA support"

    if NVCC_RELEASE is not None:
        torch_major_minor = ".".join(torch_cuda.split(".")[:2])
        if torch_major_minor != NVCC_RELEASE:
            # Not fatal any more: torch's own major.minor guard is neutralised in
            # _build_extension, and the extension only links libtorch on explicit
            # opt-in. The mismatch is reported so the log stays truthful.
            return True, (
                f"nvcc {NVCC_RELEASE}, torch CUDA {torch_cuda} "
                "(major.minor differ; torch guard overridden)"
            )

    return True, f"nvcc {NVCC_RELEASE}, torch CUDA {torch_cuda}"


def _build_extension(status):
    """Describe the CUDA extension, or return ``([], {})`` with a reason."""
    with_torch = os.environ.get("WDDM_BYPASS_WITH_TORCH") == "1"
    try:
        import torch.utils.cpp_extension as _torch_extension

        # torch aborts the build when nvcc and torch disagree on CUDA
        # major.minor. That guard protects people linking libtorch; this
        # extension consumes headers only (libtorch is linked solely when
        # WDDM_BYPASS_WITH_TORCH=1, an explicit opt-in), so the mismatch is
        # reported in the version line above and the guard is neutralised
        # instead of forcing a torch reinstall.
        _torch_extension._check_cuda_version = lambda *args, **kwargs: None
    except Exception:
        pass

    try:
        from torch.utils.cpp_extension import (
            BuildExtension,
            CUDAExtension,
            include_paths,
        )
    except Exception as error:
        print(f"[setup] {CUDA_FALLBACK_WARNING}")
        print(f"[setup] reason: torch's C++/CUDA extension API is unavailable ({error})")
        return [], {}

    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-Xcompiler=/utf-8",
        "-Xcompiler=/EHsc",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
    ]
    if _is_windows():
        # CUDA 12.1 whitelists MSVC 19.3x; 19.40+ trips the check. Every 11.x and
        # 12.x nvcc accepts the override, so it is unconditional on Windows.
        nvcc_flags.append("-allow-unsupported-compiler")

    define_macros = []
    libraries = []
    library_dirs = []
    if with_torch:
        # Opt-in: link c10 so stream_from_token(0) follows torch's current stream
        # natively. Off by default because it pins the wheel to one torch build.
        import torch

        define_macros.append(("WDDM_BYPASS_WITH_TORCH", "1"))
        libraries.extend(["c10", "c10_cuda", "torch_cuda"])
        torch_lib = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "lib")
        if os.path.isdir(torch_lib):
            library_dirs.append(torch_lib)

    try:
        extension = CUDAExtension(
            name="wddm_bypass._wddm_bypass",
            sources=SOURCES,
            include_dirs=INCLUDE_DIRS + list(include_paths()),
            define_macros=define_macros,
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": nvcc_flags},
            libraries=libraries,
            library_dirs=library_dirs,
        )
    except Exception as error:
        print(f"[setup] {CUDA_FALLBACK_WARNING}")
        print(f"[setup] reason: could not describe the CUDA extension ({error})")
        return [], {}

    if not with_torch:
        # CUDAExtension links libtorch into every extension it describes, which
        # would make the engine depend on one exact torch install at import
        # time. The engine only needs pybind11 headers and cudart, so the extra
        # libraries are removed; the header paths stay.
        dropped = [lib for lib in extension.libraries if lib in TORCH_LIBRARIES]
        extension.libraries = [lib for lib in extension.libraries if lib not in TORCH_LIBRARIES]
        if dropped:
            print(f"[setup] not linking libtorch ({', '.join(dropped)})")

    print(f"[setup] Building wddm_bypass._wddm_bypass ({status}); {MSVC_STATUS}; {SDK_STATUS}")
    return [extension], {"build_ext": BuildExtension.with_options(use_ninja=False)}


ext_modules = []
cmdclass = {}
try:
    _cuda_ok, _cuda_message = _cuda_status()
except Exception as _error:  # detection must never break packaging
    _cuda_ok, _cuda_message = False, f"CUDA detection failed ({_error})"

if _cuda_ok:
    ext_modules, cmdclass = _build_extension(_cuda_message)
else:
    print(f"[setup] {CUDA_FALLBACK_WARNING}")
    print(f"[setup] reason: {_cuda_message}")

_here = os.path.dirname(os.path.abspath(__file__))
_readme = os.path.join(_here, "README.md")
if os.path.isfile(_readme):
    with open(_readme, encoding="utf-8") as handle:
        _long_description = handle.read()
else:
    _long_description = "WDDM-Bypass - pre-instantiated CUDA graph replay for Windows."

setup(
    name="wddm-bypass",
    version="1.0.0",
    description="Pre-instantiated CUDA graph replay engine for Windows",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["wddm_bypass"],
    # v2.6.0: the Python package lives at the repository root, one level above
    # these native sources, so an in-place build drops the .pyd where a plain
    # ``import wddm_bypass`` from the repository root finds it.
    package_dir={"wddm_bypass": os.path.join("..", "wddm_bypass")},
    python_requires=">=3.8",
    install_requires=[],
    extras_require={"dev": ["pytest>=7.0"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
