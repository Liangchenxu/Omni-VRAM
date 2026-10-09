"""
vram_core Setup Configuration
==============================

Builds the optional CUDA extension module and installs the vram_core Python
package.

Packaging is **fail-safe**: when the CUDA toolkit, ``nvcc``, a host compiler,
the torch extension API or the GPU itself is missing, the build degrades to a
pure-Python wheel and prints a friendly warning -- ``pip install .`` (and
``python setup.py sdist bdist_wheel``) therefore never breaks on a machine
without a build toolchain, because every runtime path has a vectorized
NumPy/Python fallback.
"""

import os
import sys
from setuptools import setup, find_packages

# Warning emitted verbatim whenever the extension cannot be built. Kept as a
# single constant so packaging logs and CI can grep for it.
CUDA_FALLBACK_WARNING = (
    "Warning: CUDA build tools not found. "
    "Packaging/installing in pure Python mode with runtime vectorized fallback."
)


def _check_cuda_available():
    """Check if CUDA toolkit is available and version matches PyTorch."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA is not available in PyTorch"

        torch_cuda = torch.version.cuda
        if torch_cuda is None:
            return False, "PyTorch was not built with CUDA"

        # Check nvcc availability
        nvcc_path = None
        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            candidate = os.path.join(path_dir, "nvcc")
            candidate_exe = os.path.join(path_dir, "nvcc.exe")
            if os.path.isfile(candidate):
                nvcc_path = candidate
                break
            if os.path.isfile(candidate_exe):
                nvcc_path = candidate_exe
                break

        if nvcc_path is None:
            # Try CUDA_HOME
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            if cuda_home:
                nvcc_path = os.path.join(cuda_home, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc")
                if not os.path.isfile(nvcc_path):
                    return False, f"nvcc not found in CUDA_HOME={cuda_home}"

        if nvcc_path is None:
            return False, "nvcc not found in PATH or CUDA_HOME"

        # Check CUDA toolkit version matches PyTorch CUDA version
        try:
            import subprocess
            nvcc_output = subprocess.check_output([nvcc_path, "--version"], stderr=subprocess.STDOUT, text=True)
            # Extract version like "release 12.1"
            for line in nvcc_output.split("\n"):
                if "release" in line.lower():
                    # e.g. "Cuda compilation tools, release 12.1, V12.1.105"
                    parts = line.split("release")
                    if len(parts) >= 2:
                        nvcc_version = parts[1].strip().split(",")[0].strip().split(" ")[0]
                        torch_parts = torch_cuda.split(".")
                        nvcc_parts = nvcc_version.split(".")
                        torch_major_minor = ".".join(torch_parts[:2])
                        nvcc_major_minor = ".".join(nvcc_parts[:2])
                        if torch_major_minor != nvcc_major_minor:
                            return False, (
                                f"CUDA version mismatch: system nvcc={nvcc_version}, "
                                f"PyTorch CUDA={torch_cuda}. They must match (major.minor)."
                            )
                        # Also check patch version — warn if different (ABI may break)
                        torch_patch = int(torch_parts[2]) if len(torch_parts) > 2 else 0
                        nvcc_patch = int(nvcc_parts[2]) if len(nvcc_parts) > 2 else 0
                        if abs(torch_patch - nvcc_patch) > 2:
                            return False, (
                                f"CUDA patch version mismatch: system nvcc={nvcc_version}, "
                                f"PyTorch CUDA={torch_cuda}. Patch difference > 2 may cause ABI issues."
                            )
        except Exception as e:
            return False, f"Failed to check nvcc version: {e}"

        return True, f"CUDA {torch_cuda} ready"

    except ImportError:
        return False, "PyTorch is not installed"
    except Exception as e:
        return False, f"CUDA check failed: {e}"


def _build_cuda_extension(cuda_msg):
    """
    Describe the optional CUDA extension, degrading to pure Python on failure.

    Every step (torch's C++ extension API, host compiler discovery, extension
    description) is guarded so that no toolchain problem can abort packaging.

    Args:
        cuda_msg: Human readable status from :func:`_check_cuda_available`.

    Returns:
        Tuple ``(ext_modules, cmdclass)``; ``([], {})`` in pure-Python mode.
    """
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as error:  # ImportError / OSError / missing compiler stack
        print(f"[setup] {CUDA_FALLBACK_WARNING}")
        print(f"[setup] reason: torch C++/CUDA extension API unavailable ({error})")
        return [], {}

    try:
        extension = CUDAExtension(
            name='vram_core._vram_hacker',
            sources=['vram_hacker.cu'],
            extra_compile_args={'nvcc': ['-O3']},
        )
    except Exception as error:  # compiler probing raised inside torch
        print(f"[setup] {CUDA_FALLBACK_WARNING}")
        print(f"[setup] reason: could not describe the CUDA extension ({error})")
        return [], {}

    print(f"[setup] Building with CUDA extension: {cuda_msg}")
    return [extension], {'build_ext': BuildExtension}


# CUDA Extension (optional) — never fatal: falls back to a pure Python wheel
ext_modules = []
cmdclass = {}

try:
    cuda_ok, cuda_msg = _check_cuda_available()
except Exception as _error:  # defensive: detection must not break packaging
    cuda_ok, cuda_msg = False, f"CUDA detection failed: {_error}"

if cuda_ok:
    ext_modules, cmdclass = _build_cuda_extension(cuda_msg)
else:
    print(f"[setup] {CUDA_FALLBACK_WARNING}")
    print(f"[setup] reason: {cuda_msg}")

# Read README
with open('README.md', encoding='utf-8') as _f:
    _long_description = _f.read()

# Package Setup
setup(
    name='vram_core',
    version='2.6.1',
    description='vram_core - LLM Voice Interaction Framework',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='Liangchenxu',
    url='https://github.com/Liangchenxu/vram_core',
    license='MIT',

    # Python packages
    packages=['vram_core', 'vram_core.whisper', 'vram_core.chinese', 'vram_core.backends'],
    python_requires='>=3.8',

    # Dependencies
    install_requires=[
        'numpy>=1.20.0',
        'pydub>=0.25.1',
        'python-dotenv>=1.0.0',
        'requests>=2.28.0',
    ],
    extras_require={
        'audio': [
            'openai>=1.0.0',
        ],
        'realtime': [
            'pyaudio>=0.2.11',
        ],
        'tts': [
            'edge-tts>=6.1.0',
        ],
        'translation': [
            'deep-translator>=1.11.0',
        ],
        'grpc': [
            'grpcio>=1.50.0',
            'grpcio-tools>=1.50.0',
            'flask>=2.3.0',
        ],
        'dev': [
            'pytest>=7.0.0',
        ],
        'full': [
            'openai>=1.0.0',
            'pyaudio>=0.2.11',
            'edge-tts>=6.1.0',
            'deep-translator>=1.11.0',
            'grpcio>=1.50.0',
            'grpcio-tools>=1.50.0',
            'flask>=2.3.0',
        ],
    },

    # CUDA extension (empty list if CUDA not available)
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)