"""
vram_core Setup Configuration
==============================

Builds the CUDA extension module and installs the vram_core Python package.
If CUDA is not available or version mismatch, builds a pure Python package (no GPU extension).
"""

import os
import sys
from setuptools import setup, find_packages

# CUDA Extension (optional) — skip if CUDA unavailable or version mismatch
ext_modules = []
cmdclass = {}

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


cuda_ok, cuda_msg = _check_cuda_available()
if cuda_ok:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        ext_modules = [
            CUDAExtension(
                name='vram_core._vram_hacker',
                sources=['vram_hacker.cu'],
                extra_compile_args={'nvcc': ['-O3']},
            ),
        ]
        cmdclass = {'build_ext': BuildExtension}
        print(f"[setup] Building with CUDA extension: {cuda_msg}")
    except Exception as e:
        print(f"[setup] Warning: Skipping CUDA extension: {e}")
        ext_modules = []
        cmdclass = {}
else:
    print(f"[setup] Warning: Skipping CUDA extension: {cuda_msg}")
    ext_modules = []
    cmdclass = {}

# Read README
with open('README.md', encoding='utf-8') as _f:
    _long_description = _f.read()

# Package Setup
setup(
    name='vram_core',
    version='2.5.0',
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