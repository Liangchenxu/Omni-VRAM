"""
Omni-VRAM Setup Configuration
==============================

Builds the CUDA extension module and installs the vram_core Python package.
If CUDA is not available, builds a pure Python package (no GPU extension).
"""

from setuptools import setup, find_packages

# ── CUDA Extension (optional) ────────────────────────────────────
ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ext_modules = [
        CUDAExtension(
            name='vram_core._vram_hacker',
            sources=['vram_hacker.cu'],
            extra_compile_args={
                'nvcc': [
                    '-O3',
                    '-allow-unsupported-compiler',
                    '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
                ],
            },
        ),
    ]
    cmdclass = {'build_ext': BuildExtension.with_options(use_ninja=False)}
except (OSError, Exception) as e:
    print(f"Warning: CUDA not available, building without CUDA extension: {e}")
    ext_modules = []
    cmdclass = {}

# ── Package Setup ────────────────────────────────────────────────
setup(
    name='omni-vram',
    version='1.0.0',
    description='Real-time VRAM orchestration toolkit for voice-enabled LLM applications',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Liangchenxu',
    url='https://github.com/Liangchenxu/Omni-VRAM',
    license='MIT',

    # Python packages
    packages=find_packages(exclude=['tests', 'tests.*']),
    python_requires='>=3.8',

    # Dependencies
    install_requires=[
        'numpy>=1.20.0',
        'pydub>=0.25.1',
        'python-dotenv>=1.0.0',
    ],
    extras_require={
        'audio': [
            'openai>=1.0.0',
        ],
        'realtime': [
            'pyaudio>=0.2.11',
        ],
        'dev': [
            'pytest>=7.0.0',
        ],
    },

    # CUDA extension (empty list if CUDA not available)
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)