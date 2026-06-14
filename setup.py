"""
Omni-VRAM Setup Configuration
==============================

Builds the CUDA extension module and installs the vram_core Python package.
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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

    # CUDA extension
    ext_modules=[
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
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False),
    },
)