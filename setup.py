from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vram_core',
    ext_modules=[
        CUDAExtension(
            name='vram_core',
            sources=['vram_hacker.cu'],
            # Compiler flags to bypass MSVC version checks and optimize for maximum throughput
            extra_compile_args={'nvcc': [
                '-O3', 
                '-allow-unsupported-compiler',
                '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'
            ]} 
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)