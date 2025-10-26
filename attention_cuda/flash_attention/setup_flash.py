from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='attention_cuda_flash',
    ext_modules=[
        CUDAExtension(
            name='attention_cuda_flash',
            sources=[
                'attention_cuda_flash.cpp',
                'attention_kernel_flash.cu',
            ],
            extra_compile_args={
                'cxx': [
                    '-std=c++17',
                    '-O3',
                ],
                'nvcc': [
                    '-std=c++17',
                    '-O3',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-use_fast_math',
                    # Add compute capabilities - adjust based on your GPU
                    # Hopper (H100)
                    '-gencode', 'arch=compute_90,code=sm_90',
                    # Ampere (RTX 30xx, A100)
                    '-gencode', 'arch=compute_80,code=sm_80',
                    # Turing (RTX 20xx, T4)
                    '-gencode', 'arch=compute_75,code=sm_75',
                    # Volta (V100)
                    '-gencode', 'arch=compute_70,code=sm_70',
                ]
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

