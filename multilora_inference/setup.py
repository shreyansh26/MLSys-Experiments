# MAX_JOBS=256 python setup.py build_ext --inplace
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

this_dir = Path(__file__).parent

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr")  # headers at /usr/include
NVCC_BIN  = os.environ.get("NVCC", "/usr/bin/nvcc")

# Ensure nvcc is found even if CUDA_HOME is not a canonical toolkit root
os.environ["PATH"] = f"{Path(NVCC_BIN).parent}:{os.environ.get('PATH','')}"

setup(
    name="bgmv_cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="bgmv_cuda._bgmv_cuda",
            sources=[
                str(this_dir / "bgmv_cuda" / "bgmv_ext.cu"),
            ],
            include_dirs=[
                str(this_dir / "bgmv_cuda"),
                f"{CUDA_HOME}/include"
            ],  
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--extended-lambda",
                    f"-I{CUDA_HOME}/include",
                    "-I/usr/include/x86_64-linux-gnu",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)


