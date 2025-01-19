#pragma once

#include "kernels_fp32/01_naive.cuh"
#include "kernels_fp32/02_global_coalescing.cuh"
#include "kernels_fp32/03_shared_memory.cuh"
#include "kernels_fp32/04_1d_blocktiling.cuh"
#include "kernels_fp32/05_2d_blocktiling.cuh"
#include "kernels_fp32/06_vectorize.cuh"
#include "kernels_fp32/10_cuda_warptiling.cuh"