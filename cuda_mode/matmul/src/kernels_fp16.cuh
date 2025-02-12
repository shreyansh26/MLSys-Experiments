#pragma once

#include "kernels_fp16/01_hierarchical_tiling.cuh"
#include "kernels_fp16/02_hierarchical_tiling_unrolled.cuh"
#include "kernels_fp16/03_hierarcical_tiling_unrolled_vectorized.cuh"
#include "kernels_fp16/04_shared_memory_swizzling.cuh"
#include "kernels_fp16/05_async_copy.cuh"
#include "kernels_fp16/06_tuning_tile_dimensions.cuh"