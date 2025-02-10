#pragma once
#include <cuda.h>
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ constexpr unsigned int floor_log2(unsigned int x) {
    unsigned int result = 0;
    while (x >>= 1) {
        result++;
    }
    return result;
}

__device__ __forceinline__ void tileMemcpy(half* src, half* dst, const unsigned int src_stride, const unsigned int tile_rows, const unsigned int tile_cols) {
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int num_threads = blockDim.x * blockDim.y;
    
    // # of threads is multiple of # of columns in the tile
    assert(num_threads % tile_cols == 0);
    
    // assign each thread a row/column in the tile, calculate the column step
    const unsigned int row_step = num_threads / tile_cols;
    const unsigned int thread_row = thread_idx / tile_cols;
    const unsigned int thread_col = thread_idx % tile_cols;
    
    for(unsigned int r = thread_row; r < tile_rows; r+=row_step) {
        dst[r * tile_cols + thread_col] =  src[r * src_stride + thread_col];
    }
}

template<unsigned int TILE_ROWS, unsigned int TILE_COLS, unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpyUnrolled(half* src, half* dst, const unsigned int src_stride) {
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    // # of threads is multiple of # of columns in the tile
    static_assert(NUM_THREADS % TILE_COLS == 0);
    
    // assign each thread a row/column in the tile, calculate the column step
    constexpr unsigned int row_step = NUM_THREADS / TILE_COLS;
    unsigned int thread_row = thread_idx / TILE_COLS;
    const unsigned int thread_col = thread_idx % TILE_COLS;
    constexpr unsigned int num_iters = TILE_ROWS / row_step;
    
    #pragma unroll
    for(unsigned int iter = 0; iter < num_iters; iter++) {
        dst[(thread_row + iter * row_step) * TILE_COLS + thread_col] =  src[(thread_row + iter * row_step) * src_stride + thread_col];
    }
}

template<unsigned int TILE_ROWS, unsigned int TILE_COLS, unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpyUnrolledVectorized(half* src, half* dst, const unsigned int src_stride) {
    // reinterpret src and dst as float4
    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    const unsigned int src_stride_float4 = src_stride / 8;
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    // # of threads is multiple of # of columns in the tile
    const unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
    // assign each thread a row/column in the tile, calculate the column step
    constexpr unsigned int row_step = NUM_THREADS / TILE_COLS_VECTORIZED;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    constexpr unsigned int num_iters = TILE_ROWS / row_step;
    
    #pragma unroll
    for(unsigned int iter = 0; iter < num_iters; iter++) {
        dst_float4[(thread_row + iter * row_step) * TILE_COLS_VECTORIZED + thread_col] =  src_float4[(thread_row + iter * row_step) * src_stride_float4 + thread_col];
    }
}

template<unsigned int TILE_ROWS, unsigned int TILE_COLS, unsigned int NUM_THREADS, unsigned int SWIZZLE_BITS>
__device__ __forceinline__ void tileMemcpySwizzled(half* src, half* dst, const unsigned int src_stride) {
    const unsigned int SWIZZLE_MASK = 0b111 << SWIZZLE_BITS;
    // reinterpret src and dst as float4
    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    const unsigned int src_stride_float4 = src_stride / 8;
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    // # of threads is multiple of # of columns in the tile
    const unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
    // assign each thread a row/column in the tile, calculate the column step
    constexpr unsigned int row_step = NUM_THREADS / TILE_COLS_VECTORIZED;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    constexpr unsigned int num_iters = TILE_ROWS / row_step;
    
    #pragma unroll
    for(unsigned int iter = 0; iter < num_iters; iter++) {
        unsigned int dst_index = (thread_row + iter * row_step) * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK) >> SWIZZLE_BITS);
        unsigned int src_index = (thread_row + iter * row_step) * src_stride_float4 + thread_col;
        dst_float4[dst_index] =  src_float4[src_index];
    }
}

template<unsigned int TILE_ROWS, unsigned int TILE_COLS, unsigned int NUM_THREADS, unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpyLoad(half* src, float4 (&dst_reg)[ELEMENTS_PER_THREAD], const unsigned int src_stride) {
    // reinterpret src as float4
    float4* src_float4 = reinterpret_cast<float4*>(src);
    const unsigned int src_stride_float4 = src_stride / 8;
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    // # of threads is multiple of # of columns in the tile
    const unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
    // assign each thread a row/column in the tile, calculate the column step
    constexpr unsigned int row_step = NUM_THREADS / TILE_COLS_VECTORIZED;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    constexpr unsigned int num_iters = TILE_ROWS / row_step;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == num_iters);
    
    #pragma unroll
    for(unsigned int iter = 0; iter < num_iters; iter++) {
        unsigned int src_index = (thread_row + iter * row_step) * src_stride_float4 + thread_col;
        dst_reg[iter] =  src_float4[src_index];
    }
}

template<unsigned int TILE_ROWS, unsigned int TILE_COLS, unsigned int NUM_THREADS, unsigned int SWIZZLE_BITS, unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpySwizzledStore(float4 (&src_reg)[ELEMENTS_PER_THREAD], half* dst) {
    const unsigned int SWIZZLE_MASK = 0b111 << SWIZZLE_BITS;
    // reinterpret src as float4
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    // # of threads is multiple of # of columns in the tile
    const unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
    // assign each thread a row/column in the tile, calculate the column step
    constexpr unsigned int row_step = NUM_THREADS / TILE_COLS_VECTORIZED;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    constexpr unsigned int num_iters = TILE_ROWS / row_step;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == num_iters);
    
    #pragma unroll
    for(unsigned int iter = 0; iter < num_iters; iter++) {
        unsigned int dst_index = (thread_row + iter * row_step) * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK) >> SWIZZLE_BITS);
        dst_float4[dst_index] =  src_reg[iter];
    }
}