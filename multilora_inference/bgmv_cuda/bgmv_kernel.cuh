#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cassert>
#include <stdexcept>

#include <cub/util_type.cuh>

namespace cg = cooperative_groups;

__host__ __device__ inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// Device-side conversion helpers for generic arithmetic over T in {float, __half, __nv_bfloat16}
template <typename U>
__device__ __forceinline__ float to_float_device(U x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float to_float_device<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float_device<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename U>
__device__ __forceinline__ U from_float_device(float x);

template <>
__device__ __forceinline__ float from_float_device<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ __half from_float_device<__half>(float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float_device<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <int F_in, int F_out, typename T>
__global__ void bgmv_shrink_kernel(T* Y,
                                   const T* X,
                                   const T* W,
                                   const int* indices,
                                   const int seqlen,
                                   const int num_layers,
                                   const int layer_idx,
                                   const int num_lora_adapters,
                                   const T scale) {
    auto block = cg::this_thread_block();

    const int b = blockIdx.x;
    const int j = blockIdx.y;

    const int b_seq = b / seqlen;

    constexpr int vec_size = 16 / sizeof(T);
    constexpr size_t tx = 32; // threadIdx.x is also 32
    constexpr size_t ty = 4; // threadIdx.y is also 4
    constexpr size_t tile_size = tx * ty * vec_size;
    constexpr size_t num_pipeline_stages = 2;

    __shared__ T W_shared[num_pipeline_stages * tile_size];
    __shared__ T X_shared[num_pipeline_stages * tile_size];

    __shared__ float y_warpwise[ty];

    if (indices[b_seq] == num_lora_adapters) {
        return;
    }

    const int idx = indices[b_seq] * num_layers + layer_idx;

    size_t W_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
    size_t X_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};

    auto pipe = cuda::make_pipeline();

    // Pipeline the load of W and X with computation of WX
    pipe.producer_acquire();
    // tile_idx is 0 here so not included in the load indexing
    cuda::memcpy_async(W_shared + (threadIdx.y * tx + threadIdx.x) * vec_size, 
                       W + (idx * F_out + j) * F_in + (threadIdx.y * tx + threadIdx.x) * vec_size,
                       cuda::aligned_size_t<16>(16), 
                       pipe);
    cuda::memcpy_async(X_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                       X + b * F_in + (threadIdx.y * tx + threadIdx.x) * vec_size,
                       cuda::aligned_size_t<16>(16),
                       pipe);
    pipe.producer_commit();

    size_t copy_buffer_idx, compute_idx;

    float y = 0.f;
    size_t tile_idx;

#pragma unroll
    for(tile_idx = 1; tile_idx < cdiv(F_in, tile_size); tile_idx++) {
        copy_buffer_idx = tile_idx % num_pipeline_stages;
        // pipeline stage: async copy W and X fragment
        pipe.producer_acquire();

        if(tile_idx * tile_size + (threadIdx.y * tx) * vec_size < F_in) {
            cuda::memcpy_async(W_shared + W_shared_offset[copy_buffer_idx] + (threadIdx.y * tx + threadIdx.x) * vec_size, 
                        W + (idx * F_out + j) * F_in + tile_idx * tile_size + (threadIdx.y * tx + threadIdx.x) * vec_size,
                        cuda::aligned_size_t<16>(16), 
                        pipe);
            cuda::memcpy_async(X_shared + X_shared_offset[copy_buffer_idx] + (threadIdx.y * tx + threadIdx.x) * vec_size,
                        X + b * F_in + tile_idx * tile_size + (threadIdx.y * tx + threadIdx.x) * vec_size,
                        cuda::aligned_size_t<16>(16),
                        pipe);
        }
        pipe.producer_commit();

        compute_idx = (tile_idx - 1) % num_pipeline_stages;
        // pipeline stage: compute WX
        pipe.consumer_wait();
        block.sync();

        const T* x_ptr = X_shared + X_shared_offset[compute_idx] + (threadIdx.y * tx + threadIdx.x) * vec_size;
        const T* w_ptr = W_shared + W_shared_offset[compute_idx] + (threadIdx.y * tx + threadIdx.x) * vec_size;

        float sum = 0.f;

#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            sum += to_float_device<T>(w_ptr[i]) * to_float_device<T>(x_ptr[i]) * to_float_device<T>(scale);
        }

#pragma unroll
        // intra-warp reduction
        for(size_t offset = tx / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        // stores warp wise sum (one lane per warp)
        if (threadIdx.x == 0) {
            y_warpwise[threadIdx.y] = sum;
        }
        block.sync();

#pragma unroll
        for(size_t i = 0; i < ty; i++) {
            y += y_warpwise[i];
        }

        block.sync();
        pipe.consumer_release();
    }

    // Last stage is only compute WX
    compute_idx = (tile_idx - 1) % num_pipeline_stages;
    // pipeline stage: compute WX
    pipe.consumer_wait();
    block.sync();

    const T* x_ptr = X_shared + X_shared_offset[compute_idx] + (threadIdx.y * tx + threadIdx.x) * vec_size;
    const T* w_ptr = W_shared + W_shared_offset[compute_idx] + (threadIdx.y * tx + threadIdx.x) * vec_size;

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
        sum += to_float_device<T>(w_ptr[i]) * to_float_device<T>(x_ptr[i]) * to_float_device<T>(scale);
    }

#pragma unroll
    // intra-warp reduction
    for(size_t offset = tx / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // stores warp wise sum - but check bounds since it might be out of bounds
    if (threadIdx.x == 0) {
        if ((tile_idx - 1) * tile_size + (threadIdx.y * tx) * vec_size < F_in) {
            y_warpwise[threadIdx.y] = sum;
        } else {
            y_warpwise[threadIdx.y] = 0.f;
        }
    }
    block.sync();

#pragma unroll
    for(size_t i = 0; i < ty; i++) {
        y += y_warpwise[i];
    }

    block.sync();
    pipe.consumer_release();

    // write Y
    if(block.thread_rank() == 0) {
        // Read-modify-write with conversion to preserve accumulation in T
        float prev = to_float_device<T>(Y[b * F_out + j]);
        prev += y;
        Y[b * F_out + j] = from_float_device<T>(prev);
    }   
}

template <int F_in, int F_out, typename T>
__global__ void bgmv_expand_kernel(T* Y,
                                   const T* X,
                                   const T* W,
                                   const int* indices,
                                   const int seqlen,
                                   const int num_layers,
                                   const int layer_idx,
                                   const int num_lora_adapters,
                                   const T scale) {
    auto block = cg::this_thread_block();
    const int b = blockIdx.x;
    const int tile_idx = blockIdx.y;

    const int b_seq = b / seqlen;

    constexpr int vec_size = 16 / sizeof(T);
    static_assert(F_in % vec_size == 0);
    constexpr int tx = F_in / vec_size;
    static_assert(32 % tx == 0);
    constexpr int ty = 32 / tx;
    constexpr int tz = 4;

    if (indices[b_seq] == num_lora_adapters) {
        return;
    }

    const int idx = indices[b_seq] * num_layers + layer_idx;
    
    // load X
    const T* x_ptr = X + b * F_in + threadIdx.x * vec_size;

    const T* w_ptr = W + ((idx * F_out) + (tile_idx * tz * ty) + (threadIdx.z * ty) + threadIdx.y) * F_in + threadIdx.x * vec_size;
    // const T* w_ptr = W + (idx * F_out + tile_idx * tz * ty) * F_in + block.thread_rank() * vec_size;

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
        sum += to_float_device<T>(w_ptr[i]) * to_float_device<T>(x_ptr[i]) * to_float_device<T>(scale);
    }
    
    // tiled_partition<tx> forms disjoint groups of size tx inside the block; each group corresponds to one output row being computed
    // Within that group, do a warp-shuffle reduction producing the full dot product for that output row
    cg::thread_block_tile g = cg::tiled_partition<tx>(block);

#pragma unroll
    // intra-tile reduction over tx threads: sum partial dot-products along F_in
    for(size_t offset = tx / 2; offset > 0; offset /= 2) {
        sum += g.shfl_down(sum, offset);
    }
    // move the reduced sum to lane 0 within the tile (tile leader)
    sum = g.shfl(sum, 0);

    if (threadIdx.x == 0) {
        const int out_idx = b * F_out + (tile_idx * tz * ty) + (threadIdx.z * ty) + threadIdx.y;
        const float prev = to_float_device<T>(Y[out_idx]);
        Y[out_idx] = from_float_device<T>(prev + sum);
    }
}

template <int F_in, int F_out, typename T>
void bgmv_kernel(T* Y,
                 const T* X,
                 const T* W,
                 const int* indices,
                 const int seqlen,
                 const int num_layers,
                 const int layer_idx,
                 const int num_lora_adapters,
                 const T scale,
                 const int batch_size) {
    if constexpr (F_in < F_out) {
        constexpr int vec_size = 16 / sizeof(T);
        int tx = F_in / vec_size;
        int ty = 32 / tx;
        int tz = 4;
        dim3 grid(batch_size, F_out / (tz * ty));
        dim3 block(tx, ty, tz);
        bgmv_expand_kernel<F_in, F_out, T><<<grid, block>>>(Y, X, W, indices, seqlen, num_layers, layer_idx, num_lora_adapters, scale);
    }
    else {
        constexpr int vec_size = 16 / sizeof(T);
        assert(F_in % (vec_size * 32) == 0);
        dim3 grid(batch_size, F_out);
        dim3 block(32, 4);
        bgmv_shrink_kernel<F_in, F_out, T><<<grid, block>>>(Y, X, W, indices, seqlen, num_layers, layer_idx, num_lora_adapters, scale);
    }
}