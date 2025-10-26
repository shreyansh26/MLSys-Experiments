#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>
#include <cassert>

// Tile sizes for Flash Attention
#define Br 32  // Query block size
#define Bc 32  // Key/Value block size
#define WARP_SIZE 32

// Helper function for warp-level reduction (max)
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Helper function for warp-level reduction (sum)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Type conversion helpers
template<typename T>
__device__ __forceinline__ float to_float(T val);

template<>
__device__ __forceinline__ float to_float<float>(float val) {
    return val;
}

template<>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val);

template<>
__device__ __forceinline__ float from_float<float>(float val) {
    return val;
}

template<>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

// Flash Attention forward kernel
// Each block processes one query tile across all KV tiles
// Grid: (B, H, num_query_tiles)
// Shared memory: Q_tile, K_tile, V_tile, S_tile
// Registers: Running O, m, l per thread (for assigned queries)
template<typename T>
__global__ void flash_attention_forward_kernel(
    const T* Q,
    const T* K,
    const T* V,
    T* output,
    int B,
    int H,
    int N,
    int D,
    bool is_causal,
    float scale
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int query_tile_idx = blockIdx.z;
    
    if (batch_idx >= B || head_idx >= H) return;
    
    const int num_query_tiles = (N + Br - 1) / Br;
    if (query_tile_idx >= num_query_tiles) return;
    
    // Base pointer for this (batch, head) pair
    const int offset = (batch_idx * H + head_idx) * N * D;
    const T* Q_bh = Q + offset;
    const T* K_bh = K + offset;
    const T* V_bh = V + offset;
    T* O_bh = output + offset;
    
    // Shared memory allocation
    extern __shared__ float smem[];
    float* Q_tile = smem;                              // Br × D
    float* K_tile = Q_tile + Br * D;                   // Bc × D
    float* V_tile = K_tile + Bc * D;                   // Bc × D
    float* S_tile = V_tile + Bc * D;                   // Br × Bc
    float* reduction_buffer = S_tile + Br * Bc;        // 32 floats
    
    // Thread indices
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Query range for this tile
    const int q_start = query_tile_idx * Br;
    const int q_end = min(q_start + Br, N);
    const int num_queries_in_tile = q_end - q_start;
    
    // Number of KV tiles
    const int num_kv_tiles = (N + Bc - 1) / Bc;
    
    // Shared memory for running statistics (shared across all threads)
    float* O_acc = reduction_buffer + 32;              // Br × D
    float* m_shared = O_acc + Br * D;                  // Br
    float* l_shared = m_shared + Br;                   // Br
    
    // Initialize running statistics in shared memory
    for (int i = tid; i < Br; i += num_threads) {
        m_shared[i] = -INFINITY;
        l_shared[i] = 0.0f;
    }
    for (int i = tid; i < Br * D; i += num_threads) {
        O_acc[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q_tile once (reused across all KV tiles)
    for (int i = tid; i < Br * D; i += num_threads) {
        int q_local = i / D;
        int d = i % D;
        int q_idx = q_start + q_local;
        
        if (q_idx < q_end) {
            Q_tile[q_local * D + d] = to_float(Q_bh[q_idx * D + d]);
        } else {
            Q_tile[q_local * D + d] = 0.0f;
        }
    }
    __syncthreads();
    
    // Loop over KV tiles
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        const int k_start = kv_tile_idx * Bc;
        const int k_end = min(k_start + Bc, N);
        const int num_keys_in_tile = k_end - k_start;
        
        // Load K_tile and V_tile
        for (int i = tid; i < Bc * D; i += num_threads) {
            int k_local = i / D;
            int d = i % D;
            int k_idx = k_start + k_local;
            
            if (k_idx < k_end) {
                K_tile[k_local * D + d] = to_float(K_bh[k_idx * D + d]);
                V_tile[k_local * D + d] = to_float(V_bh[k_idx * D + d]);
            } else {
                K_tile[k_local * D + d] = 0.0f;
                V_tile[k_local * D + d] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute S = Q @ K^T for this tile (Br × Bc)
        for (int i = tid; i < Br * Bc; i += num_threads) {
            int q_local = i / Bc;
            int k_local = i % Bc;
            int q_idx = q_start + q_local;
            int k_idx = k_start + k_local;
            
            if (q_idx < q_end && k_idx < k_end) {
                float score = 0.0f;
                for (int d = 0; d < D; d++) {
                    score += Q_tile[q_local * D + d] * K_tile[k_local * D + d];
                }
                score *= scale;
                
                // Apply causal mask
                if (is_causal && k_idx > q_idx) {
                    score = -INFINITY;
                }
                
                S_tile[q_local * Bc + k_local] = score;
            } else {
                S_tile[q_local * Bc + k_local] = -INFINITY;
            }
        }
        __syncthreads();
        
        // For each query in the tile, perform online softmax update
        for (int q_local = 0; q_local < num_queries_in_tile; q_local++) {
            int q_idx = q_start + q_local;
            if (q_idx >= N) continue;
            
            // Find max score for this query across current KV tile
            float local_max = -INFINITY;
            for (int k_local = tid; k_local < num_keys_in_tile; k_local += num_threads) {
                local_max = fmaxf(local_max, S_tile[q_local * Bc + k_local]);
            }
            
            // Reduce max across threads
            int warp_id = tid / WARP_SIZE;
            int lane_id = tid % WARP_SIZE;
            int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
            
            local_max = warp_reduce_max(local_max);
            
            if (lane_id == 0) {
                reduction_buffer[warp_id] = local_max;
            }
            __syncthreads();
            
            float tile_max = -INFINITY;
            if (warp_id == 0) {
                float val = (lane_id < num_warps) ? reduction_buffer[lane_id] : -INFINITY;
                val = warp_reduce_max(val);
                if (lane_id == 0) {
                    reduction_buffer[0] = val;
                }
            }
            __syncthreads();
            tile_max = reduction_buffer[0];
            
            // Compute m_new and rescaling factor
            float m_old_val = m_shared[q_local];
            float m_new = fmaxf(m_old_val, tile_max);
            float rescale_factor = expf(m_old_val - m_new);
            
            // Rescale previous output and sum (parallelized)
            for (int d = tid; d < D; d += num_threads) {
                O_acc[q_local * D + d] *= rescale_factor;
            }
            if (tid == 0) {
                l_shared[q_local] *= rescale_factor;
                m_shared[q_local] = m_new;
            }
            __syncthreads();
            
            // Compute P = exp(S - m_new) and sum
            float local_sum = 0.0f;
            for (int k_local = tid; k_local < num_keys_in_tile; k_local += num_threads) {
                float p = expf(S_tile[q_local * Bc + k_local] - m_new);
                S_tile[q_local * Bc + k_local] = p;  // Store P in S_tile
                local_sum += p;
            }
            
            // Reduce sum across threads
            local_sum = warp_reduce_sum(local_sum);
            
            if (lane_id == 0) {
                reduction_buffer[warp_id] = local_sum;
            }
            __syncthreads();
            
            float tile_sum = 0.0f;
            if (warp_id == 0) {
                float val = (lane_id < num_warps) ? reduction_buffer[lane_id] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane_id == 0) {
                    reduction_buffer[0] = val;
                }
            }
            __syncthreads();
            tile_sum = reduction_buffer[0];
            
            // Update running sum
            if (tid == 0) {
                l_shared[q_local] += tile_sum;
            }
            __syncthreads();
            
            // Accumulate O += P @ V (parallelized across D)
            for (int d = tid; d < D; d += num_threads) {
                float acc = 0.0f;
                for (int k_local = 0; k_local < num_keys_in_tile; k_local++) {
                    acc += S_tile[q_local * Bc + k_local] * V_tile[k_local * D + d];
                }
                O_acc[q_local * D + d] += acc;
            }
            __syncthreads();
        }
    }
    
    // Final normalization and write output (parallelized)
    for (int i = tid; i < num_queries_in_tile * D; i += num_threads) {
        int q_local = i / D;
        int d = i % D;
        int q_idx = q_start + q_local;
        
        if (q_idx < N) {
            float val = O_acc[q_local * D + d] / l_shared[q_local];
            O_bh[q_idx * D + d] = from_float<T>(val);
        }
    }
}

// Host function to launch kernel
template<typename T>
void launch_flash_attention_kernel(
    const T* Q,
    const T* K,
    const T* V,
    T* output,
    int B,
    int H,
    int N,
    int D,
    bool is_causal
) {
    // Scale factor
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    
    // Calculate number of query tiles
    int num_query_tiles = (N + Br - 1) / Br;
    
    // Launch configuration
    dim3 grid(B, H, num_query_tiles);
    int num_threads = 256;
    
    // Shared memory size
    size_t shared_mem_size = 0;
    shared_mem_size += Br * D * sizeof(float);      // Q_tile
    shared_mem_size += Bc * D * sizeof(float);      // K_tile
    shared_mem_size += Bc * D * sizeof(float);      // V_tile
    shared_mem_size += Br * Bc * sizeof(float);     // S_tile
    shared_mem_size += 32 * sizeof(float);          // reduction_buffer
    shared_mem_size += Br * D * sizeof(float);      // O_acc
    shared_mem_size += Br * sizeof(float);          // m_shared
    shared_mem_size += Br * sizeof(float);          // l_shared
    
    // Check shared memory limits and opt-in if needed
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    size_t max_shared_mem = deviceProp.sharedMemPerBlockOptin > 0 ? 
                            deviceProp.sharedMemPerBlockOptin : 
                            deviceProp.sharedMemPerBlock;
    
    if (shared_mem_size > max_shared_mem) {
        std::cerr << "Error: Required shared memory (" << shared_mem_size 
                  << " bytes) exceeds device limit (" << max_shared_mem 
                  << " bytes)" << std::endl;
        return;
    }
    
    // Request more shared memory if needed
    if (shared_mem_size > deviceProp.sharedMemPerBlock) {
        cudaFuncSetAttribute(
            flash_attention_forward_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size
        );
    }
    
    flash_attention_forward_kernel<T><<<grid, num_threads, shared_mem_size>>>(
        Q, K, V, output, B, H, N, D, is_causal, scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

// Explicit template instantiations
template void launch_flash_attention_kernel<float>(
    const float*, const float*, const float*, float*, int, int, int, int, bool
);

template void launch_flash_attention_kernel<__half>(
    const __half*, const __half*, const __half*, __half*, int, int, int, int, bool
);

template void launch_flash_attention_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, bool
);

