#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>
#include <cassert>

// Tile size for shared memory
#define TILE_SIZE 32
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

// Main attention forward kernel
// Each block handles one (batch, head) pair
// Q, K, V: [B, H, N, D]
// Output: [B, H, N, D]
template<typename T>
__global__ void attention_forward_kernel(
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
    
    if (batch_idx >= B || head_idx >= H) return;
    
    const int offset = (batch_idx * H + head_idx) * N * D;
    const T* Q_bh = Q + offset;
    const T* K_bh = K + offset;
    const T* V_bh = V + offset;
    T* O_bh = output + offset;
    
    extern __shared__ float shared_mem[];
    float* s_scores = shared_mem; // N elements per row
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // For each query position
    for (int q_idx = 0; q_idx < N; q_idx++) {
        // Step 1: Compute attention scores QK^T for this query
        // Parallelized across threads
        for (int k_idx = tid; k_idx < N; k_idx += num_threads) {
            float score = 0.0f;
            
            // Dot product Q[q_idx] · K[k_idx]
            for (int d = 0; d < D; d++) {
                float q_val = to_float(Q_bh[q_idx * D + d]);
                float k_val = to_float(K_bh[k_idx * D + d]);
                score += q_val * k_val;
            }
            
            score *= scale;
            
            // Apply causal mask
            if (is_causal && k_idx > q_idx) {
                score = -INFINITY;
            }
            
            s_scores[k_idx] = score;
        }
        __syncthreads();
        
        // Step 2: Compute softmax over scores
        // Find max for numerical stability
        float local_max = -INFINITY;
        for (int k_idx = tid; k_idx < N; k_idx += num_threads) {
            local_max = fmaxf(local_max, s_scores[k_idx]);
        }
        
        // Reduce max across block using optimized two-level reduction
        __shared__ float shared_max[32]; // Max 32 warps per block
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
        
        // Level 1: Warp-level reduction
        local_max = warp_reduce_max(local_max);
        
        // Level 2: Each warp leader writes result
        if (lane_id == 0) {
            shared_max[warp_id] = local_max;
        }
        __syncthreads();
        
        // Level 3: First warp does final reduction
        float global_max;
        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? shared_max[lane_id] : -INFINITY;
            val = warp_reduce_max(val);
            global_max = val;
        }
        __syncthreads();
        
        // Broadcast result from first warp to all threads
        if (warp_id == 0 && lane_id == 0) {
            shared_max[0] = global_max;
        }
        __syncthreads();
        global_max = shared_max[0];
        
        // Compute exp(score - max) and sum
        float local_sum = 0.0f;
        for (int k_idx = tid; k_idx < N; k_idx += num_threads) {
            float exp_score = expf(s_scores[k_idx] - global_max);
            s_scores[k_idx] = exp_score;
            local_sum += exp_score;
        }
        
        // Reduce sum across block using optimized two-level reduction
        __shared__ float shared_sum[32];
        
        // Level 1: Warp-level reduction
        local_sum = warp_reduce_sum(local_sum);
        
        // Level 2: Each warp leader writes result
        if (lane_id == 0) {
            shared_sum[warp_id] = local_sum;
        }
        __syncthreads();
        
        // Level 3: First warp does final reduction
        float global_sum;
        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
            val = warp_reduce_sum(val);
            global_sum = val;
        }
        __syncthreads();
        
        // Broadcast result from first warp to all threads
        if (warp_id == 0 && lane_id == 0) {
            shared_sum[0] = global_sum;
        }
        __syncthreads();
        global_sum = shared_sum[0];
        
        // Normalize to get attention weights
        for (int k_idx = tid; k_idx < N; k_idx += num_threads) {
            s_scores[k_idx] /= global_sum;
        }
        __syncthreads();
        
        // Step 3: Multiply attention weights by V
        for (int d = tid; d < D; d += num_threads) {
            float output_val = 0.0f;
            
            for (int k_idx = 0; k_idx < N; k_idx++) {
                float attn_weight = s_scores[k_idx];
                float v_val = to_float(V_bh[k_idx * D + d]);
                output_val += attn_weight * v_val;
            }
            
            O_bh[q_idx * D + d] = from_float<T>(output_val);
        }
        __syncthreads();
    }
}

// Host function to launch kernel
template<typename T>
void launch_attention_kernel(
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
    
    // Launch configuration: each block handles one (batch, head) pair
    dim3 grid(B, H);
    int num_threads = 256; // Can be tuned
    
    // Shared memory size: N floats for scores
    size_t shared_mem_size = N * sizeof(float);
    
    attention_forward_kernel<T><<<grid, num_threads, shared_mem_size>>>(
        Q, K, V, output, B, H, N, D, is_causal, scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

// Explicit template instantiations
template void launch_attention_kernel<float>(
    const float*, const float*, const float*, float*, int, int, int, int, bool
);

template void launch_attention_kernel<__half>(
    const __half*, const __half*, const __half*, __half*, int, int, int, int, bool
);

template void launch_attention_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, bool
);

