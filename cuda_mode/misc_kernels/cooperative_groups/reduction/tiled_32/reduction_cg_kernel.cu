#include <stdio.h>
#include <iostream>
#include <cooperative_groups.h>
#include "reduction.h"

using namespace cooperative_groups;
using namespace std;
namespace cg = cooperative_groups;

#define NUM_LOAD 4

__device__ float reduce_sum(thread_group g, float *temp, float val) { 
    int lane = g.thread_rank();
    // printf("lane: %d\n", lane); // prints values from 0 to 31

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane < i) 
            val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

__device__ float thread_sum(float *input, unsigned int n) {
    float sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n / 4; i += blockDim.x * gridDim.x) {
        float4 in = ((float4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }

    int leftover_start = (n / 4) * 4;
    if(idx == 0) {
        for(int i = leftover_start; i < n; i++) {
            sum += input[i];
        }
    }
    return sum;
}

// cuda thread synchronization
__global__ void reduction_kernel(float *g_out, float *g_in, unsigned int size) {
    extern __shared__ float s_data[];
    float sum = thread_sum(g_in, size);

    auto g = cg::this_thread_block();
    int tileIdx = g.thread_rank() / 32;
    float *t = &s_data[32 * tileIdx];

    auto tile32 = cg::tiled_partition(g, 32);
    float tile_sum = reduce_sum(tile32, t, sum);

    if(tile32.thread_rank() == 0) {
        atomicAdd(g_out, tile_sum);
    }
}

void reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads){
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel, n_threads, n_threads*sizeof(float)); // number of blocks per sm is 2
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);
    size_t shared_mem_size = n_threads * sizeof(float);
    
    reduction_kernel<<<n_blocks, n_threads, shared_mem_size>>>(g_outPtr, g_inPtr, size);
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error" << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
