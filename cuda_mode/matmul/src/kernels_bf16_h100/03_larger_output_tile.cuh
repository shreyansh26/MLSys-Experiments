#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include "hopper_utils.cuh"
#include "data_utils.cuh"

typedef __nv_bfloat16 bf16;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

template<int BM, int BN, int BK, int NUM_THREADS, bool DBG>
__global__ void __launch_bounds__(NUM_THREADS) larger_output_tile(int M, int N, int K, bf16* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB, float alpha, float beta, int *DB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int B_WG_M = BM / (NUM_THREADS / 128);

    extern __shared__  SMem<BM, BN, BK> smem;
    bf16 *sA = smem.A;
    bf16 *sB = smem.B;

    float d[B_WG_M/WGMMA_M][WGMMA_N/16][8];
    static_assert(sizeof(d) * NUM_THREADS == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    const int num_blocks_k = K / BK;
    int num_block_n = blockIdx.x % (N / BN);
    int num_block_m = blockIdx.x / (N / BN);
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA;
    __shared__ barrier barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    int wg_idx = threadIdx.x / 128;

    int sumLoad = 0;
    int countLoad = 0;
    int sumCompute = 0;
    int countCompute = 0;
    int sumStore = 0;
    int countStore = 0;

    barrier::arrival_token tokenA, tokenB;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
        clock_t start = clock();
        // Load
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter*BK, num_block_m*BM, barA);
            // tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, BK*BM*sizeof(bf16));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter*BK, num_block_n*BN, barB);
            // tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, BK*BN*sizeof(bf16));
        } 
        else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        if constexpr (DBG) {
            sumLoad += clock() - start;
            countLoad++;
            start = clock();
        }
    
        // Compute
        warpgroup_arrive();
        #pragma unroll
        for(int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
            bf16 *wgmma_sA = sA + (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M * BK;
            #pragma unroll 
            for (int k_it = 0; k_it < BK/WGMMA_K; ++k_it) {
                wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K], &sB[k_it * WGMMA_K]);
                if constexpr (DBG) {
                    countCompute++;
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        if constexpr (DBG) {
            sumCompute += clock() - start;
            // countCompute++;
        }
    }

    // Store
    {
        clock_t start = clock();
        int tid = threadIdx.x % 128;
        int lane = tid % 32;
        int warp = tid / 32;
        uint32_t row = warp*16 + lane / 4;
        
        bf16 *block_C = C + num_block_n*BN*M + num_block_m*BM;

        #pragma unroll
        for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
            int offset_m = m_it * WGMMA_M + wg_idx * B_WG_M;
            #pragma unroll
            for (int w = 0; w < WGMMA_N/16; ++w) {
                int col = 16*w + 2*(tid % 4);
                #define IDX(i, j) ((j)*M + ((i) + offset_m))

                block_C[IDX(row, col)] = d[m_it][w][0];
                block_C[IDX(row, col+1)] = d[m_it][w][1];
                block_C[IDX(row+8, col)] = d[m_it][w][2];
                block_C[IDX(row+8, col+1)] = d[m_it][w][3];

                block_C[IDX(row, col+8)] = d[m_it][w][4];
                block_C[IDX(row, col+9)] = d[m_it][w][5];
                block_C[IDX(row+8, col+8)] = d[m_it][w][6];
                block_C[IDX(row+8, col+9)] = d[m_it][w][7];

                #undef IDX
            }
        }

        if constexpr (DBG) {
            sumStore += clock() - start;
            countStore++;
            if(threadIdx.x == 63) {
                int i = blockIdx.x * 6;
                DB[i] = sumLoad;
                DB[i+1] = countLoad;
                DB[i+2] = sumCompute;
                DB[i+3] = countCompute;
                DB[i+4] = sumStore;
                DB[i+5] = countStore;
            }
        }
    }
}