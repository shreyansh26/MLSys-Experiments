#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include "hopper_utils.cuh"

typedef __nv_bfloat16 bf16;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

template<int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) tensor_core_matmul_col_major(int M, int N, int K, bf16* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB, float alpha, float beta) {
    __shared__ alignas(128) bf16 sA[BM*BK];
    __shared__ alignas(128) bf16 sB[BK*BN];
    float d[WGMMA_N/16][8];
    static_assert(sizeof(d) * 128 == BM * BN * sizeof(float));
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

    barrier::arrival_token tokenA, tokenB;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
        // Load
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter*BK, num_block_m*BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter*BK, num_block_n*BN, barB);
            // cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, num_block_n*BN, block_k_iter*BK, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();
    
        // Compute
        warpgroup_arrive();
        wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[2*WGMMA_K], &sB[2*WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[3*WGMMA_K], &sB[3*WGMMA_K]);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    // Store
    {
        int tid = threadIdx.x;
        int lane = tid % 32;
        int warp = tid / 32;
        uint32_t row = warp*16 + lane / 4;
        bf16 *block_C = C + num_block_n*BN*M + num_block_m*BM;
        // bf16 *block_C = C + num_block_m*BM*N + num_block_n*BN;

        for (int m_it = 0; m_it < BM/WGMMA_M; ++m_it) {
            for (int n_it = 0; n_it < BN/WGMMA_N; ++n_it) {
                for (int w = 0; w < WGMMA_N/16; ++w) {
                    int col = 16*w + 2*(tid % 4);
                    #define IDX(i, j) ((j + n_it*WGMMA_N)*M + ((i) + m_it*WGMMA_M))
                    // #define IDX(i, j) ((i + m_it * WGMMA_M) * N + (j + n_it * WGMMA_N))

                    block_C[IDX(row, col)] = d[w][0];
                    block_C[IDX(row, col+1)] = d[w][1];
                    block_C[IDX(row+8, col)] = d[w][2];
                    block_C[IDX(row+8, col+1)] = d[w][3];
    
                    block_C[IDX(row, col+8)] = d[w][4];
                    block_C[IDX(row, col+9)] = d[w][5];
                    block_C[IDX(row+8, col+8)] = d[w][6];
                    block_C[IDX(row+8, col+9)] = d[w][7];

                    #undef IDX
                }
            }
        }
    }
}