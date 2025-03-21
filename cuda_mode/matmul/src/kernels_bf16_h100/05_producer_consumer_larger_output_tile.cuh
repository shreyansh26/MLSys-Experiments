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

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE>
__global__ void __launch_bounds__(NUM_THREADS) producer_consumer_larger_output_tile(int M, int N, int K, bf16* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB, float alpha, float beta, int *DB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1; // number of consumer warp groups
    constexpr int B_WG_M = BM / num_consumers;

    // extern __shared__  SMemQueue<BM, BN, BK, QSIZE> smem_queue;
    // bf16 *sA = smem_queue.A;
    // bf16 *sB = smem_queue.B;

    extern __shared__ __align__(128) uint8_t smem_queue[];
    SMemQueue<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMemQueue<BM, BN, BK, QSIZE>*>(smem_queue);
    bf16 *sA = s.A;
    bf16 *sB = s.B;

    const int num_blocks_k = K / BK;
    int num_block_n = blockIdx.x % (N / BN);
    int num_block_m = blockIdx.x / (N / BN);
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier full[QSIZE];
    __shared__ barrier empty[QSIZE];

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            init(&full[i], num_consumers * 128 + 1); // all consumer + 1 producer
            init(&empty[i], num_consumers * 128 + 1); // all consumer + 1 producer
        }
        cde::fence_proxy_async_shared_cta(); // barrier and memory initialization visible to all warps
    }
    __syncthreads();

    if(wg_idx == 0) { // producer
        /*
        “full” barrier signals that data has been loaded by the producer
        “empty” barrier signals that the consumer has used the data and the slot is available
        */
        constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
        warpgroup_reg_dealloc<num_regs>();
        
        if(tid == 0) {
            int qidx = 0;
            for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                if(qidx == QSIZE) // ring buffer
                    qidx = 0;
                empty[qidx].wait(empty[qidx].arrive());
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[qidx*BM*BK], tensorMapA, block_k_iter*BK, num_block_m*BM, full[qidx]); // col offset, row offset -> sA is row major
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[qidx*BK*BN], tensorMapB, block_k_iter*BK, num_block_n*BN, full[qidx]); // col offset, row offset -> sB is row major
                barrier::arrival_token _ = cuda::device::barrier_arrive_tx(full[qidx], 1, (BK*BN+BK*BM)*sizeof(bf16));
            }
        }
    }
    else {
        constexpr int num_regs = (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
        warpgroup_reg_alloc<num_regs>();
        --wg_idx;

        for(int i = 0; i < QSIZE; ++i) {
            barrier::arrival_token _ = empty[i].arrive();
        }

        float d[B_WG_M/WGMMA_M][WGMMA_N/16][8];
        memset(d, 0, sizeof(d));
        int qidx = 0;
        for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
            if(qidx == QSIZE) 
                qidx = 0;
            full[qidx].wait(full[qidx].arrive());
            warpgroup_arrive();
            #pragma unroll
            for(int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
                bf16 *wgmma_sA = sA + qidx*BK*BM + (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M * BK;
                #pragma unroll 
                for (int k_it = 0; k_it < BK/WGMMA_K; ++k_it) {
                    wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K], &sB[qidx*BK*BN + k_it*WGMMA_K]);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            barrier::arrival_token _ = empty[qidx].arrive();
        }

        // Store
        {
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
        }
    }
}