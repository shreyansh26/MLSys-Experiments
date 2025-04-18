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

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM, int CLUSTER_M, int CLUSTER_N>
__global__ __launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1) thread_block_cluster(int M, int N, int K, bf16* C, const __grid_constant__ CUtensorMap tensorMapA, const __grid_constant__ CUtensorMap tensorMapB, float alpha, float beta, int *DB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1; // number of consumer warp groups
    constexpr int B_WG_M = BM / num_consumers;
    constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;
    assert((M / BM) % CLUSTER_M == 0);
    assert((N / BN) % CLUSTER_N == 0);

    extern __shared__ __align__(128) uint8_t smem_queue[];
    SMemQueue<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMemQueue<BM, BN, BK, QSIZE>*>(smem_queue);
    bf16 *sA = s.A;
    bf16 *sB = s.B;

    const int num_blocks_k = K / BK;
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    __shared__ __align__(8) uint64_t full[QSIZE], empty[QSIZE];
    uint32_t rank;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(rank) :);

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            init_barrier(&full[i], 0, 1); // all consumer + 1 producer
            init_barrier(&empty[i], 0, num_consumers * CLUSTERS); // all consumer + 1 producer
        }
    }
    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    ScheduleTogether<1, NUM_SM/CLUSTERS, BM*CLUSTER_M, BN*CLUSTER_N, 16/CLUSTER_M, 8/CLUSTER_N> schedule(M, N, rank);

    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
    uint32_t rank_m = rank / CLUSTER_N;
    uint32_t rank_n = rank % CLUSTER_N;

    if(wg_idx == 0) { // producer
        /*
        “full” barrier signals that data has been loaded by the producer
        “empty” barrier signals that the consumer has used the data and the slot is available
        */
        constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
        warpgroup_reg_dealloc<num_regs>();
        
        if(tid == 0) {
            int p = 0;
            int qidx = 0;
            uint32_t col_mask = 0;
            for (int i = 0; i < CLUSTER_M; ++i) {
                col_mask |= (1 << (i * CLUSTER_N));
            }
            int num_block_m, num_block_n;
            while(schedule.next(num_block_m, num_block_n)) {
                num_block_m = num_block_m * CLUSTER_M + rank_m;
                num_block_n = num_block_n * CLUSTER_N + rank_n;
                for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                    if(qidx == QSIZE) { // ring buffer
                        qidx = 0;
                        p ^= 1;
                    }
                    wait(&empty[qidx], p);
                    expect_bytes(&full[qidx], (BK*BN+BK*BM)*sizeof(bf16));

                    if constexpr (CLUSTER_N > 1) {
                        uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
                        if (rank_n == 0) {
                            load_async_multicast(&sA[qidx*BK*BM], &tensorMapA, &full[qidx], block_k_iter*BK, num_block_m*BM, mask);
                        }
                    }
                    else {
                        load_async(&sA[qidx*BK*BM], &tensorMapA, &full[qidx], block_k_iter*BK, num_block_m*BM);
                    }

                    if constexpr (CLUSTER_M > 1) {
                        if (rank_m == 0) {
                            load_async_multicast(&sB[qidx*BK*BN], &tensorMapB, &full[qidx], block_k_iter*BK, num_block_n*BN, col_mask << rank_n);
                        }
                    }
                    else {
                        load_async(&sB[qidx*BK*BN], &tensorMapB, &full[qidx], block_k_iter*BK, num_block_n*BN);
                    }
                }
            }
        }
    }
    else {
        constexpr int num_regs = (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
        warpgroup_reg_alloc<num_regs>();
        --wg_idx;

        for(int i = 0; i < QSIZE; ++i) {
            if(tid < CLUSTERS) {
                arrive_cluster(&empty[i], tid);
            }
        }

        int p = 0;
        int qidx = 0;
        int num_block_m, num_block_n;

        float d[B_WG_M/WGMMA_M][WGMMA_N/16][8];

        while(schedule.next(num_block_m, num_block_n)) {
            num_block_m = num_block_m * CLUSTER_M + rank_m;
            num_block_n = num_block_n * CLUSTER_N + rank_n;
            memset(d, 0, sizeof(d));

            for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                if(qidx == QSIZE) {
                    qidx = 0;
                    p ^= 1;
                }
                wait(&full[qidx], p);
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
                if(tid < CLUSTERS) {
                    arrive_cluster(&empty[qidx], tid);
                }
            }

            // Store
            int tid = threadIdx.x % 128;
            int lane = tid % 32;
            int warp = tid / 32;
            uint32_t row = warp*16 + lane / 4;
            
            bf16 *block_C = C + num_block_n*BN*M + num_block_m*BM;

            #pragma unroll
            for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
                int offset_m = m_it * WGMMA_M + wg_idx * B_WG_M;
                if(row + 8 + offset_m + num_block_m*BM >= M) {
                    continue;
                }
                #pragma unroll
                for (int w = 0; w < WGMMA_N/16; ++w) {
                    if (w*16 + num_block_n*BN < N) {
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
}