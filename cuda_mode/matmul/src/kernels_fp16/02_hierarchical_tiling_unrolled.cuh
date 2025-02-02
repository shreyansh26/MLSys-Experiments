#pragma once
#include <cuda.h>
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "memcpy_utils.cuh"
#include "load_store_utils.cuh"

template <unsigned int BM_dim, unsigned int BN_dim, unsigned int BK_dim, unsigned int WM_dim, unsigned int WN_dim, unsigned int WK_dim, unsigned int NUM_THREADS>
__global__ void hierarchical_tiling_unrolled(half* A, half* B, half* C, half* D, const float alpha, const float beta, const unsigned int M, const unsigned int N, unsigned int K) {
    constexpr unsigned int MMA_M_dim = 16;
    constexpr unsigned int MMA_N_dim = 8;
    constexpr unsigned int MMA_K_dim = 8;

    // for convenience/readability in index calculations
    const unsigned int A_stride = K;
    const unsigned int B_stride = N;
    const unsigned int CD_stride = N;

    // loop bounds, constexpr where possible allows for loop unrolling
    constexpr unsigned int mma_tiles_per_warp_k = WK_dim / MMA_K_dim;
    constexpr unsigned int mma_tiles_per_warp_m = WM_dim / MMA_M_dim;
    constexpr unsigned int mma_tiles_per_warp_n = WN_dim / MMA_N_dim;
    constexpr unsigned int warp_tiles_per_block_k = BK_dim / WK_dim;
    const unsigned int num_block_tiles_k = K / BK_dim;
  
    // calculate block/warp indices
    const unsigned int block_m = blockIdx.y;
    const unsigned int block_n = blockIdx.x;
    const unsigned int warp_m = threadIdx.y;
    const unsigned int warp_n = threadIdx.x / 32;
    
    extern __shared__ half shmem[];
    half* A_block_smem = shmem;
    // offset by BM_dim * BK_dim as A_block_smem is of shape [BM_dim, BK_dim]
    // total size of shared memory is (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(half)
    // so B_block_smem is of shape [BK_dim, BN_dim]
    half* B_block_smem = &shmem[BM_dim * BK_dim];

    // declare register storage
    // ptx instructions expect uint32_t registers, where each uint32_t is 2 halfs packed together  
    // the tiles are of shape [MMA_M_dim, MMA_N_dim]
    // and the tensor core matmul is done on tiles of shape [MMA_M_dim, MMA_K_dim] and [MMA_K_dim, MMA_N_dim] => [MMA_M_dim, MMA_N_dim]
    uint32_t acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];
    
    // convenience cast to half for accumulator registers
    half (&acc_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_n][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4]>(acc_register);

    uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2];
    uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n];

    // accumulators start at 0
    for(unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++) {
        for(unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++) {
            acc_register_[mma_m][mma_n][0] = 0;
            acc_register_[mma_m][mma_n][1] = 0;
            acc_register_[mma_m][mma_n][2] = 0;
            acc_register_[mma_m][mma_n][3] = 0;
        }
    }

    for(unsigned int block_k = 0; block_k < num_block_tiles_k; block_k++) {
        half* A_block_gmem = A + (block_m * BM_dim * A_stride) + (block_k * BK_dim);
        half* B_block_gmem = B + (block_k * BK_dim * B_stride) + (block_n * BN_dim);
        tileMemcpyUnrolled<BM_dim, BK_dim, NUM_THREADS>(A_block_gmem, A_block_smem, K);
        tileMemcpyUnrolled<BK_dim, BN_dim, NUM_THREADS>(B_block_gmem, B_block_smem, N);
        __syncthreads();


        for(unsigned int warp_k = 0; warp_k < warp_tiles_per_block_k; warp_k++) {
            // preload tiles of a into registers
            half* A_warp_tile = A_block_smem + (warp_m * WM_dim * BK_dim) + (warp_k * WK_dim);
            half* B_warp_tile = B_block_smem + (warp_k * WK_dim * BN_dim) + (warp_n * WN_dim);
            uint32_t A_warp_tile_byte_offset = cvta_to_shared_u32(A_warp_tile);
            uint32_t B_warp_tile_byte_offset = cvta_to_shared_u32(B_warp_tile);

            for(unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++) {
                for(unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++) {
                    // byte offset to the top left of the mma tile
                    const unsigned int mma_tile_byte_offset = ((mma_m * MMA_M_dim * BK_dim) + (mma_k * MMA_K_dim)) * sizeof(half);
                    
                    // byte offset to the start of this thread's slice of the mma tile
                    const unsigned int thread_byte_offset = (threadIdx.x % MMA_M_dim) * BK_dim * sizeof(half);
                    
                    // calculate offset in bytes WRT to the start of our shared memory allocation
                    const unsigned int thread_offset_bytes = A_warp_tile_byte_offset + mma_tile_byte_offset + thread_byte_offset;
                    
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                        "{%0, %1}, [%2];"
                        : "=r"(A_register[mma_m][mma_k][0]), "=r"(A_register[mma_m][mma_k][1])
                        : "r"(thread_offset_bytes)
                    );
                }
            }

            // preload tiles of b into registers
            for(unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++) {
                for(unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++) {
                    const unsigned int mma_tile_byte_offset = ((mma_k * MMA_K_dim * BN_dim) + (mma_n * MMA_N_dim)) * sizeof(half);
                    const unsigned int thread_byte_offset = (threadIdx.x % MMA_K_dim) * BN_dim * sizeof(half);
                    const unsigned int thread_offset_bytes = B_warp_tile_byte_offset + mma_tile_byte_offset + thread_byte_offset;
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
                        "{%0}, [%1];"
                        : "=r"(B_register[mma_k][mma_n])
                        : "r"(thread_offset_bytes)
                    );
                }
            }

            // outer product between mma tiles
            for(unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++) {
                for(unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++) {
                    for(unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++) {
                        asm volatile (
                            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                            "{%0, %1}, "
                            "{%2, %3}, "
                            "{%4}, "
                            "{%5, %6};"
                            : "=r"(acc_register[mma_m][mma_n][0]), "=r"(acc_register[mma_m][mma_n][1])
                            : "r"(A_register[mma_m][mma_k][0]), "r"(A_register[mma_m][mma_k][1]),
                                "r"(B_register[mma_k][mma_n])
                                "r"(acc_register[mma_m][mma_n][0]), "r"(acc_register[mma_m][mma_n][1])
                        );
                    }
                }
            }
        }
        __syncthreads();
    }

    //////////////
    // epilogue //
    //////////////
    half alpha_ = __float2half(alpha);
    half beta_ = __float2half(beta);
    half C_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];
    
    // calculate pointers for this warps C and D tiles
    half* C_block_gmem = C + (block_m * BM_dim * CD_stride) + (block_n * BN_dim);
    half* C_warp_gmem = C_block_gmem + (warp_m * WM_dim * CD_stride) + (warp_n * WN_dim);
    half* D_block_gmem = D + (block_m * BM_dim * CD_stride) + (block_n * BN_dim);
    half* D_warp_gmem = D_block_gmem + (warp_m * WM_dim * CD_stride) + (warp_n * WN_dim);

    for(unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++) {
        for(unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++) {
            half* C_mma_tile = C_warp_gmem + (mma_m * MMA_M_dim * CD_stride) + (mma_n * MMA_N_dim);
            ldmatrix_m16n8_gmem(C_mma_tile, C_register[mma_m][mma_n], N * sizeof(half));
            
            // scale C by beta
            acc_register_[mma_m][mma_n][0] = __hadd(__hmul(acc_register_[mma_m][mma_n][0], alpha_),
                                                   __hmul(C_register[mma_m][mma_n][0], beta_));
            acc_register_[mma_m][mma_n][1] = __hadd(__hmul(acc_register_[mma_m][mma_n][1], alpha_),
                                                   __hmul(C_register[mma_m][mma_n][1], beta_));
            acc_register_[mma_m][mma_n][2] = __hadd(__hmul(acc_register_[mma_m][mma_n][2], alpha_),
                                                   __hmul(C_register[mma_m][mma_n][2], beta_));
            acc_register_[mma_m][mma_n][3] = __hadd(__hmul(acc_register_[mma_m][mma_n][3], alpha_),
                                                   __hmul(C_register[mma_m][mma_n][3], beta_));
        }
    }

    for(unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++) {
        for(unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++) {
            half* D_mma_tile = D_warp_gmem + (mma_m * MMA_M_dim * CD_stride) + (mma_n * MMA_N_dim);
            stmatrix_m16n8(D_mma_tile, acc_register_[mma_m][mma_n], N * sizeof(half));
        }
    }
}