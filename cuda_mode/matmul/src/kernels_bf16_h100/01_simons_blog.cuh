#pragma once
#include <cuda.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;
const int WARPSIZE = 32;

template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ void load_from_global_memory(int N, int K, bf16 *A, bf16 *B, bf16 *As, bf16 *Bs, int innerRowA, int innerColA, int innerRowB, int innerColB) {
    for(int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        bf16 tmp[4];
        float2 x = reinterpret_cast<float2 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
        memcpy(&tmp[0], &x, sizeof(bf16) * 4);

        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp[0];
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp[1];
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp[2];
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp[3];
    }

    for(int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        reinterpret_cast<float2 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] = reinterpret_cast<float2 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER, const int WNITER, const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void process_from_smem(bf16 *regM, bf16 *regN, float *threadRes, bf16 *As, bf16 *Bs, int warpRow, int warpCol, int threadRowInWarp, int threadColInWarp) {
    for(int bdim = 0; bdim < BK; bdim++) {
        // populate registers for whole warptile
        for(int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
            for(int i=0; i<TM; i++) {
                regM[wSubRowIdx * TM + i] = As[(bdim * BM) + warpRow * WM + wSubRowIdx * WSUBM +  threadRowInWarp * TM + i];
            }
        }

        for(int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
            for(int i=0; i<TN; i++) {
                regN[wSubColIdx * TN + i] = Bs[(bdim * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            }
        }

        // execute warptile matmul
        for(int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
            for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                // calculate per-thread results
                for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadRes[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] += __bfloat162float(regM[wSubRowIdx * TM + resIdxM]) * __bfloat162float(regN[wSubColIdx * TN + resIdxN]) ;
                    }
                }
            }
        }
    }
}

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template<const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) sgemm_warptiling(int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    const int warpIdx = threadIdx.x / WARPSIZE;
    const int warpRow = warpIdx / (BN / WN);
    const int warpCol = warpIdx % (BN / WN);

    // size of the warp subtile
    const int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    const int WSUBM = WM / WMITER;
    const int WSUBN = WN / WNITER;

    // Placement of the thread in the warp subtile
    const int threadIdxInWarp = threadIdx.x % WARPSIZE;
    const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);
    const int threadColInWarp = threadIdxInWarp % (WSUBN / TN);

    // allocate space for the current blocktile in SMEM
    __shared__ bf16 As[BM * BK];
    __shared__ bf16 Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int rowStrideA = NUM_THREADS / (BK / 4); // NUM_THREADS / (BK / 4) (NUM_THREADS * 4) / BK
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    const int rowStrideB = NUM_THREADS / (BN / 4); // NUM_THREADS / (BN / 4) (NUM_THREADS * 4) / BN

    // allocate thread-local cache for results in registerfile
    float threadRes[WMITER * TM * WNITER * TN] = {0.f};
    // register caches for As and Bs
    bf16 regM[WMITER * TM] = {0.0};
    bf16 regN[WNITER * TN] = {0.0};

    // outer-most loop over block tiles
    for(int blck = 0; blck < K; blck += BK) {
        // load from global memory
        load_from_global_memory<BM, BN, BK, rowStrideA, rowStrideB>(N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);

        __syncthreads();

        process_from_smem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(regM, regN, threadRes, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);

        __syncthreads();

        A += BK;        // move BK columns to right
        B += BK * N;    // move BK rows down
    }

    // write out the results
    for(int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
        for(int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
            // move C pointer to current warp subtile
            bf16 *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for(int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for(int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    // load C vector into registers
                    bf16 tmp[4] = {C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN],
                                   C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN + 1],
                                   C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN + 2],
                                   C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN + 3]};
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    tmp[0] = alpha * threadRes[i + 0] + beta * __bfloat162float(tmp[0]);
                    tmp[1] = alpha * threadRes[i + 1] + beta * __bfloat162float(tmp[1]);
                    tmp[2] = alpha * threadRes[i + 2] + beta * __bfloat162float(tmp[2]);
                    tmp[3] = alpha * threadRes[i + 3] + beta * __bfloat162float(tmp[3]);
                    
                    float2 x;
                    memcpy(&x, &tmp, sizeof(bf16) * 4);
                    // write back
                    reinterpret_cast<float2 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N +  threadColInWarp * TN + resIdxN])[0] = x;
                }
            }
        }
    }
}