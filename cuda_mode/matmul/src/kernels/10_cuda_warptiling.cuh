#pragma once

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

const int WARPSIZE = 32;

template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ void load_from_global_memory(int N, int K, float *A, float *B, float *As, float *Bs, int innerRowA, int innerColA, int innerRowB, int innerColB) {
    for(int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        float4 tmp = reinterpret_cast<float4 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for(int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] = reinterpret_cast<float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER, const int WNITER, const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void process_from_smem(float *regM, float *regN, float *threadRes, float *As, float *Bs, int warpRow, int warpCol, int threadRowInWarp, int threadColInWarp) {
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
                        threadRes[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] += regM[wSubRowIdx * TM + resIdxM] * regN[wSubColIdx * TN + resIdxN];
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
__global__ void __launch_bounds__(NUM_THREADS) sgemm_warptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
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
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

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
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

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
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for(int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for(int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    // load C vector into registers
                    float4 tmp = reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    tmp.x = alpha * threadRes[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadRes[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadRes[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadRes[i + 3] + beta * tmp.w;
                    // write back
                    reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N +  threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}