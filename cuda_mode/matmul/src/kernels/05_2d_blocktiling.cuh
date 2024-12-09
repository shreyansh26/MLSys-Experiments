#pragma once

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    const int totalResultsBlocktile = BM * BN;
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    const int strideA = numThreadsBlocktile / BK;
    const int strideB = numThreadsBlocktile / BN;

    assert(numThreadsBlocktile == blockDim.x);

    // the inner row & col that we're accessing in this thread
    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);

    // int threadRowA = threadIdx.x / BM;
    // int threadColB = threadIdx.x % BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;

    // allocate thread-local cache for results in registerfile
    float threadRes[TM * TN] = {0.f};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for(int blck = 0; blck < K; blck += BK) {
        for(int localOffsetA = 0; localOffsetA < BM; localOffsetA += strideA) {
            As[(innerRowA + localOffsetA) * BK + innerColA] = A[(innerRowA + localOffsetA) * K + innerColA];
        }

        for(int localOffsetB = 0; localOffsetB < BK; localOffsetB += strideB) {
            Bs[(innerRowB + localOffsetB) * BN + innerColB] = B[(innerRowB + localOffsetB) * N + innerColB];
        }

        __syncthreads();

        // Calculate per-thread results
        for(int bdim = 0; bdim < BK; bdim++) {
            // store in registers
            for(int i = 0; i < TM; i++) {
                regM[i] = As[(threadRow * TM + i) * BK + bdim];
            }

            for(int i = 0; i < TN; i++) {
                regN[i] = Bs[bdim * BN + threadCol * TN + i];
            }

            for(int resIdxM = 0; resIdxM < TM; resIdxM++) {
                for(int resIdxN = 0; resIdxN < TN; resIdxN++) {
                    threadRes[resIdxM * TN + resIdxN] += (regM[resIdxM] * regN[resIdxN]);
                }
            }
        }
        __syncthreads();

        A += BK;
        B += BK * N;
    }

    for(int resIdxM = 0; resIdxM < TM; resIdxM++) {
        for(int resIdxN = 0; resIdxN < TN; resIdxN++) {
            C[(threadRow * TM + resIdxM) * N + (threadCol * TN + resIdxN)] = alpha * threadRes[resIdxM * TN + resIdxN] + beta * C[(threadRow * TM + resIdxM) * N + (threadCol * TN + resIdxN)];
        }
    }
}