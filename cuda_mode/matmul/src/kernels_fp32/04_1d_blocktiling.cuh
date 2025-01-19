#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template<const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    // the inner row & col that we're accessing in this thread
    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;

    int threadRowA = threadIdx.x / BM;
    int threadColB = threadIdx.x % BN;

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

    float threadRes[TM] = {0.f};

    for(int blck = 0; blck < K; blck += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        for(int bdim = 0; bdim < BK; bdim++) {
            float tmpB = Bs[bdim * BN + threadColB];

            for(int resIdx = 0; resIdx < TM; resIdx++) {
                threadRes[resIdx] += As[(threadRowA * TM + resIdx) * BK + bdim] * tmpB;
            }
        }
        __syncthreads();

        A += BK;
        B += BK * N;
    }

    for(int resIdx = 0; resIdx < TM; resIdx++) {
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadRes[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}