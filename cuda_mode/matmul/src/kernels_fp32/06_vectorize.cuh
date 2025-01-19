#pragma once

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorize(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    // the inner row & col that we're accessing in this thread
    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);

    // int threadRowA = threadIdx.x / BM;
    // int threadColB = threadIdx.x % BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadRes[TM * TN] = {0.f};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for(int blck = 0; blck < K; blck += BK) {
        // load As
        // transpose A while loading
        float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        // load Bs
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];

        __syncthreads();

        // Calculate per-thread results
        for(int bdim = 0; bdim < BK; bdim++) {
            // store in registers
            for(int i = 0; i < TM; i++) {
                regM[i] = As[bdim * BM + threadRow * TM + i];
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
        for(int resIdxN = 0; resIdxN < TN; resIdxN+=4) {
            float4 tmp = reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + (threadCol * TN + resIdxN)])[0];

            tmp.x = alpha * threadRes[resIdxM * TN + resIdxN + 0] + beta * tmp.x;
            tmp.y = alpha * threadRes[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
            tmp.z = alpha * threadRes[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
            tmp.w = alpha * threadRes[resIdxM * TN + resIdxN + 3] + beta * tmp.w;

            reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + (threadCol * TN + resIdxN)])[0] = tmp;
        }
    }
}