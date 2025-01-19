#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BLOCKDIM>
__global__ void sgemm_global_coalescing(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int i = blockIdx.x * BLOCKDIM + (threadIdx.x / BLOCKDIM);
    int j = blockIdx.y * BLOCKDIM + (threadIdx.x % BLOCKDIM);

    if(i < M && j < N) {
        float sum = 0.f;
        for(int k=0; k<K; k++) {
            sum += A[i*K + k] * B[k*N + j];
        }

        C[i*N + j] = alpha * sum + beta * C[i*N + j];
    }
}