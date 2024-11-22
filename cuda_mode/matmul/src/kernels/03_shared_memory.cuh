#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BLOCKDIM>
__global__ void sgemm_shared_memory(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    A += cRow * BLOCKDIM * K;
    B += cCol * BLOCKDIM;
    C += cRow * BLOCKDIM * N + cCol * BLOCKDIM;

    int threadRow = threadIdx.x / BLOCKDIM;
    int threadCol = threadIdx.x % BLOCKDIM;

    __shared__ float As[BLOCKDIM * BLOCKDIM];
    __shared__ float Bs[BLOCKDIM * BLOCKDIM];

    float sum = 0.f;
    for(int blck = 0; blck < K; blck += BLOCKDIM) {
        // Make threadCol the consecutive index for global memory coalescing
        As[threadRow * BLOCKDIM + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKDIM + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();

        for(int bdim = 0; bdim < BLOCKDIM; bdim++) {
            sum += As[threadRow * BLOCKDIM + bdim] * Bs[bdim * BLOCKDIM + threadCol];
        }

        __syncthreads();

        A += BLOCKDIM;
        B += BLOCKDIM * N;
    }

    C[threadRow * N + threadCol] = alpha * sum + beta * C[threadRow * N + threadCol];
}