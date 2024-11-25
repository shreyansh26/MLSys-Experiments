#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BLOCKDIM>
__global__ void sgemm_shared_memory(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    // the inner row & col that we're accessing in this thread
    int threadRow = threadIdx.x / BLOCKDIM;
    int threadCol = threadIdx.x % BLOCKDIM;

    __shared__ float As[BLOCKDIM * BLOCKDIM];
    __shared__ float Bs[BLOCKDIM * BLOCKDIM];
    
    // advance pointers to the starting positions
    A += cRow * BLOCKDIM * K;                       // row=cRow, col=0
    B += cCol * BLOCKDIM;                           // row=0, col=cCol
    C += cRow * BLOCKDIM * N + cCol * BLOCKDIM;     // row=cRow, col=cCol

    float sum = 0.f;
    for(int blck = 0; blck < K; blck += BLOCKDIM) {
        // Make threadCol the consecutive index for global memory coalescing
        As[threadRow * BLOCKDIM + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKDIM + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();

        // execute the dotproduct on the currently cached block
        for(int bdim = 0; bdim < BLOCKDIM; bdim++) {
            sum += As[threadRow * BLOCKDIM + bdim] * Bs[bdim * BLOCKDIM + threadCol];
        }

        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();

        A += BLOCKDIM;
        B += BLOCKDIM * N;
    }

    // update the C matrix
    C[threadRow * N + threadCol] = alpha * sum + beta * C[threadRow * N + threadCol];
}