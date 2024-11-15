#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// A is MxK, B is KxN, C is MxN (in row major order)
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if(i < M && j < K) {
        float sum = 0.f;
        for(int k=0; k<K; k++) {
            sum += A[i*K + k] * B[k*N + j];
        }

        C[i*N + j] = alpha * sum + beta * C[i*N + j];
    }
}