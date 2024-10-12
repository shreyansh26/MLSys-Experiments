// From the CUDA Programming Guide - https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pragma-unroll
#include <stdio.h>

struct S1_t { static const int value = 4; };
template <int X, typename T2>
__device__ void foo(int *p1, int *p2) {

// no argument specified, loop will be completely unrolled
#pragma unroll
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i]*2;

// unroll value = 8
#pragma unroll (X+1) // Use a loop for the remaining 12-8 = 4 iterations
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i]*4;

// unroll value = 1, loop unrolling disabled
#pragma unroll 1
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i]*8;

// unroll value = 4
#pragma unroll (T2::value) // Use a loop for the remaining 12-4 = 8 iterations
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i]*16;
}


__global__ void bar(int *p1, int *p2) {
    foo<7, S1_t>(p1, p2);
}

int main() {
    int h_p1[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int h_p2[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int *d_p1, *d_p2;

    // Allocate device memory
    cudaMalloc(&d_p1, 12 * sizeof(int));
    cudaMalloc(&d_p2, 12 * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_p1, h_p1, 12 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2, h_p2, 12 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    bar<<< 1, 1 >>>(d_p1, d_p2);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_p1, d_p1, 12 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < 12; ++i) {
        printf("%d ", h_p1[i]);
    }

    // Free device memory
    cudaFree(d_p1);
    cudaFree(d_p2);

    return 0;
}
