#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>    
#include <chrono>  

// CUDA kernel to increment each element of an array by 1
__global__ void incrementKernel(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] += 1;
    }
}

// Warmup kernel that doesn't modify the array
__global__ void warmupKernel(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Just read the value and perform a dummy operation
        volatile int dummy = d_array[idx] + 1;
    }
}

void calculate_with_cpu(int* h_ref_array, int size) {
    for (int i = 0; i < size; i++) {
        h_ref_array[i] += 1;
    }
}

// Host function to set up and launch the kernel
void incrementArrayCUDA(int* h_array, int* h_ref_array, int size) {
    int* d_array;
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Allocate device memory
    cudaMalloc((void**)&d_array, size * sizeof(int));

    // Copy host array to device
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the warmup kernel
    warmupKernel<<<numBlocks, blockSize>>>(d_array, size);

    // Launch the kernel
    // This is non-blocking
    incrementKernel<<<numBlocks, blockSize>>>(d_array, size);

    // Hence, can launch another CPU code which will overlap with the kernel
    calculate_with_cpu(h_ref_array, size);

    // Whether the host function or device kernel completes first doesn’t affect the subsequent device-to-host transfer, which will begin only after the kernel completes.
    // Copy the result back to host
    cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    // // Can launch another CPU code which will execute after the kernel
    // calculate_with_cpu(h_ref_array, size);

    // Verify the result
    for (int i = 0; i < size; i++) {
        assert(h_array[i] == h_ref_array[i]);
    }

    // Free device memory
    cudaFree(d_array);
}

// Main function for testing
int main() {
    const int SIZE = 1 << 24;
    int* h_array = new int[SIZE];
    int* h_ref_array = new int[SIZE];
    // Initialize the array
    for (int i = 0; i < SIZE; i++) {
        h_array[i] = i;
        h_ref_array[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Call the CUDA function to increment the array
    incrementArrayCUDA(h_array, h_ref_array, SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Time taken: %f seconds\n", elapsed.count());

    // Verify the result (just checking a few elements)
    printf("First few elements after increment:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    delete[] h_array;
    return 0;
}

