#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// cuda runtime
#include <cuda_runtime.h>

#include "reduction.h"

void run_benchmark(void (*reduce)(float*, float*, int, int), float *d_outPtr, float *d_inPtr, int size);
void init_input(float* data, int size);
float get_cpu_result(float *data, int size);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    float *h_inPtr;
    float *d_inPtr, *d_outPtr;
    float *h_outPtr;  // New host output pointer

    unsigned int size = 1 << 24 + 1;

    float result_host;

    srand(2019);

    // Allocate memory
    h_inPtr = (float*)malloc(size * sizeof(float));
    h_outPtr = (float*)malloc(sizeof(float));
    memset(h_outPtr, 0.0f, sizeof(float));
    
    // Data initialization with random values
    init_input(h_inPtr, size);

    // Prepare GPU resource
    cudaMalloc((void**)&d_inPtr, size * sizeof(float));
    cudaMalloc((void**)&d_outPtr, sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outPtr, h_outPtr, sizeof(float), cudaMemcpyHostToDevice);

    // Get reduction result from GPU
    run_benchmark(reduction, d_outPtr, d_inPtr, size);
    
    cudaMemcpy(h_outPtr, d_outPtr, sizeof(float), cudaMemcpyDeviceToHost);

    // Get all sum from CPU
    result_host = get_cpu_result(h_inPtr, size);
    printf("host: %f, device %f\n", result_host, *h_outPtr);
    
    // Terminates memory
    cudaFree(d_outPtr);
    cudaFree(d_inPtr);
    free(h_inPtr);
    free(h_outPtr);  // Free the new host output pointer

    return 0;
}

void run_reduction(void (*reduce)(float*, float*, int, int), float *d_outPtr, float *d_inPtr, int size, int n_threads) {
    reduce(d_outPtr, d_inPtr, size, n_threads);
}

void run_benchmark(void (*reduce)(float*, float*, int, int), float *d_outPtr, float *d_inPtr, int size) {
    int num_threads = 1024;
    int test_iter = 100;

    // warm-up
    reduce(d_outPtr, d_inPtr, size, num_threads);
    
    // initialize timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ////////
    // Operation body
    ////////
    cudaEventRecord(start);
    for(int i = 0; i < test_iter; i++) {
        // reset d_outPtr to 0
        cudaMemset(d_outPtr, 0.0, sizeof(float));
        run_reduction(reduce, d_outPtr, d_inPtr, size, num_threads);
    }
    cudaEventRecord(stop);
    // getting elapsed time
    cudaEventSynchronize(stop);

    // Compute and print the performance
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= test_iter;
    float bandwidth = size * sizeof(float) / milliseconds / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", milliseconds, bandwidth);
}

void init_input(float *data, int size) {
    for(int i = 0; i < size; i++) {
        // Keep the numbers small so we don't get truncation error in the sum
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float get_cpu_result(float *data, int size) {
    double result = 0.f;
    for(int i = 0; i < size; i++)
        result += data[i];

    return (float)result;
}