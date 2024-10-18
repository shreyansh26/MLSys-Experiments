#include <cuda.h>
#include <iostream>

#include "cuda_utils.hpp"

#define NUM_ELEMENTS 1024 
#define BLOCK_SIZE 1024
#define NUM_REPEATS 10

template <typename T>
__global__ void brent_kung_scan_kernel(T* X, T* Y, unsigned int N) {
    __shared__ T XY_s[BLOCK_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N) {
        XY_s[threadIdx.x] = X[i];
    }    
    if(i + blockDim.x < N) {
        XY_s[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    }

    for(unsigned int stride=1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index < BLOCK_SIZE) {
            XY_s[index] += XY_s[index - stride];
        }
    }

    for(int stride=BLOCK_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index + stride < BLOCK_SIZE) {
            XY_s[index + stride] += XY_s[index];
        }
    }
    
    __syncthreads();
    if(i < N) {
        Y[i] = XY_s[threadIdx.x];
    }
    if(i + blockDim.x < N) {
        Y[i + blockDim.x] = XY_s[threadIdx.x + blockDim.x];
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1)/b;
}

template <typename T>
void compute_scan(T* X_d, T* Y_d, unsigned int N) {
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(cdiv(N, BLOCK_SIZE));

    brent_kung_scan_kernel<T><<<gridSize, blockSize>>>(X_d, Y_d, N);
    
    CHECK_LAST_CUDA_ERROR();    
}

template <typename T>
void profile_scan(T* X_d, T* Y_d, unsigned int N) {
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(cdiv(N, BLOCK_SIZE));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for(int cntr=0; cntr<NUM_REPEATS; cntr++) {
        brent_kung_scan_kernel<T><<<gridSize, blockSize>>>(X_d, Y_d, N);
    }
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Time taken: " << milliseconds/NUM_REPEATS << " ms\n";
}

template <typename T>
void compute_cpu_scan(T* X_h, T* Y_h, unsigned int N) {
    Y_h[0] = X_h[0];
    for(unsigned int i=1; i<N; i++) {
        Y_h[i] = Y_h[i-1] + X_h[i];
    }
}

template <typename T>
void run_engine(unsigned int N, T abs_tol, double ref_tol) {
    T* X_h = nullptr;
    T* Y_h = nullptr;
    T* Y_cpu_ref = nullptr;

    CHECK_CUDA_ERROR(cudaMallocHost(&X_h, N*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&Y_h, N*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&Y_cpu_ref, N*sizeof(T)));

    random_initialize_array(X_h, N, 100);
    random_initialize_array(Y_h, N, 101);
    random_initialize_array(Y_cpu_ref, N, 102);

    T *X_d, *Y_d;

    CHECK_CUDA_ERROR(cudaMalloc(&X_d, N*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&Y_d, N*sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemcpy(X_d, X_h, N*sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(Y_d, Y_h, N*sizeof(T), cudaMemcpyHostToDevice));

    compute_scan<T>(X_d, Y_d, N);

    compute_cpu_scan<T>(X_h, Y_cpu_ref, N);

    CHECK_CUDA_ERROR(cudaMemcpy(Y_h, Y_d, N*sizeof(T), cudaMemcpyDeviceToHost));
    
    print_array<T>(X_h, N, "Original Array");
    print_array<T>(Y_h, N, "GPU Computation");
    print_array<T>(Y_cpu_ref, N, "CPU Computation");

    std::cout   << "GPU vs CPU allclose: "
                << (all_close<T>(Y_h, Y_cpu_ref, N, abs_tol, ref_tol) ? "true" : "false")
                << std::endl;

    profile_scan<T>(X_d, Y_d, N);

    CHECK_CUDA_ERROR(cudaFree(X_d));
    CHECK_CUDA_ERROR(cudaFree(Y_d));
    CHECK_CUDA_ERROR(cudaFreeHost(X_h));
    CHECK_CUDA_ERROR(cudaFreeHost(Y_h));
    CHECK_CUDA_ERROR(cudaFreeHost(Y_cpu_ref));
}

int main() {
    unsigned int N = NUM_ELEMENTS;
    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;

    run_engine<float>(N, abs_tol, rel_tol);

    return 0;
}