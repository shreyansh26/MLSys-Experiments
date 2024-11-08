#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>

#define M 3
#define N 4
#define K 7

void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}

void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status << std::endl;
        exit(1);
    }
}

int main() {
    // Host matrices
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices (example with simple values)
    for(int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for(int i = 0; i < K * N; i++) h_B[i] = 2.0f;
    for(int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Device matrices
    float *d_A, *d_B, *d_C;
    checkCudaStatus(cudaMalloc(&d_A, M * K * sizeof(float)));
    checkCudaStatus(cudaMalloc(&d_B, K * N * sizeof(float)));
    checkCudaStatus(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data to device
    checkCudaStatus(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkCublasStatus(cublasCreate(&handle));

    // Perform SGEMM
    // C = α*A*B + β*C
    float alpha = 1.0f;
    float beta = 0.0f;
    
    checkCublasStatus(cublasSgemm(
        handle,
        CUBLAS_OP_N,    // no transpose A
        CUBLAS_OP_N,    // no transpose B
        M,              // rows of A and C
        N,              // columns of B and C
        K,              // columns of A and rows of B
        &alpha,         // scaling factor for multiplication
        d_A,           // matrix A
        M,             // leading dimension of A
        d_B,           // matrix B
        K,             // leading dimension of B
        &beta,         // scaling factor for C
        d_C,           // matrix C
        M              // leading dimension of C
    ));

    // Copy result back to host
    checkCudaStatus(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print A
    std::cout << "Matrix A:" << std::endl;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            std::cout << h_A[i * K + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print B
    std::cout << "Matrix B:" << std::endl;
    for(int i = 0; i < K; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << h_B[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print the result
    std::cout << "Result matrix C:" << std::endl;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}