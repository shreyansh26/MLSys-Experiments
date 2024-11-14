#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line);

void CudaDeviceInfo();

void randomize_matrix(float *mat, int N);

void print_matrix(const float *A, int M, int N, std::ofstream &fs);

bool verify_matrix(float *mat1, float *mat2, int N);

void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle);