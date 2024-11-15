#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line);

void CudaDeviceInfo();

void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle);