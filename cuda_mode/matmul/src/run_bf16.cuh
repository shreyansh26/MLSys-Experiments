#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>

typedef __nv_bfloat16 bf16;

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line);

void cuda_device_info();

void run_kernel_bf16(int kernel_num, int m, int n, int k, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C, cublasHandle_t handle, int trans_b = 0);