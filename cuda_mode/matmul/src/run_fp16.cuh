#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line);

void cuda_device_info();

void run_kernel_fp16(int kernel_num, int m, int n, int k, float alpha, half *A, half *B, float beta, half *C, half *D, cublasHandle_t handle);