#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line);
void cuda_device_info();
int cdiv(int a, int b);