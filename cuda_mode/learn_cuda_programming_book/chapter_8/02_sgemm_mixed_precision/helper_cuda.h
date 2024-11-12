#include <iostream>
#include <cuda_runtime.h>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Add this function for cuBLAS error checking
#define checkCublasErrors(val) check_cublas((val), #val, __FILE__, __LINE__)
void check_cublas(cublasStatus_t status, const char *msg, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Error: %s\n%s\n%s Line: %d\n", 
                msg, cublasGetStatusString(status), file, line);
        exit(-1);
    }
}