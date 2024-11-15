#include "run.cuh"

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

// A is MxK, B is KxN, C is MxN (in row major order)
void run_cublas_fp32(cublasHandle_t handle, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    // cuBLAS uses column-major order. So we change the order of our row-major A & B, since (B^T*A^T)^T = (A*B)
    // This runs cuBLAS in full fp32 mode
    // C (row-major) = C^T (column-major)
    //  = (B^T @ A^T) (column-major)
    //  = A @ B (row-major)
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                &alpha, 
                B, CUDA_R_32F, n, 
                A, CUDA_R_32F, k, 
                &beta, 
                C, CUDA_R_32F, n, 
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            // std::cout << "cuBLAS FP32" << std::endl;
            run_cublas_fp32(handle, m, n, k, alpha, A, B, beta, C);
            break;
        default:
            throw std::invalid_argument("Invalid kernel number");
    }
}