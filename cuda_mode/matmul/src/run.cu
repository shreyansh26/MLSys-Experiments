#include "run.cuh"
#include "kernels.cuh"

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void cuda_device_info() {
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
void run_cublas_fp32(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // cuBLAS uses column-major order. So we change the order of our row-major A & B, since (B^T*A^T)^T = (A*B)
    // This runs cuBLAS in full fp32 mode
    // C (row-major) = C^T (column-major)
    //  = (B^T @ A^T) (column-major)
    //  = A @ B (row-major)
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                &alpha, 
                B, CUDA_R_32F, N, 
                A, CUDA_R_32F, K, 
                &beta, 
                C, CUDA_R_32F, N, 
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(cdiv(M, 32), cdiv(N, 32));
    dim3 blockDim(32, 32);

    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_global_coalescing(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(cdiv(M, 32), cdiv(N, 32));
    dim3 blockDim(32 * 32);

    sgemm_global_coalescing<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_shared_memory(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(cdiv(M, 32), cdiv(N, 32));
    dim3 blockDim(32 * 32);

    sgemm_shared_memory<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_1d_blocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;

    dim3 gridDim(cdiv(N, BN), cdiv(M, BM));
    dim3 blockDim((BM * BN) / TM);

    sgemm_1d_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_2d_blocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    dim3 gridDim(cdiv(N, BN), cdiv(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));

    sgemm_2d_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_vectorize(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    dim3 gridDim(cdiv(N, BN), cdiv(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));

    sgemm_vectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            // std::cout << "cuBLAS FP32" << std::endl;
            run_cublas_fp32(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            // std:: cout << "Kernel 1 - Naive" << std::endl;
            run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
            break;
        case 2:
            // std::cout << Kernel 2 - Gloab Coalescing << std::endl;
            run_sgemm_global_coalescing(M, N, K, alpha, A, B, beta, C);
            break;
        case 3:
            // std::cout << "Kernel 3 - Shared Memory" << std::endl;
            run_sgemm_shared_memory(M, N, K, alpha, A, B, beta, C);
            break;
        case 4:
            // std::cout << "Kernel 4 - 1D Blocktiling" << std::endl;
            run_sgemm_1d_blocktiling(M, N, K, alpha, A, B, beta, C);
            break;
        case 5:
            // std::cout << "Kernel 5 - 2D Blocktiling" << std::endl;
            run_sgemm_2d_blocktiling(M, N, K, alpha, A, B, beta, C);
            break;
        case 6:
            // std::cout << "Kernel 6 - Vectorize" << std::endl;
            run_sgemm_vectorize(M, N, K, alpha, A, B, beta, C);
            break;
        default:
            throw std::invalid_argument("Invalid kernel number");
    }
}