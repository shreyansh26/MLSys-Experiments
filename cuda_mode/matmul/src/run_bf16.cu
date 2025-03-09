#include "run_bf16.cuh"
#include "kernels_bf16.cuh"
#include "cuda_utils.cuh"

typedef __nv_bfloat16 bf16;
#define cudaCheck(val) check_cuda((val), #val, __FILE__, __LINE__)

void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C) {
    // A is MxK, B is KxN, C is MxN (row major)
    // So if B (first argument) is column major - KxN
    // Similarly A is simply column major - KxM
    // So C is NxM (column major) -> MxN (row major)
    cublasStatus_t status = cublasGemmEx(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasGemmEx failed with status " << status << std::endl;
        exit(1);
    }
}

void run_kernel_bf16(int kernel_num, int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C, cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            // std::cout << "cuBLAS BF16" << std::endl;
            run_cublas(handle, M, N, K, alpha, A, B, beta, C);
            break;
        default:
            throw std::invalid_argument("Invalid kernel number");
    }
}