#include "run_fp16.cuh"
#include "kernels_fp16.cuh"
#include "cuda_utils.cuh"

// A is MxK, B is KxN, C is MxN (in row major order)
void run_cublas(cublasHandle_t handle, int M, int N, int K, half alpha, half *A, half *B, half beta, half *C) {
    // cuBLAS uses column-major order. So we change the order of our row-major A & B, since (B^T*A^T)^T = (A*B)
    // This runs cuBLAS in full fp16 mode
    // C (row-major) = C^T (column-major)
    //  = (B^T @ A^T) (column-major)
    //  = A @ B (row-major)
    // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
    //             &alpha, 
    //             B, CUDA_R_16F, N, 
    //             A, CUDA_R_16F, K, 
    //             &beta, 
    //             C, CUDA_R_16F, N, 
    //             CUBLAS_COMPUTE_16F,
    //             CUBLAS_GEMM_DEFAULT);

    cublasStatus_t status = cublasHgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );
}

void run_kernel_fp16(int kernel_num, int M, int N, int K, half alpha, half *A, half *B, half beta, half *C, cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            // std::cout << "cuBLAS FP16" << std::endl;
            run_cublas(handle, M, N, K, alpha, A, B, beta, C);
            break;
        default:
            throw std::invalid_argument("Invalid kernel number");
    }
}