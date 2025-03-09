#include "run_bf16.cuh"
#include "kernels_bf16.cuh"
#include "cuda_utils.cuh"

typedef __nv_bfloat16 bf16;
#define cudaCheck(val) check_cuda((val), #val, __FILE__, __LINE__)

void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C, bool trans_b = false) {
    cublasStatus_t status;
    // A is MxK, B is KxN, C is MxN (row major)
    // So if B (first argument) is column major - NxK
    // Similarly A is simply column major - KxM
    // So C is NxM (column major) -> MxN (row major)
    if (!trans_b) {
        status = cublasGemmEx(handle,
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
    }
    // A is MxK, B is NxK, C is MxN (row major)
    // So if B (first argument) is column major - KxN, transpose -> NxK
    // Similarly A is simply column major - KxM
    // So C is NxM (column major) -> MxN (row major)
    else {
        status = cublasGemmEx(handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, CUDA_R_16BF, K,
            A, CUDA_R_16BF, K,
            &beta,
            C, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasGemmEx failed with status " << status << std::endl;
        exit(1);
    }
}

void run_sgemm_cuda_warptiling(int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C) {
    // Settings for H100
    const uint K10_NUM_THREADS = 128;
    const uint K10_BN = 128;
    const uint K10_BM = 64;
    const uint K10_BK = 8;
    const uint K10_WN = 32;
    const uint K10_WM = 64;
    const uint K10_WNITER = 1;
    const uint K10_TN = 8;
    const uint K10_TM = 4;
    
    dim3 blockDim(K10_NUM_THREADS);

    constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

    // warptile in threadblocktile
    static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
    static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

    // threads in warpsubtile
    static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                    0);
    constexpr uint K10_WMITER =
        (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
    // warpsubtile in warptile
    static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

    static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                    "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                    "issues during GMEM->SMEM tiling (loading only parts of the "
                    "final row of Bs during each iteraion)");
    static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                    "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                    "issues during GMEM->SMEM tiling (loading only parts of the "
                    "final row of As during each iteration)");
    static_assert(K10_BN % (16 * K10_TN) == 0,
                    "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(K10_BM % (16 * K10_TM) == 0,
                    "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                    "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                    "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim(cdiv(N, K10_BN), cdiv(M, K10_BM));
    sgemm_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_kernel_bf16(int kernel_num, int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C, cublasHandle_t handle, bool trans_b) {
    switch (kernel_num) {
        case 0:
            // std::cout << "cuBLAS BF16" << std::endl;
            run_cublas(handle, M, N, K, alpha, A, B, beta, C, trans_b);
            break;
        default:
            throw std::invalid_argument("Invalid kernel number");
    }
}