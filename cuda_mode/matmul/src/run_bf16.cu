#include "run_bf16.cuh"
#include "kernels_bf16.cuh"
#include "cuda_utils.cuh"

typedef __nv_bfloat16 bf16;
#define cudaCheck(val) check_cuda((val), #val, __FILE__, __LINE__)

void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C, int trans_b = 0) {
    cublasStatus_t status;
    // A is MxK, B is KxN, C is MxN (row major)
    // So if B (first argument) is column major - NxK
    // Similarly A is simply column major - KxM
    // So C is NxM (column major) -> MxN (row major)
    if (trans_b == 0) {
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
    else if (trans_b == 1) {
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
    else if (trans_b == 2) {
        status = cublasGemmEx(handle,
            CUBLAS_OP_N,       // A is not transposed (column-major A is stored as A[i + k*M])
            CUBLAS_OP_N,       // B remains as stored (but note that you want Bᵀ—so B should have been supplied transposed)
            M,                 // number of rows of C
            N,                 // number of columns of C
            K,
            &alpha,
            A, CUDA_R_16BF, M,
            B, CUDA_R_16BF, K,
            &beta,
            C, CUDA_R_16BF, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
    }
    else {
        std::cerr << "Invalid trans_b value: " << trans_b << std::endl;
        exit(EXIT_FAILURE);
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
    static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) == 0);
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

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m=0, _prev_n=0, _prev_k=0;

void run_tensor_core_row_major(int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    CUtensorMap *d_tma_map_A = 0;
    CUtensorMap *d_tma_map_B = 0;

    if (!d_tma_map_A) {
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        // d_tma_map_B = allocate_and_create_tensor_map<BK, BN>(B, K / BK, N / BN);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
    }

    tensor_core_matmul_row_major<
    /*BM*/ BM,
    /*BN*/ BN,
    /*BK*/ BK,
    /*WGMMA_M*/ 64,
    /*WGMMA_N*/ 64,
    /*WGMMA_K*/ 16,
    /*NUM_THREADS*/ NUM_THREADS>
    <<<(M/BM) * (N/BN), NUM_THREADS>>>(M, N, K, C, d_tma_map_A, d_tma_map_B, alpha, beta);
}

void run_tensor_core_col_major(int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    CUtensorMap *d_tma_map_A = 0;
    CUtensorMap *d_tma_map_B = 0;

    if (!d_tma_map_A) {
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        // d_tma_map_B = allocate_and_create_tensor_map<BK, BN>(B, K / BK, N / BN);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
    }

    tensor_core_matmul_col_major<
    /*BM*/ BM,
    /*BN*/ BN,
    /*BK*/ BK,
    /*WGMMA_M*/ 64,
    /*WGMMA_N*/ 64,
    /*WGMMA_K*/ 16,
    /*NUM_THREADS*/ NUM_THREADS>
    <<<(M/BM) * (N/BN), NUM_THREADS>>>(M, N, K, C, d_tma_map_A, d_tma_map_B, alpha, beta);
}

void run_larger_output_tile(int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C, int *DB) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    CUtensorMap *d_tma_map_A = 0;
    CUtensorMap *d_tma_map_B = 0;

    if (!d_tma_map_A) {
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
    }

    size_t smem_size = sizeof(SMem<BM, BN, BK>);
    
    if (DB) {
        cudaCheck(cudaFuncSetAttribute(larger_output_tile<BM, BN, BK, NUM_THREADS, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        larger_output_tile<BM, BN, BK, NUM_THREADS, true><<<(M/BM) * (N/BN), NUM_THREADS, smem_size>>>(M, N, K, C, d_tma_map_A, d_tma_map_B, alpha, beta, DB);
    } else {
        cudaCheck(cudaFuncSetAttribute(larger_output_tile<BM, BN, BK, NUM_THREADS, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        larger_output_tile<BM, BN, BK, NUM_THREADS, false><<<(M/BM) * (N/BN), NUM_THREADS, smem_size>>>(M, N, K, C, d_tma_map_A, d_tma_map_B, alpha, beta, DB);
    }
}

void run_producer_consumer(int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C, int *DB) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128 * 2;
    constexpr int QSIZE = 5;

    CUtensorMap *d_tma_map_A = 0;
    CUtensorMap *d_tma_map_B = 0;

    if (!d_tma_map_A) {
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
    }

    size_t smem_size = sizeof(SMemQueue<BM, BN, BK, QSIZE>);
    
    cudaCheck(cudaFuncSetAttribute(producer_consumer<BM, BN, BK, NUM_THREADS, QSIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    producer_consumer<BM, BN, BK, NUM_THREADS, QSIZE><<<(M/BM) * (N/BN), NUM_THREADS, smem_size>>>(M, N, K, C, d_tma_map_A, d_tma_map_B, alpha, beta, DB);
}

void run_kernel_bf16(int kernel_num, int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C, cublasHandle_t handle, int trans_b, int *DB) {
    switch (kernel_num) {
        case 0:
            // std::cout << "cuBLAS BF16" << std::endl;
            run_cublas(handle, M, N, K, alpha, A, B, beta, C, trans_b);
            break;
        case 1:
            // std::cout << "CUDA Warptiling BF16" << std::endl; // From Simon's blog
            // C = alpha * A @ B + beta * C (A = MxK, B = KxN, C = MxN)
            run_sgemm_cuda_warptiling(M, N, K, alpha, A, B, beta, C);
            break;
        case 20:
            // std::cout << "Tensor Core BF16" << std::endl;
            run_tensor_core_row_major(M, N, K, alpha, A, B, beta, C);
            break;
        case 21:
            // std::cout << "Tensor Core BF16" << std::endl;
            run_tensor_core_col_major(M, N, K, alpha, A, B, beta, C);
            break;
        case 3:
            // std::cout << "Larger Output Tile BF16" << std::endl;
            run_larger_output_tile(M, N, K, alpha, A, B, beta, C, DB);
            break;
        case 4:
            // std::cout << "Producer Consumer BF16" << std::endl;
            run_producer_consumer(M, N, K, alpha, A, B, beta, C, DB);
            break;
        default:
            throw std::invalid_argument("Invalid kernel number");
    }
}