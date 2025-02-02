#include "run_fp16.cuh"
#include "kernels_fp16.cuh"
#include "cuda_utils.cuh"

#define cudaCheck(val) check_cuda((val), #val, __FILE__, __LINE__)

// A is MxK, B is KxN, C is MxN (in row major order)
void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, half *A, half *B, float beta, half *C) {
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

    half alpha_h = __float2half(alpha);
    half beta_h = __float2half(beta);
    cublasStatus_t status = cublasHgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        B, N,
        A, K,
        &beta_h,
        C, N
    );
}

void hierarchical_tiling_launch(cublasHandle_t handle, int M, int N, int K, float alpha, half *A, half *B, float beta, half *C, half *D) {
    constexpr unsigned int BM_dim = 128;
    constexpr unsigned int BN_dim = 128;
    constexpr unsigned int BK_dim = 64;
    
    constexpr unsigned int WARPS_PER_BLOCK_M = 4;
    constexpr unsigned int WARPS_PER_BLOCK_N = 2;
    constexpr unsigned int WARPS_PER_BLOCK_K = 2;

    constexpr unsigned int WM_dim = BM_dim / WARPS_PER_BLOCK_M;
    constexpr unsigned int WN_dim = BN_dim / WARPS_PER_BLOCK_N;
    constexpr unsigned int WK_dim = BK_dim / WARPS_PER_BLOCK_K;

    assert(M % BM_dim == 0);
    assert(N % BN_dim == 0);
    assert(K % BK_dim == 0);
    
    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int BlocksM = M / BM_dim;
    const unsigned int BlocksN = N / BN_dim;
    constexpr unsigned int ThreadsM = WARPS_PER_BLOCK_M;
    constexpr unsigned int ThreadsN = WARP_SIZE * WARPS_PER_BLOCK_N;
    constexpr unsigned int NumThreads = ThreadsM * ThreadsN;
    const unsigned int shmem_bytes = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(half);

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);
    
    cudaCheck(cudaFuncSetAttribute(hierarchical_tiling<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, NumThreads>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)); // set shared memory limit to 64KB which is maximum for sm_75

    hierarchical_tiling<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, NumThreads><<<gridDim, blockDim, shmem_bytes>>>(
            A,
            B,
            C,
            D,
            alpha,
            beta,
            M,
            N,
            K
        );
}

void hierarchical_tiling_unrolled_launch(cublasHandle_t handle, int M, int N, int K, float alpha, half *A, half *B, float beta, half *C, half *D) {
    constexpr unsigned int BM_dim = 128;
    constexpr unsigned int BN_dim = 128;
    constexpr unsigned int BK_dim = 64;
    
    constexpr unsigned int WARPS_PER_BLOCK_M = 4;
    constexpr unsigned int WARPS_PER_BLOCK_N = 2;
    constexpr unsigned int WARPS_PER_BLOCK_K = 2;

    constexpr unsigned int WM_dim = BM_dim / WARPS_PER_BLOCK_M;
    constexpr unsigned int WN_dim = BN_dim / WARPS_PER_BLOCK_N;
    constexpr unsigned int WK_dim = BK_dim / WARPS_PER_BLOCK_K;

    assert(M % BM_dim == 0);
    assert(N % BN_dim == 0);
    assert(K % BK_dim == 0);
    
    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int BlocksM = M / BM_dim;
    const unsigned int BlocksN = N / BN_dim;
    constexpr unsigned int ThreadsM = WARPS_PER_BLOCK_M;
    constexpr unsigned int ThreadsN = WARP_SIZE * WARPS_PER_BLOCK_N;
    constexpr unsigned int NumThreads = ThreadsM * ThreadsN;
    const unsigned int shmem_bytes = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(half);

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);
    
    cudaCheck(cudaFuncSetAttribute(hierarchical_tiling_unrolled<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, NumThreads>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)); // set shared memory limit to 64KB which is maximum for sm_75

    hierarchical_tiling_unrolled<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, NumThreads><<<gridDim, blockDim, shmem_bytes>>>(
            A,
            B,
            C,
            D,
            alpha,
            beta,
            M,
            N,
            K
        );
}

void hierarchical_tiling_unrolled_vectorized_launch(cublasHandle_t handle, int M, int N, int K, float alpha, half *A, half *B, float beta, half *C, half *D) {
    constexpr unsigned int BM_dim = 128;
    constexpr unsigned int BN_dim = 128;
    constexpr unsigned int BK_dim = 64;
    
    constexpr unsigned int WARPS_PER_BLOCK_M = 4;
    constexpr unsigned int WARPS_PER_BLOCK_N = 2;
    constexpr unsigned int WARPS_PER_BLOCK_K = 2;

    constexpr unsigned int WM_dim = BM_dim / WARPS_PER_BLOCK_M;
    constexpr unsigned int WN_dim = BN_dim / WARPS_PER_BLOCK_N;
    constexpr unsigned int WK_dim = BK_dim / WARPS_PER_BLOCK_K;

    assert(M % BM_dim == 0);
    assert(N % BN_dim == 0);
    assert(K % BK_dim == 0);
    
    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int BlocksM = M / BM_dim;
    const unsigned int BlocksN = N / BN_dim;
    constexpr unsigned int ThreadsM = WARPS_PER_BLOCK_M;
    constexpr unsigned int ThreadsN = WARP_SIZE * WARPS_PER_BLOCK_N;
    constexpr unsigned int NumThreads = ThreadsM * ThreadsN;
    const unsigned int shmem_bytes = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(half);

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);
    
    cudaCheck(cudaFuncSetAttribute(hierarchical_tiling_unrolled<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, NumThreads>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)); // set shared memory limit to 64KB which is maximum for sm_75

    hierarchical_tiling_unrolled_vectorized<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, NumThreads><<<gridDim, blockDim, shmem_bytes>>>(
            A,
            B,
            C,
            D,
            alpha,
            beta,
            M,
            N,
            K
        );
}

void run_kernel_fp16(int kernel_num, int M, int N, int K, float alpha, half *A, half *B, float beta, half *C, half *D, cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            // std::cout << "cuBLAS FP16" << std::endl;
            run_cublas(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            // std::cout << "Hierarchical Tiling FP16" << std::endl;
            hierarchical_tiling_launch(handle, M, N, K, alpha, A, B, beta, C, D);
            break;
        case 2:
            // std::cout << "Hierarchical Tiling Unrolled Memcpy" << std::endl;
            hierarchical_tiling_unrolled_launch(handle, M, N, K, alpha, A, B, beta, C, D);
            break;
        case 3:
            // std::cout << "Hierarchical Tiling Unrolled and Vectorized Memcpy" << std::endl;
            hierarchical_tiling_unrolled_vectorized_launch(handle, M, N, K, alpha, A, B, beta, C, D);
            break;
        default:
            throw std::invalid_argument("Invalid kernel number");
    }
}