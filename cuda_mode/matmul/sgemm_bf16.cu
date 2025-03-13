#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

#include <run_bf16.cuh>
#include <matrix_utils.cuh>

typedef __nv_bfloat16 bf16;
#define cudaCheck(val) check_cuda((val), #val, __FILE__, __LINE__)

const std::string errLogFile = "matrixValidationFailure.txt";

void gemm_cpu_ref(int M, int N, int K, float alpha, bf16 *A, bf16 *B, float beta, bf16 *C, int trans_b = 0) {
    if (trans_b == 0) {
        // A is MxK, B is KxN, C is MxN
        float alpha_f = __bfloat162float(alpha);
        float beta_f = __bfloat162float(beta);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float c = beta_f * __bfloat162float(C[i * N + j]);
                for(int k = 0; k < K; k++) {
                    c += alpha_f * __bfloat162float(A[i * K + k]) * __bfloat162float(B[k * N + j]);
                }
                C[i * N + j] = __float2bfloat16(c);
            }
        }
    }
    else if(trans_b == 1) {
        // A is MxK, B is NxK, C is MxN
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                
                // Dot product of A's row i and B's row j
                for (int k = 0; k < K; ++k) {
                    sum += __bfloat162float(A[i * K + k]) * __bfloat162float(B[j * K + k]);
                }
                
                // Apply scaling factors
                const float old_c = __bfloat162float(C[i * N + j]);
                C[i * N + j] = __float2bfloat16(alpha * sum + beta * old_c);
            }
        }
    }
    else if(trans_b == 2) {
        // Use column-major indexing on C.
        // Here, C is stored column-major (i.e. element (i, j) is at C[i + j*M])
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    // For A [MxK] in column-major, element (i,k) is A[i + k*M]
                    // For B [KxN] in column-major, but to compute A x Bᵀ we use Bᵀ(i,k)=B[k + i*K]
                    sum += __bfloat162float(A[i + k*M]) * __bfloat162float(B[k + j*K]); 
                }
                float old_c = __bfloat162float(C[i + j*M]);
                C[i + j*M] = __float2bfloat16(alpha * sum + beta * old_c);
            }
        }
    }
    else {
        std::cerr << "Invalid trans_b value: " << trans_b << std::endl;
        exit(EXIT_FAILURE);
    }
}

// void gemm_cpu_ref(int M, int N, int K, float alpha, const bf16* A, const bf16* B, float beta, bf16* C) {
    
// }

int main(int argc, char **argv) {
    if (argc != 2 and argc != 3) {
        std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS) and optionally a trans_b flag (0 or 1)" << std::endl;
        exit(EXIT_FAILURE);
    }

    // get kernel number
    int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 12) {
        std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
        exit(EXIT_FAILURE);
    }

    int trans_b = 0;
    if (argc == 3) {
        trans_b = std::stoi(argv[2]);
    }

    // get environment variable for device
    int deviceIdx = 0;
    // if (getenv("DEVICES") != NULL) {
    //     deviceIdx = atoi(getenv("DEVICES"));
    // }
    cudaCheck(cudaSetDevice(deviceIdx));

    printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

    // print some device info
    cuda_device_info();

    // Declare the handle, create the handle, cublasCreate will return a value of
    // type cublasStatus_t to determine whether the handle was created
    // successfully (the value is 0)
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // cuBLAS FLOPs ceiling is reached at 8192
    std::vector<int> SIZE = {256, 512, 1024, 2048, 4096, 8192};

    long m, n, k, max_size;
    max_size = SIZE[SIZE.size() - 1];
    std::cout << "Max size: " << max_size << std::endl;

    float alpha = 1.0, beta = 0.0; // GEMM input parameters, C=α*AB+β*C

    bf16 *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr, *C_orig = nullptr, *C_cpu_ref = nullptr; // host matrices
    bf16 *A_d = nullptr, *B_d = nullptr, *C_d = nullptr, *C_ref_d = nullptr, *C_orig_d = nullptr; // device matrices

    A = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
    B = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
    C = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
    C_orig = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
    C_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
    C_cpu_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);

    randomize_matrix<bf16>(A, max_size * max_size);
    randomize_matrix<bf16>(B, max_size * max_size);
    randomize_matrix<bf16>(C, max_size * max_size);
    memcpy(C_orig, C, sizeof(bf16) * max_size * max_size);
    memcpy(C_cpu_ref, C, sizeof(bf16) * max_size * max_size);

    cudaCheck(cudaMalloc((void **)&A_d, sizeof(bf16) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&B_d, sizeof(bf16) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&C_d, sizeof(bf16) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&C_ref_d, sizeof(bf16) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&C_orig_d, sizeof(bf16) * max_size * max_size));
    cudaCheck(cudaMemcpy(A_d, A, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(B_d, B, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(C_d, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(C_orig_d, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(C_ref_d, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));

    int repeat_times = 50;
    for(int size : SIZE) {
        m = n = k = size;

        std::cout << "dimensions(m=n=k) " << m << ", alpha: " << (float)alpha << ", beta: " << (float)beta << std::endl;
        // Verify the correctness of the calculation, and execute it once before the
        // kernel function timing to avoid cold start errors

        // For kernel 0 i.e. cuBLAS, we only check correctness for small matrices as CPU does not support fp16
        if(kernel_num == 0 and m <= 512) {
            run_kernel_bf16(0, m, n, k, alpha, A_d, B_d, beta, C_ref_d, handle, trans_b); // cuBLAS
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
            cudaMemcpy(C_ref, C_ref_d, sizeof(bf16) * m * n, cudaMemcpyDeviceToHost);
                        
            gemm_cpu_ref(m, n, k, alpha, A, B, beta, C_cpu_ref, trans_b); // perform reference calculation on CPU

            if(!verify_matrix<bf16>(C_ref, C_cpu_ref, m * n)) {
                std::cout << "Failed to pass the correctness verification against CPU computation." << std::endl;
                if(m <= 128) {
                    std::cout << " Logging faulty output into " << errLogFile << "\n";
                    std::ofstream fs;
                    fs.open(errLogFile);
                    fs << "A:\n";
                    print_matrix<bf16>(A, m, k, fs);
                    fs << "B:\n";
                    print_matrix<bf16>(B, k, n, fs);
                    fs << "C_orig:\n";
                    print_matrix<bf16>(C_orig, m, n, fs);
                    fs << "Should (CPU):\n";
                    print_matrix<bf16>(C_cpu_ref, m, n, fs);
                    fs << "Kernel out:\n";
                    print_matrix<bf16>(C_ref, m, n, fs);
                }
                exit(EXIT_FAILURE);
            }
        }
        
        // For other kernels, larger errors at certain indices start to show up at 1024
        if(kernel_num != 0 and m <= 512) {
            run_kernel_bf16(0, m, n, k, alpha, A_d, B_d, beta, C_ref_d, handle, trans_b); // cuBLAS
            run_kernel_bf16(kernel_num, m, n, k, alpha, A_d, B_d, beta, C_d, handle); // Executes the kernel, modifies the result matrix
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
            cudaMemcpy(C_ref, C_ref_d, sizeof(bf16) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C, C_d, sizeof(bf16) * m * n, cudaMemcpyDeviceToHost);
            
            // Transpose C for verification
            // bf16* C_transposed = (bf16*)malloc(sizeof(bf16) * m * n);
            // for (int i = 0; i < m; i++) {
            //     for (int j = 0; j < n; j++) {
            //         C_transposed[j * m + i] = C[i * n + j];
            //     }
            // }
            // Copy transposed result back to C
            // memcpy(C, C_transposed, sizeof(bf16) * m * n);
            // free(C_transposed);            

            if(!verify_matrix<bf16>(C_ref, C, m * n)) {
                std::cout << "Failed to pass the correctness verification against NVIDIA cuBLAS." << std::endl;
                if(m <= 512) {
                    std::cout << " Logging faulty output into " << errLogFile << "\n";
                    std::ofstream fs;
                    fs.open(errLogFile);
                    fs << "A:\n";
                    print_matrix<bf16>(A, m, k, fs);
                    fs << "B:\n";
                    print_matrix<bf16>(B, k, n, fs);
                    fs << "C_orig:\n";
                    print_matrix<bf16>(C_orig, m, n, fs);
                    fs << "Should (Cublas):\n";
                    print_matrix<bf16>(C_ref, m, n, fs);
                    fs << "Kernel out:\n";
                    print_matrix<bf16>(C, m, n, fs);
                }
                exit(EXIT_FAILURE);
            }
        }

        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++) {
            // We don't reset C_d between runs to save time and correctness check was done above
            run_kernel_bf16(kernel_num, m, n, k, alpha, A_d, B_d, beta, C_d, handle, trans_b);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.; // Convert to seconds

        long flops = 2 * m * n * k;
        std::cout << "Average elapsed time: (" << std::fixed << std::setprecision(6) << elapsed_time / repeat_times 
                  << ") s, performance: (" << std::fixed << std::setprecision(1) << (repeat_times * flops * 1e-9) / elapsed_time 
                  << ") GFLOPS. size: (" << m << ")" << std::endl;
        std::cout << std::flush;
        // make C_d and C_ref_d equal again (we modified C_d while calling our kernel for benchmarking)
        cudaCheck(cudaMemcpy(C_d, C_ref_d, sizeof(bf16) * m * n, cudaMemcpyDeviceToDevice));
    }

    // Free up CPU and GPU space
    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(C_orig);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(C_ref_d);
    cublasDestroy(handle);

    return 0;
};