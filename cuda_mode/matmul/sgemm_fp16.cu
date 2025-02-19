#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

#include <run_fp16.cuh>
#include <matrix_utils.cuh>

#define cudaCheck(val) check_cuda((val), #val, __FILE__, __LINE__)

const std::string errLogFile = "matrixValidationFailure.txt";

void gemm_cpu_ref(int m, int n, int k, float alpha, half *A, half *B, float beta, half *C) {
    float alpha_f = __half2float(alpha);
    float beta_f = __half2float(beta);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float c = beta_f * __half2float(C[i * n + j]);
            for(int l = 0; l < k; l++) {
                c += alpha_f * __half2float(A[i * k + l]) * __half2float(B[l * n + j]);
            }
            C[i * n + j] = __float2half(c);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)" << std::endl;
        exit(EXIT_FAILURE);
    }

    // get kernel number
    int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 12) {
        std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
        exit(EXIT_FAILURE);
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
    std::vector<int> SIZE = {256, 4096};

    long m, n, k, max_size;
    max_size = SIZE[SIZE.size() - 1];
    std::cout << "Max size: " << max_size << std::endl;

    float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

    half *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr, *C_ref = nullptr, *C_orig = nullptr, *C_cpu_ref = nullptr; // host matrices
    half *A_d = nullptr, *B_d = nullptr, *C_d = nullptr, *C_ref_d = nullptr, *D_d = nullptr; // device matrices

    A = (half *)malloc(sizeof(half) * max_size * max_size);
    B = (half *)malloc(sizeof(half) * max_size * max_size);
    C = (half *)malloc(sizeof(half) * max_size * max_size);
    D = (half *)malloc(sizeof(half) * max_size * max_size);
    C_orig = (half *)malloc(sizeof(half) * max_size * max_size);
    C_ref = (half *)malloc(sizeof(half) * max_size * max_size);
    C_cpu_ref = (half *)malloc(sizeof(half) * max_size * max_size);

    randomize_matrix<half>(A, max_size * max_size);
    randomize_matrix<half>(B, max_size * max_size);
    randomize_matrix<half>(C, max_size * max_size);
    randomize_matrix<half>(D, max_size * max_size);
    memcpy(C_orig, C, sizeof(half) * max_size * max_size);
    memcpy(C_cpu_ref, C, sizeof(half) * max_size * max_size);

    cudaCheck(cudaMalloc((void **)&A_d, sizeof(half) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&B_d, sizeof(half) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&C_d, sizeof(half) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&C_ref_d, sizeof(half) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&D_d, sizeof(half) * max_size * max_size));
    cudaCheck(cudaMemcpy(A_d, A, sizeof(half) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(B_d, B, sizeof(half) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(C_d, C, sizeof(half) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(C_ref_d, C, sizeof(half) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(D_d, D, sizeof(half) * max_size * max_size, cudaMemcpyHostToDevice));
    int repeat_times = 10;
    for(int size : SIZE) {
        m = n = k = size;

        std::cout << "dimensions(m=n=k) " << m << ", alpha: " << (float)alpha << ", beta: " << (float)beta << std::endl;
        // Verify the correctness of the calculation, and execute it once before the
        // kernel function timing to avoid cold start errors

        // For kernel 0 i.e. cuBLAS, we only check correctness for small matrices as CPU does not support fp16
        if(kernel_num == 0 and m <= 256) {
            run_kernel_fp16(0, m, n, k, alpha, A_d, B_d, beta, C_ref_d, D_d, handle); // cuBLAS
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
            cudaMemcpy(C_ref, C_ref_d, sizeof(half) * m * n, cudaMemcpyDeviceToHost);
                        
            gemm_cpu_ref(m, n, k, alpha, A, B, beta, C_cpu_ref); // perform reference calculation on CPU

            if(!verify_matrix<half>(C_ref, C_cpu_ref, m * n)) {
                std::cout << "Failed to pass the correctness verification against CPU computation." << std::endl;
                if(m <= 128) {
                    std::cout << " Logging faulty output into " << errLogFile << "\n";
                    std::ofstream fs;
                    fs.open(errLogFile);
                    fs << "A:\n";
                    print_matrix<half>(A, m, k, fs);
                    fs << "B:\n";
                    print_matrix<half>(B, k, n, fs);
                    fs << "C_orig:\n";
                    print_matrix<half>(C_orig, m, n, fs);
                    fs << "Should (CPU):\n";
                    print_matrix<half>(C_cpu_ref, m, n, fs);
                    fs << "Kernel out:\n";
                    print_matrix(C_ref, m, n, fs);
                }
                exit(EXIT_FAILURE);
            }
        }
        
        // For other kernels, larger errors at certain indices start to show up at 1024
        if(kernel_num != 0 and m <= 512) {
            run_kernel_fp16(0, m, n, k, alpha, A_d, B_d, beta, C_ref_d, D_d, handle); // cuBLAS
            run_kernel_fp16(kernel_num, m, n, k, alpha, A_d, B_d, beta, C_d, D_d, handle); // Executes the kernel, modifies the result matrix
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
            cudaMemcpy(D, D_d, sizeof(half) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref, C_ref_d, sizeof(half) * m * n, cudaMemcpyDeviceToHost);

            if(!verify_matrix<half>(C_ref, D, m * n)) {
                std::cout << "Failed to pass the correctness verification against NVIDIA cuBLAS." << std::endl;
                if(m <= 128) {
                    std::cout << " Logging faulty output into " << errLogFile << "\n";
                    std::ofstream fs;
                    fs.open(errLogFile);
                    fs << "A:\n";
                    print_matrix<half>(A, m, k, fs);
                    fs << "B:\n";
                    print_matrix<half>(B, k, n, fs);
                    fs << "C_orig:\n";
                    print_matrix<half>(C_orig, m, n, fs);
                    fs << "Should (Cublas):\n";
                    print_matrix<half>(C_ref, m, n, fs);
                    fs << "Kernel out:\n";
                    print_matrix<half>(D, m, n, fs);
                }
                exit(EXIT_FAILURE);
            }
        }

        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++) {
            // We don't reset C_d/D_d between runs to save time and correctness check was done above
            run_kernel_fp16(kernel_num, m, n, k, alpha, A_d, B_d, beta, C_d, D_d, handle);
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
        cudaCheck(cudaMemcpy(C_d, C_ref_d, sizeof(half) * m * n, cudaMemcpyDeviceToDevice));
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