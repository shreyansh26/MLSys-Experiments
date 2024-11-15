#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

#include <run.cuh>
#include <matrix_utils.cuh>

#define cudaCheck(val) check_cuda((val), #val, __FILE__, __LINE__)

const std::string errLogFile = "matrixValidationFailure.txt";

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
    if (getenv("DEVICE") != NULL) {
        deviceIdx = atoi(getenv("DEVICE"));
    }
    cudaCheck(cudaSetDevice(deviceIdx));

    printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

    // print some device info
    // CudaDeviceInfo();

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
    std::vector<int> SIZE = {2, 128, 256, 512, 1024, 2048, 4096};

    long m, n, k, max_size;
    max_size = SIZE[SIZE.size() - 1];
    std::cout << "Max size: " << max_size << std::endl;

    float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr, *C_orig = nullptr; // host matrices
    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr, *C_ref_d = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_orig = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);
    memcpy(C_orig, C, sizeof(float) * max_size * max_size);

    cudaCheck(cudaMalloc((void **)&A_d, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&B_d, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&C_d, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&C_ref_d, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(A_d, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(B_d, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(C_d, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(C_ref_d, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    int repeat_times = 50;
    for(int size : SIZE) {
        m = n = k = size;

        std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha << ", beta: " << beta << std::endl;
        // Verify the correctness of the calculation, and execute it once before the
        // kernel function timing to avoid cold start errors
        if(kernel_num != 0) {
            run_kernel(0, m, n, k, alpha, A_d, B_d, beta, C_ref_d, handle); // cuBLAS
            run_kernel(kernel_num, m, n, k, alpha, A_d, B_d, beta, C_d, handle); // Executes the kernel, modifies the result matrix
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
            cudaMemcpy(C, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref, C_ref_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

            if(!verify_matrix(C_ref, C, m * n)) {
                std::cout << "Failed to pass the correctness verification against NVIDIA cuBLAS." << std::endl;
                if(m <= 128) {
                    std::cout << " Logging faulty output into " << errLogFile << "\n";
                    std::ofstream fs;
                    fs.open(errLogFile);
                    fs << "A:\n";
                    print_matrix(A, m, k, fs);
                    fs << "B:\n";
                    print_matrix(B, k, n, fs);
                    fs << "C_orig:\n";
                    print_matrix(C_orig, m, n, fs);
                    fs << "Should (Cublas):\n";
                    print_matrix(C_ref, m, n, fs);
                    fs << "Kernel out:\n";
                    print_matrix(C, m, n, fs);
                }
                exit(EXIT_FAILURE);
            }
        }

        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++) {
            // We don't reset C_d between runs to save time and correctness check was done above
            run_kernel(kernel_num, m, n, k, alpha, A_d, B_d, beta, C_d, handle);
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
        cudaCheck(cudaMemcpy(C_d, C_ref_d, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
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