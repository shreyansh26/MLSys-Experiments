#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <functional>
#include <iomanip>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
bool is_equal(T const* data_1, T const* data_2, size_t size) {
    for (size_t i{0}; i < size; ++i) {
        if (data_1[i] != data_2[i]) {
            return false;
        }
    }
    return true;
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function, cudaStream_t stream, size_t num_repeats = 10, size_t num_warmups = 10) {
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (size_t i{0}; i < num_warmups; ++i) {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i) {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

template <typename T>
float profile_transpose_implementation(std::function<void(T*, T*, unsigned int, unsigned int, cudaStream_t)> transpose_function, unsigned int M, unsigned int N) {
    constexpr int num_repeats{100};
    constexpr int num_warmups{10};
    cudaStream_t stream;
    unsigned int const matrix_size{M * N};
    T* d_matrix;
    T* d_matrix_transposed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> const transpose_function_wrapped{std::bind(transpose_function, d_matrix, d_matrix_transposed, M, N, std::placeholders::_1)}; // stream to be supplied later hence 1 placeholder
    float const transpose_function_latency{measure_performance(transpose_function_wrapped, stream, num_repeats, num_warmups)};
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return transpose_function_latency;
}

void print_latency(std::string const& kernel_name, float latency) {
    std::cout << kernel_name << ": " << std::fixed << std::setprecision(2) << latency << " ms" << std::endl;
}

unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

template<typename T>
__global__ void transpose_naive_kernel(T *input, T *output, unsigned int M, unsigned int N) {
    int global_read_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_read_col = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_read_row < M && global_read_col < N) {
        int global_write_row = global_read_col;
        int global_write_col = global_read_row;
        output[global_write_row * M + global_write_col] = input[global_read_row * N + global_read_col];
    }
}

template<typename T, unsigned int BLOCK_SIZE_M = 32, unsigned int BLOCK_SIZE_N = 32, unsigned int SHARED_MEMORY_PADDING = 0>
__global__ void transpose_shared_kernel(T *input, T *output, unsigned int M, unsigned int N) {
    __shared__ T shared_mem[BLOCK_SIZE_M][BLOCK_SIZE_N + SHARED_MEMORY_PADDING];

    unsigned int global_read_row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int global_read_col = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int shared_write_row = threadIdx.y; // ty
    unsigned int shared_write_col = threadIdx.x; // tx

    if(global_read_row < M && global_read_col < N) {
        shared_mem[shared_write_row][shared_write_col] = input[global_read_row * N + global_read_col];
    }

    __syncthreads();

    unsigned int block_thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int shared_read_col = block_thread_idx / BLOCK_SIZE_M; // ty'
    unsigned int shared_read_row = block_thread_idx % BLOCK_SIZE_M; // tx'

    unsigned int global_write_row = shared_read_row + blockIdx.y * blockDim.y;
    unsigned int global_write_col = shared_read_col + blockIdx.x * blockDim.x;

    if(global_write_row < M && global_write_col < N) {
        output[global_write_col * M + global_write_row] = shared_mem[shared_read_row][shared_read_col];
    }
}

template<typename T, unsigned int BLOCK_SIZE_M = 32, unsigned int BLOCK_SIZE_N = 32>
__global__ void transpose_shared_swizzle_kernel(T *input, T *output, unsigned int M, unsigned int N) {
    __shared__ T shared_mem[BLOCK_SIZE_M][BLOCK_SIZE_N];

    unsigned int global_read_row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int global_read_col = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int shared_write_row = threadIdx.y; // ty
    unsigned int shared_write_col = threadIdx.x; // tx
    unsigned int shared_write_col_swizzled = (shared_write_col ^ shared_write_row) % BLOCK_SIZE_N;

    if(global_read_row < M && global_read_col < N) {
        shared_mem[shared_write_row][shared_write_col_swizzled] = input[global_read_row * N + global_read_col];
    }

    __syncthreads();

    unsigned int block_thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int shared_read_col = block_thread_idx / BLOCK_SIZE_M; // ty'
    unsigned int shared_read_row = block_thread_idx % BLOCK_SIZE_M; // tx'

    unsigned int shared_read_col_unswizzled = (shared_read_col ^ shared_read_row) % BLOCK_SIZE_N;

    unsigned int global_write_row = shared_read_row + blockIdx.y * blockDim.y;
    unsigned int global_write_col = shared_read_col + blockIdx.x * blockDim.x;

    if(global_write_row < M && global_write_col < N) {
        output[global_write_col * M + global_write_row] = shared_mem[shared_read_row][shared_read_col_unswizzled];
    }
}

template<typename T>
void transpose_naive(T *input, T *output, unsigned int M, unsigned int N, cudaStream_t stream) {   
    constexpr unsigned int BLOCK_SIZE_M = 32;
    constexpr unsigned int BLOCK_SIZE_N = 32;

    dim3 block(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 grid(cdiv(N, BLOCK_SIZE_N), cdiv(M, BLOCK_SIZE_M));
    transpose_naive_kernel<T><<<grid, block, 0, stream>>>(input, output, M, N);

    CHECK_LAST_CUDA_ERROR();
}

template<typename T>
void transpose_shared(T *input, T *output, unsigned int M, unsigned int N, cudaStream_t stream) {
    constexpr unsigned int BLOCK_SIZE_M = 32;
    constexpr unsigned int BLOCK_SIZE_N = 32;
    unsigned int const SHARED_MEMORY_PADDING = 0;

    dim3 block(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 grid(cdiv(N, BLOCK_SIZE_N), cdiv(M, BLOCK_SIZE_M));
    transpose_shared_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, SHARED_MEMORY_PADDING><<<grid, block, 0, stream>>>(input, output, M, N);

    CHECK_LAST_CUDA_ERROR();
}

template<typename T>
void transpose_shared_padding(T *input, T *output, unsigned int M, unsigned int N, cudaStream_t stream) {
    constexpr unsigned int BLOCK_SIZE_M = 32;
    constexpr unsigned int BLOCK_SIZE_N = 32;
    unsigned int const SHARED_MEMORY_PADDING = 1;

    dim3 block(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 grid(cdiv(N, BLOCK_SIZE_N), cdiv(M, BLOCK_SIZE_M));
    transpose_shared_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, SHARED_MEMORY_PADDING><<<grid, block, 0, stream>>>(input, output, M, N);

    CHECK_LAST_CUDA_ERROR();
}

template<typename T>
void transpose_shared_swizzle(T *input, T *output, unsigned int M, unsigned int N, cudaStream_t stream) {
    constexpr unsigned int BLOCK_SIZE_M = 32;
    constexpr unsigned int BLOCK_SIZE_N = 32;

    dim3 block(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 grid(cdiv(N, BLOCK_SIZE_N), cdiv(M, BLOCK_SIZE_M));
    transpose_shared_swizzle_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N><<<grid, block, 0, stream>>>(input, output, M, N);

    CHECK_LAST_CUDA_ERROR();
}

template<typename T>
bool verify_transpose(std::function<void(T*, T*, unsigned int, unsigned int, cudaStream_t)> transpose_function, unsigned int M, unsigned int N) {
    // Fixed random seed for reproducibility
    std::mt19937 gen{0};
    cudaStream_t stream;
    
    unsigned int const matrix_size{M * N};
    std::vector<T> matrix(matrix_size, 0.0f);
    std::vector<T> matrix_transposed(matrix_size, 1.0f);
    std::vector<T> matrix_transposed_reference(matrix_size, 2.0f);
    std::uniform_real_distribution<T> uniform_dist(-256, 256);
    
    for(unsigned int i{0}; i < matrix_size; ++i) {
        matrix[i] = uniform_dist(gen);
    }
    // Create the reference transposed matrix using CPU.
    for(unsigned int i{0}; i < M; ++i) {
        for (unsigned int j{0}; j < N; ++j) {
            unsigned int const from_idx = i * N + j;
            unsigned int const to_idx = j * M + i;
            matrix_transposed_reference[to_idx] = matrix[from_idx];
        }
    }

    T* d_matrix;
    T* d_matrix_transposed;

    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, matrix.data(), matrix_size * sizeof(T), cudaMemcpyHostToDevice));
    transpose_function(d_matrix, d_matrix_transposed, M, N, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(matrix_transposed.data(), d_matrix_transposed, matrix_size * sizeof(T), cudaMemcpyDeviceToHost));
    bool const correctness = is_equal(matrix_transposed.data(), matrix_transposed_reference.data(), matrix_size);
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return correctness;
}


int main() {
    for (unsigned int m{1}; m <= 64; ++m) {
        for (unsigned int n{1}; n <= 64; ++n) {
            // std::cout << "m: " << m << " n: " << n << std::endl;
            assert(verify_transpose<float>(&transpose_naive<float>, m, n));
            assert(verify_transpose<float>(&transpose_shared<float>, m, n));
            assert(verify_transpose<float>(&transpose_shared_padding<float>, m, n));
            assert(verify_transpose<float>(&transpose_shared_swizzle<float>, m, n));
        }
    }

    // M: Number of rows.
    unsigned int const M{8192};
    // N: Number of columns.
    unsigned int const N{8192};
    
    std::cout << M << " x " << N << " Matrix" << std::endl;
    float const latency_naive = profile_transpose_implementation<float>(&transpose_naive<float>, M, N);
    print_latency("Transpose Naive", latency_naive);
    float const latency_with_shm_bank_conflict = profile_transpose_implementation<float>(&transpose_shared<float>, M, N);
    print_latency("Transpose with Shared Memory Bank Conflict", latency_with_shm_bank_conflict);
    float const latency_without_shm_bank_conflict_via_padding = profile_transpose_implementation<float>(&transpose_shared_padding<float>, M, N);
    print_latency("Transpose without Shared Memory Bank Conflict via Padding", latency_without_shm_bank_conflict_via_padding);
    float const latency_without_shm_bank_conflict_via_swizzling = profile_transpose_implementation<float>(&transpose_shared_swizzle<float>, M, N);
    print_latency("Transpose without Shared Memory Bank Conflict via Swizzling", latency_without_shm_bank_conflict_via_swizzling);

    return 0;
}