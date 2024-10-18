#include <iostream>
#include <random>
#include <cuda_runtime.h>

#include "cuda_utils.hpp"

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void check_cuda_last(const char* const file, const int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
void random_initialize_array(T* A, unsigned int N, unsigned int seed) {
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() {return dis(eng);};

    for(unsigned int i=0; i<N; i++)
        A[i] = static_cast<T>(rand());
}

template <typename T>
void print_array(T* A, unsigned int N, std::string msg) {
    std::cout<<msg<<std::endl;
    for(unsigned int i=0; i<N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout<<std::endl;
}

template <typename T>
bool all_close(T* A, T* A_ref, unsigned int N, T abs_tol, double rel_tol) {
    bool is_close = true;
    for(unsigned int i=0; i<N; i++) {
        double A_val = static_cast<double>(A[i]);
        double A_ref_val = static_cast<double>(A_ref[i]);

        double diff_val = std::abs(A_val - A_ref_val);

        if(diff_val > std::max(static_cast<double>(abs_tol), static_cast<double>(std::abs(A_ref_val)) * rel_tol)) {
            std::cout   << "A[" << i << "] = " << A_val
                        << ", A_ref[" << i << "] = " << A_ref_val
                        << ", Abs Diff Threshold: "
                        << static_cast<double>(abs_tol)
                        << ", Rel->Abs Diff Threshold: "
                        << static_cast<double>(static_cast<double>(std::abs(A_ref_val)) * rel_tol)
                        << std::endl;
            is_close = false;
            return is_close;
        }
    }
    return is_close;
}

template void random_initialize_array<float>(float*, unsigned int, unsigned int);
template void print_array<float>(float*, unsigned int, std::string);
template bool all_close<float>(float*, float*, unsigned int, float, double);

template void random_initialize_array<double>(double*, unsigned int, unsigned int);
template void print_array<double>(double*, unsigned int, std::string);
template bool all_close<double>(double*, double*, unsigned int, double, double);