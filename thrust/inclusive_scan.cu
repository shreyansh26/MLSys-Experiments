#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/random.h>

#include "cuda_utils.hpp"

#define NUM_ELEMENTS 1048576 // 1024 * 1024 
#define NUM_REPEATS 10

typedef thrust::device_vector<float> d_vec;
typedef thrust::host_vector<float> h_vec;

template <typename D, typename H>
H compute_scan(D A_d, D out_d, H out_h) {
    thrust::inclusive_scan(thrust::device, A_d.begin(), A_d.end(), out_d.begin());
    CHECK_LAST_CUDA_ERROR();    
    
    thrust::copy(out_d.begin(), out_d.end(), out_h.begin());
    return out_h;
}

template <typename D, typename H>
H profile_scan(D A_d, D out_d, H out_h) {
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for(int cntr=0; cntr<NUM_REPEATS; cntr++) {
        thrust::inclusive_scan(thrust::device, A_d.begin(), A_d.end(), out_d.begin());
    }
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Time taken: " << milliseconds/NUM_REPEATS << " ms\n";
    thrust::copy(out_d.begin(), out_d.end(), out_h.begin());
    return out_h;
}

template <typename H, typename T>
bool all_close(H A, H A_ref, unsigned int N, T abs_tol, double rel_tol) {
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

template <typename H>
void compute_cpu_scan(H& A_h, H& Y_cpu_ref, unsigned int N) {
    Y_cpu_ref[0] = A_h[0];
    for(unsigned int i=1; i<N; i++) {
        Y_cpu_ref[i] = Y_cpu_ref[i-1] + A_h[i];
    }
}

int main() {
    // Generate 32M random numbers
    int seed = 100;
    unsigned int N = NUM_ELEMENTS;
    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;

    thrust::default_random_engine rng(seed);
    thrust::uniform_int_distribution<float> dist(-1.0f, 1.0f);
    h_vec A_h(N);
    h_vec out_h(N);
    h_vec out_cpu_ref(N);
    thrust::generate(A_h.begin(), A_h.end(), [&rng, &dist]() {return dist(rng);});
    thrust::generate(out_h.begin(), out_h.end(), [&rng, &dist]() {return dist(rng);});
    thrust::generate(out_cpu_ref.begin(), out_cpu_ref.end(), [&rng, &dist]() {return dist(rng);});

    // Transfer data to GPU (device)
    d_vec A_d = A_h;
    d_vec out_d = out_h;

    out_h = compute_scan<d_vec, h_vec>(A_d, out_d, out_h);
    out_h = profile_scan<d_vec, h_vec>(A_d, out_d, out_h);
    compute_cpu_scan<h_vec>(A_h, out_cpu_ref, N);
    
    std::cout   << "GPU vs CPU allclose: "
                << (all_close<h_vec, float>(out_h, out_cpu_ref, N, abs_tol, rel_tol) ? "true" : "false")
                << std::endl;

    std::cout<<"Original Array\n";
    for(int i=0; i<N; i++) {
        std::cout<<A_h[i]<<" ";
    }
    std::cout<<std::endl;
    
    std::cout<<"Inclusive Scan GPU (Thrust)\n";
    for(int i=0; i<N; i++) {
        std::cout<<out_h[i]<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"Inclusive Scan CPU\n";
    for(int i=0; i<N; i++) {
        std::cout<<out_cpu_ref[i]<<" ";
    }
    std::cout<<std::endl;
}