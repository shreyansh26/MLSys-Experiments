#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/random.h>

#include "cuda_utils.hpp"

typedef thrust::device_vector<int> d_vec;
typedef thrust::host_vector<int> h_vec;

template <typename D>
void compute_sort(D A_d) {
    thrust::sort(A_d.begin(), A_d.end());
    CHECK_LAST_CUDA_ERROR();    
}

template <typename D>
void profile_sort(D A_d) {
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    thrust::sort(A_d.begin(), A_d.end());
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Time taken: " << milliseconds << " ms\n";
}

int main() {
    // Generate 32M random numbers
    int seed = 100;
    unsigned int N = 1000;
    thrust::default_random_engine rng(seed);
    thrust::uniform_int_distribution<int> dist;
    h_vec A_h(N);
    thrust::generate(A_h.begin(), A_h.end(), [&rng, &dist]() {return dist(rng);});

    // Transfer data to GPU (device)
    d_vec A_d = A_h;

    profile_sort<d_vec>(A_d);

    // Transfer data back to host
    thrust::copy(A_d.begin(), A_d.end(), A_h.begin());

    for(int i=0; i<N; i++) {
        std::cout<<A_h[i]<<" ";
    }
    std::cout<<std::endl;
}