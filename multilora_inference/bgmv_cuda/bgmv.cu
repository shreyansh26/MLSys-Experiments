#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "bgmv_kernel.cuh"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at "         \
                      << __FILE__ << ":" << __LINE__ << std::endl;                      \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

int main() {
    constexpr int B = 16;            // batch size
    constexpr int num_layers = 8;   // layers per adapter
    constexpr int L = 10;           // number of adapters
    constexpr int layer_idx = 2;    // selected layer within adapter
    constexpr float scale = 0.25f;  // alpha / r

    // Shrink case: F_in > F_out
    constexpr int F_in = 1024;
    constexpr int F_out = 16;

    using T = float;  // Use float for simplicity and broad device support

    std::mt19937 rng(1023);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Host buffers
    std::vector<T> hX(static_cast<size_t>(B) * F_in);
    std::vector<T> hW(static_cast<size_t>(L) * num_layers * F_out * F_in);
    std::vector<T> hY(static_cast<size_t>(B) * F_out, T(0));
    std::vector<int> hIndices(B);

    for(auto& x : hX) 
        x = static_cast<T>(dist(rng));
    for(auto& w : hW) 
        w = static_cast<T>(dist(rng));
    for (int b = 0; b < B; ++b) 
        hIndices[b] = rng() % L;

    // Device buffers
    T* dX = nullptr; 
    T* dW = nullptr; 
    T* dY = nullptr; 
    int* dIndices = nullptr;

    CUDA_CHECK(cudaMalloc(&dX, hX.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dW, hW.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dY, hY.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dIndices, hIndices.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, hY.data(), hY.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dIndices, hIndices.data(), hIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Launch shrink kernel via the templated wrapper
    bgmv_kernel<F_in, F_out, T>(dY, dX, dW, dIndices, num_layers, layer_idx, static_cast<T>(scale), B);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(hY.data(), dY, hY.size() * sizeof(T), cudaMemcpyDeviceToHost));

    // Simple printout of Y to confirm execution
    std::cout << "BGMV CUDA\n";
    std::cout << "Y (first few elements):\n";
    for (int b = 0; b < B; ++b) {
        std::cout << "b=" << b << ": ";
        for (int j = 0; j < std::min(F_out, 8); ++j) {
            std::cout << hY[static_cast<size_t>(b) * F_out + j] << (j + 1 < std::min(F_out, 8) ? ", " : "\n");
        }
    }

    // Optional: basic CPU verification (float)
    std::vector<float> refY(static_cast<size_t>(B) * F_out, 0.0f);
    for (int b = 0; b < B; ++b) {
        const int idx = hIndices[b] * num_layers + layer_idx;
        for (int j = 0; j < F_out; ++j) {
            float acc = 0.0f;
            const size_t wBase = static_cast<size_t>(idx) * F_out * F_in + static_cast<size_t>(j) * F_in;
            const size_t xBase = static_cast<size_t>(b) * F_in;
            for (int i = 0; i < F_in; ++i) {
                acc += static_cast<float>(hW[wBase + i]) * static_cast<float>(hX[xBase + i]) * scale;
            }
            refY[static_cast<size_t>(b) * F_out + j] += acc;
        }
    }

    std::cout << "BGMV CPU\n";
    std::cout << "Ref Y (first few elements):\n";
    for (int b = 0; b < B; ++b) {
        std::cout << "b=" << b << ": ";
        for (int j = 0; j < std::min(F_out, 8); ++j) {
            std::cout << refY[static_cast<size_t>(b) * F_out + j] << (j + 1 < std::min(F_out, 8) ? ", " : "\n");
        }
    }

    // Compute max abs diff
    float max_abs_diff = 0.0f;
    for (size_t k = 0; k < refY.size(); ++k) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(refY[k] - static_cast<float>(hY[k])));
    }
    std::cout << "Max abs diff vs CPU: " << max_abs_diff << "\n";

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dIndices));

    return 0;
}