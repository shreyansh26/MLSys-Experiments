#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>
#include <cassert>

// Forward declarations of CUDA kernel launchers
template<typename T>
void launch_attention_kernel(
    const T* Q,
    const T* K,
    const T* V,
    T* output,
    int B,
    int H,
    int N,
    int D,
    bool is_causal
);

// C++ interface function
torch::Tensor attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool is_causal
) {
    // Input validation
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D (B, H, N, D)");
    TORCH_CHECK(K.dim() == 4, "K must be 4D (B, H, N, D)");
    TORCH_CHECK(V.dim() == 4, "V must be 4D (B, H, N, D)");
    
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have the same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have the same shape");
    
    TORCH_CHECK(Q.dtype() == K.dtype(), "Q and K must have the same dtype");
    TORCH_CHECK(Q.dtype() == V.dtype(), "Q and V must have the same dtype");
    
    // Extract dimensions
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    
    // Create output tensor
    auto output = torch::empty_like(Q);
    
    // Dispatch based on dtype
    if (Q.dtype() == torch::kFloat32) {
        launch_attention_kernel<float>(
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            output.data_ptr<float>(),
            B, H, N, D, is_causal
        );
    } else if (Q.dtype() == torch::kFloat16) {
        launch_attention_kernel<__half>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            B, H, N, D, is_causal
        );
    } else if (Q.dtype() == torch::kBFloat16) {
        launch_attention_kernel<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            B, H, N, D, is_causal
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Supported: float32, float16, bfloat16");
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "Attention forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("is_causal") = false);
}

