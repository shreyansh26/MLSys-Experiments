#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <vector>
#include <string>
#include <type_traits>

#include "bgmv_kernel.cuh"

namespace {

template <typename T>
inline T host_cast_from_float(float x);

template <>
inline float host_cast_from_float<float>(float x) { return x; }

template <>
inline __half host_cast_from_float<__half>(float x) { return __float2half(x); }

template <>
inline __nv_bfloat16 host_cast_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

inline void check_contiguous(const torch::Tensor& t, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + " must be contiguous");
    }
}

template <typename T, int F_IN, int F_OUT>
inline void launch_impl(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor indices_i32, int64_t seqlen, int num_layers, int layer_idx, int num_lora_adapters, float scale, int batch_size) {
    T* y_ptr = reinterpret_cast<T*>(Y.data_ptr());
    const T* x_ptr = reinterpret_cast<const T*>(X.data_ptr());
    const T* w_ptr = reinterpret_cast<const T*>(W.data_ptr());
    const int* idx_ptr = indices_i32.data_ptr<int>();
    const T scaleT = host_cast_from_float<T>(scale);
    bgmv_kernel<F_IN, F_OUT, T>(y_ptr, x_ptr, w_ptr, idx_ptr, seqlen, num_layers, layer_idx, num_lora_adapters, scaleT, batch_size);
}

template <typename T>
inline void dispatch_dims_and_launch(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor indices_i32, int64_t seqlen, int num_layers, int layer_idx, int num_lora_adapters, float scale) {
    // X and Y are 2D: [B*n, F_in] and [B*n, F_out]
    const int B = static_cast<int>(X.size(0));
    const int F_in = static_cast<int>(X.size(1));
    const int F_out = static_cast<int>(Y.size(1));

    if (F_in >= F_out) {
        // Shrink/equal cases. Keep the original supported set.
        switch (F_in) {
            case 256:
                switch (F_out) {
                    case 128:  return launch_impl<T, 256, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 64:  return launch_impl<T, 256, 64>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 32:  return launch_impl<T, 256, 32>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 16:  return launch_impl<T, 256, 16>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    default: break;
                }
                break;
            case 512:
                switch (F_out) {
                    case 512: return launch_impl<T, 512, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 256: return launch_impl<T, 512, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 128: return launch_impl<T, 512, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    default: break;
                }
                break;
            case 1024:
                switch (F_out) {
                    case 1024: return launch_impl<T, 1024, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 512:  return launch_impl<T, 1024, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 256:  return launch_impl<T, 1024, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 64:  return launch_impl<T, 1024, 64>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 16:  return launch_impl<T, 1024, 16>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    default: break;
                }
                break;
            case 2048:
                switch (F_out) {
                    case 2048: return launch_impl<T, 2048, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 1024: return launch_impl<T, 2048, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 512:  return launch_impl<T, 2048, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    default: break;
                }
                break;
            case 3072:
                switch (F_out) {
                    case 2048: return launch_impl<T, 3072, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 1024: return launch_impl<T, 3072, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 512:  return launch_impl<T, 3072, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 64:  return launch_impl<T, 3072, 64>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    default: break;
                }
                break;
            case 4096:
                switch (F_out) {
                    case 4096: return launch_impl<T, 4096, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 2048: return launch_impl<T, 4096, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 1024: return launch_impl<T, 4096, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 16:  return launch_impl<T, 4096, 16>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    default: break;
                }
                break;
            case 8192:
                switch (F_out) {
                    case 8192: return launch_impl<T, 8192, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 4096: return launch_impl<T, 8192, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 2048: return launch_impl<T, 8192, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 64: return launch_impl<T, 8192, 64>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    default: break;
                }
                break;
            case 16384:
                switch (F_out) {
                    case 16384: return launch_impl<T, 16384, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 8192:  return launch_impl<T, 16384, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    case 4096:  return launch_impl<T, 16384, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                    default: break;
                }
                break;
            default:
                break;
        }
    } else { // F_in < F_out → expand
        // Expand cases must satisfy kernel tiling divisibility.
        if constexpr (std::is_same<T, float>::value) {
            switch (F_in) {
                case 4:
                    switch (F_out) {
                        case 128:   return launch_impl<T, 4, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 256:   return launch_impl<T, 4, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 4, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 4, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 4, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 4, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 4, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 4, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 8:
                    switch (F_out) {
                        case 64:    return launch_impl<T, 8, 64>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 128:   return launch_impl<T, 8, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 256:   return launch_impl<T, 8, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 8, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 8, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 8, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 8, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 8, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 8, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 16:
                    switch (F_out) {
                        case 32:    return launch_impl<T, 16, 32>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 64:    return launch_impl<T, 16, 64>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 128:   return launch_impl<T, 16, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 256:   return launch_impl<T, 16, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 16, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 16, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 16, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 16, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 16, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 16, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 32:
                    switch (F_out) {
                        case 64:    return launch_impl<T, 32, 64>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 128:   return launch_impl<T, 32, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 256:   return launch_impl<T, 32, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 32, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 32, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 32, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 32, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 32, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 32, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 64:
                    switch (F_out) {
                        case 128:   return launch_impl<T, 64, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 256:   return launch_impl<T, 64, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 64, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 64, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 64, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 64, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 64, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 64, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 128:
                    switch (F_out) {
                        case 256:   return launch_impl<T, 128, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 128, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 128, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 128, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 128, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 128, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 128, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                default:
                    break;
            }
        } else { // half/bfloat16
            switch (F_in) {
                case 8:
                    switch (F_out) {
                        case 128:   return launch_impl<T, 8, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 256:   return launch_impl<T, 8, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 8, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 8, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 8, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 8, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 8, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 8, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 16:
                    switch (F_out) {
                        case 64:    return launch_impl<T, 16, 64>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 128:   return launch_impl<T, 16, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 256:   return launch_impl<T, 16, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 16, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 16, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 16, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 16, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 16, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 16, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 32:
                    switch (F_out) {
                        case 64:    return launch_impl<T, 32, 64>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 128:   return launch_impl<T, 32, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 256:   return launch_impl<T, 32, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 32, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 32, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 32, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 32, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 32, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 32, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 64:
                    switch (F_out) {
                        case 128:   return launch_impl<T, 64, 128>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 256:   return launch_impl<T, 64, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 64, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 64, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 64, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 3072:  return launch_impl<T, 64, 3072>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 64, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 64, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 64, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 128:
                    switch (F_out) {
                        case 256:   return launch_impl<T, 128, 256>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 512:   return launch_impl<T, 128, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 128, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 128, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 128, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 128, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 128, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                case 256:
                    switch (F_out) {
                        case 512:   return launch_impl<T, 256, 512>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 1024:  return launch_impl<T, 256, 1024>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 2048:  return launch_impl<T, 256, 2048>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 4096:  return launch_impl<T, 256, 4096>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 8192:  return launch_impl<T, 256, 8192>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        case 16384: return launch_impl<T, 256, 16384>(Y, X, W, indices_i32, seqlen, num_layers, layer_idx, num_lora_adapters, scale, B);
                        default: break;
                    }
                    break;
                default:
                    break;
            }
        }
    }
    throw std::runtime_error("Unsupported (F_in=" + std::to_string(F_in) + ", F_out=" + std::to_string(F_out) + ") for bgmv");
}

} // namespace

void bgmv_forward(torch::Tensor Y,
                  torch::Tensor X,
                  torch::Tensor W,
                  torch::Tensor indices,
                  int64_t seqlen,
                  int64_t num_layers,
                  int64_t layer_idx,
                  int64_t num_lora_adapters,
                  double scale) {
    TORCH_CHECK(Y.device().is_cuda() && X.device().is_cuda() && W.device().is_cuda() && indices.device().is_cuda(), "All tensors must be CUDA");
    check_contiguous(Y, "Y");
    check_contiguous(X, "X");
    check_contiguous(W, "W");
    check_contiguous(indices, "indices");

    TORCH_CHECK(Y.dim() == 2 && X.dim() == 2 && W.dim() == 3, "Invalid dims: Y[B*n,F_out], X[B*n,F_in], W[L*num_layers,F_out,F_in]");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D [B]");

    const int64_t Bn = X.size(0); // B*n tokens
    const int64_t F_in = X.size(1);
    const int64_t F_out = Y.size(1);
    TORCH_CHECK(Y.size(0) == Bn, "Y and X must have same batch size (B*n)");
    TORCH_CHECK(W.size(1) == F_out && W.size(2) == F_in, "W must be [L*num_layers, F_out, F_in]");
    TORCH_CHECK(seqlen > 0 && (Bn % seqlen == 0), "seqlen must divide X.size(0)");
    const int64_t B = Bn / seqlen; // true batch size
    TORCH_CHECK(indices.size(0) == B, "indices length must match batch size");

    torch::Tensor indices_i32 = indices.scalar_type() == at::kInt ? indices.contiguous() : indices.toType(at::kInt).contiguous();

    const auto st = X.scalar_type();
    if (st == at::kFloat) {
        dispatch_dims_and_launch<float>(Y, X, W, indices_i32, seqlen, static_cast<int>(num_layers), static_cast<int>(layer_idx), static_cast<int>(num_lora_adapters), static_cast<float>(scale));
    } else if (st == at::kHalf) {
        dispatch_dims_and_launch<__half>(Y, X, W, indices_i32, seqlen, static_cast<int>(num_layers), static_cast<int>(layer_idx), static_cast<int>(num_lora_adapters), static_cast<float>(scale));
    } else if (st == at::kBFloat16) {
        dispatch_dims_and_launch<__nv_bfloat16>(Y, X, W, indices_i32, seqlen, static_cast<int>(num_layers), static_cast<int>(layer_idx), static_cast<int>(num_lora_adapters), static_cast<float>(scale));
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Use float32, float16, or bfloat16");
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bgmv_forward", &bgmv_forward, "BGMV forward (accumulate into Y)",
          pybind11::arg("Y"), pybind11::arg("X"), pybind11::arg("W"), pybind11::arg("indices"),
          pybind11::arg("seqlen"),
          pybind11::arg("num_layers"), pybind11::arg("layer_idx"), pybind11::arg("num_lora_adapters"), pybind11::arg("scale"));
}