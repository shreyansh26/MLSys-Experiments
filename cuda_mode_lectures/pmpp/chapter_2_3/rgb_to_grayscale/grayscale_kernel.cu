#include <cuda.h>
#include <stdio.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ 
void rgb_to_grayscale_kernel(unsigned char* inp, unsigned char* out, int width, int height) {
    const int channels = 3;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < height && col < width) {
        int inp_offset = ((row * width) + col) * channels;
        int out_offset = (row * width + col);

        unsigned char r = inp[inp_offset + 0];
        unsigned char g = inp[inp_offset + 1];
        unsigned char b = inp[inp_offset + 2];
        
        out[out_offset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b); 
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 blockSize(16, 16);
    dim3 gridSize(cdiv(width, blockSize.x), cdiv(height, blockSize.y));

    rgb_to_grayscale_kernel<<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
        image.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width,
        height
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}