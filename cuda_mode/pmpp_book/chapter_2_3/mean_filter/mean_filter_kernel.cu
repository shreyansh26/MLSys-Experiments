#include <cuda.h>
#include <stdio.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__
void mean_filter_kernel(unsigned char* inp, unsigned char* out, int width, int height, int radius) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int channel = threadIdx.z;

    if(row < height && col < width) {
        int base_offset = (height * width) * channel;
        int pixSum = 0;
        int pixCnt = 0;

        for(int i=-radius; i<=radius; i++) {
            for(int j=-radius; j<=radius; j++) {
                int r = row + i;
                int c = col + j;
                if(r >= 0 and r < height and c >= 0 and c < width) {
                    int offset = base_offset + r*width + c;
                    pixSum += inp[offset];
                    pixCnt += 1;
                }
            }
        }
        out[base_offset + row*width + col] = (unsigned char)((float)pixSum / pixCnt);
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor mean_filter(torch::Tensor image, int radius) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto result = torch::empty_like(image);

    dim3 blockSize(16, 16, channels);
    dim3 gridSize(cdiv(width, blockSize.x), cdiv(height, blockSize.y));

    mean_filter_kernel<<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
        image.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width,
        height,
        radius
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}