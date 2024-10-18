from torch.utils.cpp_extension import load_inline
import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING']='1'

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define TILE_WIDTH 32
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_src = cuda_begin + r'''__global__
void matmul_kernel(float* O, float* A, float* B, int M, int K, int N) {
    int c = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int r = blockIdx.y * TILE_WIDTH + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0f;
    for(int i=0; i<ceil(K/(float)TILE_WIDTH); i++) {
        if(r < M and (i*TILE_WIDTH + tx) < K) {
            As[ty][tx] = A[r*K + i*TILE_WIDTH + tx];
        }
        else
            As[ty][tx] = 0.0f;

        if(c < N and (i*TILE_WIDTH + ty) < K) {
            Bs[ty][tx] = B[c*K + i*TILE_WIDTH + ty];
        }
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();

        for(int ii=0; ii<TILE_WIDTH; ii++) {
            sum += As[ty][ii] * Bs[ii][tx];
        }
        __syncthreads();
    }
    if(r < M and c < N)
        O[r*N + c] = sum;
}

torch::Tensor matmul(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0);
    int w = n.size(0);
    int k = m.size(1);
    TORCH_CHECK(k==n.size(1), "Size mismatch!");
    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(TILE_WIDTH,TILE_WIDTH);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    matmul_kernel<<<blocks, tpb>>>(
        output.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(), h, k, w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}'''

def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")

cpp_src = "torch::Tensor matmul(torch::Tensor m, torch::Tensor n);"

module = load_cuda(cuda_src, cpp_src, ['matmul'])

m1 = torch.randn(512, 1088)
m2 = torch.randn(256, 1088)

m1c, m2c = m1.contiguous().cuda(), m2.contiguous().cuda()

tr = torch.matmul(m1, m2.T)

print(tr)
print(module.matmul(m1c, m2c).cpu())
print(module.matmul(m1c, m2c).shape)

print(torch.isclose(tr, module.matmul(m1c, m2c).cpu(), atol=1e-4).all())