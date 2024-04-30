#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024
#define N 40960000

__global__ void segmented_reduce_sum_kernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_SIZE];

    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int idx = segment + threadIdx.x;
    unsigned int t_idx = threadIdx.x;

    input_s[t_idx] = input[idx] + input[idx + BLOCK_SIZE];

    for(unsigned int stride=blockDim.x/2; stride >= 1; stride/=2) {
        __syncthreads();
        if(t_idx < stride)
            input_s[t_idx] += input_s[t_idx + stride];
    }

    if(t_idx == 0)
        atomicAdd(output, input_s[0]);
}

unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a+b-1)/b;
}

float get_random_float() {
    float r = (float)rand() / (float)RAND_MAX;
    return r;
}

void reduce_sum_kernel(float* inp_h, float* out_h) {
    float *inp_d, *out_d;

    cudaMalloc((void**)&inp_d, N*sizeof(float));
    cudaMalloc((void**)&out_d, sizeof(float));

    cudaMemcpy(inp_d, inp_h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out_h, sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(cdiv(N, 2*blockSize.x));

    segmented_reduce_sum_kernel<<<gridSize, blockSize>>>(inp_d, out_d);

    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
}

int main() {
    srand(time(NULL));
    float *inp, out;
    inp = new float[N];

    for(unsigned int i=0; i<N; i++)
        inp[i] = get_random_float();

    double sum = 0;
    for(unsigned int i=0; i<N; i++) {
        sum += inp[i];
    }
    printf("Reference CPU sum -\t%f\n", sum);

    reduce_sum_kernel(inp, &out);

    printf("Calculated GPU sum -\t%f\n", out);

    return 0;
}