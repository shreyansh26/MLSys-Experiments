#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define COARSENING_FACTOR 2
#define BLOCK_SIZE 1024
#define N 3432342

__global__ void segmented_coarsening_reduce_sum_kernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_SIZE];

    unsigned int segment = 2*COARSENING_FACTOR*blockDim.x*blockIdx.x;
    unsigned int idx = segment + threadIdx.x;
    unsigned int t_idx = threadIdx.x;

    float sum = 0.0;
    if(idx < N)
        sum = input[idx];

    for(int c=1; c<COARSENING_FACTOR*2; c++) {
        if((idx + c*BLOCK_SIZE) < N)
            sum += input[idx + c*BLOCK_SIZE];
    }
    
    input_s[t_idx] = sum;
    for(unsigned int stride=blockDim.x/2; stride >= 1; stride/=2) {
        __syncthreads();
        if(t_idx < stride)
            input_s[t_idx] += input_s[t_idx + stride];
    }

    if(t_idx == 0)
        atomicAdd(output, input_s[0]);
}

__device__ float atomicMaxf(float* address, float val) {
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}

__global__ void segmented_coarsening_reduce_max_kernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_SIZE];

    unsigned int segment = 2*COARSENING_FACTOR*blockDim.x*blockIdx.x;
    unsigned int idx = segment + threadIdx.x;
    unsigned int t_idx = threadIdx.x;

    float maxVal = -100000.0;
    if(idx < N)
        maxVal = input[idx];

    for(int c=1; c<COARSENING_FACTOR*2; c++) {
        if((idx + c*BLOCK_SIZE) < N)
            maxVal = max(maxVal, input[idx + c*BLOCK_SIZE]);
    }
    
    input_s[t_idx] = maxVal;
    for(unsigned int stride=blockDim.x/2; stride >= 1; stride/=2) {
        __syncthreads();
        if(t_idx < stride)
            input_s[t_idx] = max(maxVal, input_s[t_idx + stride]);
    }

    if(t_idx == 0)
        atomicMaxf(output, input_s[0]);
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
    dim3 gridSize(cdiv(N, 2*COARSENING_FACTOR*blockSize.x));

    segmented_coarsening_reduce_sum_kernel<<<gridSize, blockSize>>>(inp_d, out_d);

    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
}

void reduce_max_kernel(float* inp_h, float* out_h) {
    float *inp_d, *out_d;

    cudaMalloc((void**)&inp_d, N*sizeof(float));
    cudaMalloc((void**)&out_d, sizeof(float));

    cudaMemcpy(inp_d, inp_h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out_h, sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(cdiv(N, 2*COARSENING_FACTOR*blockSize.x));

    segmented_coarsening_reduce_max_kernel<<<gridSize, blockSize>>>(inp_d, out_d);

    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
}

int main() {
    srand(time(NULL));
    float *inp, out, maxGPU;
    inp = new float[N];

    for(unsigned int i=0; i<N; i++)
        inp[i] = get_random_float();

    double sum = 0;
    for(unsigned int i=0; i<N; i++) {
        sum += inp[i];
    }
    printf("Reference CPU sum -\t%f\n", sum);

    reduce_sum_kernel(inp, &out);

    cudaError_t err = cudaGetLastError();        // Get error code

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    printf("Calculated GPU sum -\t%f\n", out);


    double maxVal = -10000.0;
    for(unsigned int i=0; i<N; i++) {
        maxVal = max(maxVal, (double)inp[i]);
    }
    printf("Reference CPU max -\t%lf\n", maxVal);

    reduce_max_kernel(inp, &maxGPU);

    err = cudaGetLastError();        // Get error code

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    printf("Calculated GPU max -\t%f\n", maxGPU);

    return 0;
}