#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BIN_COUNT 128
#define BLOCK_SIZE 256
#define LENGTH 524288

__device__ int get_bin_device(int x) {
    return x % BIN_COUNT;
}

int get_bin(int x) {
    return x % BIN_COUNT;
}

// A is array of length N with ints
__global__ void histo_kernel(int *data, unsigned int N, unsigned int *histo) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned int histo_s[BIN_COUNT];

    for(int bin=threadIdx.x; bin<BIN_COUNT; bin+=blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    unsigned int accumulator = 0;
    int prevBinIdx = -1;

    for(unsigned int i=tid; i<N; i+=(blockDim.x * gridDim.x)) {
        int idx = get_bin_device(data[i]);
        if(idx == prevBinIdx) {
            accumulator++;
        }
        else {
            if(accumulator > 0) {
                atomicAdd(&(histo_s[prevBinIdx]), accumulator);
            }
            accumulator = 1;
            prevBinIdx = idx;
        }
    }
    if(accumulator > 0) {
        atomicAdd(&(histo_s[prevBinIdx]), accumulator);
    }

    __syncthreads();
    for(int bin=threadIdx.x; bin<BIN_COUNT; bin+=blockDim.x) {
        unsigned int binVal = histo_s[bin];
        if(binVal > 0) {
            atomicAdd(&histo[bin], binVal);
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1)/b;
}

void compute_histogram(int *data_h, unsigned int N, unsigned int *histo_h) {
    int *data_d;
    unsigned int *histo_d;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&data_d, N*sizeof(int));
    cudaMalloc((void**)&histo_d, BIN_COUNT*sizeof(unsigned int));

    cudaMemcpy(data_d, data_h, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(histo_d, histo_h, BIN_COUNT*sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int multiprocessor_count = deviceProp.multiProcessorCount;

    dim3 gridSize(16 * multiprocessor_count);
    // dim3 gridSize(cdiv(N, BLOCK_SIZE));
    dim3 blockSize(BLOCK_SIZE);

    cudaEventRecord(start);
    histo_kernel<<<gridSize, blockSize>>>(data_d, N, histo_d);
    cudaEventRecord(stop);

    cudaMemcpy(histo_h, histo_d, BIN_COUNT*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);

    cudaFree(data_d);
    cudaFree(histo_d);
}

int main() {
    srand(time(NULL));
    unsigned int N = LENGTH;

    int *data = new int[N];
    unsigned int *histo = new unsigned int[BIN_COUNT];
    unsigned int *histo_ref = new unsigned int[BIN_COUNT];

    for(unsigned int i=0; i<N; i++) {
        int random_number = rand() % 1000;
        data[i] = random_number;
    }
    for(unsigned int i=0; i<BIN_COUNT; i++) {
        histo[i] = 0;
    }
    for(unsigned int i=0; i<BIN_COUNT; i++) {
        histo_ref[i] = 0;
    }

    printf("Reference CPU\n");
    for(unsigned int i=0; i<N; i++) {
        histo_ref[get_bin(data[i])]++;
    }
    for(int i=0; i<BIN_COUNT; i++) {
        printf("%d, ", histo_ref[i]);
    }
    printf("\n");

    compute_histogram(data, N, histo);

    printf("Calculated CUDA\n");
    for(int i=0; i<BIN_COUNT; i++) {
        printf("%d, ", histo[i]);
    }
    printf("\n");

    return 0;
}