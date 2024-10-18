#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BIN_COUNT 128

__device__ int get_bin_device(int x) {
    return x % BIN_COUNT;
}

int get_bin(int x) {
    return x % BIN_COUNT;
}

// A is array of length N with ints
__global__ void histo_kernel(int *data, unsigned int N, unsigned int *histo) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < N) {
        int idx = get_bin_device(data[i]);
        atomicAdd(&(histo[idx]), 1);
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

    dim3 blockSize(1024);
    dim3 gridSize(cdiv(N, blockSize.x));

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
    unsigned int N = 524288;

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