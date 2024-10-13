#include <stdio.h>

__global__ void basic_reduction_ballot(int* input, int* output, int numElements) {
    unsigned int mask = __ballot_sync(0xffffffff, threadIdx.x < numElements);
    int laneId = threadIdx.x & 0x1f;
    int val;
    if(threadIdx.x < numElements) {
        val = input[threadIdx.x];
        for(int offset=16; offset>0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if(laneId == 0) {
            output[blockIdx.x] = val;
        }
    }
}

int main() {
    int* h_inp, *h_out;
    int* d_inp, *d_out;
    int numElements = 29;
    int numBlocks = (numElements + 31) / 32;

    h_inp = (int*)malloc(numElements * sizeof(int));
    h_out = (int*)malloc(numBlocks * sizeof(int));
    cudaMalloc((void**)&d_inp, numElements * sizeof(int));
    cudaMalloc((void**)&d_out, numBlocks * sizeof(int));

    for(int i=0; i<numElements; i++) {
        h_inp[i] = 1;
    }

    cudaMemcpy(d_inp, h_inp, numElements * sizeof(int), cudaMemcpyHostToDevice);
    basic_reduction_ballot<<<numBlocks, 32>>>(d_inp, d_out, numElements);
    cudaMemcpy(h_out, d_out, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<numBlocks; i++) {
        printf("Block %d: %d\n", i, h_out[i]);
    }

    free(h_inp);
    free(h_out);
    cudaFree(d_inp);
    cudaFree(d_out);

    return 0;
}