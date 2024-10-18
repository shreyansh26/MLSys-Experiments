#include <cuda.h>
#include <stdio.h>

__global__
void vector_add_kernel(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
        C[idx] = A[idx] + B[idx];
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

void vector_add(float* A_h, float* B_h, float* C_h, int n) {
    float *A_d, *B_d, *C_d;
    size_t sz = n * sizeof(float);

    cudaMalloc((void **)&A_d, sz);
    cudaMalloc((void **)&B_d, sz);
    cudaMalloc((void **)&C_d, sz);

    cudaMemcpy(A_d, A_h, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sz, cudaMemcpyHostToDevice);

    const unsigned int threadCnt = 256;
    const unsigned int numBlocks = cdiv(n, threadCnt);
    dim3 blockSize(threadCnt);
    dim3 gridSize(numBlocks);

    vector_add_kernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, sz, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main() {
    const int n = 1000;
    float A[n];
    float B[n];
    float C[n];

    // generate some dummy vectors to add
    for (int i = 0; i < n; i += 1) {
        A[i] = float(i);
        B[i] = A[i] / 1000.0f;
    }

    vector_add(A, B, C, n);

    // print result
    for(int i = 0; i < n; i += 1) {
        if (i > 0) {
            printf(", ");
            if (i % 10 == 0) {
                printf("\n");
            }
        }
        printf("%8.3f", C[i]);
    }
    printf("\n");
    return 0;
}