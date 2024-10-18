#include <cuda.h>
#include <stdio.h>

// A is M X K matrix and B is K x N matrix
// O is M x N matrix
__global__
void matmul_kernel(float* O, float* A, float* B, int M, int K, int N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if(r < M && c < N) {
        float sum = 0.0f;
        for(int k=0; k<K; k++) {
            sum += A[r*K + k] * B[k*N + c];
        }
        O[r*N + c] = sum;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1)/b;
}

void matmul(float* O_h, float* A_h, float* B_h, int M, int K, int N, bool bench=true) {
    float *O_d;
    float *A_d; 
    float *B_d;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&O_d, M*N*sizeof(float));
    cudaMalloc((void **)&A_d, M*K*sizeof(float));
    cudaMalloc((void **)&B_d, K*N*sizeof(float));

    cudaMemcpy(A_d, A_h, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize(cdiv(N, blockSize.x), cdiv(M, blockSize.y));

    if(bench) {
        for(int i=0; i<20; i++) {
            matmul_kernel<<<gridSize, blockSize>>>(O_d, A_d, B_d, M, K, N);
        }
    }

    cudaEventRecord(start);
    matmul_kernel<<<gridSize, blockSize>>>(O_d, A_d, B_d, M, K, N);
    cudaEventRecord(stop);

    cudaMemcpy(O_h, O_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);

    cudaFree(O_d);
    cudaFree(A_d);
    cudaFree(B_d);
}

int main() {
    int M = 2048;
    int K = 2047;
    int N = 2048;

    float  *mat1 = new float[M*K];
    float  *mat2 = new float[K*N];
    float  *out = new float[M*N];

    for (int h = 0; h < M; h++){
        for (int w = 0; w < K; w++)
            mat1[K * h + w] = w;
    }

    for (int h = 0; h < K; h++){
        for (int w = 0; w < N; w++)
            mat2[N * h + w] = w;
    }

    matmul(out, mat1, mat2, M, K, N);

    // for(int i=0; i<M; i++) {
    //     for(int j=0; j<K; j++) {
    //         if(j > 0)
    //             printf(", ");
    //         printf("%8.3f", mat1[i*K+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // for(int i=0; i<M; i++) {
    //     for(int j=0; j<N; j++) {
    //         if(j > 0)
    //             printf(", ");
    //         printf("%8.3f", out[i*N+j]);
    //     }
    //     printf("\n");
    // }
    printf("\n");
    return 0;
}