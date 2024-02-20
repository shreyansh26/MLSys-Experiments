#include <cuda.h>
#include <stdio.h>

#define TILE_WIDTH 64

// A is M X K matrix and B is N x K matrix
// O is A @ B.T and shape is M x N matrix
__global__
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
    cudaMalloc((void **)&B_d, N*K*sizeof(float));

    cudaMemcpy(A_d, A_h, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N*K*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
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
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    printf("Max threads per block %d\n", devProp.maxThreadsPerBlock);
    printf("Max shared memory per block %lu\n", devProp.sharedMemPerBlock);
    size_t requiredSize = devProp.maxThreadsPerBlock * 2 * sizeof(float); 
    size_t size = min(devProp.sharedMemPerBlock, requiredSize);
    printf("Required size = %lu, Shared Mem size taken = %lu\n", requiredSize, size);

    unsigned int shmem_size = TILE_WIDTH * TILE_WIDTH * 2 * sizeof(float);
    printf("shmem_size = %u\n", shmem_size);

    // int M = 64;
    // int K = 64;
    // int N = 128;
    int M = 2048;
    int K = 4096;
    int N = 1024;

    float  *mat1 = new float[M*K];
    float  *mat2 = new float[N*K];
    float  *out = new float[M*N];

    printf("M: %d\nN: %d\nK: %d\n", M, N, K);
    for (int h = 0; h < M; h++){
        for (int w = 0; w < K; w++)
            mat1[K * h + w] = w;
    }

    for (int h = 0; h < N; h++){
        for (int w = 0; w < K; w++)
            mat2[K * h + w] = w;
    }

    matmul(out, mat1, mat2, M, K, N);

    // for(int i=0; i<M; i++) {
    //     for(int j=0; j<N; j++) {
    //         if(j > 0)
    //             printf(", ");
    //         printf("%8.3f", out[i*N+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    return 0;
}