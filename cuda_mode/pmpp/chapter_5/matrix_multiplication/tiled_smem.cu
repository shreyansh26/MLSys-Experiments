#include <cuda.h>
#include <stdio.h>

#define TILE_WIDTH 32

// A is M X K matrix and B is K x N matrix
// O is M x N matrix
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
        if(r < M and (i*TILE_WIDTH + tx) < K)
            As[ty][tx] = A[r*K + i*TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0f;

        if((i*TILE_WIDTH + ty) < K and c < N)
            Bs[ty][tx] = B[(i*TILE_WIDTH + ty)*N + c];
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

__global__
void matmul_kernel2(float* O, float* A, float* B, int M, int K, int N) {
    int c = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int r = blockIdx.y * TILE_WIDTH + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float smem[];
    float *As = (float*) smem;
    float *Bs = &smem[TILE_WIDTH*TILE_WIDTH];

    float sum = 0.0f;
    for(int i=0; i<ceil(K/(float)TILE_WIDTH); i++) {
        if(r < M and (i*TILE_WIDTH + tx) < K)
            As[ty*TILE_WIDTH+tx] = A[r*K + i*TILE_WIDTH + tx];
        else
            As[ty*TILE_WIDTH+tx] = 0.0f;

        if((i*TILE_WIDTH + ty) < K and c < N)
            Bs[ty*TILE_WIDTH+tx] = B[(i*TILE_WIDTH + ty)*N + c];
        else
            Bs[ty*TILE_WIDTH+tx] = 0.0f;
            
        __syncthreads();

        for(int ii=0; ii<TILE_WIDTH; ii++) {
            sum += As[ty*TILE_WIDTH + ii] * Bs[ii*TILE_WIDTH + tx];
        }
        __syncthreads();
    }
    if(r < M and c < N)
        O[r*N + c] = sum;
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1)/b;
}

void matmul(float* O_h, float* A_h, float* B_h, int M, int K, int N) {
    float *O_d;
    float *A_d; 
    float *B_d;

    cudaMalloc((void **)&O_d, M*N*sizeof(float));
    cudaMalloc((void **)&A_d, M*K*sizeof(float));
    cudaMalloc((void **)&B_d, K*N*sizeof(float));

    cudaMemcpy(A_d, A_h, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(cdiv(N, blockSize.x), cdiv(M, blockSize.y));

    matmul_kernel<<<gridSize, blockSize>>>(O_d, A_d, B_d, M, K, N);

    cudaMemcpy(O_h, O_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(O_d);
    cudaFree(A_d);
    cudaFree(B_d);
}

void matmul2(float* O_h, float* A_h, float* B_h, int M, int K, int N, unsigned int shmem_size) {
    float *O_d;
    float *A_d; 
    float *B_d;

    cudaMalloc((void **)&O_d, M*N*sizeof(float));
    cudaMalloc((void **)&A_d, M*K*sizeof(float));
    cudaMalloc((void **)&B_d, K*N*sizeof(float));

    cudaMemcpy(A_d, A_h, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(cdiv(N, blockSize.x), cdiv(M, blockSize.y));

    matmul_kernel2<<<gridSize, blockSize, shmem_size>>>(O_d, A_d, B_d, M, K, N);

    cudaMemcpy(O_h, O_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);

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

    int M = 64;
    int K = 64;
    int N = 128;

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

    // matmul(out, mat1, mat2, M, K, N);
    matmul2(out, mat1, mat2, M, K, N, shmem_size);

    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            if(j > 0)
                printf(", ");
            printf("%8.3f", out[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}