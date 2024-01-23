#include <cuda.h>
#include <stdio.h>

// in_mat is M X N matrix and in_vec is N dim vector
// out_vec is M dim vector
__global__
void matrix_vector_mult_kernel(float* out_vec, float* in_mat, float* in_vec, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < M) {
        float sum = 0.0f;
        for(int j=0; j<N; j++) {
            sum += in_mat[idx*N + j] * in_vec[j];
        }
        out_vec[idx] = sum;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1)/b;
}

void matrix_vector_mult(float* O_h, float* A_h, float* b_h, int M, int N) {
    float *O_d, *A_d, *b_d;

    cudaMalloc((void **)&O_d, M*sizeof(float));
    cudaMalloc((void **)&A_d, M*N*sizeof(float));
    cudaMalloc((void **)&b_d, N*sizeof(float));

    cudaMemcpy(A_d, A_h, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32);
    dim3 gridSize(cdiv(M, blockSize.x));

    matrix_vector_mult_kernel<<<gridSize, blockSize>>>(O_d, A_d, b_d, M, N);

    cudaMemcpy(O_h, O_d, M*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(O_d);
    cudaFree(A_d);
    cudaFree(b_d);
}

int main() {
    int M = 64;
    int N = 128;

    float  *mat = new float[M*N];
    float  *vec = new float[N];
    float  *out = new float[M];

    for (int h = 0; h < M; h++){
        for (int w = 0; w < N; w++)
            mat[N * h + w] = w;
    }

    for (int w = 0; w < N; w++) {
        vec[w] = w;
    }

    matrix_vector_mult(out, mat, vec, M, N);

    for(int i=0; i<N; i++) {
        printf("%8.3f", vec[i]);
        printf(" ");
    }
    printf("\n");
    for(int i=0; i<M; i++) {
        printf("%8.3f", out[i]);
        printf(" ");
    }
    printf("\n");
    return 0;
}