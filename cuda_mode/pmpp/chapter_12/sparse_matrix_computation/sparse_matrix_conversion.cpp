#include <iostream>
#include <vector>

#include "cuda_utils.hpp"

#define ROWS 10
#define COLUMNS 10
#define SPARSITY_RATIO 0.1

template<typename T>
struct SparseMatrix {
    T* mat;
    unsigned int R;
    unsigned int C;
    unsigned int num_nonzero;
};

template <typename T>
struct COOMatrix {
    unsigned int* rowIdx;
    unsigned int* colIdx;
    T* value;
    unsigned int R;
    unsigned int C;
    unsigned int num_nonzero;
};

template <typename T>
int get_num_nonzero(T*A, unsigned int R, unsigned int C) {
    unsigned int num_nonzero = 0;
    for(unsigned int row=0; row<R; row++) {
        for(unsigned int col=0; col<C; col++) {
            if(A[row*C + col] != 0) {
                num_nonzero++;
            }
        }
    }
    return num_nonzero;
}

template <typename T>
SparseMatrix<T> generate_sparse_matrix(float sparsity_ratio, unsigned int R, unsigned int C) {
    T* A = (T*)malloc(R*C*sizeof(T));

    random_initialize_sparse_matrix<T>(A, sparsity_ratio, R, C);
    unsigned int num_nonzero = get_num_nonzero(A, R, C);

    SparseMatrix<T> sparse_matrix = {A, R, C, num_nonzero};
    return sparse_matrix;
}

template <typename T>
COOMatrix<T> sparse_to_coo(SparseMatrix<T> A) {
    unsigned int* rowIdx = (unsigned int*)malloc(A.num_nonzero * sizeof(unsigned int));
    unsigned int* colIdx = (unsigned int*)malloc(A.num_nonzero * sizeof(unsigned int));
    T* value = (T*)malloc(A.num_nonzero * sizeof(T));

    int cntr = 0;
    for(unsigned int row=0; row<A.R; row++) {
        for(unsigned int col=0; col<A.C; col++) {
            if(A.mat[row*A.C + col] != 0) {
                rowIdx[cntr] = row;
                colIdx[cntr] = col;
                value[cntr] = A.mat[row*A.C + col];
                cntr++;
            }
        }
    }
    COOMatrix<T> coo_matrix = {rowIdx, colIdx, value, A.R, A.C, A.num_nonzero};
    return coo_matrix;
}

template <typename T>
SparseMatrix<T> coo_to_sparse(COOMatrix<T> A) {
    T* mat = (T*)malloc(A.R * A.C * sizeof(T));

    std::fill(mat, mat+A.R*A.C, static_cast<T>(0));

    for(int i=0; i<A.num_nonzero; i++) {
        mat[A.rowIdx[i]*A.C + A.colIdx[i]] = A.value[i];
    }

    SparseMatrix<T> sparse_matrix = {mat, A.R, A.C, A.num_nonzero};
    return sparse_matrix;
}

int main() {
    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;

    SparseMatrix<float> sparse_matrix = generate_sparse_matrix<float>(SPARSITY_RATIO, ROWS, COLUMNS);
    
    print_matrix<float>(sparse_matrix.mat, sparse_matrix.R, sparse_matrix.C, "Sparse Matrix");    

    // COO Representation
    COOMatrix<float> coo_matrix = sparse_to_coo<float>(sparse_matrix);
    print_array<unsigned int>(coo_matrix.rowIdx, sparse_matrix.num_nonzero, "rowIdx");
    print_array<unsigned int>(coo_matrix.colIdx, sparse_matrix.num_nonzero, "colIdx");
    print_array<float>(coo_matrix.value, sparse_matrix.num_nonzero, "value");

    SparseMatrix<float> sparse_matrix_from_coo = coo_to_sparse<float>(coo_matrix);
    print_matrix<float>(sparse_matrix_from_coo.mat, sparse_matrix_from_coo.R, sparse_matrix_from_coo.C, "Sparse Matric from COO");

    std::cout   << "(Original Sparse) vs (COO->Sparse) allclose: "
                << (all_close<float>(sparse_matrix.mat, sparse_matrix_from_coo.mat, sparse_matrix.R * sparse_matrix.C, abs_tol, rel_tol) ? "true" : "false")
                << std::endl; 
}