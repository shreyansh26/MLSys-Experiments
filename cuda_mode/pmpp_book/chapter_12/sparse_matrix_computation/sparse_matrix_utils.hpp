#pragma once

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
struct CSRMatrix {
    unsigned int* rowPtrs;
    unsigned int* colIdx;
    T* value;
    unsigned int R;
    unsigned int C;
    unsigned int num_nonzero;
};

template <typename T>
struct ELLMatrix {
    unsigned int* rowPtrs;
    unsigned int* colIdx;
    T* value;
    unsigned int max_nz_in_row;
    unsigned int R;
    unsigned int C;
    unsigned int num_nonzero;
};

template <typename T>
int get_num_nonzero(T*A, unsigned int R, unsigned int C);

template <typename T>
SparseMatrix<T> generate_sparse_matrix(float sparsity_ratio, unsigned int R, unsigned int C);

template <typename T>
COOMatrix<T> sparse_to_coo(SparseMatrix<T> A);

template <typename T>
SparseMatrix<T> coo_to_sparse(COOMatrix<T> A);

template <typename T>
CSRMatrix<T> sparse_to_csr(SparseMatrix<T> A);

template <typename T>
SparseMatrix<T> csr_to_sparse(CSRMatrix<T> A);

template <typename T>
ELLMatrix<T> sparse_to_ell(SparseMatrix<T> A);

template <typename T>
SparseMatrix<T> ell_to_sparse(ELLMatrix<T> A);