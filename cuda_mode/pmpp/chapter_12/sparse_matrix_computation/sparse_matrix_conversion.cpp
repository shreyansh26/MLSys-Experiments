#include <iostream>
#include <vector>
#include <climits>

#include "cuda_utils.hpp"

#define ROWS            10
#define COLUMNS         10
#define SPARSITY_RATIO  0.2
#define PAD_VAL         (1<<20)

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

    for(unsigned int i=0; i<A.num_nonzero; i++) {
        mat[A.rowIdx[i]*A.C + A.colIdx[i]] = A.value[i];
    }

    SparseMatrix<T> sparse_matrix = {mat, A.R, A.C, A.num_nonzero};
    return sparse_matrix;
}

template <typename T>
CSRMatrix<T> sparse_to_csr(SparseMatrix<T> A) {
    unsigned int* rowPtrs = (unsigned int*)malloc((A.R+1) * sizeof(unsigned int));
    unsigned int* colIdx = (unsigned int*)malloc(A.num_nonzero * sizeof(unsigned int));
    T* value = (T*)malloc(A.num_nonzero * sizeof(T));

    unsigned int row_cntr = 0;
    unsigned int cntr = 0;
    for(unsigned int row=0; row<A.R; row++) {
        rowPtrs[row_cntr] = cntr;
        for(unsigned int col=0; col<A.C; col++) {
            if(A.mat[row*A.C + col] != 0) {
                colIdx[cntr] = col;
                value[cntr] = A.mat[row*A.C + col];
                cntr++;
            }
        }
        row_cntr++;
    }
    rowPtrs[row_cntr] = cntr;
    CSRMatrix<T> csr_matrix = {rowPtrs, colIdx, value, A.R, A.C, A.num_nonzero};
    return csr_matrix;
}

template <typename T>
SparseMatrix<T> csr_to_sparse(CSRMatrix<T> A) {
    T* mat = (T*)malloc(A.R * A.C * sizeof(T));

    std::fill(mat, mat+A.R*A.C, static_cast<T>(0));

    for(unsigned int i=0; i<A.R; i++) {
        for(unsigned int j=A.rowPtrs[i]; j<A.rowPtrs[i+1]; j++) {
            unsigned int row_idx = i;
            unsigned int col_idx = A.colIdx[j];
            mat[row_idx*A.C + col_idx] = A.value[j];
        }
    }

    SparseMatrix<T> sparse_matrix = {mat, A.R, A.C, A.num_nonzero};
    return sparse_matrix;
}

template <typename T>
ELLMatrix<T> sparse_to_ell(SparseMatrix<T> A) {
    unsigned int max_nz_in_row = 0;

    for(unsigned int i=0; i<A.R; i++) {
        unsigned int nz_in_row = 0;
        for(unsigned int j=0; j<A.C; j++) {
            if(A.mat[i*A.C + j] != 0) {
                nz_in_row++;
            }
        }
        max_nz_in_row = std::max(max_nz_in_row, nz_in_row);
    }
    unsigned int* rowPtrs = (unsigned int*)malloc((A.R+1) * sizeof(unsigned int));
    unsigned int* colIdx = (unsigned int*)malloc((A.R * max_nz_in_row) * sizeof(unsigned int));
    T* value = (T*)malloc((A.R * max_nz_in_row) * sizeof(T));

    std::fill(colIdx, colIdx+(A.R * max_nz_in_row), static_cast<unsigned int>(PAD_VAL));
    std::fill(value, value+(A.R * max_nz_in_row), static_cast<T>(PAD_VAL));

    unsigned int row_cntr = 0;
    unsigned int col_cntr = 0;
    unsigned int cntr = 0;
    for(unsigned int row=0; row<A.R; row++) {
        rowPtrs[row_cntr] = cntr;
        for(unsigned int col=0; col<A.C; col++) {
            if(A.mat[row*A.C + col] != 0) {
                // i, j -> row, col_cntr
                // => j, i -> col_cntr, row
                // colIdx[row*max_nz_in_row + col_cntr] = col;
                // value[row*max_nz_in_row + col_cntr] = A.mat[row*A.C + col];
                colIdx[col_cntr*A.R + row] = col;
                value[col_cntr*A.R + row] = A.mat[row*A.C + col];
                cntr++;
                col_cntr++;
            }
        }
        col_cntr = 0;
        row_cntr++;
    }
    rowPtrs[row_cntr] = cntr;

    ELLMatrix<T> ell_matrix = {rowPtrs, colIdx, value, max_nz_in_row, A.R, A.C, A.num_nonzero};
    return ell_matrix;
}

template <typename T>
SparseMatrix<T> ell_to_sparse(ELLMatrix<T> A) {
    T* mat = (T*)malloc(A.R * A.C * sizeof(T));

    std::fill(mat, mat+A.R*A.C, static_cast<T>(0));

    for(unsigned int i=0; i<A.R; i++) {
        for(unsigned int j=i; j<A.max_nz_in_row*A.R; j+=A.R) {
            unsigned int col_idx = A.colIdx[j];
            if(col_idx != static_cast<T>(PAD_VAL))
                mat[i*A.C + col_idx] = A.value[j];
        }
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
    printf(R"EOF(
 ## ##    ## ##    ## ##   
##   ##  ##   ##  ##   ##  
##       ##   ##  ##   ##  
##       ##   ##  ##   ##  
##       ##   ##  ##   ##  
##   ##  ##   ##  ##   ##  
 ## ##    ## ##    ## ## 

)EOF");
    COOMatrix<float> coo_matrix = sparse_to_coo<float>(sparse_matrix);
    print_array<unsigned int>(coo_matrix.rowIdx, coo_matrix.num_nonzero, "rowIdx");
    print_array<unsigned int>(coo_matrix.colIdx, coo_matrix.num_nonzero, "colIdx");
    print_array<float>(coo_matrix.value, coo_matrix.num_nonzero, "value");

    SparseMatrix<float> sparse_matrix_from_coo = coo_to_sparse<float>(coo_matrix);
    print_matrix<float>(sparse_matrix_from_coo.mat, sparse_matrix_from_coo.R, sparse_matrix_from_coo.C, "Sparse Matric from COO");

    std::cout   << "(Original Sparse) vs (COO->Sparse) allclose: "
                << (all_close<float>(sparse_matrix.mat, sparse_matrix_from_coo.mat, sparse_matrix.R * sparse_matrix.C, abs_tol, rel_tol) ? "true" : "false")
                << std::endl;


    // CSR Representation
    printf(R"EOF(
 ## ##    ## ##   ### ##   
##   ##  ##   ##   ##  ##  
##       ####      ##  ##  
##        #####    ## ##   
##           ###   ## ##   
##   ##  ##   ##   ##  ##  
 ## ##    ## ##   #### ##  

)EOF");
    CSRMatrix<float> csr_matrix = sparse_to_csr<float>(sparse_matrix);
    print_array<unsigned int>(csr_matrix.rowPtrs, csr_matrix.R, "rowPtrs");
    print_array<unsigned int>(csr_matrix.colIdx, csr_matrix.num_nonzero, "colIdx");
    print_array<float>(csr_matrix.value, csr_matrix.num_nonzero, "value");

    SparseMatrix<float> sparse_matrix_from_csr = csr_to_sparse<float>(csr_matrix);
    print_matrix<float>(sparse_matrix_from_csr.mat, sparse_matrix_from_csr.R, sparse_matrix_from_csr.C, "Sparse Matric from CSR");

    std::cout   << "(Original Sparse) vs (CSR->Sparse) allclose: "
                << (all_close<float>(sparse_matrix.mat, sparse_matrix_from_csr.mat, sparse_matrix.R * sparse_matrix.C, abs_tol, rel_tol) ? "true" : "false")
                << std::endl; 


    // ELL Representation
    printf(R"EOF(
### ###  ####     ####     
 ##  ##   ##       ##      
 ##       ##       ##      
 ## ##    ##       ##      
 ##       ##       ##      
 ##  ##   ##  ##   ##  ##  
### ###  ### ###  ### ###   

)EOF");
    ELLMatrix<float> ell_matrix = sparse_to_ell<float>(sparse_matrix);
    print_array<unsigned int>(ell_matrix.rowPtrs, ell_matrix.R, "rowPtrs");
    print_array<unsigned int>(ell_matrix.colIdx, ell_matrix.R*ell_matrix.max_nz_in_row, "colIdx");
    print_array<float>(ell_matrix.value, ell_matrix.R*ell_matrix.max_nz_in_row, "value");

    SparseMatrix<float> sparse_matrix_from_ell = ell_to_sparse<float>(ell_matrix);
    print_matrix<float>(sparse_matrix_from_ell.mat, sparse_matrix_from_ell.R, sparse_matrix_from_ell.C, "Sparse Matric from ELL");

    std::cout   << "(Original Sparse) vs (ELL->Sparse) allclose: "
                << (all_close<float>(sparse_matrix.mat, sparse_matrix_from_ell.mat, sparse_matrix.R * sparse_matrix.C, abs_tol, rel_tol) ? "true" : "false")
                << std::endl; 
}