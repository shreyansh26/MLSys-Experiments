#include <iostream>
#include <vector>
#include <climits>

#include "sparse_matrix_utils.hpp"
#include "cuda_utils.hpp"

#define ROWS            10
#define COLUMNS         10
#define SPARSITY_RATIO  0.2
#define PAD_VAL         (1<<20)

int main() {
    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;

    printf(R"EOF(
 ## ##   ### ##     ##     ### ##    ## ##   ### ###           ##   ##    ##     #### ##  ### ##     ####   ##  ##   
##   ##   ##  ##     ##     ##  ##  ##   ##   ##  ##            ## ##      ##    # ## ##   ##  ##     ##    ### ##   
####      ##  ##   ## ##    ##  ##  ####      ##               # ### #   ## ##     ##      ##  ##     ##     ###     
 #####    ##  ##   ##  ##   ## ##    #####    ## ##            ## # ##   ##  ##    ##      ## ##      ##      ###    
    ###   ## ##    ## ###   ## ##       ###   ##               ##   ##   ## ###    ##      ## ##      ##       ###   
##   ##   ##       ##  ##   ##  ##  ##   ##   ##  ##           ##   ##   ##  ##    ##      ##  ##     ##    ##  ###  
 ## ##   ####     ###  ##  #### ##   ## ##   ### ###           ##   ##  ###  ##   ####    #### ##    ####   ##   ##  
                                                                                                                     
)EOF");
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
    print_matrix<float>(sparse_matrix_from_coo.mat, sparse_matrix_from_coo.R, sparse_matrix_from_coo.C, "Sparse Matrix from COO");

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
    print_matrix<float>(sparse_matrix_from_csr.mat, sparse_matrix_from_csr.R, sparse_matrix_from_csr.C, "Sparse Matrix from CSR");

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
    print_matrix<float>(sparse_matrix_from_ell.mat, sparse_matrix_from_ell.R, sparse_matrix_from_ell.C, "Sparse Matrix from ELL");

    std::cout   << "(Original Sparse) vs (ELL->Sparse) allclose: "
                << (all_close<float>(sparse_matrix.mat, sparse_matrix_from_ell.mat, sparse_matrix.R * sparse_matrix.C, abs_tol, rel_tol) ? "true" : "false")
                << std::endl; 
}