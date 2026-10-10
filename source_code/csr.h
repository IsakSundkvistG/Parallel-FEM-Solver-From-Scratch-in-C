#ifndef CSR_H
#define CSR_H

#include "mesh.h"

typedef struct 
{
    int n;        // Nr of rows
    int nnz;      // Nr of nonzeros
    int *row_ptr; // size n+1
    int *col_idx; // size nnz
    double *val;  // size nnz 
} CSRMatrix;

// Build sparsity pattern from mesh (val not filled yet): 
int build_csr_pattern(const Mesh *mesh, CSRMatrix *A);

// Free mem: 
void free_csr(CSRMatrix *A); 

#endif

