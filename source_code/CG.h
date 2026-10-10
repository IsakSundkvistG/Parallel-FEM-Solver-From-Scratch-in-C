#ifndef CG_H
#define CG_H

# include "csr.h"

void csr_matvec(const CSRMatrix *A, const double *x, double *y);

int conjugate_gradient(const CSRMatrix *A, const double *f, double *x, double tol,
                    int max_iter, int *num_iters, double *relative_residual);


#endif