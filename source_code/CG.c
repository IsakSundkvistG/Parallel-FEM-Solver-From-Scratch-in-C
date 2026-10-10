#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "CG.h"


static inline double csr_row_product(const CSRMatrix *A, const double *x, int row)
{
    double sum = 0.0;
    
    for (int k = A->row_ptr[row]; k < A->row_ptr[row + 1]; k++)
        sum += A->val[k] * x[A->col_idx[k]];

    return sum;
}


void csr_matvec(const CSRMatrix *A, const double *x, double *y)
{
    #pragma omp parallel for schedule(static) 
    for (int i = 0; i < A->n; i++)
        y[i] = csr_row_product(A, x, i);
}


// Compute Ap and pAp in the same matrix traversal:
static double csr_matvec_dot(const CSRMatrix *A, const double *x, double *y)
{
    double result = 0.0;

    #pragma omp parallel for reduction(+ : result) schedule(static)
    //(+ : result) ==> (reduce with addition : result is the variable being reduced)
    for (int i = 0; i < A->n; i++)
    {
        double value = csr_row_product(A, x, i);

        // Ap: 
        y[i] = value;

        // pAp:
        result += x[i] * value;
    }

    return result;
}


int conjugate_gradient(const CSRMatrix *A,
                       const double *f,
                       double *x,
                       double tol,
                       int max_iter,
                       int *num_iters,
                       double *relative_residual)
{
    if (!A || A->n <= 0 || !A->row_ptr || !A->col_idx || !A->val || !f || !x || 
        !num_iters || !relative_residual || tol <= 0.0 || max_iter <= 0)
    {
        fprintf(stderr, "conjugate_gradient error: invalid input\n");
        return 1;
    }

    int n = A->n;

    double *r = malloc((size_t)n * sizeof(double));
    double *p = malloc((size_t)n * sizeof(double));
    double *Ap = malloc((size_t)n * sizeof(double));

    if (!r || !p || !Ap)
    {
        fprintf(stderr, "conjugate_gradient error: memory allocation failed\n");
        free(r);
        free(p);
        free(Ap);

        return 1;
    }

    // Compute the initial residual r = f - Ax:
    csr_matvec(A, x, Ap);

    double rr = 0.0;
    double ff = 0.0;

    // Initialize r and p while also calculating rr and ff:
    #pragma omp parallel for reduction(+ : rr, ff) schedule(static)
    for (int i = 0; i < n; i++)
    {
        r[i] = f[i] - Ap[i];
        p[i] = r[i];

        rr += r[i] * r[i];
        ff += f[i] * f[i];
    }

    // Use an absolute residual criterion when f is zero:
    double residual_denominator = (ff == 0.0) ? 1.0 : ff;

    // Compare squared residuals to avoid sqrt() in every iteration:
    double tol_squared = tol * tol * residual_denominator;

    *num_iters = 0;
    *relative_residual = sqrt(rr / residual_denominator);

    // Check whether the initial guess already satisfies the tolerance:
    if (rr <= tol_squared)
    {
        free(r);
        free(p);
        free(Ap);

        return 0;
    }

    // Main CG loop:
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Compute Ap and pAp in the same traversal:
        double pAp = csr_matvec_dot(A, p, Ap);

        // CG requires a symmetric positive-definite matrix:
        if (pAp <= 0.0)
        {
            fprintf(stderr, "conjugate_gradient error: matrix is not SPD\n");
            free(r);
            free(p);
            free(Ap);

            return 1;
        }

        double alpha = rr / pAp;
        double rr_new = 0.0;

        // Update x and r while also calculating the new rr:
        #pragma omp parallel for reduction(+ : rr_new) schedule(static)
        for (int i = 0; i < n; i++)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];

            rr_new += r[i] * r[i];
        }

        *num_iters = iter + 1;

        if (rr_new <= tol_squared)
        {
            *relative_residual =
                sqrt(rr_new / residual_denominator);

            free(r);
            free(p);
            free(Ap);

            return 0;
        }

        double beta = rr_new / rr;

        // Update the search direction:
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta * p[i];

        rr = rr_new;
    }

    *relative_residual = sqrt(rr / residual_denominator);

    fprintf(stderr, "conjugate_gradient error: did not converge in %d iterations.\n", max_iter);

    free(r);
    free(p);
    free(Ap);

    return 1;
}