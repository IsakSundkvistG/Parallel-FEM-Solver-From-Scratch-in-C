#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "FEM.h"
#include "csr.h"

/* 
2/8/2026: 
    This module is now both serially optimized and parallelized!

*/

// The original find_position() with binary search was removed on 1/8/2026. 

/*
find_element_positions() was added on 1/8/2026: 

Find the CSR positions of all thre element nodes in one traversal of the 
specified row. This is more efficient than calling find_position three times, 
since we only traverse the row once. 
*/
static void find_element_positions(int row, const int nodes[3], int positions[3], const CSRMatrix *A)
{
    positions[0] = -1;
    positions[1] = -1;
    positions[2] = -1;

    int remaining = 3;
    
    for (int k = A->row_ptr[row]; k < A->row_ptr[row + 1]; k++)
    {
        int col = A->col_idx[k]; 

        if (col == nodes[0])
        {
            positions[0] = k; 
            remaining--; 
        }
        else if (col == nodes[1])
        {
            positions[1] = k; 
            remaining--; 
        }
        else if (col == nodes[2])
        {
            positions[2] = k; 
            remaining--; 
        }

        if (remaining == 0)
            return; 
    }

    fprintf(stderr, "find_element_positions error: not all nodes found in row %d\n", row);
    exit(EXIT_FAILURE);
}

// -------------- LOCAL 3x3 STIFFNESS MATRIX --------------
static int local_stiffness_and_load(Node n1, Node n2, Node n3, double Ke[3][3], double *load)
{
    /* 
    1/8/2026: 
    The major serial optimization we did here is to avoid computing the area of the triangle twice, once 
    for the stiffness matrix and once for the load vector. Instead, we compute it once and reuse. Then, 
    subseqently, we do both the stiffness and load calculations in one function call, too.
    */

    // Coordinate differences relative to node 1:
    double dx21 = n2.x - n1.x; 
    double dy21 = n2.y - n1.y; 
    double dx31 = n3.x - n1.x; 
    double dy31 = n3.y - n1.y; 

    // Twice the triangle area (2A = |(n2-n1) x (n3-n1)|):
    double twice_area = fabs(dx21 * dy31 - dx31 * dy21); 

    if (twice_area <= 0.0) 
    {
        fprintf(stderr, "local_stiffness_and_load error: degenerate triangle detected.\n");
        return 1; 
    }

    // Gradient coefficients of the three local basis functions: 
    double b[3] = {dy21 - dy31, dy31, -dy21};
    double c[3] = {dx31 - dx21, -dx31, dx21}; 

    // Since 4A = 2 * twice_area ==> 1/(4A) = 1/(2 * twice_area) or 0.5 / twice_area: 
    double scale = 0.5 / twice_area;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Ke[i][j] = (b[i] * b[j] + c[i] * c[j]) * scale;
        }
    }

    // For the constant source term f = 1: integral(phi_i * 1) over one triangle = A/3 = twice_area / 6.0; 
    *load = twice_area / 6.0; 

    return 0; 
}

// ==================================== GLOBAL ASSEMBLY ===============================================================
// We got a CSR sparse format going for us here. Quite beautiful but also a bit of a pain to implement. Hope it gives 
// good performance. DON'T FORGET TO PARALLELIZE THE MAIN ASSEMBLY LOOP!!!!!!!!!!!!!!!!
// ==================================================================================================================== 

// We should think about the implications of the usage of bin search in the loop when we parallelize it, with respect to cache and memory access pattern issues! 
// The issue described in the above comment has been addressed. 

/*
sparse_global_assemble() was updated on 2/8/2026: 
    Parallelized. First tried atomic operations. Comparison against to be done
    against private copy version. Concern with the first is overhead, with the
    latter is memory. 

    We will settle for the atomic version. The FEM assembly has very insignificant 
    impact on the runtime overall, even for 1.6M nodes, and we see OK scaling, so
    we wont bother with the private buffer one. Also, the memory concern of the 
    private buffer version is worse with such a fine mesh, anyways. 
*/
int sparse_global_assemble(const Mesh *mesh, CSRMatrix *A, double *f)
{
    int n = mesh->num_nodes; 

    // Initialize global arrays: 
    memset(f, 0, n * sizeof(double)); // Not on IEEE754? Too bad. 
    memset(A->val, 0, A->nnz * sizeof(double)); 

    int first_degenerate_element = mesh->num_elements;

    #pragma omp parallel for reduction (min : first_degenerate_element) schedule(static)
    for (int e = 0; e < mesh->num_elements; e++)
    {
        Element elem = mesh->elements[e]; 

        Node n1 = mesh->nodes[elem.n1]; 
        Node n2 = mesh->nodes[elem.n2];
        Node n3 = mesh->nodes[elem.n3];

        int nodes[3] = {elem.n1, elem.n2, elem.n3};

        double Ke[3][3];
        double load; 

        /*If the local calculation detects a degenerate triangle, immediately 
        stop global assembly. This prevenets Ke or fe from being used while still
        uninitialized.
        
        We cannot return directly from an OpenMP parallel loop. Instead, record the lowest-indexed
        degenerater element.*/ 
       if (local_stiffness_and_load(n1, n2, n3, Ke, &load) != 0)
        {
            if (e < first_degenerate_element)
                first_degenerate_element = e;
            
            continue;
        }

        // Global CSR stiffness assembly + dense load assembly: 
        for (int i = 0; i < 3; i++)
        {
            int I = nodes[i]; 

            // Different elements may contribute to the same load entry, so make atomic: 
            #pragma omp atomic update
            f[I] += load; 

            int positions[3];
            find_element_positions(I, nodes, positions, A);

            // Neighboring elements may contribute to the same CSR entries, so all matrix additions atomic: 
            #pragma omp atomic update
            A->val[positions[0]] += Ke[i][0];

            #pragma omp atomic update
            A->val[positions[1]] += Ke[i][1];

            #pragma omp atomic update
            A->val[positions[2]] += Ke[i][2];
        } 
    }

    if (first_degenerate_element < mesh->num_elements)
    {
        fprintf(stderr, "sparse_global_assemble error: triangle no. %d is degenerate, assembly stopped. \n", first_degenerate_element);
        return 1; 
    }

    return 0;
}


/*
apply_dirichlet_bc() was updated on 2/8/2026: 
    Parallelized. 

apply_dirichlet_bc() was updated on 2/8/2026:
    We now apply the BCs in one single row traversal.
*/
int apply_dirichlet_bc(const Mesh *mesh, CSRMatrix *A, double *f)
{
    if (!mesh || !mesh->is_boundary || !A || !A->row_ptr || !A->col_idx || !A->val || !f)
    { 
        fprintf(stderr, "apply_dirichlet_bc error: invalid input.\n");
        return 1;
    }

    int first_missing_diagonal = A->n; 
    
    #pragma omp parallel for reduction(min : first_missing_diagonal) schedule(static)
    for (int i = 0; i < A->n; i++)
    {
        int row_is_boundary = mesh->is_boundary[i];
        int diagonal_found = !row_is_boundary;

        if (row_is_boundary)
        {
            f[i] = 0.0; 
        }
        
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++)
        {
            int j = A->col_idx[k]; 
            
            if (row_is_boundary)
            {
                // Replace a boundary row with u_i = 0: 
                if (j==i)
                {
                    A->val[k] = 1.0; 
                    diagonal_found = 1;
                }
                else
                {
                    A->val[k] = 0.0; 
                }
            }
            else if (mesh->is_boundary[j])
            {
                // Zero out entire columns corresponding to boundary nodes to preserve symmetry: 
                A->val[k] = 0.0; 
            }
        }

        if (!diagonal_found && i < first_missing_diagonal)
                first_missing_diagonal = i; 
    }

    if (first_missing_diagonal < A->n)
        {
            fprintf(stderr, "apply_dirichlet_bc error: diagonal entry not found for row %d\n", first_missing_diagonal);
            return 1;
        }

   return 0; 
}