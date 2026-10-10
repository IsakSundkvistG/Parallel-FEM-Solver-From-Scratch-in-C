#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csr.h"

/*
2/8/2026: 
    Parallelized. Only the most trivially parallelizable loops were parallelized, since
    it isn't worth it to spend more time on this module which accounts for <1% of the 
    total runtime. 

*/

/*
The following function is a helper function: dynamic append. We need growing arrays. 
Regarding pointers and how we pass args to functions, rmember: 
    - Things to modify --> pass as pointers, otherwise only stack copy will be modified
    - If something is huge --> pass as pointer, might otherwise put heavy pressure on stack. 

The if block will run immediately when size = capacity, i.e the list is full. Then the capacity
gets doubled and the same thing happens once it fills again. 
*/


// - append() removed (sometime late july 2026)


// Compare function for sorting. Writing to work with qsort, it has to take generic pointers and return +-1 or 0:
static int compare_int(const void *a, const void *b)
{
    int first = *(const int *)a;
    int second = *(const int *)b;

    return (first > second) - (first < second); //overflow guard
}

void free_csr(CSRMatrix *A)
{
    free(A->row_ptr); 
    free(A->col_idx); 
    free(A->val); 

    A->row_ptr = NULL; 
    A->col_idx = NULL; 
    A->val = NULL; 
    A->n = 0; 
    A->nnz = 0; 
}

// Actual CSR implementation:
int build_csr_pattern(const Mesh *mesh, CSRMatrix *A)
{
    if (!mesh || !A || mesh->num_nodes <= 0 || mesh->num_elements <= 0 || !mesh->elements)
    {
        fprintf(stderr, "build_csr_pattern error: invalid input\n");
        return 1; 
    }

    /*

    31/7 2026:

    What we want to do is extract an adjacency array that stores the node IDs contigiously, however
    according to the CSR row order.

    Ex:
        adjacency = [45 23 65 7 4 9 1 2 34 89 3] 
                     --r1---  ---r2---- ---r3--
        temp_row_ptr = [0, 3, 8, 11]
    
    */

    int n = mesh->num_nodes; 

    if (mesh->num_elements > (INT_MAX - n) / 6)
    {
        fprintf(stderr, "build_csr_pattern error: mesh is too large\n");
        return 1;
    }
    
    // Initialize the CSR matrix to a safe empty state:
    A->n = n;
    A->nnz = 0;
    A->row_ptr = NULL;
    A->col_idx = NULL;
    A->val = NULL;

    int *row_sizes = malloc((size_t)n * sizeof(int));
    int *temp_row_ptr = malloc(((size_t)n + 1) * sizeof(int)); // The extra entry marks the end of the last row. 
    int *next = malloc((size_t)n * sizeof(int)); 

     if (!row_sizes || !temp_row_ptr || !next)
    {
        fprintf(stderr, "build_csr_pattern error: temporary allocation failed\n");
        free(row_sizes);
        free(temp_row_ptr);
        free(next);
        return 1;
    }

    // Each matrix row needs at least its diagonal entry. Start with those. No race conds here: 
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) 
        row_sizes[i] = 1;
    
    // First mesh traversal: count how many temp entries each row requires: 
    for (int e = 0; e < mesh->num_elements; e++)
    {
        Element elem = mesh->elements[e]; 
        
        /*
        For each triangle, its off-diagonal matrix contributions are simply the connections
        of each node to its other in-triangle nodes. I.e, for a it is a->b and a->c, etc. 

        This completely encodes the total amount of entries, including duplicates, for the right rows. 
        */
        row_sizes[elem.n1] += 2; 
        row_sizes[elem.n2] += 2;
        row_sizes[elem.n3] += 2;
    }

    // Convert the row sizes into offsets that specify where each row begins in the full contiguous adjacency array: 
    temp_row_ptr[0] = 0; 

    // Dep. chain remains serial
    for (int i = 0; i < n; i++)
    {
        temp_row_ptr[i+1] = temp_row_ptr[i] + row_sizes[i]; 
    }

    int temp_nnz = temp_row_ptr[n]; // Last element is the amount of total entries. 

    // Allocate one contigious array containing the temporary adjacency entries for every row:
    int *adjacency = malloc((size_t)temp_nnz * sizeof(int)); 

    if (!adjacency)
    {
        fprintf(stderr, "build_csr_pattern error: adjacency allocation failed\n");
        free(row_sizes);
        free(temp_row_ptr);
        free(next);
        return 1;
    }

    // next[i] identifies the next uninhabited position in row i. When initialized, this is the first position belonging to that row. 
     

    // Insert each diagonal entry exactly once: 
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        /*
        Advances next[i] with 1 for every time we add an entry in the adjacency array
        corresponding to row i, signifying the next position to be filled in row i.
        */
        next[i] = temp_row_ptr[i];
        adjacency[next[i]++] = i; 
    }
    
    // Second mesh traversal: insert the six off-diagonal connections contributed by each triangle: 
    for (int e = 0; e < mesh->num_elements; e++)
    {
        Element elem = mesh->elements[e]; 

        int n1 = elem.n1; 
        int n2 = elem.n2; 
        int n3 = elem.n3; 

        adjacency[next[n1]++] = n2; 
        adjacency[next[n1]++] = n3;

        adjacency[next[n2]++] = n1;
        adjacency[next[n2]++] = n3;

        adjacency[next[n3]++] = n1;
        adjacency[next[n3]++] = n2;
    }

   // Now sort each temporary row and compact its unique column indices into the beginning of that same row:
   int total_nnz = 0; 

   #pragma omp parallel for reduction(+ : total_nnz) schedule(static)
   for (int i = 0; i < n; i++)
   {
        int start = temp_row_ptr[i]; 
        int count = row_sizes[i]; 

        if (count > 1)
        {
            qsort(adjacency + start, (size_t)count, sizeof(int), compare_int);
        }

        int unique_count = 1; 

        // Same deal as in baseline, one lagging write and one leading read compacts values to the left: 
        for (int read = 1; read < count; read++)
        {
            if (adjacency[start + read] != adjacency[start + unique_count - 1])
            {
                adjacency[start + unique_count] = adjacency[start + read]; 
                unique_count++; 
            }
        }

        row_sizes[i] = unique_count; 
        total_nnz += unique_count; 
   }

   A->nnz = total_nnz;

   A->row_ptr = malloc(((size_t)n + 1) * sizeof(int)); 
   A->col_idx = malloc((size_t)A->nnz * sizeof(int)); 
   A->val = calloc((size_t)A->nnz, sizeof(double)); 

   if (!A->row_ptr || !A->col_idx || !A->val)
   {
       fprintf(stderr, "build_csr_pattern error: CSR allocation failed\n");
       free(adjacency);
       free(row_sizes);
       free(temp_row_ptr);
       free(next);
       free_csr(A);
       return 1;
    }

    // Construct the final row pointers:
    A->row_ptr[0] = 0; 

    for (int i = 0; i < n; i++)
    {
        A->row_ptr[i + 1] = A->row_ptr[i] + row_sizes[i]; 
    }

    // Copy each row column index from the adjacency array into the true final row_ptr array:
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        memcpy(A->col_idx + A->row_ptr[i], adjacency + temp_row_ptr[i], (size_t)row_sizes[i] * sizeof(int)); 
    }

    // Dont need these anymore:
    free(adjacency);
    free(row_sizes);
    free(temp_row_ptr);
    free(next);


    return 0;
}
    




/*
CSR = Compressed sparse row. 

We store the matrix row-by-row, BUT ONLY NON-ZEROS!
We use three arrays for this: 

- val[] - the actual values: 
    val = [K00, K01, K10, K11, K12, ...]

- col_idx[] - column index for each value: 
    col_idx = [0, 1, 0, 1, 2, ...]
    This mean val[k] is located at column col_idx[k]

- row_ptr[] - where each row starts, counted by index in the value array:

( OBS !!!: k IS THE INDEX OF THE NON-ZERO ELEMENTS, READ FROM LEFT TO RIGHT AND TOP DOWN IN THE MATRIX!)
( SO BASICALLY k IS INDICES OF VAL ARRAY)
Example: 
5x5 tridiagonal: 
[ 1 2 0 0 0]
[ 2 1 2 0 0]
[ 0 2 1 2 0]
[ 0 0 2 1 2]
[ 0 0 0 2 1]

col_idx = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4]
           -r1-  ---r2--  ---r3--  ---r4--  -r5-

val = [1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1]

And the row_ptr stores the index of the val array at which a new row starts if one ignores the 0s. 

row_ptr = [0, 2, 5, 8, 11, 13] 

The 13 is stored despite it denoting a next row that doesnt exist, because there needs to be an end 
index for all rows, including the last one. 

So to read one row, e.g the third row, we do: 

row[2] = 5
row[3] = 8

That means that we can identify that row 3 lives in val[5] to val[7] !!!

Row 0: [2 3 0 0]
Row 1: [3 5 7 0]
Row 2: [0 7 4 1]
Row 3: [0 0 1 6]
*/