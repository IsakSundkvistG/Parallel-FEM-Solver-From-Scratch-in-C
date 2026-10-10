#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mesh.h"
#include "csr.h"
#include "FEM.h"
#include "CG.h"


static double wall_seconds(void)
{
    struct timespec time;

    clock_gettime(CLOCK_MONOTONIC, &time);

    return (double)time.tv_sec +
           1.0e-9 * (double)time.tv_nsec;
}


static int export_solution(const Mesh *mesh,
                           const double *solution)
{
    FILE *node_file =
        fopen("solution_nodes.csv", "w");

    if (!node_file)
    {
        perror("Could not open solution_nodes.csv");
        return 1;
    }

    fprintf(node_file, "node,x,y,u\n");

    for (int i = 0; i < mesh->num_nodes; i++)
    {
        fprintf(node_file, "%d,%.17g,%.17g,%.17g\n",
                i,
                mesh->nodes[i].x,
                mesh->nodes[i].y,
                solution[i]);
    }

    fclose(node_file);

    FILE *element_file =
        fopen("solution_elements.csv", "w");

    if (!element_file)
    {
        perror("Could not open solution_elements.csv");
        return 1;
    }

    fprintf(element_file, "n1,n2,n3\n");

    for (int i = 0; i < mesh->num_elements; i++)
    {
        fprintf(element_file, "%d,%d,%d\n",
                mesh->elements[i].n1,
                mesh->elements[i].n2,
                mesh->elements[i].n3);
    }

    fclose(element_file);

    return 0;
}


int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        fprintf(stderr,
                "Usage: %s mesh.msh [--export]\n",
                argv[0]);

        return 1;
    }

    int export_requested = 0;

    if (argc == 3)
    {
        if (strcmp(argv[2], "--export") != 0)
        {
            fprintf(stderr,
                    "Unknown option: %s\n",
                    argv[2]);

            fprintf(stderr,
                    "Usage: %s mesh.msh [--export]\n",
                    argv[0]);

            return 1;
        }

        export_requested = 1;
    }

    const char *mesh_filename = argv[1];

    Mesh mesh;
    CSRMatrix A;

    double *f = NULL;
    double *x = NULL;
    double *Ax = NULL;

    double pipeline_start = wall_seconds();

    // Read the mesh:
    if (read_mesh(mesh_filename, &mesh) != 0)
        return 1;

    double after_mesh_read = wall_seconds();

    // Build the CSR sparsity pattern:
    if (build_csr_pattern(&mesh, &A) != 0)
    {
        free_mesh(&mesh);
        return 1;
    }

    double after_csr_pattern = wall_seconds();

    // Allocate the vectors:
    f = malloc((size_t)A.n * sizeof(double));
    x = calloc((size_t)A.n, sizeof(double));
    Ax = malloc((size_t)A.n * sizeof(double));

    if (!f || !x || !Ax)
    {
        fprintf(stderr,
                "main error: vector allocation failed.\n");

        free(f);
        free(x);
        free(Ax);
        free_csr(&A);
        free_mesh(&mesh);

        return 1;
    }

    double after_allocation = wall_seconds();

    // Assemble the global matrix and load vector:
    if (sparse_global_assemble(&mesh, &A, f) != 0)
    {
        free(f);
        free(x);
        free(Ax);
        free_csr(&A);
        free_mesh(&mesh);

        return 1;
    }

    double after_assembly = wall_seconds();

    // Apply homogeneous Dirichlet boundary conditions:
    if (apply_dirichlet_bc(&mesh, &A, f) != 0)
    {
        free(f);
        free(x);
        free(Ax);
        free_csr(&A);
        free_mesh(&mesh);

        return 1;
    }

    double after_boundary_conditions = wall_seconds();

    int iterations = 0;
    double reported_relative_residual = 0.0;

    // Since x was allocated using calloc, the initial guess is x = 0:
    if (conjugate_gradient(&A, f, x, 1e-10, 10000,
                           &iterations,
                           &reported_relative_residual) != 0)
    {
        free(f);
        free(x);
        free(Ax);
        free_csr(&A);
        free_mesh(&mesh);

        return 1;
    }

    double after_cg = wall_seconds();

    // Verification begins after all performance measurements:
    csr_matvec(&A, x, Ax);

    double residual_squared = 0.0;
    double f_norm_squared = 0.0;
    double max_boundary_value = 0.0;
    double max_nodal_error = 0.0;

    for (int i = 0; i < A.n; i++)
    {
        double residual = f[i] - Ax[i];

        residual_squared += residual * residual;
        f_norm_squared += f[i] * f[i];

        if (mesh.is_boundary[i] &&
            fabs(x[i]) > max_boundary_value)
        {
            max_boundary_value = fabs(x[i]);
        }

        // Exact solution for -Delta u = 1 on the unit disk:
        // u(x,y) = (1 - x^2 - y^2)/4.
        double radius_squared =
            mesh.nodes[i].x * mesh.nodes[i].x +
            mesh.nodes[i].y * mesh.nodes[i].y;

        double exact_solution =
            (1.0 - radius_squared) / 4.0;

        double nodal_error =
            fabs(x[i] - exact_solution);

        if (nodal_error > max_nodal_error)
            max_nodal_error = nodal_error;
    }

    double true_relative_residual;

    if (f_norm_squared > 0.0)
    {
        true_relative_residual =
            sqrt(residual_squared / f_norm_squared);
    }
    else
    {
        true_relative_residual =
            sqrt(residual_squared);
    }

    printf("\nProblem size:\n");
    printf("Nodes:                       %d\n",
           mesh.num_nodes);
    printf("Triangles:                   %d\n",
           mesh.num_elements);
    printf("Nonzero matrix entries:      %d\n",
           A.nnz);

    printf("\nPerformance timings:\n");
    printf("Mesh reading:                %.9f s\n",
           after_mesh_read - pipeline_start);
    printf("CSR-pattern construction:    %.9f s\n",
           after_csr_pattern - after_mesh_read);
    printf("Vector allocation:           %.9f s\n",
           after_allocation - after_csr_pattern);
    printf("FEM assembly:                %.9f s\n",
           after_assembly - after_allocation);
    printf("Boundary conditions:         %.9f s\n",
           after_boundary_conditions - after_assembly);
    printf("CG solver:                   %.9f s\n",
           after_cg - after_boundary_conditions);

    // Compute total excludes mesh-file reading:
    printf("Compute total:               %.9f s\n",
           after_cg - after_mesh_read);

    // Pipeline total includes reading but excludes verification and export:
    printf("Pipeline total:              %.9f s\n",
           after_cg - pipeline_start);

    printf("\nCorrectness results:\n");
    printf("CG iterations:               %d\n",
           iterations);
    printf("Reported relative residual:  %.12e\n",
           reported_relative_residual);
    printf("Recomputed relative residual: %.12e\n",
           true_relative_residual);
    printf("Largest boundary |x[i]|:     %.12e\n",
           max_boundary_value);
    printf("Largest nodal error:         %.12e\n",
           max_nodal_error);

    int failed =
        true_relative_residual > 1e-9 ||
        max_boundary_value > 1e-12;

    if (failed)
        printf("CG test failed.\n");
    else
        printf("CG test passed.\n");

    // Export is optional and excluded from all timings:
    if (export_requested)
    {
        if (export_solution(&mesh, x) != 0)
        {
            free(f);
            free(x);
            free(Ax);
            free_csr(&A);
            free_mesh(&mesh);

            return 1;
        }

        printf("Solution exported successfully.\n");
    }

    free(f);
    free(x);
    free(Ax);
    free_csr(&A);
    free_mesh(&mesh);

    return failed ? 1 : 0;
}