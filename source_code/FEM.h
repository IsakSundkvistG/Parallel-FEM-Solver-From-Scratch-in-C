#ifndef FEM_H
#define FEM_H

#include "mesh.h"
#include "csr.h"

// These return 0 on success, 1 if a triangle is degenerrate (zero area) 
int local_stiffness(Node n1, Node n2, Node n3, double Ke[3][3]);
int local_load(Node n1, Node n2, Node n3, double fe[3]);

int sparse_global_assemble(const Mesh *mesh, CSRMatrix *A, double *f); 
int apply_dirichlet_bc(const Mesh *mesh, CSRMatrix *A, double *f);

#endif