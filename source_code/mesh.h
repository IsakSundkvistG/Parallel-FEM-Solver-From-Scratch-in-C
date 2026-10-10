#ifndef MESH_H
#define MESH_H

typedef struct
{
    double x, y;
} Node; 

typedef struct 
{
    int n1, n2, n3; 
} Element;

typedef struct
{
    int num_nodes; 
    int num_elements; 
    Node *nodes; 
    Element *elements;
    unsigned char *is_boundary; // 1 if node is on boundary, 0 otherwise
} Mesh;

// Returns 0 on success, 1 on failure:
int read_mesh(const char *filename, Mesh *mesh);

// Frees mem owned by mesh and resets fields: 
void free_mesh(Mesh *mesh); 

#endif
