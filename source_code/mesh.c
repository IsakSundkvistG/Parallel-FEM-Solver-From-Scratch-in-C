#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mesh.h"

#define MAX_LINE 2048

static void mesh_init(Mesh *mesh)
{
    mesh->num_nodes = 0; 
    mesh->num_elements = 0; 
    mesh->nodes = NULL; 
    mesh->elements = NULL; 
    mesh->is_boundary = NULL;
}

void free_mesh(Mesh *mesh)
{
    free(mesh->nodes); 
    free(mesh->elements); 
    free(mesh->is_boundary);

    mesh->num_nodes = 0; 
    mesh->num_elements = 0; 
    mesh->nodes = NULL; 
    mesh->elements = NULL; 
    mesh->is_boundary = NULL;
}

int read_mesh(const char *filename, Mesh *mesh)
{
    if (!filename || !mesh)
    {
        fprintf(stderr, "read_mesh error: NULL input.\n");
        return 1; 
    }
    
    mesh_init(mesh); 

    FILE *file = fopen(filename, "r"); 
    if (!file) 
    {
        perror("Error opening mesh file.");
        return 1; 
    }

    char line[MAX_LINE]; 

    while(fgets(line, MAX_LINE, file))
    {

        // ---------- NODES -----------
        if (strncmp(line, "$Nodes", 6) == 0) // Check if $Nodes starting
        {   

            if (!fgets(line, MAX_LINE, file))
            {
                fprintf(stderr, "read_mesh error: failed to read node count.\n");
                fclose(file);
                free_mesh(mesh); 
                return 1; 
            }

            mesh->num_nodes = atoi(line);
            if (mesh->num_nodes <= 0) 
            {
                fprintf(stderr, "read_mesh error: invalid number of nodes.\n");
                fclose(file);
                free_mesh(mesh);
                return 1;
            }
            
            mesh->nodes = malloc(mesh->num_nodes * sizeof(Node)); 
            if (!mesh->nodes) 
            {
                fprintf(stderr, "Malloc failed for nodes.\n");
                fclose(file);
                free_mesh(mesh);
                return 1;
            }
            
            mesh->is_boundary = calloc(mesh->num_nodes, sizeof(unsigned char)); 
            if (!mesh->is_boundary) 
            {
                fprintf(stderr, "Malloc failed for is_boundary.\n");
                fclose(file);
                free_mesh(mesh);
                return 1;
            }

            for (int i = 0; i < mesh->num_nodes; i++)
            {
                int id; 
                double x, y, z; 

                if (!fgets(line, MAX_LINE, file)) 
                {
                    fprintf(stderr, "read_mesh error: unexpected end while reading nodes.\n");
                    fclose(file);
                    free_mesh(mesh);
                    return 1;
                }

                if (sscanf(line, "%d %lf %lf %lf", &id, &x, &y, &z) != 4) 
                {
                    fprintf(stderr, "read_mesh error: malformed node line.\n");
                    fclose(file);
                    free_mesh(mesh);
                    return 1;
                }

                if (id < 1 || id > mesh->num_nodes) 
                {
                    fprintf(stderr, "read_mesh error: node id out of range.\n");
                    fclose(file);
                    free_mesh(mesh);
                    return 1;
                }

                mesh->nodes[id - 1].x = x; 
                mesh->nodes[id - 1].y = y; 
            }
        }

        // ----------- ELEMENTS ------------
        if (strncmp(line, "$Elements", 9) == 0)
        {   

            if (!fgets(line, MAX_LINE, file)) 
            {
                fprintf(stderr, "read_mesh error: failed to read element count.\n");
                fclose(file);
                free_mesh(mesh);
                return 1;
            }

            int total_elements = atoi(line); // Total elements different from the number of relevant ones
            if (total_elements <= 0) 
            {
                fprintf(stderr, "read_mesh error: invalid number of elements.\n");
                fclose(file);
                free_mesh(mesh);
                return 1;
            }

            mesh->elements = malloc(total_elements * sizeof(Element)); 
            if (!mesh->elements) 
            {
                fprintf(stderr, "Malloc failed for elements.\n");
                fclose(file);
                free_mesh(mesh);
                return 1;
            }


            int count = 0; 

            for (int i = 0; i < total_elements; i++)
            {

                if (!fgets(line, MAX_LINE, file))
                {
                    fprintf(stderr, "read_mesh error: unexpected end while reading elements.\n");
                    fclose(file);
                    free_mesh(mesh);
                    return 1;
                } 

                char *ptr = line; 

                
                /* strtrol() parses a char from ptr, converts it to a long int
                and returns it. By passing &ptr, strtol makes it so that ptr now
                points to the end where it stopped parsing. So basically it walks 
                through a string, parsing each token then advancing the ptr to the 
                next token. 10 is just for base-10: */

                // Not using id:
                (void)strtol(ptr, &ptr, 10); 

                // Read element type (important): 
                int type = strtol(ptr, &ptr, 10);

                // Read number of tags (so we can skip em):
                int num_tags = strtol(ptr, &ptr, 10); 

                // Skip tags (idk what they are and we don't need em so):
                for (int i = 0; i < num_tags; i++)
                {
                    strtol(ptr, &ptr, 10);
                }

                /* gmsh exports not only the triangles that we consider to be 
                "elements" in the FEM sense. It includes points and lines as well,
                and it calls these elements. This is why there is a number denoting
                the type of element. Triangles are elements of type 2; and crucially, 
                boundary line segments (two nodes) are type 1, and these are the only ones we care about,
                so we only store the triangles and mark the boundary nodes as such. 
                */
                

                if (type == 1) // Boundary line segment 
                {
                    // Read the two node ids of the boundary segment:
                    int n1 = strtol(ptr, &ptr, 10);
                    int n2 = strtol(ptr, &ptr, 10);

                    // Mark these nodes as boundary nodes:
                    if (n1 < 1 || n1 > mesh->num_nodes || n2 < 1 || n2 > mesh->num_nodes)
                    {
                        fprintf(stderr, "read_mesh error: boundary node id out of range.\n");
                        fclose(file);
                        free_mesh(mesh);
                        return 1;
                    }  
                
                    mesh->is_boundary[n1 - 1] = 1; 
                    mesh->is_boundary[n2 - 1] = 1; 
                }
                
                else if (type == 2)
                {
                    int n1 = strtol(ptr, &ptr, 10);
                    int n2 = strtol(ptr, &ptr, 10);
                    int n3 = strtol(ptr, &ptr, 10);

                    if (n1 < 1 || n1 > mesh->num_nodes || n2 < 1 || n2 > mesh->num_nodes || n3 < 1 || n3 > mesh->num_nodes) 
                    {
                        fprintf(stderr, "read_mesh error: triangle node id out of range.\n");
                        fclose(file);
                        free_mesh(mesh);
                        return 1;
                    }

                    mesh->elements[count].n1 = n1 - 1; // gmsh uses 1-based indexing :(
                    mesh->elements[count].n2 = n2 - 1; 
                    mesh->elements[count].n3 = n3 - 1;

                    count ++; 
                }

            }

            if (count == 0)
            {
                fprintf(stderr, "read_mesh error: no triangles found in mesh.\n");
                fclose(file);
                free_mesh(mesh);
                return 1;
            }

            mesh->num_elements = count; // Actual relevant element (triangle) count

            // SHRINK ALLOCATION to actual triangle count (since we allocated for all elements, but only kept type 2 (triangles)):
            Element *tmp = realloc(mesh->elements, count * sizeof(Element));
            if (tmp)
                mesh->elements = tmp; 
            // I figured this could potentially help with reducing RAM usage and better cache usage etc.
            // If it needs to copy the array to a new location, it won't hurt performance that much, relatively. According to Claude at least. 
        }   
    }

    fclose(file);

    
    if (!mesh->nodes)
    {
        fprintf(stderr, "read_mesh error: no $Nodes section found.\n");
        free_mesh(mesh);
        return 1;
    }

    if (!mesh->elements) 
    {
        fprintf(stderr, "read_mesh error: no $Elements section found.\n");
        free_mesh(mesh);
        return 1;
    }
    
    return 0; 
}

           