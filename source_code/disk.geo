SetFactory("OpenCASCADE"); 

Disk(1) = {0, 0, 0, 1}; 

// Disable automatic sizing effects: 
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// Force uniform size:
Mesh.MeshSizeMax = 0.006;
Mesh.MeshSizeMin = 0.006;

Mesh.MshFileVersion = 2.2;