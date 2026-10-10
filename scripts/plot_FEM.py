from pathlib import Path

import matplotlib

# Allows plotting on remote Linux hosts without a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
NODES_FILE = SCRIPT_DIRECTORY / "solution_nodes.csv"
ELEMENTS_FILE = SCRIPT_DIRECTORY / "solution_elements.csv"
OUTPUT_FILE = SCRIPT_DIRECTORY / "FEM_solution_surface.png"

nodes = np.loadtxt(
    NODES_FILE,
    delimiter=",",
    skiprows=1
)

elements = np.loadtxt(
    ELEMENTS_FILE,
    delimiter=",",
    skiprows=1,
    dtype=int
)

x_coordinate = nodes[:, 1]
y_coordinate = nodes[:, 2]
solution = nodes[:, 3]

triangulation = mtri.Triangulation(
    x_coordinate,
    y_coordinate,
    triangles=elements
)

figure = plt.figure(figsize=(9, 7))
axis = figure.add_subplot(111, projection="3d")

surface = axis.plot_trisurf(
    triangulation,
    solution,
    cmap="viridis",
    edgecolor="none",
    antialiased=True
)

axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_zlabel("u(x, y)")
axis.set_title("Finite-element solution")

figure.colorbar(
    surface,
    ax=axis,
    shrink=0.7,
    label="u(x, y)"
)

figure.tight_layout()
figure.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

print(f"Plot saved as {OUTPUT_FILE}")
