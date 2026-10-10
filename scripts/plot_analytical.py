from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIRECTORY / "analytical_solution_surface.png"

x = np.linspace(-1, 1, 300)
y = np.linspace(-1, 1, 300)
x, y = np.meshgrid(x, y)

r_squared = x**2 + y**2
u = np.where(r_squared <= 1, (1 - r_squared) / 4, np.nan)

figure = plt.figure(figsize=(9, 7))
axis = figure.add_subplot(111, projection="3d")
surface = axis.plot_surface(x, y, u, cmap="viridis", edgecolor="none")

axis.set(xlabel="x", ylabel="y", zlabel="u(x, y)", title="Analytical solution")
figure.colorbar(surface, ax=axis, shrink=0.7, label="u(x, y)")
figure.tight_layout()
figure.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

print(f"Plot saved as {OUTPUT_FILE}")
