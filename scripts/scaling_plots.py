from pathlib import Path

import matplotlib

# Allows plotting on remote Linux hosts without a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_FILE = SCRIPT_DIRECTORY / "parallel_scaling_results.csv"
OUTPUT_FILE = SCRIPT_DIRECTORY / "scaling_results.png"


results = pd.read_csv(RESULTS_FILE, skipinitialspace=True)
successful_results = results[results["success"] == 1].copy()

# Normalize each individual run before taking the median. This ensures every
# runtime is divided by the iteration count from the same CSV row.
successful_results["CG_seconds_per_iteration"] = (
    successful_results["CG_runtime"] / successful_results["iterations"]
)
successful_results["total_seconds_per_iteration"] = (
    successful_results["tot_runtime"] / successful_results["iterations"]
)

runtime_columns = [
    "CG_runtime",
    "tot_runtime",
    "CG_seconds_per_iteration",
    "total_seconds_per_iteration",
]

# Use the median of the five runs for each parallel data point.
median_results = (
    successful_results.groupby(
        ["experiment_type", "threads", "nodes"],
        as_index=False,
    )[runtime_columns]
    .median()
    .sort_values(["experiment_type", "threads"])
)

strong_results = median_results[
    median_results["experiment_type"] == "strong"
].copy()
weak_results = median_results[
    median_results["experiment_type"] == "weak"
].copy()
# Calculate unnormalized and iteration-normalized strong-scaling speedups.
strong_results["CG_speedup"] = (
    strong_results.iloc[0]["CG_runtime"] / strong_results["CG_runtime"]
)
strong_results["total_speedup"] = (
    strong_results.iloc[0]["tot_runtime"] / strong_results["tot_runtime"]
)
strong_results["normalized_CG_speedup"] = (
    strong_results.iloc[0]["CG_seconds_per_iteration"]
    / strong_results["CG_seconds_per_iteration"]
)
strong_results["normalized_total_speedup"] = (
    strong_results.iloc[0]["total_seconds_per_iteration"]
    / strong_results["total_seconds_per_iteration"]
)

# Conventional weak-scaling efficiency uses the smallest one-thread problem as
# the reference: E(p) = T_1 / T_p.
weak_results["CG_efficiency"] = (
    weak_results.iloc[0]["CG_runtime"] / weak_results["CG_runtime"]
)
weak_results["total_efficiency"] = (
    weak_results.iloc[0]["tot_runtime"] / weak_results["tot_runtime"]
)
weak_results["normalized_CG_efficiency"] = (
    weak_results.iloc[0]["CG_seconds_per_iteration"]
    / weak_results["CG_seconds_per_iteration"]
)
weak_results["normalized_total_efficiency"] = (
    weak_results.iloc[0]["total_seconds_per_iteration"]
    / weak_results["total_seconds_per_iteration"]
)

figure, axes = plt.subplots(1, 2, figsize=(14, 6))

# Strong scaling plot
axes[0].plot(
    strong_results["threads"],
    strong_results["CG_speedup"],
    marker="o",
    label="CG speedup",
)
axes[0].plot(
    strong_results["threads"],
    strong_results["total_speedup"],
    marker="o",
    label="Total runtime speedup",
)
axes[0].plot(
    strong_results["threads"],
    strong_results["normalized_CG_speedup"],
    marker="s",
    label="Normalized CG speedup",
)
axes[0].plot(
    strong_results["threads"],
    strong_results["normalized_total_speedup"],
    marker="s",
    label="Normalized total speedup",
)
axes[0].plot(
    strong_results["threads"],
    strong_results["threads"],
    color="black",
    linestyle="--",
    label="Ideal speedup",
)
axes[0].set_title("Strong scaling")
axes[0].set_xlabel("Number of threads")
axes[0].set_ylabel("Speedup")
axes[0].set_xticks(strong_results["threads"])
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Weak scaling efficiency plot
axes[1].plot(
    weak_results["threads"],
    weak_results["CG_efficiency"],
    marker="o",
    label="CG efficiency",
)
axes[1].plot(
    weak_results["threads"],
    weak_results["total_efficiency"],
    marker="o",
    label="Total efficiency",
)
axes[1].plot(
    weak_results["threads"],
    weak_results["normalized_CG_efficiency"],
    marker="s",
    label="Normalized CG efficiency",
)
axes[1].plot(
    weak_results["threads"],
    weak_results["normalized_total_efficiency"],
    marker="s",
    label="Normalized total efficiency",
)
axes[1].axhline(
    1.0,
    color="black",
    linestyle="--",
    label="Ideal efficiency",
)
axes[1].set_title("Weak scaling")
axes[1].set_xlabel("Number of threads")
axes[1].set_ylabel("Efficiency")
axes[1].set_xticks(weak_results["threads"])
axes[1].grid(True, alpha=0.3)
axes[1].legend()

figure.tight_layout()
figure.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

print(f"Plot saved as {OUTPUT_FILE}")
