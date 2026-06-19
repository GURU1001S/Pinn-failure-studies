import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Import plotting func
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp8_collocation_starvation import plot_l2_vs_count_errorbars, L2_FAILURE_THRESHOLD

RESULTS_FILE = Path(r"d:\Games\FAILURE STUDIES\results\exp8\exp8_results.json")
PLOT_FILE = Path(r"d:\Games\FAILURE STUDIES\results\exp8\l2_vs_count.pdf")

with open(RESULTS_FILE, "r") as f:
    data = json.load(f)

# Rebuild count_results with integer keys and recomputed status
count_results = {}
counts = [50, 100, 200, 500, 1000, 2000, 5000]

for n in counts:
    mean_l2 = data["count_sweep"][str(n)]["mean_l2"]
    # Recompute status with 0.10 threshold!
    status = "FAIL" if mean_l2 > 0.10 else "PASS"
    count_results[n] = {
        "mean_l2": mean_l2,
        "std_l2": data["count_sweep"][str(n)]["std_l2"],
        "mean_status": status
    }

min_viable = data.get("minimum_viable_count", 500)

print("Regenerating plot...")
plot_l2_vs_count_errorbars(count_results, counts, L2_FAILURE_THRESHOLD, min_viable, PLOT_FILE)
print("Done.")
