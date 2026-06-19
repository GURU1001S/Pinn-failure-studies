import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp25_unified_failure_phase_space import fit_boundary

with open("results/exp25/exp25_results.json", "r") as f:
    data = json.load(f)

BETA_VALS  = [1, 5, 10, 20, 30, 40, 50]
N_COL_VALS = [100, 250, 500, 750, 1000, 2000, 5000]

code_A = np.array(data["grid_A"]["regime_codes"])

xb_A, yb_A, best_A, fits_A, r2_A = fit_boundary(
    BETA_VALS, N_COL_VALS, code_A, x_name="β", y_name="N*")

print("Grid A Boundary:")
print("x:", xb_A)
print("y:", yb_A)
print("best:", best_A)
print("fits:", fits_A)
