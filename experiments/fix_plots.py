import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp25_unified_failure_phase_space as exp25

print("Loading JSON...")
with open(r"d:\Games\FAILURE STUDIES\results\exp25\exp25_results.json", "r") as f:
    results = json.load(f)

BETA_VALS  = [1, 5, 10, 20, 30, 40, 50]
N_COL_VALS = [100, 250, 500, 750, 1000, 2000, 5000]
NU_VALS    = [0.1, 0.05, 0.01, 0.005, 0.001]
WIDTH_VALS = [16, 32, 64, 128, 256]

code_A = np.array(results["grid_A"]["regime_codes"])
code_B = np.array(results["grid_B"]["regime_codes"])
code_C = np.array(results["grid_C"]["regime_codes"])
l2_A = np.array(results["grid_A"]["l2_matrix"])
l2_B = np.array(results["grid_B"]["l2_matrix"])
l2_C = np.array(results["grid_C"]["l2_matrix"])

print("Fitting Grid A...")
xb_A, yb_A, best_A, fits_A, r2_A = exp25.fit_boundary(BETA_VALS, N_COL_VALS, code_A, x_name="β", y_name="N*")
print("Fitting Grid B...")
xb_B, yb_B, best_B, fits_B, r2_B = exp25.fit_boundary(NU_VALS, N_COL_VALS, code_B, x_name="ν", y_name="N*")
print("Fitting Grid C...")
xb_C, yb_C, best_C, fits_C, r2_C = exp25.fit_boundary(BETA_VALS, WIDTH_VALS, code_C, x_name="β", y_name="Width")

results["grid_A"]["boundary"] = {"x": xb_A, "y": yb_A, "best_fit": best_A, "r2": r2_A}
results["grid_B"]["boundary"] = {"x": xb_B, "y": yb_B, "best_fit": best_B, "r2": r2_B}
results["grid_C"]["boundary"] = {"x": xb_C, "y": yb_C, "best_fit": best_C, "r2": r2_C}

with open(r"d:\Games\FAILURE STUDIES\results\exp25\exp25_results.json", "w") as f:
    json.dump(results, f, indent=2)

OUT_DIR = Path(r"d:\Games\FAILURE STUDIES\results\exp25")

print("Plotting A...")
exp25.plot_phase_map(code_A, l2_A, BETA_VALS, N_COL_VALS, "β", "N_collocation",
    "Grid A: Failure Phase Space — β vs N_collocation (Advection)", xb_A, yb_A, best_A, fits_A, OUT_DIR / "phase_map_beta_N.png")

print("Plotting B...")
exp25.plot_phase_map(code_B, l2_B, NU_VALS, N_COL_VALS, "ν", "N_collocation",
    "Grid B: Failure Phase Space — ν vs N_collocation (Burgers)", xb_B, yb_B, best_B, fits_B, OUT_DIR / "phase_map_nu_N.png")

print("Plotting C...")
exp25.plot_phase_map(code_C, l2_C, BETA_VALS, WIDTH_VALS, "β", "Network Width",
    "Grid C: Failure Phase Space — β vs Network Width (Advection)", xb_C, yb_C, best_C, fits_C, OUT_DIR / "phase_map_beta_width.png")

print("Plotting fits...")
results_dict = {
    "A": {"x_boundary": xb_A, "y_boundary": yb_A, "best_fit": best_A, "fits": fits_A, "r2": r2_A},
    "B": {"x_boundary": xb_B, "y_boundary": yb_B, "best_fit": best_B, "fits": fits_B, "r2": r2_B},
    "C": {"x_boundary": xb_C, "y_boundary": yb_C, "best_fit": best_C, "fits": fits_C, "r2": r2_C},
}
exp25.plot_boundary_fits(results_dict, OUT_DIR / "boundary_fits.png")

print("Done! Check results.")
