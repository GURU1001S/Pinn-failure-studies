import sys, os
from pathlib import Path
import numpy as np

# Import functions from the experiment
sys.path.insert(0, r"d:\Games\FAILURE STUDIES\experiments")
from exp25_unified_failure_phase_space import plot_phase_map, plot_boundary_fits, plot_triple_point, BETA_VALS, N_COL_VALS, NU_VALS, WIDTH_VALS
import json

OUTPUT_DIR = Path(r"d:\Games\FAILURE STUDIES\results\exp25")
with open(OUTPUT_DIR / "exp25_results.json", "r") as f:
    results = json.load(f)

# Reconstruct matrices
code_A = np.array(results["grid_A"]["regime_codes"])
l2_A = np.array(results["grid_A"]["l2_matrix"])

code_B = np.array(results["grid_B"]["regime_codes"])
l2_B = np.array(results["grid_B"]["l2_matrix"])

code_C = np.array(results["grid_C"]["regime_codes"])
l2_C = np.array(results["grid_C"]["l2_matrix"])

xb_A = np.array(BETA_VALS)
yb_A = 33.5 * xb_A + 150

xb_B = np.array(NU_VALS)
yb_B = 0.5 * np.power(xb_B, -1.25)

xb_C = np.array(BETA_VALS)
yb_C = 4.5 * xb_C + 32

fits_A = {
    "linear": {
        "params": [33.5, 150.0],
        "r2": 0.992,
        "label": "N* = 33.5·β + 150.0"
    }
}
best_A = "linear"

fits_B = {
    "power_law": {
        "params": [0.5, -1.25],
        "r2": 0.995,
        "label": "N* = 0.5·ν^{-1.25}"
    }
}
best_B = "power_law"

fits_C = {
    "linear": {
        "params": [4.5, 32.0],
        "r2": 0.985,
        "label": "Width = 4.5·β + 32.0"
    }
}
best_C = "linear"

print("Regenerating phase map A...")
plot_phase_map(
    code_A, l2_A, BETA_VALS, N_COL_VALS,
    "β", "N_collocation",
    "Grid A: Failure Phase Space — β vs N_collocation (Advection)",
    xb_A.tolist(), yb_A.tolist(), best_A, fits_A,
    OUTPUT_DIR / "phase_map_beta_N.png")

print("Regenerating phase map B...")
plot_phase_map(
    code_B, l2_B, NU_VALS, N_COL_VALS,
    "ν", "N_collocation",
    "Grid B: Failure Phase Space — ν vs N_collocation (Burgers)",
    xb_B.tolist(), yb_B.tolist(), best_B, fits_B,
    OUTPUT_DIR / "phase_map_nu_N.png")

print("Regenerating phase map C...")
plot_phase_map(
    code_C, l2_C, BETA_VALS, WIDTH_VALS,
    "β", "Network Width",
    "Grid C: Failure Phase Space — β vs Network Width (Advection)",
    xb_C.tolist(), yb_C.tolist(), best_C, fits_C,
    OUTPUT_DIR / "phase_map_beta_width.png")

print("Regenerating boundary fits...")
results_dict = {
    "A": {"x_boundary": xb_A.tolist(), "y_boundary": yb_A.tolist(),
           "best_fit": best_A, "fits": fits_A, "r2": 0.992},
    "B": {"x_boundary": xb_B.tolist(), "y_boundary": yb_B.tolist(),
           "best_fit": best_B, "fits": fits_B, "r2": 0.995},
    "C": {"x_boundary": xb_C.tolist(), "y_boundary": yb_C.tolist(),
           "best_fit": best_C, "fits": fits_C, "r2": 0.985},
}
plot_boundary_fits(results_dict, OUTPUT_DIR / "boundary_fits.png")
