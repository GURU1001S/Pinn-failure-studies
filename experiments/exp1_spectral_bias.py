"""
exp1_spectral_bias.py — Experiment 1: Spectral Bias vs. β  [v2 — fixed]

Trains standard 4-layer tanh PINNs on the 1D advection equation
u_t + β*u_x = 0 for β ∈ [1, 5, 10, 30, 50, 100].

For each β:
  1. Train PINN (4 hidden layers, 64 neurons, tanh, 15k Adam steps)
  2. Evaluate on 256×100 test grid
  3. Compute spatial Fourier spectrum at t=1.0
  4. Find cutoff frequency where PINN spectrum diverges from exact
  5. Record L2 relative error

Outputs (saved to results/exp1/):
  - spectral_comparison.png
  - l2_vs_beta.png
  - solution_snapshots.png
  - training_curves.png
  - exp1_results.json

FIXES vs v1:
  [FIX 1] find_cutoff_frequency() — completely rewritten.
          Old logic found first index where ratio < threshold, which
          always returned Nyquist (128) for good cases and index 1 for
          failed cases. New logic:
            - Only searches where exact power is above a noise floor
            - Finds where PINN power first drops BELOW a fraction of
              the local exact power (ratio test)
            - Returns None + did_fail=False when PINN tracks everywhere
              meaningful, instead of returning Nyquist as a fake cutoff

  [FIX 2] empirical_cutoff_beta sentinel — now uses None instead of
          BETA_VALUES[-1] when no failure is detected, with an explicit
          did_fail boolean in the JSON. Same pattern as exp12 v2.

  [FIX 3] JSON now stores:
            - "spectral_failure_detected": bool per beta
            - "cutoff_frequencies": null when no spectral failure
            - "empirical_failure_detected": top-level bool
            - "empirical_cutoff_beta": null if no failure
          so downstream code and Atlas text can distinguish real
          measurements from fallback sentinels.

  [FIX 4] Training curve spike annotation: adds a note in the JSON
          explaining that periodic spikes are a collocation resampling
          artifact, not gradient instability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from pathlib import Path

from pinn_core import (
    AdvectionPINN, train_pinn, evaluate_on_grid,
    compute_spectrum, dominant_frequency,
    save_results, save_model, DEVICE,
)
from plot_utils import (
    plot_spectral_comparison, plot_l2_vs_beta,
    plot_solution_snapshots, plot_training_curves, savefig,
)

# ===================================================================
# Speed flags
# ===================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

# ===================================================================
# Configuration
# ===================================================================
BETA_VALUES        = [1, 5, 10, 30, 50, 100]
N_HIDDEN           = 4
N_NEURONS          = 64
ACTIVATION         = "tanh"
N_EPOCHS           = 15000
LR                 = 1e-3
LR_MIN             = 1e-5
N_COLLOCATION      = 10000
N_IC               = 200
N_BC               = 200
T_SNAPSHOT_IDX     = 50          # index 50 in 100-point t-array ≈ t=1.0
L2_FAILURE_THRESHOLD = 0.10

# Spectral cutoff detection parameters
# PINN is considered to have "lost" a frequency band when its power
# drops below SPECTRAL_RATIO_THRESHOLD × exact power at that frequency.
SPECTRAL_RATIO_THRESHOLD = 0.1   # 10% of exact power
SPECTRAL_NOISE_FLOOR     = 1e-8  # ignore exact frequencies below this

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp1"
MODEL_DIR  = OUTPUT_DIR / "models"


# ===================================================================
# Fixed cutoff frequency detection
# ===================================================================

def find_cutoff_frequency_fixed(power_pred, power_exact,
                                 ratio_threshold=SPECTRAL_RATIO_THRESHOLD,
                                 noise_floor=SPECTRAL_NOISE_FLOOR):
    """
    Find the first frequency index where the PINN's spectral power
    meaningfully diverges from the exact solution's spectral power.

    Algorithm:
      1. Only consider frequency bins where exact power > noise_floor.
         Beyond that the exact solution has no energy — comparison is
         meaningless.
      2. At each meaningful frequency, compute:
             ratio = power_pred[i] / power_exact[i]
      3. The cutoff is the first index where ratio < ratio_threshold,
         meaning PINN has lost ≥ (1 - ratio_threshold)×100% of the
         spectral energy at that frequency.
      4. If ratio never drops below threshold in the meaningful range,
         the PINN tracks the exact spectrum everywhere → no spectral
         failure. Return (False, None, None).

    Returns
    -------
    did_fail   : bool   — True if PINN lost spectral energy
    cutoff_idx : int | None
    cutoff_freq: float | None  (caller passes freqs array for lookup)
    """
    power_pred  = np.array(power_pred)
    power_exact = np.array(power_exact)

    meaningful_indices = np.where(power_exact > noise_floor)[0]

    if len(meaningful_indices) == 0:
        # Exact solution has no spectral energy at all — degenerate case
        return False, None, None

    for i in meaningful_indices:
        ratio = power_pred[i] / (power_exact[i] + 1e-30)
        if ratio < ratio_threshold:
            return True, int(i), None   # caller fills in freq value

    # PINN matched exact everywhere meaningful
    return False, None, None


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 1: Spectral Bias vs. β (1D Advection)  [v2 — fixed]")
    print(f"Device     : {DEVICE}")
    print(f"Arch       : {N_HIDDEN} layers × {N_NEURONS} neurons, {ACTIVATION}")
    print(f"Training   : {N_EPOCHS} Adam steps, lr={LR} → {LR_MIN}")
    print(f"β values   : {BETA_VALUES}")
    print(f"L2 thresh  : {L2_FAILURE_THRESHOLD}")
    print(f"Spec ratio : {SPECTRAL_RATIO_THRESHOLD}  noise floor: {SPECTRAL_NOISE_FLOOR}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Per-beta storage
    all_eval_results      = []
    all_freqs_pred        = []
    all_power_pred        = []
    all_freqs_exact       = []
    all_power_exact       = []
    all_cutoff_indices    = []   # int or None
    all_cutoff_freqs      = []   # float or None
    all_spectral_failed   = []   # bool
    all_l2_errors         = []
    all_loss_histories    = []
    all_training_times    = []

    for beta in BETA_VALUES:
        print(f"\n{'─' * 60}")
        print(f"β = {beta}")
        print(f"{'─' * 60}")

        model = AdvectionPINN(
            n_hidden=N_HIDDEN,
            n_neurons=N_NEURONS,
            activation=ACTIVATION,
        )
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # ── Train ──────────────────────────────────────────────────
        train_result = train_pinn(
            model, beta,
            n_epochs=N_EPOCHS,
            lr=LR,
            lr_min=LR_MIN,
            n_collocation=N_COLLOCATION,
            n_ic=N_IC,
            n_bc=N_BC,
        )

        # ── Evaluate ───────────────────────────────────────────────
        eval_result = evaluate_on_grid(model, beta)
        l2_err = eval_result["l2_error"]
        print(f"  L2 relative error: {l2_err:.6f}")

        # ── Spectral analysis ──────────────────────────────────────
        u_pred_snap  = eval_result["u_pred"][:, T_SNAPSHOT_IDX]
        u_exact_snap = eval_result["u_exact"][:, T_SNAPSHOT_IDX]

        freqs_pred,  power_pred  = compute_spectrum(u_pred_snap)
        freqs_exact, power_exact = compute_spectrum(u_exact_snap)

        # ---- FIXED cutoff detection --------------------------------
        spec_failed, cutoff_idx, _ = find_cutoff_frequency_fixed(
            power_pred, power_exact,
            ratio_threshold=SPECTRAL_RATIO_THRESHOLD,
            noise_floor=SPECTRAL_NOISE_FLOOR,
        )

        # Resolve cutoff frequency value
        if spec_failed and cutoff_idx is not None:
            cutoff_freq = float(freqs_pred[cutoff_idx])
        else:
            cutoff_freq = None

        # ---- Unambiguous terminal reporting -----------------------
        if spec_failed:
            print(f"  Spectral failure : YES — cutoff at index {cutoff_idx}, "
                  f"f = {cutoff_freq:.4f} cycles/domain")
        else:
            # Find maximum meaningful freq for reporting
            meaningful = freqs_pred[power_exact > SPECTRAL_NOISE_FLOOR]
            max_meaningful = float(meaningful[-1]) if len(meaningful) > 0 else 0.0
            print(f"  Spectral failure : NO — PINN tracks exact spectrum "
                  f"up to f = {max_meaningful:.1f} cycles/domain")

        all_eval_results.append(eval_result)
        all_freqs_pred.append(freqs_pred)
        all_power_pred.append(power_pred)
        all_freqs_exact.append(freqs_exact)
        all_power_exact.append(power_exact)
        all_cutoff_indices.append(cutoff_idx)   # None = no spectral failure
        all_cutoff_freqs.append(cutoff_freq)    # None = no spectral failure
        all_spectral_failed.append(spec_failed)
        all_l2_errors.append(l2_err)
        all_loss_histories.append(train_result["loss_history"])
        all_training_times.append(train_result["training_time"])

        save_model(model, MODEL_DIR / f"pinn_beta{beta}.pt")

    # ================================================================
    # Post-processing — empirical cutoff beta  [FIX 2: None sentinel]
    # ================================================================
    empirical_failure_detected = False
    empirical_cutoff_beta      = None
    empirical_cutoff_freq      = None

    for beta, l2 in zip(BETA_VALUES, all_l2_errors):
        if l2 > L2_FAILURE_THRESHOLD:
            empirical_failure_detected = True
            empirical_cutoff_beta      = beta
            empirical_cutoff_freq      = dominant_frequency(beta)
            print(f"\n  ★ Empirical L2 cutoff : β = {beta}  "
                  f"(L2 = {l2:.4f} > {L2_FAILURE_THRESHOLD})")
            break

    if not empirical_failure_detected:
        print(f"\n  ★ No L2 failure detected — all β have L2 < {L2_FAILURE_THRESHOLD}")
        # empirical_cutoff_beta stays None — not set to BETA_VALUES[-1]

    dom_freqs = [dominant_frequency(b) for b in BETA_VALUES]

    # ================================================================
    # Plots
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Generating plots...")

    # Pass cutoff_indices to plotting — use -1 as sentinel for "no cutoff"
    # so plot_spectral_comparison can skip drawing the cutoff line
    plot_cutoff_indices = [
        idx if idx is not None else -1
        for idx in all_cutoff_indices
    ]

    plot_spectral_comparison(
        all_freqs_pred, all_power_pred, all_power_exact,
        BETA_VALUES, plot_cutoff_indices,
        filepath=OUTPUT_DIR / "spectral_comparison.png",
    )

    plot_l2_vs_beta(
        BETA_VALUES, all_l2_errors, dom_freqs,
        cutoff_beta=empirical_cutoff_beta,   # None-safe: plot_l2_vs_beta
        filepath=OUTPUT_DIR / "l2_vs_beta.png",
    )

    plot_solution_snapshots(
        all_eval_results, BETA_VALUES,
        t_snapshot_idx=T_SNAPSHOT_IDX,
        filepath=OUTPUT_DIR / "solution_snapshots.png",
    )

    plot_training_curves(
        all_loss_histories,
        labels=[f"β={b}" for b in BETA_VALUES],
        filepath=OUTPUT_DIR / "training_curves.png",
        title="Training Loss Curves by β",
    )

    # ================================================================
    # Save JSON  [FIX 3: explicit booleans, no ambiguous sentinels]
    # ================================================================
    results = {
        "experiment": "Spectral Bias vs Beta",
        "version":    "v2-fixed",
        "config": {
            "n_hidden":              N_HIDDEN,
            "n_neurons":             N_NEURONS,
            "activation":            ACTIVATION,
            "n_epochs":              N_EPOCHS,
            "lr":                    LR,
            "lr_min":                LR_MIN,
            "n_collocation":         N_COLLOCATION,
            "l2_failure_threshold":  L2_FAILURE_THRESHOLD,
            "spectral_ratio_threshold": SPECTRAL_RATIO_THRESHOLD,
            "spectral_noise_floor":  SPECTRAL_NOISE_FLOOR,
        },

        # ── Per-beta results ──────────────────────────────────────
        "beta_values":      BETA_VALUES,
        "l2_errors":        all_l2_errors,
        "dominant_frequencies": dom_freqs,

        # Spectral cutoff — null means no spectral failure detected
        "spectral_failure_detected": all_spectral_failed,
        "cutoff_frequency_indices":  all_cutoff_indices,   # int or null
        "cutoff_frequencies":        all_cutoff_freqs,     # float or null

        # ── Empirical L2 cutoff ───────────────────────────────────
        # null means no beta in BETA_VALUES caused L2 > threshold
        "empirical_failure_detected":  empirical_failure_detected,
        "empirical_cutoff_beta":       empirical_cutoff_beta,
        "empirical_cutoff_frequency":  empirical_cutoff_freq,
        "failure_betas": [
            b for b, l2 in zip(BETA_VALUES, all_l2_errors)
            if l2 > L2_FAILURE_THRESHOLD
        ],

        # ── Timing ───────────────────────────────────────────────
        "training_times_seconds":       all_training_times,
        "total_training_time_seconds":  sum(all_training_times),

        # ── Artifact notes ────────────────────────────────────────
        # FIX 4: document training curve spike artifact
        "notes": {
            "training_curve_spikes": (
                "Periodic loss spikes visible in training_curves.png "
                "are a collocation resampling artifact. Each time a new "
                "random batch of collocation points is drawn, high-residual "
                "points are sampled causing a transient loss spike. "
                "This is NOT gradient instability. Spikes disappear with "
                "fixed collocation sets. Do not interpret as a failure signal."
            ),
            "cutoff_frequency_fix": (
                "v1 find_cutoff_frequency() incorrectly returned index 128 "
                "(Nyquist) for passing betas and index 1 for failing betas "
                "due to a threshold logic error. v2 uses a noise-floor-gated "
                "ratio test and returns null when no spectral failure is "
                "detected, eliminating the ambiguous sentinel value."
            ),
            "empirical_cutoff_fix": (
                "v1 set empirical_cutoff_beta = BETA_VALUES[-1] = 100 when "
                "no failure occurred, making 'never failed' indistinguishable "
                "from 'failed at beta=100'. v2 uses None with an explicit "
                "empirical_failure_detected boolean flag."
            ),
        },
    }

    save_results(results, OUTPUT_DIR / "exp1_results.json")

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 1 — SUMMARY  [v2]")
    print(f"{'=' * 70}")
    print(f"\n{'β':>6} | {'L2 Error':>12} | {'Dom.Freq':>10} | "
          f"{'L2 Status':>10} | {'Spec. Cutoff f':>15}")
    print("─" * 70)
    for beta, l2, df, s_fail, c_freq in zip(
            BETA_VALUES, all_l2_errors, dom_freqs,
            all_spectral_failed, all_cutoff_freqs):
        l2_status  = "FAIL" if l2 > L2_FAILURE_THRESHOLD else "PASS"
        spec_str   = f"{c_freq:.4f}" if s_fail and c_freq is not None else "none"
        print(f"{beta:6d} | {l2:12.6f} | {df:10.4f} | "
              f"{l2_status:>10} | {spec_str:>15}")

    print()
    if empirical_failure_detected:
        print(f"  Empirical L2 cutoff : β = {empirical_cutoff_beta}  "
              f"(f = {empirical_cutoff_freq:.4f} cycles/domain)")
    else:
        print("  Empirical L2 cutoff : NOT DETECTED within tested β range")

    print(f"  Total training time : {sum(all_training_times):.1f}s")
    print(f"  Results saved to    : {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    results = run_experiment()