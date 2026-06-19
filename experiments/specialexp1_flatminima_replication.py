"""
specialexp1_flatminima_replication.py — Flat-Minima Inversion Replication
[30-seed statistical validation of Exp 5 finding]

Replicates the Exp 5 flat-minima inversion finding across 30 independent
seeds to determine whether it is a general principle or a coincidence
of initialization.

For each seed (0 to 29):
  Train two Burgers PINNs with identical architecture:
    - Model A: standard training, no intervention
    - Model B: same but with BC loss weighted ×10 (bias toward success)

  After training, for each model:
  (1) Classify outcome: converged (L2 < 0.1) or failed (L2 ≥ 0.1)
  (2) Estimate Hessian sharpness via 50-step power iteration (λ_max)
  (3) Record final L2 error and final training loss

After all 30 seeds:
  Split 60 model outcomes into Group C (converged) vs Group F (failed).
  Compute t-test, Cohen's d, violin plots, sharpness scatter.

Outputs (results/specialexp1/):
  - sharpness_distributions.png
  - sharpness_scatter.png
  - specialexp1_results.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from pinn_core import DEVICE, DTYPE, save_results
from pinn_equations import (
    GenericPINN, train_burgers_pinn, evaluate_burgers,
    load_burgers_reference, sample_burgers_domain, burgers_residual,
    BURGERS_NU,
)
from plot_utils import savefig, setup_style

setup_style()

# ===================================================================
# Speed flags
# ===================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

# ===================================================================
# Config
# ===================================================================
N_SEEDS           = 30
N_HIDDEN          = 4
N_NEURONS         = 64
N_EPOCHS          = 30000
LR                = 1e-3
FAILURE_THRESHOLD = 0.1      # L2 < 0.1 = converged, ≥ 0.1 = failed
HESSIAN_N_ITERS   = 50       # power iteration steps
HESSIAN_N_INT     = 3000     # interior points for Hessian loss
HESSIAN_N_IC      = 200
HESSIAN_N_BC      = 200

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "specialexp1"


# ===================================================================
# Hessian max eigenvalue via power iteration
# ===================================================================

def hessian_max_eigenvalue(model, n_iters=HESSIAN_N_ITERS, seed=0):
    """
    Estimate top Hessian eigenvalue λ_max via power iteration on
    Hessian-vector products using double backprop.

    Uses the Burgers total loss: L = L_pde + 10*L_ic + L_bc
    """
    torch.manual_seed(seed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    # Random initial vector
    v = [torch.randn_like(p) for p in params]
    v_norm = sum(vi.norm() ** 2 for vi in v).sqrt()
    v = [vi / v_norm for vi in v]

    # Sample fixed collocation points for stable Hessian estimate
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = \
        sample_burgers_domain(HESSIAN_N_INT, HESSIAN_N_IC, HESSIAN_N_BC)

    eigenvalue = 0.0
    for _ in range(n_iters):
        model.zero_grad()
        res      = burgers_residual(model, x_int, t_int, BURGERS_NU)
        loss_pde = torch.mean(res ** 2)
        loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
        loss_bc  = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
        loss     = loss_pde + 10 * loss_ic + loss_bc

        grads = torch.autograd.grad(loss, params, create_graph=True)
        Hv    = torch.autograd.grad(grads, params, grad_outputs=v,
                                    retain_graph=False)

        eigenvalue = sum((hvi * vi).sum()
                         for hvi, vi in zip(Hv, v)).item()
        v_norm     = sum(hvi.norm() ** 2 for hvi in Hv).sqrt()
        v          = [hvi / (v_norm + 1e-10) for hvi in Hv]
        v          = [vi.detach() for vi in v]

    return abs(eigenvalue)


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    t_start = time.time()
    print("=" * 70)
    print("SPECIAL EXP 1: Flat-Minima Inversion Replication (30 seeds)")
    print(f"Device          : {DEVICE}")
    print(f"Seeds           : {N_SEEDS}")
    print(f"Epochs per model: {N_EPOCHS}")
    print(f"Hessian iters   : {HESSIAN_N_ITERS}")
    print(f"Failure threshold: L2 ≥ {FAILURE_THRESHOLD}")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Burgers reference solution
    x_ref, t_ref, u_ref = load_burgers_reference()

    # ── Checkpoint loading ──────────────────────────────────────────
    checkpoint_path = OUTPUT_DIR / "specialexp1_checkpoint.json"
    all_models_data = []   # list of dicts, one per model

    if checkpoint_path.exists():
        print(f"  [Checkpoint] Loading from {checkpoint_path.name}")
        try:
            with open(checkpoint_path, 'r') as f:
                all_models_data = json.load(f)
            print(f"  [Checkpoint] Loaded {len(all_models_data)} model records")
        except Exception as e:
            print(f"  [Checkpoint] Failed: {e}")
            all_models_data = []

    # Build set of completed (seed, variant) pairs
    completed = set()
    for rec in all_models_data:
        completed.add((rec["seed"], rec["variant"]))

    # ── Training loop ───────────────────────────────────────────────
    for seed in range(N_SEEDS):
        for variant, lambda_bc in [("standard", 1.0), ("bc_weighted", 10.0)]:
            if (seed, variant) in completed:
                print(f"\n  Seed {seed} / {variant} [loaded from checkpoint]")
                continue

            print(f"\n  Seed {seed} / {variant} (λ_bc={lambda_bc})")
            t_seed = time.time()

            torch.manual_seed(seed)
            np.random.seed(seed)

            model = GenericPINN(
                in_dim=2, out_dim=1,
                n_hidden=N_HIDDEN, n_neurons=N_NEURONS,
                activation="tanh"
            ).to(DEVICE)

            # Train
            result = train_burgers_pinn(
                model, n_epochs=N_EPOCHS, lr=LR,
                n_int=10000, n_ic=200, n_bc=200,
                lambda_bc=lambda_bc,
                log_every=10000, verbose=True
            )
            trained_model = result["model"]

            # Evaluate L2
            _, l2 = evaluate_burgers(trained_model, x_ref, t_ref, u_ref)
            converged = l2 < FAILURE_THRESHOLD
            final_loss = result["loss_history"][-1] if result["loss_history"] else float("nan")

            print(f"    L2={l2:.6f}  [{'CONVERGED' if converged else 'FAILED'}]")

            # Hessian sharpness
            print(f"    Computing Hessian λ_max ({HESSIAN_N_ITERS} power iterations)...")
            lambda_max = hessian_max_eigenvalue(trained_model, seed=seed + 1000)
            print(f"    λ_max={lambda_max:.4e}  ({time.time() - t_seed:.1f}s total)")

            record = {
                "seed":       seed,
                "variant":    variant,
                "lambda_bc":  lambda_bc,
                "l2_error":   float(l2),
                "converged":  bool(converged),
                "final_loss": float(final_loss),
                "lambda_max": float(lambda_max),
            }
            all_models_data.append(record)
            completed.add((seed, variant))

            # Save checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump(all_models_data, f, indent=2)

            # Free memory
            del model, trained_model, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Statistical Analysis ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Statistical Analysis")
    print(f"{'=' * 70}")

    group_c = [r for r in all_models_data if r["converged"]]
    group_f = [r for r in all_models_data if not r["converged"]]

    n_c = len(group_c)
    n_f = len(group_f)
    print(f"  Group C (converged, L2 < {FAILURE_THRESHOLD}): {n_c} models")
    print(f"  Group F (failed,    L2 ≥ {FAILURE_THRESHOLD}): {n_f} models")

    if n_c >= 2 and n_f >= 2:
        sharp_c = [r["lambda_max"] for r in group_c]
        sharp_f = [r["lambda_max"] for r in group_f]

        mean_c, std_c = np.mean(sharp_c), np.std(sharp_c)
        mean_f, std_f = np.mean(sharp_f), np.std(sharp_f)

        # Two-sample t-test (Welch's)
        t_stat, p_value = stats.ttest_ind(sharp_c, sharp_f, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt(((n_c - 1) * std_c**2 + (n_f - 1) * std_f**2) /
                             (n_c + n_f - 2))
        cohens_d = (mean_c - mean_f) / (pooled_std + 1e-10)

        # Direction
        if mean_f < mean_c:
            direction = "CONFIRMED: failed models have LOWER λ_max (flatter minima)"
            inversion_confirmed = True
        else:
            direction = "REJECTED: failed models have HIGHER λ_max (sharper minima)"
            inversion_confirmed = False

        significant = p_value < 0.05

        print(f"\n  λ_max (converged): {mean_c:.4e} ± {std_c:.4e}")
        print(f"  λ_max (failed):    {mean_f:.4e} ± {std_f:.4e}")
        print(f"  t-statistic:       {t_stat:.4f}")
        print(f"  p-value:           {p_value:.6f}")
        print(f"  Cohen's d:         {cohens_d:.4f}")
        print(f"  Direction:         {direction}")
        print(f"  Significant (p<0.05): {significant}")
    else:
        print("  ⚠ Insufficient models in one group for t-test")
        mean_c = mean_f = std_c = std_f = float("nan")
        t_stat = p_value = cohens_d = float("nan")
        direction = "INSUFFICIENT DATA"
        inversion_confirmed = False
        significant = False
        sharp_c = [r["lambda_max"] for r in group_c] if group_c else []
        sharp_f = [r["lambda_max"] for r in group_f] if group_f else []

    # ── Plots ───────────────────────────────────────────────────────
    print("\n── Generating plots ──")

    # 1. Violin / box plots of λ_max distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: violin plot
    ax = axes[0]
    data_for_violin = []
    labels_for_violin = []
    colors_violin = []
    if sharp_c:
        data_for_violin.append(sharp_c)
        labels_for_violin.append(f"Converged\n(n={n_c})")
        colors_violin.append("#2E7D32")
    if sharp_f:
        data_for_violin.append(sharp_f)
        labels_for_violin.append(f"Failed\n(n={n_f})")
        colors_violin.append("#D32F2F")

    if len(data_for_violin) >= 2:
        parts = ax.violinplot(data_for_violin, positions=range(len(data_for_violin)),
                              showmeans=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors_violin[i])
            pc.set_alpha(0.4)

        # Overlay individual points
        for i, (dv, col) in enumerate(zip(data_for_violin, colors_violin)):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(dv))
            ax.scatter(np.full(len(dv), i) + jitter, dv,
                       color=col, alpha=0.7, s=30, zorder=5,
                       edgecolors="white", linewidths=0.5)

        ax.set_xticks(range(len(labels_for_violin)))
        ax.set_xticklabels(labels_for_violin)
    elif len(data_for_violin) == 1:
        ax.boxplot(data_for_violin)
        ax.set_xticklabels(labels_for_violin)

    ax.set_ylabel("Hessian λ_max (sharpness)", fontsize=12)
    ax.set_title(
        "Sharpness Distribution: Converged vs Failed\n"
        f"p={p_value:.4f}  Cohen's d={cohens_d:.2f}",
        fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Right: summary statistics
    ax2 = axes[1]
    if n_c >= 1 and n_f >= 1:
        x_pos = [0, 1]
        means = [mean_c, mean_f]
        stds  = [std_c, std_f]
        bar_colors = ["#2E7D32", "#D32F2F"]
        bars = ax2.bar(x_pos, means, yerr=stds, color=bar_colors,
                       alpha=0.8, edgecolor="white", capsize=8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"Converged\n(n={n_c})", f"Failed\n(n={n_f})"])

        # Significance annotation
        if significant:
            y_max = max(means[0] + stds[0], means[1] + stds[1])
            ax2.plot([0, 0, 1, 1],
                     [y_max * 1.05, y_max * 1.1, y_max * 1.1, y_max * 1.05],
                     "k-", lw=1.5)
            ax2.text(0.5, y_max * 1.12, f"p={p_value:.4f} *",
                     ha="center", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Mean λ_max ± std", fontsize=12)
    ax2.set_title(
        "Mean Hessian Sharpness\n"
        f"{'★ Flat-minima inversion CONFIRMED' if inversion_confirmed and significant else '✗ Inversion not confirmed'}",
        fontweight="bold", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"SpecialExp1: Flat-Minima Inversion Replication ({N_SEEDS} seeds × 2 models)\n"
        f"If failed models have LOWER λ_max → flat minima ≠ good generalization in PINNs",
        fontweight="bold", fontsize=13)
    savefig(fig, OUTPUT_DIR / "sharpness_distributions.png")

    # 2. Scatter: λ_max vs L2 error
    fig, ax = plt.subplots(figsize=(10, 7))

    for rec in all_models_data:
        color = "#2E7D32" if rec["converged"] else "#D32F2F"
        marker = "o" if rec["variant"] == "standard" else "s"
        ax.scatter(rec["lambda_max"], rec["l2_error"],
                   color=color, marker=marker, s=60, alpha=0.7,
                   edgecolors="white", linewidths=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E7D32",
               markersize=10, label="Converged / Standard"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2E7D32",
               markersize=10, label="Converged / BC×10"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D32F2F",
               markersize=10, label="Failed / Standard"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#D32F2F",
               markersize=10, label="Failed / BC×10"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    ax.axhline(FAILURE_THRESHOLD, color="orange", linestyle="--",
               linewidth=2, alpha=0.7,
               label=f"Failure threshold (L2={FAILURE_THRESHOLD})")

    ax.set_xlabel("Hessian λ_max (sharpness)", fontsize=12)
    ax.set_ylabel("L2 Relative Error", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        "Sharpness vs L2 Error — All 60 Models\n"
        "If flat-minima inversion is real, failed models cluster at LOWER λ_max",
        fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    savefig(fig, OUTPUT_DIR / "sharpness_scatter.png")

    # ── JSON results ────────────────────────────────────────────────
    results = {
        "experiment": "Flat-Minima Inversion Replication",
        "version": "specialexp1",
        "config": {
            "n_seeds":           N_SEEDS,
            "n_hidden":          N_HIDDEN,
            "n_neurons":         N_NEURONS,
            "n_epochs":          N_EPOCHS,
            "lr":                LR,
            "failure_threshold": FAILURE_THRESHOLD,
            "hessian_n_iters":   HESSIAN_N_ITERS,
        },
        "all_models": all_models_data,
        "statistics": {
            "n_converged":        n_c,
            "n_failed":           n_f,
            "mean_lambda_max_converged": float(mean_c),
            "std_lambda_max_converged":  float(std_c),
            "mean_lambda_max_failed":    float(mean_f),
            "std_lambda_max_failed":     float(std_f),
            "t_statistic":               float(t_stat),
            "p_value":                   float(p_value),
            "cohens_d":                  float(cohens_d),
            "direction":                 direction,
            "significant_p005":          bool(significant),
            "inversion_confirmed":       bool(inversion_confirmed),
        },
        "hypothesis_note": (
            "Exp5 found that FAILED Burgers PINNs converge to FLATTER "
            "minima (lower Hessian λ_max) than successful ones — the "
            "opposite of the deep learning folklore where flat minima = "
            "good generalization. This experiment replicates that finding "
            f"across {N_SEEDS} independent seeds with statistical testing. "
            f"{'The inversion is CONFIRMED (p<0.05).' if inversion_confirmed and significant else 'The inversion is NOT statistically confirmed.'} "
            f"Hypothesis 2 confidence: {'90%' if inversion_confirmed and significant else '50%'}."
        ),
    }

    save_results(results, OUTPUT_DIR / "specialexp1_results.json")

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"SPECIAL EXP 1 — COMPLETE")
    print(f"  Total wall time: {total_elapsed / 60:.1f} min")
    print(f"  Models: {n_c} converged, {n_f} failed")
    print(f"  Inversion confirmed: {inversion_confirmed and significant}")
    print(f"  Results → {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    return results


if __name__ == "__main__":
    run_experiment()
