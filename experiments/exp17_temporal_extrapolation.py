"""
exp17_temporal_extrapolation.py — Temporal Extrapolation Failure Study
[v2 — journal-ready fixes]

Trains PINNs on the 1D advection equation (β=10) for T_train ∈ [0.5, 1.0, 2.0]
and evaluates on [0, T_EVAL=10]. (T_train=5.0 dropped — see FIX 1.)

Protocol:
  1. For each T_train: train, evaluate on [0, T_eval], record error vs t
  2. Find extrapolation ratio R* = T_cross / T_train at error thresholds
  3. Fit growth models (linear / exponential / power-law) for t > T_train
  4. Classify failure sharpness at the extrapolation boundary

Outputs (results/exp17/):
  - error_vs_time.png
  - phase_diagram.png
  - error_growth_models.png
  - spatial_error_snapshots.png
  - exp17_results.json

FIXES vs v1 (journal-ready):
  [FIX 1] T_train=5.0 removed from default sweep. β=10 over T_train=5.0
          requires the PINN to learn ~8 full wave oscillations (β*T/(2π)≈8).
          v1 produced u≡0 for this case (mean_error_in=0.52) — the model
          failed entirely within the training domain. Extrapolation results
          from a completely failed model are meaningless. v2 trains each
          T_train and checks in-training L2; configs failing in-domain
          (mean_error_in > IN_DOMAIN_FAIL_THRESH) are excluded from
          extrapolation analysis and documented in JSON.

  [FIX 2] R2 threshold for growth model claims — v1 reported "best fit:
          power_law" at R2=0.04–0.10 for T_train=0.5 and T_train=1.0.
          All three candidate models (linear, exponential, power_law)
          fit essentially equally poorly on data that jumps to a plateau.
          v2 adds R2_MIN=0.50: if no model exceeds this threshold, reports
          "no reliable growth fit — error saturates immediately" instead
          of claiming a meaningless best model.

  [FIX 3] Sentinel value handling for R* — v1 returned T_EVAL (=10.0)
          when the error threshold was never crossed, then divided by
          T_train to get R*=20.0 for T_train=0.5. This sentinel was
          plotted as a data point on the phase diagram, creating a spike
          to R*=20 that dominated the figure visually. v2 distinguishes
          between "threshold never crossed" (R*=None, plotted as open
          marker at T_EVAL) and "threshold crossed at T_cross" so the
          phase diagram is not distorted by boundary artifacts.

  [FIX 4] Explicit seed for reproducibility. v1 had no seed control.

  [FIX 5] Growth model labels corrected — v1 labeled fits as
          "0.007t + 0.621" where t means time-since-T_train, but the
          plot x-axis showed absolute t. v2 labels as
          "0.007·(t−T_train) + 0.621" to prevent misreading of slope.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.optimize import curve_fit

# ===================================================================
# Speed flags
# ===================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp17] Device: {DEVICE}")

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp17"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Configuration
# ===================================================================
BETA         = 10
# FIX 1: T_train=5.0 removed — β=10 over 5.0 requires ~8 oscillations
# which exceeds tanh PINN capacity (mean_error_in=0.52 in v1)
T_TRAIN_LIST = [0.5, 1.0, 2.0]
T_EVAL       = 10.0
SEED         = 42        # FIX 4

N_HIDDEN   = 4
N_NEURONS  = 128
ACTIVATION = "tanh"

N_EPOCHS      = 20000
LR            = 1e-3
LR_MIN        = 1e-5
N_COLLOCATION = 8000
N_IC          = 300
N_BC          = 300

NX_EVAL = 200
NT_EVAL = 400

ERROR_THRESHOLDS = [0.10, 0.50, 1.00]

# FIX 1: if mean in-domain error exceeds this, model is considered failed
IN_DOMAIN_FAIL_THRESH = 0.10

# FIX 2: minimum R2 to claim a growth fit
R2_MIN = 0.50


# ===================================================================
# Model
# ===================================================================

class AdvectionPINN(nn.Module):
    def __init__(self, n_hidden=4, n_neurons=128, activation="tanh"):
        super().__init__()
        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
        layers  = [nn.Linear(2, n_neurons), act_map[activation]()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), act_map[activation]()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def exact_solution(x, t):
    return np.sin(x - BETA * t)


# ===================================================================
# PDE loss
# ===================================================================

def pde_residual(model, x, t):
    x = x.requires_grad_(True); t = t.requires_grad_(True)
    u   = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return u_t + BETA * u_x


def compute_loss(model, T_train, n_col, n_ic, n_bc):
    x_col = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    t_col = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * T_train
    loss_pde = (pde_residual(model, x_col, t_col) ** 2).mean()

    x_ic = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    t_ic = torch.zeros(n_ic, 1, dtype=DTYPE, device=DEVICE)
    loss_ic = ((model(x_ic, t_ic) - torch.sin(x_ic)) ** 2).mean()

    t_bc   = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE) * T_train
    x_bc_l = torch.zeros(n_bc, 1, dtype=DTYPE, device=DEVICE)
    x_bc_r = torch.full((n_bc, 1), 2 * np.pi, dtype=DTYPE, device=DEVICE)
    loss_bc = ((model(x_bc_l, t_bc) - model(x_bc_r, t_bc)) ** 2).mean()

    return loss_pde + 100 * loss_ic + 10 * loss_bc


# ===================================================================
# Training
# ===================================================================

def train_model(T_train, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n  Training T_train={T_train}  seed={seed}")
    model     = AdvectionPINN(N_HIDDEN, N_NEURONS, ACTIVATION).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = compute_loss(model, T_train, N_COLLOCATION, N_IC, N_BC)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        if epoch % 5000 == 0:
            print(f"    Epoch {epoch:>6d}: loss={loss.item():.4e}")
    return model


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_extrapolation(model, T_train):
    x_vals = np.linspace(0, 2 * np.pi, NX_EVAL)
    t_vals = np.linspace(0, T_EVAL,    NT_EVAL)
    XX, TT = np.meshgrid(x_vals, t_vals)   # (NT, NX)

    model.eval()
    with torch.no_grad():
        x_f = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        t_f = torch.tensor(TT.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u_pred = model(x_f, t_f).cpu().numpy().reshape(XX.shape)

    u_exact = exact_solution(XX, TT)
    norm    = np.abs(u_exact).max(axis=1, keepdims=True) + 1e-8
    profile = (np.abs(u_pred - u_exact) / norm).mean(axis=1)   # (NT,)

    mask_in  = t_vals <= T_train
    mask_out = t_vals >  T_train
    err_in   = float(profile[mask_in].mean())
    err_out  = float(profile[mask_out].mean()) if mask_out.any() else np.nan

    return t_vals, profile, u_pred, u_exact, x_vals, err_in, err_out


# ===================================================================
# FIX 3 — Crossing time with sentinel handling
# ===================================================================

def find_crossing_time(t_vals, profile, threshold, T_train):
    """
    Returns (t_cross: float|None, crossed: bool).
    None means threshold was never crossed in [T_train, T_EVAL].
    FIX 3: v1 returned T_EVAL as sentinel, causing R*=T_EVAL/T_train=20.
    """
    mask = t_vals >= T_train
    if not mask.any():
        return None, False
    t_ext = t_vals[mask]; p_ext = profile[mask]
    idx   = np.where(p_ext >= threshold)[0]
    if len(idx) == 0:
        return None, False
    i = idx[0]
    if i == 0:
        return float(t_ext[0]), True
    t0, t1 = t_ext[i-1], t_ext[i]
    p0, p1 = p_ext[i-1], p_ext[i]
    if abs(p1 - p0) < 1e-12:
        return float(t1), True
    return float(t0 + (threshold - p0) * (t1 - t0) / (p1 - p0)), True


# ===================================================================
# FIX 2+5 — Growth model fitting with R2 threshold and corrected labels
# ===================================================================

def fit_growth_models(t_vals, profile, T_train):
    """
    FIX 2: requires R2 >= R2_MIN to claim a fit.
    FIX 5: labels use (t - T_train) to avoid slope misreading.
    """
    mask  = (t_vals > T_train) & (profile > 0)
    if mask.sum() < 5:
        return {}, False, "insufficient_data"

    t_rel = t_vals[mask] - T_train   # time relative to boundary
    p_ext = profile[mask]
    fits  = {}

    # Linear
    try:
        def linear_fn(t, a, b): return a * t + b
        popt, _ = curve_fit(linear_fn, t_rel, p_ext,
                            p0=[0.1, p_ext[0]], maxfev=3000)
        pred   = linear_fn(t_rel, *popt)
        ss_res = np.sum((p_ext - pred) ** 2)
        ss_tot = np.sum((p_ext - p_ext.mean()) ** 2) + 1e-12
        r2     = float(1 - ss_res / ss_tot)
        # FIX 5: label uses (t-T_train)
        fits["linear"] = {
            "params": list(popt), "r2": r2,
            "label": f"linear: {popt[0]:.3f}·(t−{T_train}) + {popt[1]:.3f}"}
    except Exception:
        pass

    # Exponential
    try:
        def exp_fn(t, a, b): return a * np.exp(np.clip(b * t, -20, 20))
        popt, _ = curve_fit(exp_fn, t_rel, p_ext,
                            p0=[p_ext[0], 0.1], maxfev=5000,
                            bounds=([0, -10], [np.inf, 10]))
        pred   = exp_fn(t_rel, *popt)
        ss_res = np.sum((p_ext - pred) ** 2)
        ss_tot = np.sum((p_ext - p_ext.mean()) ** 2) + 1e-12
        r2     = float(1 - ss_res / ss_tot)
        fits["exponential"] = {
            "params": list(popt), "r2": r2,
            "label": f"exp: {popt[0]:.3f}·e^({popt[1]:.3f}·(t−{T_train}))"}
    except Exception:
        pass

    # Power law
    try:
        def power_fn(t, a, b): return a * (t + 1) ** b
        popt, _ = curve_fit(power_fn, t_rel, p_ext,
                            p0=[p_ext[0], 1.0], maxfev=5000,
                            bounds=([0, 0], [np.inf, 20]))
        pred   = power_fn(t_rel, *popt)
        ss_res = np.sum((p_ext - pred) ** 2)
        ss_tot = np.sum((p_ext - p_ext.mean()) ** 2) + 1e-12
        r2     = float(1 - ss_res / ss_tot)
        fits["power_law"] = {
            "params": list(popt), "r2": r2,
            "label": f"power: {popt[0]:.3f}·((t−{T_train})+1)^{popt[1]:.2f}"}
    except Exception:
        pass

    if not fits:
        return {}, False, "fit_failed"

    best    = max(fits, key=lambda k: fits[k]["r2"])
    max_r2  = fits[best]["r2"]
    # FIX 2: threshold check
    reliable = max_r2 >= R2_MIN
    fits["best"] = best if reliable else None
    fits["max_r2"] = max_r2
    fits["reliable"] = reliable

    reason = best if reliable else f"no_reliable_fit (max R²={max_r2:.3f} < {R2_MIN})"
    return fits, reliable, reason


def classify_failure_sharpness(t_vals, profile, T_train, window=0.5):
    mb = (t_vals >= T_train - window) & (t_vals < T_train)
    ma = (t_vals >= T_train)          & (t_vals < T_train + window)
    if not (mb.any() and ma.any()):
        return "unknown", 0.0
    gb = np.mean(np.abs(np.gradient(np.log(profile[mb] + 1e-8), t_vals[mb])))
    ga = np.mean(np.abs(np.gradient(np.log(profile[ma] + 1e-8), t_vals[ma])))
    ratio = ga / (gb + 1e-12)
    return ("sudden" if ratio > 3.0 else "gradual"), float(ratio)


# ===================================================================
# Plotting
# ===================================================================

COLORS = cm.plasma(np.linspace(0.1, 0.85, len(T_TRAIN_LIST)))


def plot_error_vs_time(all_results, failed_configs, filepath):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Temporal Extrapolation Error (β={BETA})", fontsize=13,
                 fontweight="bold")

    for i, T_train in enumerate(T_TRAIN_LIST):
        t_vals  = all_results[T_train]["t_vals"]
        profile = all_results[T_train]["profile"]
        failed  = T_train in failed_configs
        lw      = 1.0 if failed else 2.0
        ls      = ":" if failed else "-"
        label   = (f"T_train={T_train}"
                   + (" [FAILED in-domain]" if failed else ""))
        ax.semilogy(t_vals, profile + 1e-6, color=COLORS[i],
                    lw=lw, ls=ls, label=label)
        ax.axvline(T_train, color=COLORS[i], ls=":", alpha=0.4, lw=1)

    for thresh in ERROR_THRESHOLDS:
        ax.axhline(thresh, color="grey", ls="--", alpha=0.5, lw=0.8,
                   label=f"ε={int(thresh*100)}%")

    # FIX 1: annotate failed configs
    if failed_configs:
        ax.text(0.98, 0.02,
                f"⚠ Dashed/dotted = failed in-domain (err_in > {IN_DOMAIN_FAIL_THRESH})\n"
                "Extrapolation from failed model is not reported.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="#FFF9C4",
                          edgecolor="#F9A825", alpha=0.9))

    ax.set_xlabel("t", fontsize=11)
    ax.set_ylabel("Mean Relative Error", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(0, T_EVAL)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_phase_diagram(all_results, valid_configs, filepath):
    """FIX 3: None crossings shown as open markers, not sentinel values."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(f"Extrapolation Phase Diagram (β={BETA})", fontsize=13,
                 fontweight="bold")
    ls_map = {0.10: "-", 0.50: "--", 1.00: ":"}
    mk_map = {0.10: "o", 0.50: "s", 1.00: "D"}

    ax = axes[0]
    for thresh in ERROR_THRESHOLDS:
        t_crosses, t_crosses_valid = [], []
        T_valid = []
        for T_train in valid_configs:
            crossing = all_results[T_train]["crossings"][thresh]
            if crossing["crossed"]:
                t_crosses.append(crossing["t_cross"])
                T_valid.append(T_train)
            # uncrossed → open marker at T_EVAL
        if T_valid:
            ax.plot(T_valid, t_crosses,
                    ls=ls_map[thresh], marker=mk_map[thresh],
                    lw=2, label=f"ε={int(thresh*100)}%")
        # Mark uncrossed
        for T_train in valid_configs:
            if not all_results[T_train]["crossings"][thresh]["crossed"]:
                ax.scatter([T_train], [T_EVAL], marker=mk_map[thresh],
                           s=80, facecolors="none",
                           edgecolors=f"C{ERROR_THRESHOLDS.index(thresh)}",
                           linewidths=1.5,
                           label=f"ε={int(thresh*100)}% never crossed")

    ax.plot([0, max(valid_configs)]*2,
            [0, max(valid_configs)]*2, "k:", lw=1,
            label="T_cross = T_train")
    ax.set_xlabel("T_train", fontsize=11)
    ax.set_ylabel("T_cross", fontsize=11)
    ax.set_title("Crossing Time vs Training Window", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for thresh in ERROR_THRESHOLDS:
        r_stars, T_r = [], []
        for T_train in valid_configs:
            crossing = all_results[T_train]["crossings"][thresh]
            if crossing["crossed"]:
                r_stars.append(crossing["t_cross"] / T_train)
                T_r.append(T_train)
        if T_r:
            ax2.plot(T_r, r_stars,
                     ls=ls_map[thresh], marker=mk_map[thresh],
                     lw=2, label=f"ε={int(thresh*100)}%")

    ax2.axhline(1.0, color="k", ls=":", lw=1, label="R*=1")
    ax2.set_xlabel("T_train", fontsize=11)
    ax2.set_ylabel("R* = T_cross / T_train", fontsize=11)
    ax2.set_title("Extrapolation Ratio at Error Threshold", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_growth_models(all_results, valid_configs, filepath):
    n  = len(valid_configs)
    nc = min(n, 3); nr = (n + nc - 1) // nc
    fig, axes_all = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr),
                                  constrained_layout=True)
    axes_flat = np.array(axes_all).ravel() if n > 1 else [axes_all]
    fig.suptitle(
        f"Error Growth Models Beyond T_train (β={BETA})\n"
        f"(Fit only plotted if R² ≥ {R2_MIN} — FIX 2)",
        fontsize=12, fontweight="bold")

    model_colors = {"linear": "#E64040", "exponential": "#F59F00",
                    "power_law": "#3A7FD5"}

    for i, T_train in enumerate(valid_configs):
        ax     = axes_flat[i]
        t_vals = all_results[T_train]["t_vals"]
        prof   = all_results[T_train]["profile"]
        fits   = all_results[T_train]["growth_fits"]
        rel    = all_results[T_train]["fit_reliable"]
        reason = all_results[T_train]["fit_reason"]

        ax.semilogy(t_vals, prof + 1e-6, "k-", lw=1.5, alpha=0.6,
                    label="Actual error")
        ax.axvspan(0, T_train, alpha=0.08, color="green")
        ax.axvline(T_train, color="green", ls="--", lw=1.5,
                   label=f"T_train={T_train}")

        if rel:
            mask   = t_vals > T_train
            t_abs  = t_vals[mask]
            t_rel  = t_abs - T_train

            for mn in ["linear", "exponential", "power_law"]:
                if mn not in fits:
                    continue
                p  = fits[mn]["params"]
                r2 = fits[mn]["r2"]
                lbl = fits[mn]["label"] + f"  (R²={r2:.2f})"
                if mn == "linear":
                    y = p[0] * t_rel + p[1]
                elif mn == "exponential":
                    y = p[0] * np.exp(np.clip(p[1] * t_rel, -20, 20))
                elif mn == "power_law":
                    y = p[0] * (t_rel + 1) ** p[1]
                else:
                    continue
                pos = y > 0
                if pos.any():
                    ax.semilogy(t_abs[pos], y[pos],
                                color=model_colors.get(mn, "grey"),
                                lw=1.5, ls="--", label=lbl)
        else:
            ax.text(0.5, 0.5,
                    f"No reliable fit\n{reason}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="gray",
                    bbox=dict(boxstyle="round", facecolor="#F5F5F5",
                              edgecolor="gray", alpha=0.8))

        best_str = fits.get("best") or "none"
        ax.set_title(f"T_train={T_train}  (best: {best_str})", fontsize=10)
        ax.set_xlabel("t  (x-axis: absolute time)", fontsize=9)
        ax.set_ylabel("Mean Relative Error", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, which="both")
        ax.set_xlim(0, T_EVAL)

    for ax in axes_flat[len(valid_configs):]:
        ax.set_visible(False)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_spatial_snapshots(all_results, valid_configs, filepath):
    n_rows = len(valid_configs); n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3 * n_rows),
                              constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Spatial Snapshots: u_pred vs u_exact (β={BETA})",
                 fontsize=13, fontweight="bold")

    for i, T_train in enumerate(valid_configs):
        t_vals  = all_results[T_train]["t_vals"]
        x_vals  = all_results[T_train]["x_vals"]
        u_pred  = all_results[T_train]["u_pred"]
        u_exact = all_results[T_train]["u_exact"]

        snaps = [T_train,
                 min(2 * T_train, T_EVAL),
                 T_EVAL]
        lbls  = ["T_train", "2·T_train", "T_eval"]

        for j, (t_snap, lbl) in enumerate(zip(snaps, lbls)):
            ax     = axes[i, j]
            t_idx  = np.argmin(np.abs(t_vals - t_snap))
            t_actual = t_vals[t_idx]
            ax.plot(x_vals, u_exact[t_idx], "k-",  lw=1.5, label="Exact")
            ax.plot(x_vals, u_pred[t_idx],  "r--", lw=1.5, label="PINN")
            ax.set_title(f"T_train={T_train}, t={t_actual:.1f} ({lbl})",
                         fontsize=9)
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("u", fontsize=8)
            ax.legend(fontsize=7)
            ax.set_ylim(-1.5, 1.5)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 17: Temporal Extrapolation Failure  [v2]")
    print(f"Device      : {DEVICE}")
    print(f"β={BETA}  T_train={T_TRAIN_LIST}  T_eval={T_EVAL}")
    print(f"Seed        : {SEED}  (FIX 4)")
    print(f"R2_MIN      : {R2_MIN}  (FIX 2)")
    print(f"IN_FAIL_THRESH: {IN_DOMAIN_FAIL_THRESH}  (FIX 1)")
    print("=" * 70)

    t0          = time.time()
    all_results = {}
    failed_configs = []
    valid_configs  = []

    for T_train in T_TRAIN_LIST:
        print(f"\n{'━' * 60}")
        print(f"T_train = {T_train}")
        print(f"{'━' * 60}")

        model = train_model(T_train, SEED)

        (t_vals, profile, u_pred, u_exact,
         x_vals, err_in, err_out) = evaluate_extrapolation(model, T_train)

        # FIX 1: in-domain failure check
        in_domain_failed = err_in > IN_DOMAIN_FAIL_THRESH
        if in_domain_failed:
            failed_configs.append(T_train)
            print(f"  ⚠ IN-DOMAIN FAILURE: err_in={err_in:.4f} > "
                  f"{IN_DOMAIN_FAIL_THRESH}. Extrapolation not reported.")
        else:
            valid_configs.append(T_train)

        # Crossing times (FIX 3: returns dict with crossed bool)
        crossings = {}
        for thresh in ERROR_THRESHOLDS:
            t_cross, crossed = find_crossing_time(t_vals, profile, thresh, T_train)
            r_star = (t_cross / T_train) if crossed else None
            crossings[thresh] = {
                "t_cross":  t_cross,
                "crossed":  crossed,
                "r_star":   r_star,
            }
            sym = f"R*={r_star:.2f}" if crossed else "never crossed"
            print(f"  ε={int(thresh*100):3d}%: {sym}")

        # Growth fits (FIX 2+5)
        fits, fit_reliable, fit_reason = fit_growth_models(
            t_vals, profile, T_train)
        print(f"  Growth fit: {fit_reason}")

        # Sharpness
        sharp_label, sharp_ratio = classify_failure_sharpness(
            t_vals, profile, T_train)
        print(f"  Sharpness: {sharp_label}  (ratio={sharp_ratio:.2f})")
        print(f"  err_in={err_in:.4f}  err_out={err_out:.4f}")

        all_results[T_train] = {
            "t_vals":          t_vals,
            "profile":         profile,
            "u_pred":          u_pred,
            "u_exact":         u_exact,
            "x_vals":          x_vals,
            "err_in":          err_in,
            "err_out":         err_out,
            "in_domain_failed": in_domain_failed,
            "crossings":       crossings,
            "growth_fits":     fits,
            "fit_reliable":    fit_reliable,
            "fit_reason":      fit_reason,
            "sharpness":       {"label": sharp_label, "ratio": sharp_ratio},
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # ── Plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_error_vs_time(
        all_results, failed_configs,
        OUTPUT_DIR / "error_vs_time.png")
    plot_phase_diagram(
        all_results, valid_configs,
        OUTPUT_DIR / "phase_diagram.png")
    plot_growth_models(
        all_results, valid_configs,
        OUTPUT_DIR / "error_growth_models.png")
    plot_spatial_snapshots(
        all_results, valid_configs,
        OUTPUT_DIR / "spatial_error_snapshots.png")

    # ── JSON ─────────────────────────────────────────────────────────
    def safe_float(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    results_json = {
        "experiment": "Temporal Extrapolation Failure Study",
        "version":    "v2-journal-ready",
        "config": {
            "beta":              BETA,
            "seed":              SEED,
            "T_train_list":      T_TRAIN_LIST,
            "T_eval":            T_EVAL,
            "n_hidden":          N_HIDDEN,
            "n_neurons":         N_NEURONS,
            "n_epochs":          N_EPOCHS,
            "error_thresholds":  ERROR_THRESHOLDS,
            "in_domain_fail_thresh": IN_DOMAIN_FAIL_THRESH,
            "r2_min":            R2_MIN,
        },
        "fix_notes": {
            "fix1_t5_removed": (
                "T_train=5.0 was in v1 but produced mean_error_in=0.52 — "
                "the model failed completely in-domain (β=10 over 5.0 "
                "requires ~8 wave oscillations, exceeding tanh PINN capacity). "
                f"v2 excludes T_train configs with err_in > {IN_DOMAIN_FAIL_THRESH}. "
                "Failed configs are still trained and reported but excluded from "
                "extrapolation figures."
            ),
            "fix2_r2_threshold": (
                f"v1 reported 'best fit: power_law' at R2=0.04–0.10. "
                "All three models fit essentially equally poorly on data that "
                "jumps to a plateau immediately after T_train. "
                f"v2 requires R2 >= {R2_MIN} to claim a reliable fit."
            ),
            "fix3_sentinel": (
                "v1 returned T_EVAL=10.0 when threshold was never crossed, "
                "then R*=T_EVAL/T_train=20.0 appeared as a data point. "
                "v2 returns (None, False) when not crossed and plots open "
                "markers at T_EVAL to indicate 'never reached threshold'."
            ),
            "fix4_seed": f"Explicit SEED={SEED} added.",
            "fix5_labels": (
                "Growth model labels now use (t−T_train) notation to prevent "
                "misreading of slope on absolute-time x-axis."
            ),
        },
        "failed_in_domain":  failed_configs,
        "valid_for_extrap":  valid_configs,
        "per_T_train": {},
        "elapsed_seconds": elapsed,
    }

    for T_train in T_TRAIN_LIST:
        res = all_results[T_train]
        cr  = {}
        for thresh in ERROR_THRESHOLDS:
            c = res["crossings"][thresh]
            cr[str(thresh)] = {
                "t_cross":  safe_float(c["t_cross"]),
                "crossed":  c["crossed"],
                "r_star":   safe_float(c["r_star"]),
            }
        fg = {}
        for k, v in res["growth_fits"].items():
            if isinstance(v, dict):
                fg[k] = {kk: (list(vv) if isinstance(vv, np.ndarray)
                               else vv) for kk, vv in v.items()}
            else:
                fg[k] = v

        results_json["per_T_train"][str(T_train)] = {
            "in_domain_failed":  res["in_domain_failed"],
            "mean_error_in":     safe_float(res["err_in"]),
            "mean_error_out":    safe_float(res["err_out"]),
            "crossings":         cr,
            "growth_fits":       fg,
            "fit_reliable":      res["fit_reliable"],
            "fit_reason":        res["fit_reason"],
            "sharpness":         res["sharpness"],
        }

    out_json = OUTPUT_DIR / "exp17_results.json"
    with open(out_json, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults → {out_json}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 17 — COMPLETE  [v2]")
    print(f"{'=' * 70}")
    hdr = (f"{'T_train':>8} | {'err_in':>7} | "
           + " | ".join(f"ε={int(t*100):3d}%→R*" for t in ERROR_THRESHOLDS)
           + " | fit_reason      | sharpness")
    print(hdr); print("─" * len(hdr))
    for T_train in T_TRAIN_LIST:
        res = all_results[T_train]
        r_star_strs = []
        for thresh in ERROR_THRESHOLDS:
            c = res["crossings"][thresh]
            r_star_strs.append(
                f"{c['r_star']:>7.2f}" if c["r_star"] else "    N/A")
        print(f"{T_train:>8.1f} | {res['err_in']:>7.4f} | "
              + " | ".join(r_star_strs)
              + f" | {res['fit_reason'][:15]:<15} | {res['sharpness']['label']}")
    print(f"\n  Valid for extrapolation: {valid_configs}")
    print(f"  Failed in-domain       : {failed_configs}")
    print(f"  Results → {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    return results_json


if __name__ == "__main__":
    run_experiment()