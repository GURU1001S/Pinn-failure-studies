"""
specialexp3_compound_failure.py — Compound Failure Interaction Study
[Multi-signal diagnostic recording for failure mode interactions]

Studies whether spectral bias and gradient pathology are causally linked
or independent failure modes, using 5 simultaneous diagnostic signals
on the advection equation at β=30 (failure) vs β=1 (control).

Diagnostic signals recorded every 500 iterations:
  1. Spectral divergence score (FFT comparison)
  2. Gradient conflict ratio (||g_pde|| / ||g_bc||) + cosine similarity
  3. NTK eigenvalue spread (every 5000 epochs)
  4. Weight norm growth
  5. Activation saturation fraction

Analyses:
  1. Joint phase portrait (spectral_score vs gradient_ratio)
  2. Failure timeline with onset detection
  3. Cross-correlation heatmap
  4. Control (β=1) vs failure (β=30) comparison

Outputs (results/specialexp3/):
  - compound_phase_portrait.png
  - compound_failure_timeline.png
  - compound_correlation_heatmap.png
  - control_vs_failure_comparison.png
  - specialexp3_compound_results.json
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
from scipy.stats import pearsonr

from pinn_core import (
    DEVICE, DTYPE, AdvectionPINN, advection_residual,
    sample_collocation, sample_initial_condition,
    sample_boundary_condition, exact_solution,
    save_results,
)
from plot_utils import savefig, setup_style

setup_style()

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "specialexp3"

# ===================================================================
# Config
# ===================================================================
BETA_FAILURE  = 30
BETA_CONTROL  = 1
N_HIDDEN      = 4
N_NEURONS     = 64
N_EPOCHS      = 50000
LR            = 1e-3
N_COLLOCATION = 8000
N_IC          = 300
N_BC          = 300
SEED          = 42

TRACK_EVERY   = 500       # record signals every N steps
NTK_EVERY     = 5000      # NTK is expensive, do less frequently
NTK_N_POINTS  = 50        # collocation pts for NTK Jacobian

# Failure thresholds for timeline analysis
THRESH_SPECTRAL    = 0.5
THRESH_GRAD_RATIO  = 10.0
THRESH_COND_NUMBER = 1e8
THRESH_WEIGHT_NORM = 3.0   # 3× initial
THRESH_SATURATION  = 0.3


# ===================================================================
# Diagnostic signal functions
# ===================================================================

def compute_spectral_score(model, beta, nx=256):
    """
    Compute spectral divergence at t=1.0.
    score = sum_k |P_pred(k) - P_exact(k)| / sum_k P_exact(k)
    """
    x_vals = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    u_exact_np = exact_solution(x_vals, 1.0, beta)

    x_t = torch.tensor(x_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    t_t = torch.ones(nx, 1, dtype=DTYPE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        u_pred_np = model(x_t, t_t).cpu().numpy().flatten()
    model.train()

    P_pred = np.abs(np.fft.rfft(u_pred_np)) ** 2
    P_exact = np.abs(np.fft.rfft(u_exact_np)) ** 2

    score = np.sum(np.abs(P_pred - P_exact)) / (np.sum(P_exact) + 1e-10)
    return float(score)


def compute_gradient_conflict(model, x_col, t_col, x_bl, t_bl,
                               x_br, t_br, beta):
    """
    Compute gradient ratio ||g_pde|| / ||g_bc|| and cosine similarity.
    """
    with torch.enable_grad():
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]

        # PDE gradient
        model.zero_grad()
        res = advection_residual(model, x_col, t_col, beta)
        loss_pde = torch.mean(res ** 2)
        grads_pde = torch.autograd.grad(loss_pde, params, retain_graph=False,
                                         allow_unused=True)
        g_pde = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=DEVICE)
                            for g, p in zip(grads_pde, params)])

        # BC gradient
        model.zero_grad()
        u_l = model(x_bl, t_bl)
        u_r = model(x_br, t_br)
        loss_bc = torch.mean((u_l - u_r) ** 2)
        grads_bc = torch.autograd.grad(loss_bc, params, retain_graph=False,
                                        allow_unused=True)
        g_bc = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=DEVICE)
                           for g, p in zip(grads_bc, params)])

        norm_pde = g_pde.norm().item()
        norm_bc  = g_bc.norm().item()
        ratio    = norm_pde / (norm_bc + 1e-10)
        cosine   = torch.dot(g_pde, g_bc) / (norm_pde * norm_bc + 1e-10)

        return float(ratio), float(cosine.item())


def compute_ntk_condition(model, beta, n_points=NTK_N_POINTS):
    """
    Compute NTK condition number via Jacobian on n_points.
    Returns: top eigenvalue, bottom nonzero eigenvalue, condition number.
    """
    with torch.enable_grad():
        model.eval()
        x = torch.rand(n_points, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
        t = torch.rand(n_points, 1, dtype=DTYPE, device=DEVICE) * 2.0

        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)

        # Limit parameter count for memory
        max_params = 30000
        if n_params > max_params:
            # Use subset of parameters
            pass

        # Compute Jacobian row by row
        J = torch.zeros(n_points, n_params, device=DEVICE)
        for i in range(n_points):
            model.zero_grad()
            xi = x[i:i+1].requires_grad_(False)
            ti = t[i:i+1].requires_grad_(False)
            u = model(xi, ti)
            grads = torch.autograd.grad(u, params, retain_graph=False,
                                         allow_unused=True)
            row = torch.cat([g.flatten() if g is not None
                             else torch.zeros(p.numel(), device=DEVICE)
                             for g, p in zip(grads, params)])
            J[i] = row

        # NTK = J @ J.T
        K = J @ J.T
        eigvals = torch.linalg.eigvalsh(K).cpu().numpy()
        eigvals = np.abs(eigvals)
        eigvals = eigvals[eigvals > 1e-12]

        if len(eigvals) < 2:
            return 0.0, 0.0, 1.0

        top = float(eigvals[-1])
        bottom = float(eigvals[0])
        cond = top / (bottom + 1e-15)

        model.train()
        return top, bottom, float(cond)


def compute_weight_norm(model):
    """L2 norm of all parameters combined."""
    total = sum(p.data.norm() ** 2 for p in model.parameters()).sqrt()
    return float(total.item())


def compute_saturation(model, n_points=500):
    """Fraction of tanh neurons with |output| > 0.99."""
    model.eval()
    x = torch.rand(n_points, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    t = torch.rand(n_points, 1, dtype=DTYPE, device=DEVICE) * 2.0
    inp = torch.cat([x, t], dim=1)

    # Forward through individual layers to check tanh outputs
    saturated = 0
    total = 0
    current = inp
    for layer in model.net:
        current = layer(current)
        if isinstance(layer, nn.Tanh):
            sat = (current.abs() > 0.99).float().mean().item()
            n = current.numel()
            saturated += sat * n
            total += n

    model.train()
    return float(saturated / (total + 1e-10))


# ===================================================================
# Training with diagnostic recording
# ===================================================================

def train_with_diagnostics(beta, label=""):
    """Train advection PINN with all 5 diagnostic signals recorded."""
    print(f"\n  Training β={beta} ({label})...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = AdvectionPINN(n_hidden=N_HIDDEN, n_neurons=N_NEURONS,
                          activation="tanh").to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    x_col, t_col = sample_collocation(N_COLLOCATION)
    x_ic, t_ic = sample_initial_condition(N_IC)
    x_bl, t_bl, x_br, t_br = sample_boundary_condition(N_BC)
    u_ic_target = torch.sin(x_ic).detach()

    initial_weight_norm = compute_weight_norm(model)

    # Diagnostic storage
    diagnostics = {
        "epochs":          [],
        "spectral_score":  [],
        "grad_ratio":      [],
        "grad_cosine":     [],
        "weight_norm":     [],
        "weight_norm_rel": [],  # relative to initial
        "saturation":      [],
        "ntk_top":         [],
        "ntk_bottom":      [],
        "ntk_cond":        [],
        "ntk_epochs":      [],
        "loss":            [],
    }

    loss_hist = []

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        res = advection_residual(model, x_col, t_col, beta)
        loss_pde = torch.mean(res ** 2)
        u_ic_pred = model(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
        u_l = model(x_bl, t_bl)
        u_r = model(x_br, t_br)
        loss_bc = torch.mean((u_l - u_r) ** 2)
        loss = loss_pde + 10 * loss_ic + loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_hist.append(loss.item())

        # Record diagnostics
        if epoch % TRACK_EVERY == 0:
            with torch.no_grad():
                spec = compute_spectral_score(model, beta)
                gr, gc = compute_gradient_conflict(
                    model, x_col, t_col, x_bl, t_bl, x_br, t_br, beta)
                wn = compute_weight_norm(model)
                sat = compute_saturation(model)

            diagnostics["epochs"].append(epoch)
            diagnostics["spectral_score"].append(spec)
            diagnostics["grad_ratio"].append(gr)
            diagnostics["grad_cosine"].append(gc)
            diagnostics["weight_norm"].append(wn)
            diagnostics["weight_norm_rel"].append(wn / initial_weight_norm)
            diagnostics["saturation"].append(sat)
            diagnostics["loss"].append(loss.item())

            # NTK (expensive, less frequent)
            if epoch % NTK_EVERY == 0:
                try:
                    top, bot, cond = compute_ntk_condition(model, beta)
                    diagnostics["ntk_top"].append(top)
                    diagnostics["ntk_bottom"].append(bot)
                    diagnostics["ntk_cond"].append(cond)
                    diagnostics["ntk_epochs"].append(epoch)
                except Exception as e:
                    print(f"    NTK computation failed at epoch {epoch}: {e}")

            if epoch % 10000 == 0:
                print(f"    [{epoch:6d}/{N_EPOCHS}] Loss={loss.item():.4e} "
                      f"spec={spec:.3f} gr={gr:.1f} sat={sat:.3f}")

    # Final L2
    x_vals = np.linspace(0, 2 * np.pi, 256)
    t_val = 1.0
    x_t = torch.tensor(x_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    t_t = torch.full((256, 1), t_val, dtype=DTYPE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        u_pred = model(x_t, t_t).cpu().numpy().flatten()
    u_exact = exact_solution(x_vals, t_val, beta)
    l2 = np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10)
    print(f"  → β={beta}: L2={l2:.6f}")

    diagnostics["final_l2"] = float(l2)
    diagnostics["loss_history"] = loss_hist
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return diagnostics


# ===================================================================
# Analysis functions
# ===================================================================

def find_onset(signal, threshold, epochs):
    """Find first epoch where signal exceeds threshold."""
    for ep, val in zip(epochs, signal):
        if val > threshold:
            return int(ep)
    return None


def plot_phase_portrait(diag_fail, diag_ctrl, filepath):
    """Joint phase portrait: spectral_score vs gradient_ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, diag, beta_val, title in [
        (axes[0], diag_fail, BETA_FAILURE, f"β={BETA_FAILURE} (failure)"),
        (axes[1], diag_ctrl, BETA_CONTROL, f"β={BETA_CONTROL} (control)"),
    ]:
        epochs = np.array(diag["epochs"])
        spec = np.array(diag["spectral_score"])
        gr = np.array(diag["grad_ratio"])

        # Color by normalized epoch (blue=early, red=late)
        colors = cm.coolwarm(epochs / max(epochs.max(), 1))

        sc = ax.scatter(spec, gr, c=epochs, cmap="coolwarm",
                        s=15, alpha=0.7, edgecolors="none")
        ax.plot(spec, gr, color="gray", alpha=0.2, linewidth=0.5)
        plt.colorbar(sc, ax=ax, label="Epoch")

        # Threshold lines
        ax.axvline(THRESH_SPECTRAL, color="orange", ls="--", alpha=0.6,
                   label=f"Spectral thresh={THRESH_SPECTRAL}")
        ax.axhline(THRESH_GRAD_RATIO, color="red", ls="--", alpha=0.6,
                   label=f"Grad ratio thresh={THRESH_GRAD_RATIO}")

        ax.set_xlabel("Spectral Divergence Score", fontsize=11)
        ax.set_ylabel("Gradient Ratio (||g_pde|| / ||g_bc||)", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.legend(fontsize=8)
        ax.set_yscale("log")

    fig.suptitle(
        "Compound Failure Phase Portrait\n"
        "2D trajectory over training: blue=early, red=late.\n"
        "Correlated movement → coupled failure; independent → separate modes.",
        fontweight="bold", fontsize=13)
    savefig(fig, filepath)


def plot_failure_timeline(diag, beta_val, filepath):
    """All 5 signals normalized, with onset markers."""
    fig, ax = plt.subplots(figsize=(14, 6))
    epochs = np.array(diag["epochs"])

    def normalize(arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    signals = {
        "Spectral Score": (diag["spectral_score"], THRESH_SPECTRAL, "#E64040"),
        "Gradient Ratio":  (diag["grad_ratio"], THRESH_GRAD_RATIO, "#3A7FD5"),
        "Weight Norm (rel)": (diag["weight_norm_rel"], THRESH_WEIGHT_NORM, "#F59F00"),
        "Saturation": (diag["saturation"], THRESH_SATURATION, "#2E7D32"),
    }

    onsets = {}
    for name, (signal, thresh, color) in signals.items():
        normed = normalize(signal)
        ax.plot(epochs, normed, color=color, linewidth=1.5,
                label=name, alpha=0.8)

        onset = find_onset(signal, thresh, epochs)
        onsets[name] = onset
        if onset is not None:
            ax.axvline(onset, color=color, linestyle=":", alpha=0.5, linewidth=1)
            ax.scatter([onset], [0.02], color=color, s=100, marker="^", zorder=5)

    # NTK condition on separate epochs
    if diag["ntk_epochs"] and diag["ntk_cond"]:
        ntk_e = np.array(diag["ntk_epochs"])
        ntk_normed = normalize(diag["ntk_cond"])
        ax.plot(ntk_e, ntk_normed, "D-", color="#9C27B0", linewidth=1.5,
                markersize=6, label="NTK Condition", alpha=0.8)
        onset_ntk = find_onset(diag["ntk_cond"], THRESH_COND_NUMBER,
                                diag["ntk_epochs"])
        onsets["NTK Condition"] = onset_ntk

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Normalized Signal (0–1)", fontsize=12)
    ax.set_title(
        f"Failure Timeline (β={beta_val})\n"
        f"▲ = onset epoch. Ordering reveals causal chain.",
        fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    savefig(fig, filepath)

    return onsets


def plot_correlation_heatmap(diag, beta_val, filepath):
    """Pearson correlation matrix between diagnostic signals."""
    signal_names = ["Spectral", "Grad Ratio", "Grad Cosine",
                    "Weight Norm", "Saturation"]
    signal_data = [
        diag["spectral_score"],
        diag["grad_ratio"],
        diag["grad_cosine"],
        diag["weight_norm"],
        diag["saturation"],
    ]

    n = len(signal_names)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                si = np.array(signal_data[i])
                sj = np.array(signal_data[j])
                min_len = min(len(si), len(sj))
                if min_len >= 3:
                    r, p = pearsonr(si[:min_len], sj[:min_len])
                    corr_matrix[i, j] = r
                    p_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(signal_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(signal_names, fontsize=10)

    # Annotate with correlation values
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            p_val = p_matrix[i, j]
            sig = "*" if p_val < 0.05 else ""
            ax.text(j, i, f"{val:.2f}{sig}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(val) > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Pearson r (* = p<0.05)")
    ax.set_title(
        f"Signal Cross-Correlation (β={beta_val})\n"
        "High |r| between spectral & gradient → compound failure mode",
        fontweight="bold", fontsize=12)
    savefig(fig, filepath)

    return corr_matrix.tolist(), p_matrix.tolist()


def plot_comparison(diag_fail, diag_ctrl, filepath):
    """Side-by-side β=30 vs β=1 for all signals."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    signal_configs = [
        ("spectral_score", "Spectral Score", "epochs"),
        ("grad_ratio", "Gradient Ratio", "epochs"),
        ("grad_cosine", "Gradient Cosine", "epochs"),
        ("weight_norm_rel", "Weight Norm (×initial)", "epochs"),
        ("saturation", "Saturation Fraction", "epochs"),
        ("loss", "Training Loss", "epochs"),
    ]

    for ax, (key, title, ep_key) in zip(axes.ravel(), signal_configs):
        for diag, beta_val, color, ls in [
            (diag_fail, BETA_FAILURE, "#D32F2F", "-"),
            (diag_ctrl, BETA_CONTROL, "#2E7D32", "--"),
        ]:
            epochs = diag[ep_key]
            vals = diag[key]
            min_len = min(len(epochs), len(vals))
            ax.plot(epochs[:min_len], vals[:min_len],
                    color=color, linestyle=ls, linewidth=1.5,
                    label=f"β={beta_val}", alpha=0.8)

        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if key in ["grad_ratio", "loss"]:
            ax.set_yscale("log")

    fig.suptitle(
        f"Control (β={BETA_CONTROL}) vs Failure (β={BETA_FAILURE}) "
        f"Diagnostic Comparison",
        fontweight="bold", fontsize=14)
    savefig(fig, filepath)


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    t_start = time.time()
    print("=" * 70)
    print("SPECIAL EXP 3: Compound Failure Interaction Study")
    print(f"Device      : {DEVICE}")
    print(f"β_failure   : {BETA_FAILURE}")
    print(f"β_control   : {BETA_CONTROL}")
    print(f"Epochs      : {N_EPOCHS}")
    print(f"Track every : {TRACK_EVERY}")
    print(f"NTK every   : {NTK_EVERY}")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    checkpoint_path = OUTPUT_DIR / "specialexp3_checkpoint.json"

    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not serializable: {type(o).__name__}")

    ckpt = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                ckpt = json.load(f)
            print(f"  [Checkpoint] Loaded from {checkpoint_path.name}")
        except Exception:
            pass

    # Run β=30 (failure case)
    if "failure" not in ckpt:
        diag_fail = train_with_diagnostics(BETA_FAILURE, "failure")
        ckpt["failure"] = diag_fail
        with open(checkpoint_path, 'w') as f:
            json.dump(ckpt, f, default=_default)
    else:
        diag_fail = ckpt["failure"]
        print(f"\n  β={BETA_FAILURE} (failure): [loaded from checkpoint]")

    # Run β=1 (control)
    if "control" not in ckpt:
        diag_ctrl = train_with_diagnostics(BETA_CONTROL, "control")
        ckpt["control"] = diag_ctrl
        with open(checkpoint_path, 'w') as f:
            json.dump(ckpt, f, default=_default)
    else:
        diag_ctrl = ckpt["control"]
        print(f"\n  β={BETA_CONTROL} (control): [loaded from checkpoint]")

    # ── Analysis & Plots ────────────────────────────────────────────
    print("\n── Generating analyses & plots ──")

    # 1. Phase portrait
    plot_phase_portrait(diag_fail, diag_ctrl,
                        OUTPUT_DIR / "compound_phase_portrait.png")

    # 2. Failure timeline
    onsets_fail = plot_failure_timeline(
        diag_fail, BETA_FAILURE,
        OUTPUT_DIR / "compound_failure_timeline.png")

    onsets_ctrl = plot_failure_timeline(
        diag_ctrl, BETA_CONTROL,
        OUTPUT_DIR / "control_failure_timeline.png")

    # 3. Correlation heatmap
    corr_fail, p_fail = plot_correlation_heatmap(
        diag_fail, BETA_FAILURE,
        OUTPUT_DIR / "compound_correlation_heatmap.png")

    corr_ctrl, p_ctrl = plot_correlation_heatmap(
        diag_ctrl, BETA_CONTROL,
        OUTPUT_DIR / "control_correlation_heatmap.png")

    # 4. Control vs failure comparison
    plot_comparison(diag_fail, diag_ctrl,
                    OUTPUT_DIR / "control_vs_failure_comparison.png")

    # ── Causal ordering ─────────────────────────────────────────────
    onset_order = sorted(
        [(name, ep) for name, ep in onsets_fail.items() if ep is not None],
        key=lambda x: x[1]
    )
    causal_chain = " → ".join(f"{name}(ep={ep})" for name, ep in onset_order)
    print(f"\n  Causal chain (β={BETA_FAILURE}): {causal_chain}")

    # Check spectral-gradient coupling
    spec_arr = np.array(diag_fail["spectral_score"])
    grad_arr = np.array(diag_fail["grad_ratio"])
    min_len = min(len(spec_arr), len(grad_arr))
    if min_len >= 3:
        r_sg, p_sg = pearsonr(spec_arr[:min_len], grad_arr[:min_len])
    else:
        r_sg, p_sg = 0.0, 1.0

    coupling = "COUPLED" if abs(r_sg) > 0.5 and p_sg < 0.05 else "INDEPENDENT"
    print(f"  Spectral-Gradient correlation: r={r_sg:.3f}, p={p_sg:.4f} → {coupling}")

    # ── JSON ────────────────────────────────────────────────────────
    results = {
        "experiment": "Compound Failure Interaction Study",
        "version": "specialexp3",
        "config": {
            "beta_failure": BETA_FAILURE,
            "beta_control": BETA_CONTROL,
            "n_epochs": N_EPOCHS,
            "n_hidden": N_HIDDEN,
            "n_neurons": N_NEURONS,
            "track_every": TRACK_EVERY,
            "ntk_every": NTK_EVERY,
            "ntk_n_points": NTK_N_POINTS,
            "seed": SEED,
        },
        "failure_thresholds": {
            "spectral_score": THRESH_SPECTRAL,
            "grad_ratio": THRESH_GRAD_RATIO,
            "ntk_condition": THRESH_COND_NUMBER,
            "weight_norm_growth": THRESH_WEIGHT_NORM,
            "saturation": THRESH_SATURATION,
        },
        "failure_diagnostics": {
            "final_l2": diag_fail.get("final_l2"),
            "onset_epochs": {k: v for k, v in onsets_fail.items()},
            "causal_chain": causal_chain,
            "spectral_grad_correlation": float(r_sg),
            "spectral_grad_p_value": float(p_sg),
            "coupling_verdict": coupling,
        },
        "control_diagnostics": {
            "final_l2": diag_ctrl.get("final_l2"),
            "onset_epochs": {k: v for k, v in onsets_ctrl.items()},
        },
        "correlation_matrix_failure": corr_fail,
        "correlation_matrix_control": corr_ctrl,
        "p_value_matrix_failure": p_fail,
        "compound_failure_note": (
            f"Spectral bias and gradient pathology at β={BETA_FAILURE}: "
            f"Pearson r={r_sg:.3f} (p={p_sg:.4f}). "
            f"Verdict: {coupling}. "
            f"{'These are ONE compound failure mode.' if coupling == 'COUPLED' else 'These are TWO independent failure modes that co-occur.'} "
            f"Causal chain: {causal_chain}."
        ),
    }

    save_results(results, OUTPUT_DIR / "specialexp3_compound_results.json")

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"SPECIAL EXP 3 — COMPLETE")
    print(f"  Total wall time: {total_elapsed / 60:.1f} min")
    print(f"  Coupling verdict: {coupling}")
    print(f"  Causal chain: {causal_chain}")
    print(f"  Results → {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    return results


if __name__ == "__main__":
    run_experiment()
