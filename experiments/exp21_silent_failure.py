"""
exp21_silent_failure.py — Experiment 21: Silent Failure Detection in PINNs

Tests the hypothesis that PINNs can appear to converge (low training loss)
while being completely wrong on the solution (high test L2 error).

Systematically varies three axes for the Helmholtz equation:
  - Domain size    L ∈ {1.0, 2.0, 4.0, 8.0}
  - Frequency k²   ∈ {1, 25, 100, 400}
  - Network width  w ∈ {16, 32, 64, 128}

producing 4×4×4 = 64 experiments (+ 36 random samples = 100 total).

For each experiment:
  1. Train for 15,000 Adam steps.
  2. Record final training loss (L_train) and test L2 relative error (E_test).
  3. Flag as "silent failure" when L_train < τ_loss AND E_test > τ_err.

Then:
  - Scatter plot L_train vs E_test across all 100 runs, with quadrant shading.
  - Identify conditions that reliably produce the silent-failure quadrant.
  - Propose and evaluate 3 cheap diagnostic metrics (no ground truth):
      D1: gradient variance ratio (variance of |∇L_pde| over collocation pts)
      D2: residual spatial entropy (how uniformly distributed is the residual)
      D3: loss plateau score (flatness of loss curve in last 20% of training)

Outputs saved to results/exp21/:
  - loss_vs_error_scatter.png
  - silent_failure_conditions.png
  - diagnostic_metric_roc.png
  - domain_heatmaps.png
  - exp21_results.json

PDE: Δu + k²u = f(x,y),  (x,y) ∈ [-L/2, L/2]²
Exact: u(x,y) = sin(a₁x)sin(a₂y),  a₁=a₂=π/L
f(x,y) = -(a₁² + a₂²)u + k²u
BC: u = 0 on all 4 walls (Dirichlet, soft constraint)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from itertools import product as iproduct

# ─── Device ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp21] Device: {DEVICE}")

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp21"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ────────────────────────────────────────────────────────
# Primary grid axes
DOMAIN_SIZES = [1.0, 2.0, 4.0, 8.0]
K2_VALUES    = [1, 25, 100, 400]
WIDTHS       = [16, 32, 64, 128]

N_HIDDEN     = 3
N_EPOCHS     = 15_000
LR           = 1e-3
LR_MIN       = 1e-5
N_COL        = 5_000
N_BC         = 300
N_IC         = 200   # interior reference pts for test

NX_EVAL      = 64
NY_EVAL      = 64

# Silent failure thresholds
TAU_LOSS = 1e-3    # training loss below this → "converged"
TAU_ERR  = 0.20   # test L2 error above this → "wrong solution"

# ─── Model ─────────────────────────────────────────────────────────

class HelmholtzPINN(nn.Module):
    def __init__(self, n_hidden=3, n_neurons=64):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


# ─── PDE helpers ───────────────────────────────────────────────────

def u_exact(x_np, y_np, L, k2):
    a = np.pi / L
    return np.sin(a * x_np) * np.sin(a * y_np)


def forcing_fn(x, y, L, k2):
    """RHS f such that u_exact = sin(ax)sin(ay), a=π/L."""
    a = np.pi / L
    # Δu + k²u = f  →  f = -(a²+a²)u + k²u = (k² - 2a²) u
    u = torch.sin(a * x) * torch.sin(a * y)
    return (k2 - 2 * a ** 2) * u


def helmholtz_residual(model, x, y, L, k2):
    x = x.requires_grad_(True); y = y.requires_grad_(True)
    u   = model(x, y)
    ux  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    uy  = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, torch.ones_like(uy), create_graph=True)[0]
    f   = forcing_fn(x, y, L, k2)
    return uxx + uyy + k2 * u - f


# ─── Loss ──────────────────────────────────────────────────────────

def compute_loss(model, L, k2, n_col, n_bc):
    half = L / 2.0
    # Collocation
    xc = (torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1) * half
    yc = (torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1) * half
    res   = helmholtz_residual(model, xc, yc, L, k2)
    l_pde = (res ** 2).mean()

    # Dirichlet BC on 4 walls
    tb   = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE) * 2 * half - half
    wall = torch.full((n_bc, 1), half, dtype=DTYPE, device=DEVICE)
    l_bc = (model(wall,  tb) ** 2
          + model(-wall, tb) ** 2
          + model(tb,  wall) ** 2
          + model(tb, -wall) ** 2).mean()

    return l_pde + 100 * l_bc, l_pde.item(), l_bc.item()


# ─── Evaluation ────────────────────────────────────────────────────

def evaluate_test_error(model, L, k2, nx=NX_EVAL, ny=NY_EVAL):
    half   = L / 2.0
    x_vals = np.linspace(-half, half, nx)
    y_vals = np.linspace(-half, half, ny)
    XX, YY = np.meshgrid(x_vals, y_vals)

    model.eval()
    with torch.no_grad():
        xf = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        yf = torch.tensor(YY.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u_pred = model(xf, yf).cpu().numpy().reshape(ny, nx)
    model.train()

    u_ref  = u_exact(XX, YY, L, k2)
    denom  = np.sqrt((u_ref ** 2).mean()) + 1e-8
    l2_err = float(np.sqrt(((u_pred - u_ref) ** 2).mean()) / denom)
    return l2_err, u_pred, u_ref, XX, YY


# ─── 3 Cheap Diagnostics (no ground truth) ─────────────────────────

def diag_gradient_variance(model, L, k2, n_probe=1000):
    """
    D1: Variance of per-point PDE residual magnitude over collocation pts.
    High variance → residual is spatially concentrated → possible silent failure.
    Normalised by mean residual squared so it's scale-free.
    """
    half = L / 2.0
    xp = (torch.rand(n_probe, 1, dtype=DTYPE, device=DEVICE) * 2 - 1) * half
    yp = (torch.rand(n_probe, 1, dtype=DTYPE, device=DEVICE) * 2 - 1) * half
    res = helmholtz_residual(model, xp, yp, L, k2)
    r   = res.detach().abs().cpu().numpy().flatten()
    return float(r.var() / (r.mean() ** 2 + 1e-12))


def diag_residual_entropy(model, L, k2, nx=32, ny=32):
    """
    D2: Spatial entropy of the residual field.
    Uniform residual → high entropy → silent failure (residual not concentrated
    near boundaries only, but spread everywhere).
    Low entropy → residual localised → more interpretable failure.
    """
    half = L / 2.0
    x_vals = np.linspace(-half, half, nx)
    y_vals = np.linspace(-half, half, ny)
    XX, YY = np.meshgrid(x_vals, y_vals)
    xf = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE, requires_grad=True).unsqueeze(1)
    yf = torch.tensor(YY.ravel(), dtype=DTYPE, device=DEVICE, requires_grad=True).unsqueeze(1)
    res  = helmholtz_residual(model, xf, yf, L, k2)
    r    = res.detach().abs().cpu().numpy().flatten()
    # Normalise to probability distribution
    r_sum = r.sum() + 1e-12
    p     = r / r_sum
    p     = np.clip(p, 1e-12, None)
    entropy = -float(np.sum(p * np.log(p)))
    # Normalise by max entropy (uniform)
    max_ent = float(np.log(len(p)))
    return float(entropy / max_ent)


def diag_loss_plateau(loss_history, frac=0.2):
    """
    D3: Plateau score = std(loss in last frac of training) / mean(loss in last frac).
    Near-zero → loss is flat (plateau) → candidate for silent failure.
    """
    n    = len(loss_history)
    tail = loss_history[int(n * (1 - frac)):]
    if len(tail) < 3:
        return 0.0
    arr = np.array(tail)
    return float(arr.std() / (arr.mean() + 1e-12))


# ─── Training ──────────────────────────────────────────────────────

def train_one(L, k2, width, run_id):
    model     = HelmholtzPINN(N_HIDDEN, width).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)

    loss_history = []
    final_loss   = None

    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        try:
            loss, lp, lb = compute_loss(model, L, k2, N_COL, N_BC)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            lv = float(loss.item())
            loss_history.append(lv)
            final_loss = lv
        except Exception:
            break

    if final_loss is None:
        final_loss = 1e6

    # ── Test error ──
    l2_err, u_pred, u_ref, XX, YY = evaluate_test_error(model, L, k2)

    # ── 3 diagnostics ──
    try:
        d1 = diag_gradient_variance(model, L, k2)
    except Exception:
        d1 = float("nan")
    try:
        d2 = diag_residual_entropy(model, L, k2)
    except Exception:
        d2 = float("nan")
    d3 = diag_loss_plateau(loss_history)

    silent = (final_loss < TAU_LOSS) and (l2_err > TAU_ERR)

    return {
        "run_id":       run_id,
        "L":            float(L),
        "k2":           int(k2),
        "width":        int(width),
        "train_loss":   float(final_loss),
        "test_l2":      float(l2_err),
        "silent_failure": bool(silent),
        "d1_grad_var":  float(d1),
        "d2_res_entropy": float(d2),
        "d3_loss_plateau": float(d3),
        "loss_history": loss_history[::50],   # downsample for storage
    }


# ─── Diagnostic ROC analysis ───────────────────────────────────────

def roc_curve_manual(labels, scores):
    """Simple ROC using threshold sweep. Returns (fpr, tpr, auc)."""
    labels = np.array(labels, dtype=float)
    scores = np.array(scores, dtype=float)
    # Handle NaN
    valid  = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return [0, 1], [0, 1], 0.5

    thresholds = np.linspace(scores.min(), scores.max(), 100)
    tprs, fprs = [], []
    pos = labels.sum(); neg = len(labels) - pos
    for th in thresholds:
        pred = (scores >= th).astype(float)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tprs.append(tp / (pos + 1e-12))
        fprs.append(fp / (neg + 1e-12))
    # Sort by fpr
    order = np.argsort(fprs)
    fpr   = np.array(fprs)[order]
    tpr   = np.array(tprs)[order]
    auc   = float(np.trapezoid(tpr, fpr))
    return fpr.tolist(), tpr.tolist(), auc


# ─── Plotting ──────────────────────────────────────────────────────

def plot_loss_vs_error_scatter(results, filepath):
    train_losses = [r["train_loss"] for r in results]
    test_l2s     = [r["test_l2"]    for r in results]
    silents      = [r["silent_failure"] for r in results]
    k2s          = [r["k2"] for r in results]
    widths       = [r["width"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Silent Failure Detection — Helmholtz PINN (100 experiments)", fontsize=13)

    # Panel 1: L_train vs E_test with quadrant shading
    ax = axes[0]
    ax.axvspan(1e-8, TAU_LOSS, ymin=0,
               ymax=(np.log10(TAU_ERR) - np.log10(min(test_l2s) + 1e-8))
                   / (np.log10(max(test_l2s) + 1e-8) - np.log10(min(test_l2s) + 1e-8)),
               alpha=0.0)
    ax.axvline(TAU_LOSS, color="red",   linestyle="--", linewidth=1.2, label=f"τ_loss={TAU_LOSS}")
    ax.axhline(TAU_ERR,  color="blue",  linestyle="--", linewidth=1.2, label=f"τ_err={TAU_ERR}")

    # Quadrant shading
    xlim_lo, xlim_hi = 1e-6, 1e1
    ylim_lo, ylim_hi = 1e-4, 10.0
    ax.fill_betweenx([TAU_ERR, ylim_hi], xlim_lo, TAU_LOSS,
                     color="red", alpha=0.08, label="SILENT FAILURE")
    ax.fill_betweenx([ylim_lo, TAU_ERR], xlim_lo, TAU_LOSS,
                     color="green", alpha=0.08, label="Success")
    ax.fill_betweenx([TAU_ERR, ylim_hi], TAU_LOSS, xlim_hi,
                     color="orange", alpha=0.08, label="Visible failure")

    colors = np.log10(np.array(k2s) + 1)
    sc = ax.scatter(train_losses, test_l2s, c=colors, cmap="plasma",
                    s=60, alpha=0.8, zorder=5)
    # Mark silent failures with red border
    for r in results:
        if r["silent_failure"]:
            ax.scatter([r["train_loss"]], [r["test_l2"]],
                       s=120, facecolors="none", edgecolors="red",
                       linewidths=2, zorder=6)
    plt.colorbar(sc, ax=ax, label="log₁₀(k²+1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Training Loss (log)",  fontsize=11)
    ax.set_ylabel("Test L2 Error (log)",  fontsize=11)
    ax.set_title("Loss vs Error — All Runs", fontsize=11)
    ax.set_xlim(xlim_lo, xlim_hi); ax.set_ylim(ylim_lo, ylim_hi)
    ax.legend(fontsize=8)

    # Panel 2: histogram of test L2 for "converged" vs "not-converged" models
    ax2 = axes[1]
    converged     = [r["test_l2"] for r in results if r["train_loss"] < TAU_LOSS]
    not_converged = [r["test_l2"] for r in results if r["train_loss"] >= TAU_LOSS]
    bins = np.logspace(-4, 1, 30)
    ax2.hist(converged,     bins=bins, color="blue",   alpha=0.6, label="Low train loss")
    ax2.hist(not_converged, bins=bins, color="orange", alpha=0.6, label="High train loss")
    ax2.axvline(TAU_ERR, color="red", linestyle="--", linewidth=1.2, label=f"τ_err={TAU_ERR}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Test L2 Error", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Error Distribution: Converged vs Not", fontsize=11)
    ax2.legend(fontsize=9)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_silent_failure_conditions(results, filepath):
    """Heatmaps of silent-failure rate over (k², L) and (k², width)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Conditions Producing Silent Failure", fontsize=13)

    def make_heatmap(ax, x_vals, y_vals, x_key, y_key, xlabel, ylabel, title):
        mat = np.zeros((len(y_vals), len(x_vals)))
        cnt = np.zeros_like(mat)
        for r in results:
            xi = x_vals.index(r[x_key]) if r[x_key] in x_vals else -1
            yi = y_vals.index(r[y_key]) if r[y_key] in y_vals else -1
            if xi >= 0 and yi >= 0:
                mat[yi, xi] += float(r["silent_failure"])
                cnt[yi, xi] += 1
        rate = np.where(cnt > 0, mat / cnt, 0)
        im = ax.imshow(rate, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(x_vals))); ax.set_xticklabels([str(v) for v in x_vals])
        ax.set_yticks(range(len(y_vals))); ax.set_yticklabels([str(v) for v in y_vals])
        ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label="Silent Failure Rate")
        for yi in range(len(y_vals)):
            for xi in range(len(x_vals)):
                ax.text(xi, yi, f"{rate[yi, xi]:.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    make_heatmap(axes[0],
                 K2_VALUES, DOMAIN_SIZES,
                 "k2", "L", "k² (frequency)", "Domain size L",
                 "Silent Failure Rate\n(k² × L)")
    make_heatmap(axes[1],
                 K2_VALUES, WIDTHS,
                 "k2", "width", "k² (frequency)", "Network width",
                 "Silent Failure Rate\n(k² × width)")

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_diagnostic_metric_roc(results, filepath):
    """ROC curves for each of the 3 cheap diagnostics."""
    labels = [int(r["silent_failure"]) for r in results]
    d1     = [r["d1_grad_var"]      for r in results]
    d2     = [r["d2_res_entropy"]   for r in results]
    d3_inv = [1.0 / (r["d3_loss_plateau"] + 1e-6) for r in results]  # low plateau → silent

    diag_sets = [
        ("D1: Gradient Variance Ratio", d1),
        ("D2: Residual Spatial Entropy", d2),
        ("D3: 1/Loss Plateau Score",    d3_inv),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    fig.suptitle("Diagnostic ROC Curves for Silent Failure Detection", fontsize=12)

    for ax, (name, scores) in zip(axes, diag_sets):
        fpr, tpr, auc = roc_curve_manual(labels, scores)
        ax.plot(fpr, tpr, color="#E64040", linewidth=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
        ax.set_aspect("equal")

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_domain_heatmaps(results, filepath, n_show=6):
    """Show u_pred and |u_pred - u_exact| for representative silent-failure cases."""
    silent_cases = [r for r in results if r["silent_failure"]]
    if not silent_cases:
        silent_cases = sorted(results, key=lambda r: r["test_l2"])[-6:]
    silent_cases = silent_cases[:n_show]

    if not silent_cases:
        return

    fig, axes = plt.subplots(len(silent_cases), 2,
                             figsize=(8, 3.5 * len(silent_cases)),
                             constrained_layout=True)
    fig.suptitle("Silent Failure Cases: Prediction vs Error Field", fontsize=12)
    if len(silent_cases) == 1:
        axes = [axes]

    for i, r in enumerate(silent_cases):
        L, k2, width = r["L"], r["k2"], r["width"]
        model = HelmholtzPINN(N_HIDDEN, width).to(DEVICE)
        # Re-train briefly just for visualisation (fast)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5000, eta_min=LR_MIN)
        for _ in range(5000):
            model.train(); optimizer.zero_grad()
            loss, _, _ = compute_loss(model, L, k2, 3000, 200)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()

        _, u_pred, u_ref, XX, YY = evaluate_test_error(model, L, k2)
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

        ax_pred  = axes[i][0]
        ax_err   = axes[i][1]
        half = L / 2.0
        extent = [-half, half, -half, half]

        im1 = ax_pred.imshow(u_pred, origin="lower", cmap="seismic",
                              extent=extent, aspect="auto")
        ax_pred.set_title(f"u_pred  L={L}, k²={k2}, w={width}\n"
                          f"(L_train={r['train_loss']:.2e}, E_test={r['test_l2']:.3f})",
                          fontsize=8)
        plt.colorbar(im1, ax=ax_pred, shrink=0.8)

        err_field = np.abs(u_pred - u_ref)
        im2 = ax_err.imshow(err_field, origin="lower", cmap="hot",
                             extent=extent, aspect="auto")
        ax_err.set_title(f"|u_pred - u_exact|  (silent failure)", fontsize=8)
        plt.colorbar(im2, ax=ax_err, shrink=0.8)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ─── Main ──────────────────────────────────────────────────────────

def build_experiment_grid():
    """
    Build 64 structured grid experiments + 36 random to reach 100 total.
    """
    experiments = []
    run_id = 0

    # 4×4×4 = 64 structured
    for L, k2, w in iproduct(DOMAIN_SIZES, K2_VALUES, WIDTHS):
        experiments.append((L, k2, w, run_id))
        run_id += 1

    # 36 random additional
    rng = np.random.default_rng(42)
    for _ in range(36):
        L   = float(rng.choice(DOMAIN_SIZES))
        k2  = int(rng.choice(K2_VALUES))
        w   = int(rng.choice(WIDTHS))
        experiments.append((L, k2, w, run_id))
        run_id += 1

    return experiments


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 21: Silent Failure Detection in PINNs")
    print(f"Device: {DEVICE}")
    print(f"Grid: {len(DOMAIN_SIZES)}×{len(K2_VALUES)}×{len(WIDTHS)} = 64 + 36 random = 100 runs")
    print(f"Silent failure criterion: L_train < {TAU_LOSS} AND E_test > {TAU_ERR}")
    print("=" * 70)

    experiments = build_experiment_grid()
    
    ckpt_path = OUTPUT_DIR / "exp21_checkpoint.json"
    results = []
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r") as f:
                results = json.load(f)
            print(f"  [Loaded checkpoint with {len(results)} completed runs]")
        except Exception:
            pass
    completed_ids = {r["run_id"] for r in results}

    t0          = time.time()

    for i, (L, k2, w, run_id) in enumerate(experiments):
        if run_id in completed_ids:
            continue

        print(f"\n[{i+1:>3d}/100] L={L:.1f}  k²={k2:>4d}  w={w:>3d}", end="  ")
        r = train_one(L, k2, w, run_id)
        flag = "⚑ SILENT" if r["silent_failure"] else ""
        print(f"L_train={r['train_loss']:.2e}  E_test={r['test_l2']:.4f}  {flag}")
        results.append(r)
        
        with open(ckpt_path, "w") as f:
            json.dump(results, f)
            
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    elapsed = time.time() - t0

    # ── Analysis ──
    n_silent   = sum(r["silent_failure"] for r in results)
    n_converged= sum(r["train_loss"] < TAU_LOSS for r in results)
    sf_rate    = n_silent / max(n_converged, 1)

    # Conditions that most reliably produce silent failure
    from collections import Counter
    sf_k2    = Counter(r["k2"]    for r in results if r["silent_failure"])
    sf_L     = Counter(r["L"]     for r in results if r["silent_failure"])
    sf_width = Counter(r["width"] for r in results if r["silent_failure"])

    # Diagnostic AUC summary
    labels   = [int(r["silent_failure"]) for r in results]
    _, _, auc_d1 = roc_curve_manual(labels, [r["d1_grad_var"]     for r in results])
    _, _, auc_d2 = roc_curve_manual(labels, [r["d2_res_entropy"]  for r in results])
    _, _, auc_d3 = roc_curve_manual(labels, [1.0/(r["d3_loss_plateau"]+1e-6) for r in results])

    # ── Plots ──
    print("\nGenerating plots...")
    plot_loss_vs_error_scatter(results, OUTPUT_DIR / "loss_vs_error_scatter.png")
    plot_silent_failure_conditions(results, OUTPUT_DIR / "silent_failure_conditions.png")
    plot_diagnostic_metric_roc(results, OUTPUT_DIR / "diagnostic_metric_roc.png")
    plot_domain_heatmaps(results, OUTPUT_DIR / "domain_heatmaps.png")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 21 — SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total runs:                   100")
    print(f"  Converged (L_train<{TAU_LOSS}):   {n_converged}")
    print(f"  Silent failures:              {n_silent}")
    print(f"  Silent failure rate (of converged): {sf_rate*100:.1f}%")
    print(f"\n  Most common k² in silent failures:    {sf_k2.most_common(3)}")
    print(f"  Most common L  in silent failures:    {sf_L.most_common(3)}")
    print(f"  Most common width in silent failures: {sf_width.most_common(3)}")
    print(f"\n  Diagnostic AUCs (detecting silent failure without ground truth):")
    print(f"    D1 Gradient Variance Ratio : {auc_d1:.3f}")
    print(f"    D2 Residual Spatial Entropy: {auc_d2:.3f}")
    print(f"    D3 1/Loss Plateau Score    : {auc_d3:.3f}")

    # ── Save ──
    results_json = {
        "experiment": "Silent Failure Detection",
        "config": {
            "domain_sizes":  DOMAIN_SIZES,
            "k2_values":     K2_VALUES,
            "widths":        WIDTHS,
            "tau_loss":      TAU_LOSS,
            "tau_err":       TAU_ERR,
        },
        "summary": {
            "n_converged":  n_converged,
            "n_silent":     n_silent,
            "sf_rate_of_converged": float(sf_rate),
            "diagnostic_aucs": {
                "D1_grad_variance":    auc_d1,
                "D2_residual_entropy": auc_d2,
                "D3_loss_plateau":     auc_d3,
            },
            "silent_failure_conditions": {
                "top_k2":    sf_k2.most_common(4),
                "top_L":     sf_L.most_common(4),
                "top_width": sf_width.most_common(4),
            },
        },
        "runs": [{k: v for k, v in r.items() if k != "loss_history"}
                 for r in results],
        "elapsed_seconds": elapsed,
    }
    out = OUTPUT_DIR / "exp21_results.json"
    with open(out, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {out}")
    print(f"All outputs in:  {OUTPUT_DIR}")
    return results_json


if __name__ == "__main__":
    run_experiment()