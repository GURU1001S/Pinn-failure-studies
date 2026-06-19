"""
exp16_causality_failure.py — Causality Violation in Standard PINNs
[v2 — journal-ready fixes]

Reproduces and extends the causality failure experiment from Wang et al. 2022.
Trains a PINN on the 1D advection equation at β=10 (marginal failure case)
using:
  (A) Standard uniform collocation in time
  (B) Causal training with a time-band causal weighting scheme

Protocol:
  1. Train both variants recording snapshots at [1000, 5000, 10000, 30000, 50000]
  2. At each checkpoint, compute pointwise residual |u - u_exact| on fine grid
  3. Compare temporal error profiles u(t) := mean_x |u - u_exact|
  4. Quantify causality-violation lag metric

Outputs (results/exp16/):
  - residual_maps_standard.png
  - residual_maps_causal.png
  - temporal_error_comparison.png
  - causality_lag_analysis.png
  - exp16_results.json

FIXES vs v1 (journal-ready):
  [FIX 1] β changed from 50 to 10. At β=50 both models fail completely
          (L2 > 0.83) due to spectral bias — the experiment was comparing
          two zero-function predictors, not measuring causality effects.
          β=10 is the marginal failure case (L2≈0.035 in Exp 1) where
          the PINN partially succeeds and causality differences are visible.

  [FIX 2] Causal weight formula fixed. v1 used multiplicative cascade:
          w[b] = w[b-1] * gate, which with steepness=10 and CAUSAL_EPS=0.01
          collapses to w[3]≈3×10⁻⁷ by band 3. The network trained almost
          exclusively on t≈0, making causal training worse than standard
          (improvement_pct = -73%). v2 uses per-band gate without cascade:
          w[b] = sigmoid(steepness * (eps - mean_residual_{b-1}))
          so each band is gated independently on its predecessor.

  [FIX 3] CAUSAL_EPS raised from 0.01 to 0.3. At β=10, training residuals
          are in the 0.1–1.0 range early in training. eps=0.01 caused
          gate ≈ sigmoid(10*(0.01-0.5)) ≈ 0 for all bands from the start.
          eps=0.3 gives a meaningful threshold that gates open as residuals
          improve during training.

  [FIX 4] Explicit seed for reproducibility. v1 had no seed control.

  [FIX 5] Lag metric fallback documented. If both models fail completely
          (lag=t_max for all checkpoints), the JSON records
          lag_metric_informative=False and includes a note.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ===================================================================
# Speed flags
# ===================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp16] Device: {DEVICE}")

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp16"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Configuration
# ===================================================================
BETA    = 10          # FIX 1: was 50 (complete failure), now 10 (marginal)
SEED    = 42          # FIX 4: explicit seed
N_HIDDEN   = 4
N_NEURONS  = 128
ACTIVATION = "tanh"

CHECKPOINTS   = [1000, 5000, 10000, 30000, 50000]
N_EPOCHS      = CHECKPOINTS[-1]
LR            = 1e-3
LR_MIN        = 1e-5

N_COLLOCATION = 10000
N_IC          = 400
N_BC          = 400

# FIX 2+3: causal parameters
N_TIME_BANDS      = 20
CAUSAL_EPS        = 0.3    # was 0.01 — too tight, collapsed all weights
CAUSAL_STEEPNESS  = 5.0    # was 10 — reduced to avoid over-hard gating
WEIGHT_UPDATE_FREQ = 200

# Evaluation grid
NX_EVAL = 200
NT_EVAL = 200
T_MAX   = 2.0

# Lag analysis threshold
LAG_THRESHOLD = 0.10   # 10% — raised from 5% to be reachable at β=10


# ===================================================================
# Model
# ===================================================================

class AdvectionPINN(nn.Module):
    def __init__(self, n_hidden=4, n_neurons=128, activation="tanh"):
        super().__init__()
        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
        act_fn  = act_map[activation]()
        layers  = [nn.Linear(2, n_neurons), act_fn]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), act_fn]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def exact_solution(x_np, t_np, beta):
    return np.sin(x_np - beta * t_np)


# ===================================================================
# Loss components
# ===================================================================

def pde_residual(model, x, t, beta):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u   = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                               create_graph=True)[0]
    return u_t + beta * u_x


def sample_collocation(n_col, n_ic, n_bc):
    x_col = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    t_col = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * T_MAX
    x_ic  = torch.rand(n_ic,  1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    t_ic  = torch.zeros(n_ic, 1, dtype=DTYPE, device=DEVICE)
    t_bc  = torch.rand(n_bc,  1, dtype=DTYPE, device=DEVICE) * T_MAX
    x_bc_l = torch.zeros(n_bc, 1, dtype=DTYPE, device=DEVICE)
    x_bc_r = torch.full((n_bc, 1), 2 * np.pi, dtype=DTYPE, device=DEVICE)
    return (x_col, t_col), (x_ic, t_ic), (t_bc, x_bc_l, x_bc_r)


def compute_loss(model, beta, n_col, n_ic, n_bc,
                 causal_weights=None):
    """
    Unified loss. If causal_weights is None → standard training.
    If causal_weights is a (n_bands,) tensor → causal training.
    """
    (x_col, t_col), (x_ic, t_ic), (t_bc, x_bc_l, x_bc_r) = \
        sample_collocation(n_col, n_ic, n_bc)

    res      = pde_residual(model, x_col, t_col, beta)
    res_sq   = res ** 2

    if causal_weights is not None:
        # Assign each collocation point to its time band
        band_edges = torch.linspace(0, T_MAX, N_TIME_BANDS + 1,
                                    device=DEVICE)
        t_flat = t_col.squeeze(1)
        w      = torch.ones_like(t_flat)
        for b in range(N_TIME_BANDS):
            mask = (t_flat >= band_edges[b]) & (t_flat < band_edges[b + 1])
            w[mask] = causal_weights[b]
        # Last edge
        mask = t_flat >= band_edges[-1]
        w[mask] = causal_weights[-1]
        loss_pde = (w.unsqueeze(1).detach() * res_sq).mean()
    else:
        loss_pde = res_sq.mean()

    u_ic_pred = model(x_ic, t_ic)
    u_ic_true = torch.sin(x_ic)
    loss_ic   = ((u_ic_pred - u_ic_true) ** 2).mean()

    loss_bc = ((model(x_bc_l, t_bc) - model(x_bc_r, t_bc)) ** 2).mean()

    return loss_pde + 100 * loss_ic + 10 * loss_bc


# ===================================================================
# FIX 2+3 — Corrected causal weight update
# ===================================================================

def update_causal_weights(model, beta):
    """
    Per-band causal gate (FIX 2: not cumulative product).

    w[0] = 1.0  (anchored at IC, always active)
    w[b] = sigmoid(steepness * (eps - mean_residual_{b-1}))
           for b >= 1

    Each band is gated independently on the mean PDE residual in the
    previous band. As training progresses and residuals decrease,
    gates open sequentially from early to late time.

    FIX 3: CAUSAL_EPS=0.3 gives a meaningful threshold reachable at β=10.
    """
    band_edges = np.linspace(0, T_MAX, N_TIME_BANDS + 1)
    n_probe    = 100
    weights    = torch.ones(N_TIME_BANDS, dtype=DTYPE, device=DEVICE)

    band_residuals = []
    for b in range(N_TIME_BANDS):
        # Probe this band's residual
        x_p = torch.rand(n_probe, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
        t_p = (torch.rand(n_probe, 1, dtype=DTYPE, device=DEVICE)
               * (band_edges[b + 1] - band_edges[b])
               + band_edges[b])
        x_p.requires_grad_(True)
        t_p.requires_grad_(True)
        u_p   = model(x_p, t_p)
        u_t_p = torch.autograd.grad(u_p, t_p, torch.ones_like(u_p),
                                    create_graph=False,
                                    retain_graph=True)[0]
        u_x_p = torch.autograd.grad(u_p, x_p, torch.ones_like(u_p),
                                    create_graph=False)[0]
        res_b = float((u_t_p + beta * u_x_p).abs().mean().item())
        band_residuals.append(res_b)

        # Compute weight for NEXT band based on this band's residual
        if b + 1 < N_TIME_BANDS:
            gate_val = float(torch.sigmoid(
                torch.tensor(CAUSAL_STEEPNESS * (CAUSAL_EPS - res_b),
                             dtype=DTYPE)).item())
            weights[b + 1] = gate_val   # FIX 2: independent gate, not cascade

    return weights.detach(), band_residuals


# ===================================================================
# Evaluation helpers
# ===================================================================

def build_eval_grid():
    x_vals = np.linspace(0, 2 * np.pi, NX_EVAL)
    t_vals = np.linspace(0, T_MAX,     NT_EVAL)
    XX, TT = np.meshgrid(x_vals, t_vals)   # shape (NT_EVAL, NX_EVAL)
    return XX, TT, x_vals, t_vals


def eval_model_on_grid(model, XX, TT):
    model.eval()
    with torch.no_grad():
        x_f = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        t_f = torch.tensor(TT.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u   = model(x_f, t_f).cpu().numpy().reshape(XX.shape)
    return u


def compute_error_map(u_pred, XX, TT, beta):
    u_exact = exact_solution(XX, TT, beta)
    norm    = np.abs(u_exact).mean() + 1e-8
    return np.abs(u_pred - u_exact) / norm


def temporal_error_profile(error_map):
    return error_map.mean(axis=1)   # (NT_EVAL,)


# ===================================================================
# Training runners
# ===================================================================

def train_model(mode, beta, seed, ckpt=None, ckpt_path=None, model_path=None):
    """
    mode: "standard" or "causal"
    Returns (model, snapshots_dict, losses_list, x_vals, t_vals)
    """
    assert mode in ("standard", "causal")
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n  [{mode.upper()} Training]  β={beta}  seed={seed}")

    model     = AdvectionPINN(N_HIDDEN, N_NEURONS, ACTIVATION).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)

    XX, TT, x_vals, t_vals = build_eval_grid()
    snapshots = {}
    losses    = []

    # Initial causal weights (all ones)
    causal_weights = torch.ones(N_TIME_BANDS, dtype=DTYPE, device=DEVICE)

    start_epoch = 1
    if ckpt is not None and f"{mode}_completed_epoch" in ckpt:
        start_epoch = ckpt[f"{mode}_completed_epoch"] + 1
        losses = ckpt.get(f"{mode}_losses", [])
        snaps = ckpt.get(f"{mode}_snapshots", {})
        for k, v in snaps.items():
            snapshots[int(k)] = np.array(v)
        
        if model_path and model_path.exists():
            model.load_state_dict(torch.load(model_path))
            print(f"    [Resuming {mode} from epoch {start_epoch - 1}]")
            
        for _ in range(start_epoch - 1):
            scheduler.step()

        if start_epoch > N_EPOCHS:
            return model, snapshots, losses, x_vals, t_vals

    for epoch in range(start_epoch, N_EPOCHS + 1):
        # Update causal weights periodically
        if mode == "causal" and epoch % WEIGHT_UPDATE_FREQ == 1:
            model.eval()
            with torch.enable_grad():
                causal_weights, _ = update_causal_weights(model, beta)
            model.train()

        model.train()
        optimizer.zero_grad()
        loss = compute_loss(
            model, beta, N_COLLOCATION, N_IC, N_BC,
            causal_weights=(causal_weights if mode == "causal" else None))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

        if epoch in CHECKPOINTS:
            u_pred  = eval_model_on_grid(model, XX, TT)
            err_map = compute_error_map(u_pred, XX, TT, beta)
            snapshots[epoch] = err_map
            w_info  = (f"mean_w={causal_weights.mean().item():.3f}"
                       if mode == "causal" else "")
            print(f"    Epoch {epoch:>6d}: loss={loss.item():.4e}  "
                  f"mean_err={err_map.mean():.4f}  {w_info}")

        # Periodic checkpoint
        if epoch % 5000 == 0 or epoch == N_EPOCHS:
            if model_path:
                torch.save(model.state_dict(), model_path)
            if ckpt is not None and ckpt_path:
                ckpt[f"{mode}_completed_epoch"] = epoch
                ckpt[f"{mode}_losses"] = losses
                snap_to_save = {str(k): v.tolist() for k, v in snapshots.items()}
                ckpt[f"{mode}_snapshots"] = snap_to_save
                with open(ckpt_path, "w") as f:
                    json.dump(ckpt, f)

    return model, snapshots, losses, x_vals, t_vals


# ===================================================================
# Plotting
# ===================================================================

def plot_residual_maps(snapshots, x_vals, t_vals, title, filepath):
    n    = len(CHECKPOINTS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4),
                              constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Shared colormap range (capped at 2.0)
    vmax = min(max(snapshots[ck].max() for ck in CHECKPOINTS), 2.0)

    for ax, ck in zip(axes, CHECKPOINTS):
        err = snapshots[ck]
        im  = ax.pcolormesh(x_vals, t_vals, err,
                            cmap="hot_r", vmin=0, vmax=vmax,
                            shading="auto")
        ax.set_title(f"Iter {ck}", fontsize=10)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("t", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8, label="|error|")

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_temporal_comparison(std_snaps, caus_snaps, t_vals, filepath):
    n = len(CHECKPOINTS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4),
                              constrained_layout=True)
    fig.suptitle(
        f"Temporal Error Profile: Standard vs Causal  (β={BETA})\n"
        f"FIX 1: β={BETA} (was β=50 — complete spectral failure)",
        fontsize=11, fontweight="bold")

    for ax, ck in zip(axes, CHECKPOINTS):
        p_std  = temporal_error_profile(std_snaps[ck])
        p_caus = temporal_error_profile(caus_snaps[ck])
        ax.plot(t_vals, p_std,  color="#E64040", lw=1.8, label="Standard")
        ax.plot(t_vals, p_caus, color="#3A7FD5", lw=1.8, ls="--",
                label="Causal")
        ax.axhline(LAG_THRESHOLD, color="grey", ls=":", lw=0.8,
                   label=f"ε={LAG_THRESHOLD}")
        ax.set_title(f"Iter {ck}", fontsize=10)
        ax.set_xlabel("t", fontsize=9)
        ax.set_ylabel("Mean |error|", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylim(0, None)

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def compute_lag_metrics(snapshots, t_vals):
    """Last t where mean error > LAG_THRESHOLD. t_max if always above."""
    lags           = {}
    all_at_t_max   = True
    for ck, err_map in snapshots.items():
        prof    = temporal_error_profile(err_map)
        crossed = np.where(prof > LAG_THRESHOLD)[0]
        lag_t   = float(t_vals[crossed[-1]]) if len(crossed) > 0 else 0.0
        lags[ck] = lag_t
        if lag_t < T_MAX:
            all_at_t_max = False
    return lags, all_at_t_max


def plot_causality_lag_analysis(std_snaps, caus_snaps, t_vals, filepath):
    std_lags,  std_maxed  = compute_lag_metrics(std_snaps,  t_vals)
    caus_lags, caus_maxed = compute_lag_metrics(caus_snaps, t_vals)

    epochs        = CHECKPOINTS
    std_lag_vals  = [std_lags[e]  for e in epochs]
    caus_lag_vals = [caus_lags[e] for e in epochs]
    both_maxed    = std_maxed and caus_maxed

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              constrained_layout=True)
    fig.suptitle(
        f"Causality Lag Analysis  (β={BETA})\n"
        "FIX 2+3: per-band gate (not cascade), CAUSAL_EPS=0.3",
        fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, std_lag_vals,  "o-",  color="#E64040",
            label="Standard", lw=2)
    ax.plot(epochs, caus_lag_vals, "s--", color="#3A7FD5",
            label="Causal",   lw=2)
    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel(f"Last t where error > {LAG_THRESHOLD}",
                  fontsize=11)
    ax.set_title("Error Convergence Lag", fontsize=11)
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    if both_maxed:
        ax.text(0.5, 0.5,
                f"Both models have error > {LAG_THRESHOLD}\nacross "
                f"full t-domain at all checkpoints.\n"
                "Lag metric is at maximum (uninformative).\n"
                "Consider using β < 10 or more epochs.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#FFF9C4",
                          edgecolor="#F9A825", alpha=0.9))

    ax2 = axes[1]
    final_ck  = CHECKPOINTS[-1]
    p_std  = temporal_error_profile(std_snaps[final_ck])
    p_caus = temporal_error_profile(caus_snaps[final_ck])
    ax2.semilogy(t_vals, p_std  + 1e-8, color="#E64040",
                 label="Standard", lw=2)
    ax2.semilogy(t_vals, p_caus + 1e-8, color="#3A7FD5",
                 label="Causal",   lw=2, ls="--")
    ax2.axhline(LAG_THRESHOLD, color="grey", ls=":", lw=1,
                label=f"ε={LAG_THRESHOLD}")
    ax2.set_xlabel("t", fontsize=11)
    ax2.set_ylabel("Mean Relative Error (log)", fontsize=11)
    ax2.set_title(f"Final Error Profile (iter={final_ck})",
                  fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")

    return std_lags, caus_lags, both_maxed


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 16: Causality Violation in Standard PINNs  [v2]")
    print(f"Device         : {DEVICE}")
    print(f"β = {BETA}  (was 50 — FIX 1: β=10 to avoid full spectral failure)")
    print(f"CAUSAL_EPS     : {CAUSAL_EPS}  (was 0.01 — FIX 3)")
    print(f"Weight scheme  : per-band gate (was cascade — FIX 2)")
    print(f"Seed           : {SEED}  (FIX 4)")
    print("=" * 70)

    t0 = time.time()

    # Checkpoint setup
    ckpt_path = OUTPUT_DIR / "exp16_checkpoint.json"
    model_std_path = OUTPUT_DIR / "model_std.pt"
    model_caus_path = OUTPUT_DIR / "model_caus.pt"
    
    ckpt = {}
    if ckpt_path.exists():
        try:
            with open(ckpt_path, 'r') as f:
                ckpt = json.load(f)
        except Exception:
            pass

    # Train both variants
    (model_std,  std_snaps,  std_losses,  x_vals, t_vals) = \
        train_model("standard", BETA, SEED, ckpt, ckpt_path, model_std_path)
    (model_caus, caus_snaps, caus_losses, _,      _     ) = \
        train_model("causal",   BETA, SEED, ckpt, ckpt_path, model_caus_path)

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── Plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_residual_maps(
        std_snaps, x_vals, t_vals,
        title=f"Residual Map — Standard PINN (β={BETA})",
        filepath=OUTPUT_DIR / "residual_maps_standard.png")

    plot_residual_maps(
        caus_snaps, x_vals, t_vals,
        title=f"Residual Map — Causal PINN (β={BETA})\n"
              "(per-band gate, CAUSAL_EPS=0.3)",
        filepath=OUTPUT_DIR / "residual_maps_causal.png")

    plot_temporal_comparison(
        std_snaps, caus_snaps, t_vals,
        filepath=OUTPUT_DIR / "temporal_error_comparison.png")

    std_lags, caus_lags, lag_uninformative = plot_causality_lag_analysis(
        std_snaps, caus_snaps, t_vals,
        filepath=OUTPUT_DIR / "causality_lag_analysis.png")

    # ── Quantitative summary ─────────────────────────────────────────
    final_ck      = CHECKPOINTS[-1]
    std_final_err  = float(std_snaps[final_ck].mean())
    caus_final_err = float(caus_snaps[final_ck].mean())
    improvement    = (std_final_err - caus_final_err) / (std_final_err + 1e-12)

    p_std  = temporal_error_profile(std_snaps[final_ck])
    p_caus = temporal_error_profile(caus_snaps[final_ck])
    std_viol_frac  = float((p_std  > LAG_THRESHOLD).mean())
    caus_viol_frac = float((p_caus > LAG_THRESHOLD).mean())

    print(f"\n{'=' * 70}")
    print("EXPERIMENT 16 — SUMMARY  [v2]")
    print(f"{'=' * 70}")
    print(f"Final mean error  | Standard={std_final_err:.4f} | "
          f"Causal={caus_final_err:.4f}")
    print(f"Improvement (causal over standard): {100*improvement:.1f}%")
    print(f"Fraction of t-domain above ε={LAG_THRESHOLD}:")
    print(f"  Standard={100*std_viol_frac:.1f}%  |  Causal={100*caus_viol_frac:.1f}%")
    if improvement < 0:
        print(f"\n  ⚠ Causal is still WORSE — causal gating may be too aggressive.")
        print(f"    Try increasing CAUSAL_EPS further or reducing CAUSAL_STEEPNESS.")

    # ── JSON ─────────────────────────────────────────────────────────
    results = {
        "experiment": "Causality Violation in Standard PINNs",
        "version":    "v2-journal-ready",
        "config": {
            "beta":             BETA,
            "seed":             SEED,
            "n_hidden":         N_HIDDEN,
            "n_neurons":        N_NEURONS,
            "checkpoints":      CHECKPOINTS,
            "n_epochs":         N_EPOCHS,
            "n_time_bands":     N_TIME_BANDS,
            "causal_eps":       CAUSAL_EPS,
            "causal_steepness": CAUSAL_STEEPNESS,
            "lag_threshold":    LAG_THRESHOLD,
        },

        "fix_notes": {
            "fix1_beta": (
                f"v1 used β=50 (confirmed complete spectral failure in Exp 1). "
                f"Both models predicted u≡0 (L2≈0.83–1.44). Comparing two "
                f"zero-function predictors cannot measure causality effects. "
                f"v2 uses β={BETA} (marginal case, Exp 1 L2≈0.035) where "
                "PINNs partially succeed and causality differences are visible."
            ),
            "fix2_weight_formula": (
                "v1 used w[b] = w[b-1] * gate (cumulative product). "
                "With steepness=10, gate≈0.007 for typical residuals, "
                "so w[3]≈3×10⁻⁷ — network trained only on t≈0. "
                "v2 uses w[b] = sigmoid(steepness*(eps - residual_{b-1})) "
                "independently per band. No cascade multiplication."
            ),
            "fix3_causal_eps": (
                f"v1 CAUSAL_EPS=0.01. At β=50, PDE residuals are 0.1–1.0 "
                "early in training, so gate=sigmoid(10*(0.01-0.5))≈0 always. "
                f"v2 CAUSAL_EPS={CAUSAL_EPS}, giving a reachable threshold "
                "that gates open as training progresses."
            ),
            "fix4_seed": f"Explicit seed={SEED} added for reproducibility.",
            "fix5_lag_fallback": (
                "If both models fail to bring error below LAG_THRESHOLD at "
                "any time t, the lag metric is stuck at T_MAX and uninformative. "
                "lag_metric_informative=False is flagged in results."
            ),
        },

        "final_mean_error": {
            "standard":       std_final_err,
            "causal":         caus_final_err,
            "improvement_pct": float(100 * improvement),
            "causal_better":   bool(improvement > 0),
        },
        "causality_violation_fraction": {
            "standard": std_viol_frac,
            "causal":   caus_viol_frac,
        },
        "lag_metric_informative": bool(not lag_uninformative),
        "lag_metrics": {
            "standard": {str(k): float(v) for k, v in std_lags.items()},
            "causal":   {str(k): float(v) for k, v in caus_lags.items()},
        },
        "checkpoint_mean_errors": {
            "standard": {str(ck): float(std_snaps[ck].mean())
                         for ck in CHECKPOINTS},
            "causal":   {str(ck): float(caus_snaps[ck].mean())
                         for ck in CHECKPOINTS},
        },
        "elapsed_seconds": elapsed,
    }

    out_json = OUTPUT_DIR / "exp16_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_json}")
    print(f"Plots   → {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    run_experiment()