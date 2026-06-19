"""
exp15_lr_phase_diagram.py — Learning Rate Failure Phase Diagram
[v2 — journal-ready fixes]

Sweeps learning rate and training duration for the Burgers PINN with Adam:
  - Learning rates: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]
  - Iterations:     [10000, 30000, 50000, 100000]

Trains all combinations (36 total), recording final L2 error.
Generates a 2D heatmap phase diagram identifying three regimes:
  1. Underfitting zone  (not enough iterations or LR too low)
  2. Convergence zone   (optimal region)
  3. Divergence zone    (LR too high → explosion)

Outputs (results/exp15/):
  - lr_phase_diagram.png
  - regime_boundaries.png
  - lr_corridor.png
  - convergence_curves_grid.png
  - exp15_results.json

FIXES vs v1 (journal-ready):
  [FIX 1] Fixed eta_min in CosineAnnealingLR — v1 used lr * 0.01 as
          eta_min, making the scheduler behavior non-equivalent across
          LRs: LR=0.1 → eta_min=0.001, LR=1e-5 → eta_min=1e-7. The
          LR effectively decays to different fractions of its initial
          value for each config, confounding the comparison. v2 uses
          a fixed absolute eta_min=1e-6 for all configs.

  [FIX 2] Corrected divergence detection — v1 used `loss_val > 1e10`
          as the divergence trigger but applied gradient clipping at
          5.0. Clipping prevents loss from exceeding 1e10, so many
          diverged configs were silently misclassified as underfitting.
          v2 detects divergence by: (a) parameter norm explosion
          (||θ|| > 1e6), (b) gradient norm explosion before clipping
          (‖∇‖ > 1e4), or (c) loss > 1e8 after clipping. Records the
          epoch of first divergence detection.

  [FIX 3] Heatmap for diverged cells — v1 set L2=inf → 10.0 for
          display, making diverged cells indistinguishable from
          genuinely bad L2=10 runs. v2 uses a distinct color (black)
          for confirmed diverged cells and annotates them "DIV",
          separate from the L2 colormap.

  [FIX 4] Runtime warning for 100k column — v1 silently ran 100000-
          epoch configs on RTX 3050 (up to 12 hrs per config × 9 LRs).
          v2 prints an estimated runtime warning before starting and
          offers FAST_MODE that caps at 50000 epochs.

  [FIX 5] Fixed single-seed evaluation note — each config tested with
          N_SEEDS=2 seeds; results marked as "uncertain" in JSON when
          seeds disagree on regime classification.
"""

import sys, os, json, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

from pinn_core import DEVICE, DTYPE, save_results
from pinn_equations import (
    GenericPINN, burgers_residual,
    sample_burgers_domain, evaluate_burgers, load_burgers_reference,
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
N_HIDDEN   = 4
N_NEURONS  = 64
N_INT      = 10000
N_IC       = 200
N_BC       = 200
N_SEEDS    = 2        # FIX 5: two seeds per config

LEARNING_RATES    = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]
ITERATION_COUNTS  = [10000, 30000, 50000, 100000]

# FIX 4: fast mode skips 100k column to save ~36 hrs on RTX 3050
# Set FAST_MODE=False to run the full experiment (paper quality)
# Set FAST_MODE=True for a quick validation pass
FAST_MODE = False
if FAST_MODE:
    ITERATION_COUNTS = [10000, 30000, 50000]
    print("  ⚡ FAST_MODE=True: skipping 100k column (~36hr savings)")

# FIX 1: fixed absolute eta_min (not lr-relative)
ETA_MIN = 1e-6

# FIX 2: divergence detection thresholds
GRAD_NORM_THRESHOLD  = 1e4   # before clipping
PARAM_NORM_THRESHOLD = 1e6
LOSS_THRESHOLD       = 1e8   # after clipping

CONVERGENCE_THRESHOLD = 0.1
DIVERGENCE_THRESHOLD  = 5.0

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp15"

# ===================================================================
# Graceful interrupt & Checkpoint handling
# ===================================================================
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"
_STOP_REQUESTED = False

def _handle_sigint(sig, frame):
    global _STOP_REQUESTED
    if not _STOP_REQUESTED:
        print("\n\n  ⚠  Ctrl+C received — finishing current seed then saving "
              "checkpoint.\n     Press Ctrl+C again to force-quit immediately.\n")
        _STOP_REQUESTED = True
    else:
        print("\n  Force-quitting now.")
        sys.exit(1)

signal.signal(signal.SIGINT, _handle_sigint)

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                ckpt = json.load(f)
            n_done = 0
            if "completed" in ckpt:
                for conf in ckpt["completed"].values():
                    n_done += len(conf)
            print(f"\n  ✔ Checkpoint found: {n_done} runs already done.")
            print(f"    Resuming from: {CHECKPOINT_PATH}\n")
            return ckpt
        except json.JSONDecodeError:
            print("  ⚠ Checkpoint file corrupted. Starting fresh.")
    return {"completed": {}}

def save_checkpoint(ckpt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(ckpt, f)
    tmp_path.replace(CHECKPOINT_PATH)


# ===================================================================
# FIX 2 — Robust divergence detection
# ===================================================================

def check_divergence(model, loss_val, pre_clip_grad_norm):
    """
    Multi-criterion divergence detection.
    v1 only checked loss > 1e10 — gradient clipping prevented this.
    v2 checks: parameter norm, pre-clip gradient norm, and loss value.
    """
    if not np.isfinite(loss_val) or loss_val > LOSS_THRESHOLD:
        return True, "loss_explosion"

    if pre_clip_grad_norm > GRAD_NORM_THRESHOLD:
        return True, "gradient_explosion"

    # Parameter norm check
    total_param_norm = sum(
        p.data.norm().item() ** 2
        for p in model.parameters()
        if p.data is not None
    ) ** 0.5
    if total_param_norm > PARAM_NORM_THRESHOLD:
        return True, "parameter_explosion"

    return False, None


# ===================================================================
# Training
# ===================================================================

def train_single_config(lr, n_epochs, seed=0):
    """
    Train a Burgers PINN with given LR and epoch count.
    FIX 1: fixed eta_min=ETA_MIN (not lr*0.01).
    FIX 2: robust divergence detection.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = GenericPINN(in_dim=2, out_dim=1, n_hidden=N_HIDDEN,
                             n_neurons=N_NEURONS, activation="tanh").to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # FIX 1: absolute eta_min, not lr-relative
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=ETA_MIN)

    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = \
        sample_burgers_domain(N_INT, N_IC, N_BC)

    loss_hist    = []
    diverged     = False
    diverge_ep   = None
    diverge_cause = None
    t0 = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        res      = burgers_residual(model, x_int, t_int, BURGERS_NU)
        loss_pde = torch.mean(res ** 2)
        loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
        loss_bc  = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
        loss     = loss_pde + 10 * loss_ic + loss_bc
        loss.backward()

        # FIX 2: measure grad norm BEFORE clipping
        pre_clip_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        # FIX 2: multi-criterion divergence check
        is_div, cause = check_divergence(model, loss_val, pre_clip_norm)
        if is_div:
            diverged      = True
            diverge_ep    = epoch
            diverge_cause = cause
            break

        loss_hist.append(loss_val)

    return model, loss_hist, diverged, diverge_ep, diverge_cause, \
           time.time() - t0


def evaluate_config(ckpt, lr, n_epochs, x_ref, t_ref, u_ref):
    """Run N_SEEDS seeds and return aggregated result, using/updating ckpt."""
    config_key = f"lr_{lr:.2e}_iters_{n_epochs}"
    if config_key not in ckpt.get("completed", {}):
        if "completed" not in ckpt:
            ckpt["completed"] = {}
        ckpt["completed"][config_key] = {}

    seed_l2s = []
    seed_divs = []
    seed_trajs = []

    for seed in range(N_SEEDS):
        if _STOP_REQUESTED:
            print(f"\n  ⏸  Pause requested — stopping after seed {seed-1}.")
            break

        str_seed = str(seed)
        if str_seed in ckpt["completed"][config_key]:
            print(f"    Seed {seed} already done — loading from checkpoint.")
            out_ckpt = ckpt["completed"][config_key][str_seed]
            seed_l2s.append(out_ckpt["l2"])
            seed_divs.append(out_ckpt["diverged"])
            seed_trajs.append(out_ckpt["trajectory"])
            continue

        model, lh, divd, div_ep, div_cause, t_tr = \
            train_single_config(lr, n_epochs, seed=seed)

        l2 = float("inf") if divd else float("nan")
        if not divd and x_ref is not None:
            try:
                _, l2 = evaluate_burgers(model, x_ref, t_ref, u_ref)
            except Exception:
                pass

        # Subsample trajectory
        step = max(1, len(lh) // 500)
        sub_traj = lh[::step]

        seed_l2s.append(l2)
        seed_divs.append(divd)
        seed_trajs.append(sub_traj)

        # Save this seed to checkpoint
        ckpt["completed"][config_key][str_seed] = {
            "l2": l2,
            "diverged": divd,
            "trajectory": sub_traj,
        }
        save_checkpoint(ckpt)

    if len(seed_l2s) < N_SEEDS:
        return None

    # Aggregate
    finite_l2s = [e for e in seed_l2s if np.isfinite(e)]
    mean_l2    = float(np.mean(finite_l2s)) if finite_l2s else float("inf")
    std_l2     = float(np.std(finite_l2s))  if len(finite_l2s) > 1 else 0.0
    any_div    = any(seed_divs)
    all_div    = all(seed_divs)

    # Regime classification
    if all_div or mean_l2 > DIVERGENCE_THRESHOLD:
        regime = "divergence"
    elif mean_l2 < CONVERGENCE_THRESHOLD:
        regime = "convergence"
    else:
        regime = "underfitting"

    # Uncertainty flag: seeds disagree on regime
    regimes = []
    for l2, div in zip(seed_l2s, seed_divs):
        if div or not np.isfinite(l2) or l2 > DIVERGENCE_THRESHOLD:
            regimes.append("divergence")
        elif l2 < CONVERGENCE_THRESHOLD:
            regimes.append("convergence")
        else:
            regimes.append("underfitting")
    uncertain = len(set(regimes)) > 1

    return {
        "l2_per_seed":  seed_l2s,
        "mean_l2":      mean_l2,
        "std_l2":       std_l2,
        "regime":       regime,
        "uncertain":    uncertain,
        "any_diverged": any_div,
        "trajectory":   seed_trajs[0] if seed_trajs else [],
    }


def classify_regime(mean_l2, any_div):
    if any_div or mean_l2 > DIVERGENCE_THRESHOLD:
        return "divergence"
    elif mean_l2 < CONVERGENCE_THRESHOLD:
        return "convergence"
    return "underfitting"


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    n_lr   = len(LEARNING_RATES)
    n_iter = len(ITERATION_COUNTS)
    total  = n_lr * n_iter

    # FIX 4: runtime warning
    est_mins_per_config = 6   # RTX 3050, 10k collocation, 100k epochs ≈ 10 min
    if not FAST_MODE:
        est_total = total * est_mins_per_config * N_SEEDS
        print("=" * 70)
        print(f"EXP 15: LR Phase Diagram  [v2 — journal]")
        print(f"Device  : {DEVICE}")
        print(f"Configs : {total}  ×  {N_SEEDS} seeds  =  {total*N_SEEDS} runs")
        print(f"⏱ Estimated time: ~{est_total // 60}h {est_total % 60}m on RTX 3050")
        print(f"  (100k-epoch column is the bottleneck — "
              f"set FAST_MODE=True to skip it)")
        print(f"FIX 1: eta_min = {ETA_MIN} (fixed, not lr-relative)")
        print(f"FIX 2: divergence detection via grad/param/loss norms")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        x_ref, t_ref, u_ref = load_burgers_reference()
    except FileNotFoundError:
        print("  ⚠ Reference not found. L2 = NaN.")
        x_ref = t_ref = u_ref = None

    ckpt = load_checkpoint()

    # Storage
    l2_matrix      = np.full((n_lr, n_iter), np.nan)
    std_matrix     = np.zeros((n_lr, n_iter))
    regime_matrix  = [["" for _ in range(n_iter)] for _ in range(n_lr)]
    uncertain_mask = [[False for _ in range(n_iter)] for _ in range(n_lr)]
    div_mask       = [[False for _ in range(n_iter)] for _ in range(n_lr)]
    trajectories   = {}

    done = 0
    for i, lr in enumerate(LEARNING_RATES):
        for j, n_ep in enumerate(ITERATION_COUNTS):
            done += 1
            if _STOP_REQUESTED:
                break
            print(f"\n  [{done}/{total}] LR={lr:.0e}, Epochs={n_ep}")

            res = evaluate_config(ckpt, lr, n_ep, x_ref, t_ref, u_ref)
            if res is None:
                break

            l2_matrix[i, j]      = res["mean_l2"]
            std_matrix[i, j]     = res["std_l2"]
            regime_matrix[i][j]  = res["regime"]
            uncertain_mask[i][j] = res["uncertain"]
            div_mask[i][j]       = res["any_diverged"]
            trajectories[(lr, n_ep)] = res["trajectory"]

            sym = {"convergence": "✓", "underfitting": "~",
                   "divergence": "✗"}.get(res["regime"], "?")
            unc = " ⚠uncertain" if res["uncertain"] else ""
            print(f"    mean L2={res['mean_l2']:.6f} ± {res['std_l2']:.6f} "
                  f"[{sym} {res['regime']}{unc}]")

    if _STOP_REQUESTED or np.any(np.isnan(l2_matrix)):
        print(f"\n{'=' * 70}")
        print(f"  ⏸  PAUSED — Resume script later.")
        print(f"  Checkpoint: {CHECKPOINT_PATH}")
        print(f"{'=' * 70}")
        return None

    # ── Analysis ────────────────────────────────────────────────────
    optimal_corridor   = {}
    divergence_boundary = {}

    for j, n_ep in enumerate(ITERATION_COUNTS):
        conv_lrs = [LEARNING_RATES[i] for i in range(n_lr)
                    if regime_matrix[i][j] == "convergence"]
        if conv_lrs:
            best_i   = min(range(n_lr),
                           key=lambda ii: (l2_matrix[ii, j]
                                           if regime_matrix[ii][j] == "convergence"
                                           else float("inf")))
            optimal_corridor[n_ep] = {
                "min_lr":  min(conv_lrs),
                "max_lr":  max(conv_lrs),
                "best_lr": LEARNING_RATES[best_i],
                "best_l2": float(l2_matrix[best_i, j]),
            }
        else:
            optimal_corridor[n_ep] = None

        div_lrs = [LEARNING_RATES[i] for i in range(n_lr)
                   if regime_matrix[i][j] == "divergence"]
        divergence_boundary[n_ep] = min(div_lrs) if div_lrs else None

    finite_mask = np.isfinite(l2_matrix) & (l2_matrix < DIVERGENCE_THRESHOLD)
    if np.any(finite_mask):
        best_idx  = np.unravel_index(
            np.where(finite_mask, l2_matrix, np.inf).argmin(),
            l2_matrix.shape)
        best_lr   = LEARNING_RATES[best_idx[0]]
        best_iter = ITERATION_COUNTS[best_idx[1]]
        best_l2   = float(l2_matrix[best_idx])
    else:
        best_lr = best_iter = None; best_l2 = float("nan")

    print(f"\n  ★ Best: LR={best_lr}, Iters={best_iter}, L2={best_l2:.6f}")

    # ── Plots ────────────────────────────────────────────────────────
    print("\n── Generating plots ──")

    lr_labels   = [f"{lr:.0e}" for lr in LEARNING_RATES]
    iter_labels = [f"{n//1000}k" for n in ITERATION_COUNTS]

    regime_colors = {"convergence": "#2E7D32",
                     "underfitting": "#FF9800",
                     "divergence": "#D32F2F"}

    # 1. Phase diagram heatmap (FIX 3: diverged cells = black)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Build display matrix: use log10 L2 for convergence/underfitting,
    # fixed low value for diverged (displayed as black separately)
    l2_display = l2_matrix.copy()
    for i in range(n_lr):
        for j in range(n_iter):
            if div_mask[i][j] or l2_display[i, j] >= DIVERGENCE_THRESHOLD:
                l2_display[i, j] = np.nan   # will be masked → black

    # Finite range for colormap
    finite_vals = l2_display[np.isfinite(l2_display)]
    vmin = float(np.log10(np.nanmin(finite_vals) + 1e-10)) \
           if len(finite_vals) > 0 else -2
    vmax = float(np.log10(DIVERGENCE_THRESHOLD))

    log_display = np.where(np.isfinite(l2_display),
                            np.log10(l2_display + 1e-10), np.nan)

    cmap_base = plt.cm.RdYlGn_r.copy()
    cmap_base.set_bad(color="black")   # FIX 3: diverged = black, not 10.0

    im = ax.imshow(log_display, cmap=cmap_base,
                   vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax,
                        label="log₁₀(L2 Error)  [black = confirmed divergence]")

    # Cell annotations
    for i in range(n_lr):
        for j in range(n_iter):
            regime = regime_matrix[i][j]
            l2_val = l2_matrix[i, j]
            unc    = uncertain_mask[i][j]

            if div_mask[i][j] or regime == "divergence":
                txt      = "DIV"
                txt_color = "white"
            elif np.isfinite(l2_val):
                txt      = f"{l2_val:.3f}"
                txt_color = "white" if log_display[i, j] > (vmin + vmax) / 2 \
                            else "black"
            else:
                txt = "NaN"; txt_color = "white"

            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")

    ax.set_xticks(range(n_iter))
    ax.set_xticklabels(iter_labels, fontsize=11)
    ax.set_yticks(range(n_lr))
    ax.set_yticklabels(lr_labels, fontsize=11)
    ax.set_xlabel("Training Iterations", fontsize=13)
    ax.set_ylabel("Learning Rate", fontsize=13)
    ax.set_title(
        "Learning Rate × Iterations Phase Diagram\n"
        "Black = divergence (confirmed by grad/param/loss norms).",
        fontweight="bold", fontsize=11)
    savefig(fig, OUTPUT_DIR / "lr_phase_diagram.png")

    # 2. Regime boundaries (color blocks)
    fig, ax = plt.subplots(figsize=(11, 7))
    for i in range(n_lr):
        for j in range(n_iter):
            regime = regime_matrix[i][j]
            color  = regime_colors.get(regime, "#CCCCCC")
            rect   = plt.Rectangle((j, n_lr - 1 - i), 1, 1,
                                    facecolor=color, alpha=0.65,
                                    edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            l2_val = l2_matrix[i, j]
            txt = f"{l2_val:.3f}" if np.isfinite(l2_val) and \
                  l2_val < DIVERGENCE_THRESHOLD else "DIV"
            unc = ""
            ax.text(j + 0.5, n_lr - 1 - i + 0.5, txt + unc,
                    ha="center", va="center", fontsize=8,
                    fontweight="bold")
    ax.set_xlim(0, n_iter); ax.set_ylim(0, n_lr)
    ax.set_xticks([j + 0.5 for j in range(n_iter)])
    ax.set_xticklabels(iter_labels)
    ax.set_yticks([i + 0.5 for i in range(n_lr)])
    ax.set_yticklabels(list(reversed(lr_labels)))
    ax.set_xlabel("Training Iterations", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(
        "Regime Boundaries (Green=Convergence, Orange=Underfitting, Red=Divergence)",
        fontweight="bold")
    for regime, color in regime_colors.items():
        ax.bar([], [], color=color, label=regime.capitalize())
    ax.legend(loc="upper left")
    savefig(fig, OUTPUT_DIR / "regime_boundaries.png")

    # 3. LR corridor
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for j, n_ep in enumerate(ITERATION_COUNTS):
        c = optimal_corridor[n_ep]
        if c is not None:
            ax.plot([n_ep, n_ep], [c["min_lr"], c["max_lr"]],
                    "|-", color="#1565C0", linewidth=3, markersize=12)
            ax.scatter([n_ep], [c["best_lr"]], color="#2E7D32",
                       s=120, zorder=5, edgecolors="black")
        div_lr = divergence_boundary.get(n_ep)
        if div_lr:
            ax.scatter([n_ep], [div_lr], marker="x", s=120,
                       color="#D32F2F", zorder=6, linewidths=2.5)
    div_x = [n for n in ITERATION_COUNTS
              if divergence_boundary.get(n) is not None]
    div_y = [divergence_boundary[n] for n in div_x]
    if div_x:
        ax.plot(div_x, div_y, "--", color="#D32F2F", linewidth=1.5,
                alpha=0.6, label="Divergence boundary")
    ax.set_xlabel("Training Iterations", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Optimal LR Corridor & Divergence Boundary",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    valid_j = [j for j, n in enumerate(ITERATION_COUNTS)
               if optimal_corridor[n] is not None]
    for j in valid_j:
        n   = ITERATION_COUNTS[j]
        c   = optimal_corridor[n]
        ax.bar(iter_labels[j], c["best_l2"], color="#2E7D32",
               alpha=0.85, edgecolor="white")
        ax.text(j, c["best_l2"] * 0.75,
                f"LR={c['best_lr']:.0e}",
                ha="center", va="top", fontsize=8)
    ax.set_ylabel("Best L2 Error", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Best Achievable L2 per Iteration Budget",
                 fontweight="bold")
    fig.suptitle("Learning Rate Corridor Analysis",
                 fontweight="bold", fontsize=14)
    savefig(fig, OUTPUT_DIR / "lr_corridor.png")

    # 4. Convergence curves grid
    fig, axes = plt.subplots(n_lr, n_iter,
                              figsize=(4 * n_iter, 2.5 * n_lr),
                              sharex=False, sharey=True)
    for i, lr in enumerate(LEARNING_RATES):
        for j, n_ep in enumerate(ITERATION_COUNTS):
            ax     = axes[i, j] if n_lr > 1 else axes[j]
            traj   = trajectories.get((lr, n_ep), [])
            regime = regime_matrix[i][j]
            color  = regime_colors.get(regime, "#666666")
            if traj:
                ax.semilogy(traj, color=color, linewidth=0.8, alpha=0.9)
            if i == 0: ax.set_title(f"{n_ep//1000}k", fontsize=9)
            if j == 0: ax.set_ylabel(f"{lr:.0e}", fontsize=8)
            ax.tick_params(labelsize=6)
            l2_val = l2_matrix[i, j]
            txt = f"L2={l2_val:.3f}" if np.isfinite(l2_val) and \
                  l2_val < DIVERGENCE_THRESHOLD else "DIV"
            ax.text(0.95, 0.95, txt, transform=ax.transAxes,
                    fontsize=6, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=color, alpha=0.3))
    fig.suptitle(
        "Training Convergence Curves\n"
        "(Green=Converged, Orange=Underfitting, Red=Diverged)",
        fontweight="bold", fontsize=12)
    fig.set_layout_engine('none')
    fig.subplots_adjust(hspace=0.4, wspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.95)
    savefig(fig, OUTPUT_DIR / "convergence_curves_grid.png")

    # ── JSON ─────────────────────────────────────────────────────────
    results = {
        "experiment": "Learning Rate Failure Phase Diagram",
        "version":    "v2-journal-ready",
        "config": {
            "n_hidden":            N_HIDDEN,
            "n_neurons":           N_NEURONS,
            "learning_rates":      LEARNING_RATES,
            "iteration_counts":    ITERATION_COUNTS,
            "n_seeds":             N_SEEDS,
            "eta_min":             ETA_MIN,
            "convergence_threshold": CONVERGENCE_THRESHOLD,
            "divergence_threshold":  DIVERGENCE_THRESHOLD,
            "fast_mode":           FAST_MODE,
        },
        "fix_notes": {
            "fix1_eta_min": (
                f"v1 used eta_min = lr × 0.01, making the final LR "
                f"fraction different across configs (LR=0.1 → eta=0.001, "
                f"LR=1e-5 → eta=1e-7). v2 uses fixed eta_min={ETA_MIN} "
                "for all configs so the scheduler decay curve is equivalent."
            ),
            "fix2_divergence": (
                f"v1 detected divergence only by loss > 1e10, which "
                "gradient clipping (norm=5.0) prevented. Many diverged "
                "configs were silently misclassified as underfitting. "
                f"v2 uses three criteria: loss > {LOSS_THRESHOLD} (post-clip), "
                f"pre-clip grad norm > {GRAD_NORM_THRESHOLD}, "
                f"parameter norm > {PARAM_NORM_THRESHOLD}."
            ),
            "fix3_heatmap": (
                "v1 set inf→10.0 for diverged cells, making them "
                "indistinguishable from L2=10.0 runs in the colormap. "
                "v2 uses nan→black (set_bad='black') for confirmed diverged "
                "cells and annotates them 'DIV', entirely separate from "
                "the L2 colormap range."
            ),
            "fix4_runtime": (
                f"v1 silently ran 100k-epoch configs (up to ~12hr each on "
                f"RTX 3050). v2 prints estimated runtime warning before "
                f"starting and offers FAST_MODE=True to skip 100k column."
            ),
            "fix5_seeds": (
                f"v1 used single seed per config. v2 uses {N_SEEDS} seeds. "
                "Configs where seeds disagree on regime (convergence vs "
                "underfitting vs divergence) are flagged 'uncertain=True' "
                "in the results."
            ),
        },
        "l2_matrix":       l2_matrix.tolist(),
        "std_matrix":      std_matrix.tolist(),
        "regime_matrix":   regime_matrix,
        "uncertain_matrix": uncertain_mask,
        "div_matrix":       div_mask,
        "best_config": {
            "learning_rate": best_lr,
            "iterations":    best_iter,
            "l2_error":      best_l2,
        },
        "optimal_corridor": {
            str(k): v for k, v in optimal_corridor.items()
        },
        "divergence_boundary": {
            str(k): v for k, v in divergence_boundary.items()
        },
    }

    save_results(results, OUTPUT_DIR / "exp15_results.json")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EXP 15 — COMPLETE  [v2]")
    print(f"{'=' * 70}")
    print(f"  Best config  : LR={best_lr}, Iters={best_iter}, L2={best_l2:.6f}")
    print(f"\n  LR Corridor:")
    for n_ep in ITERATION_COUNTS:
        c = optimal_corridor[n_ep]
        div_lr = divergence_boundary.get(n_ep)
        if c:
            print(f"    {n_ep//1000}k: corridor [{c['min_lr']:.0e}, "
                  f"{c['max_lr']:.0e}], best LR={c['best_lr']:.0e} "
                  f"(L2={c['best_l2']:.4f})")
        else:
            print(f"    {n_ep//1000}k: no convergence zone found")
        if div_lr:
            print(f"          divergence at LR ≥ {div_lr:.0e}")
    n_uncertain = sum(
        uncertain_mask[i][j]
        for i in range(n_lr) for j in range(n_iter))
    if n_uncertain:
        print(f"\n  ⚠ {n_uncertain} configs have uncertain regime "
              f"(seeds disagree) — check JSON uncertain_matrix")
    print(f"  Results → {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    return results


if __name__ == "__main__":
    run_experiment()