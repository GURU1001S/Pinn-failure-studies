"""
specialexp5_sensor_noise.py — Experiment 23: Industrial Robustness
[Sensor Noise Shatters the PINN Loss Landscape]

Tests whether Gaussian noise on IC/BC targets acts as a regularizer
(standard DL claim) or actively accelerates gradient conflict in PINNs.

PDE:  u_t + 30·u_x = 0,  x ∈ [0, 2π], t ∈ [0, 1]
IC:   u(x, 0) = sin(x)  + noise
BC:   u(0, t) = u(2π, t)  (periodic, soft constraint + noise)
Exact: u(x, t) = sin(x − 30t)

Noise levels:  0% (control), 1%, 5%, 10%
Architecture:  4 hidden layers, 64 neurons, tanh
Training:      Adam lr=1e-3, 40,000 epochs

Critical diagnostic (every 500 epochs):
  - Cosine similarity between ∇_θ L_pde and ∇_θ L_ic
  - Pathology onset: first epoch where cos_sim < -0.90

Outputs (results/exp23/):
  - exp23_noise_results.json
  - gradient_conflict_noise.png      (2×2 grid of cos_sim trajectories)
  - noise_onset_acceleration.png     (noise% vs pathology onset epoch)
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
from pathlib import Path

# ===================================================================
# Speed flags
# ===================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp23] Device: {DEVICE}")

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp23"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Configuration
# ===================================================================
BETA        = 30
SEED        = 42
N_HIDDEN    = 4
N_NEURONS   = 64
N_EPOCHS    = 40_000
LR          = 1e-3
LR_MIN      = 1e-5
T_MAX       = 1.0
X_MAX       = 2 * np.pi

N_PDE       = 5_000
N_IC        = 500
N_BC        = 300

TRACK_EVERY = 500           # gradient conflict recording interval
PATHOLOGY_THRESHOLD = -0.90 # cosine similarity threshold for "severe conflict"
CHECKPOINT_EVERY    = 5000

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]   # fraction of signal amplitude


# ===================================================================
# JSON serializer
# ===================================================================
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


# ===================================================================
# Model
# ===================================================================
class AdvectionPINN(nn.Module):
    def __init__(self, n_hidden=4, n_neurons=64):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
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


# ===================================================================
# PDE residual
# ===================================================================
def pde_residual(model, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u  = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return u_t + BETA * u_x


# ===================================================================
# Sampling (with noise injection on IC/BC targets)
# ===================================================================
def sample_points(noise_std, rng):
    """
    Returns (x_pde, t_pde), (x_ic, t_ic, u_ic_noisy), (x_bc_lo, x_bc_hi, t_bc).
    noise_std: absolute standard deviation of Gaussian noise.
    """
    # PDE interior
    x_pde = torch.rand(N_PDE, 1, dtype=DTYPE, device=DEVICE) * X_MAX
    t_pde = torch.rand(N_PDE, 1, dtype=DTYPE, device=DEVICE) * T_MAX

    # IC: u(x, 0) = sin(x) + noise
    x_ic = torch.rand(N_IC, 1, dtype=DTYPE, device=DEVICE) * X_MAX
    t_ic = torch.zeros(N_IC, 1, dtype=DTYPE, device=DEVICE)
    u_ic_clean = torch.sin(x_ic)
    if noise_std > 0:
        noise = torch.tensor(
            rng.normal(0, noise_std, size=(N_IC, 1)),
            dtype=DTYPE, device=DEVICE
        )
        u_ic = u_ic_clean + noise
    else:
        u_ic = u_ic_clean

    # BC: periodic, u(0,t) = u(2π,t)  — target difference = 0 + noise
    t_bc   = torch.rand(N_BC, 1, dtype=DTYPE, device=DEVICE) * T_MAX
    x_bc_lo = torch.zeros(N_BC, 1, dtype=DTYPE, device=DEVICE)
    x_bc_hi = torch.full((N_BC, 1), X_MAX, dtype=DTYPE, device=DEVICE)

    return (x_pde, t_pde), (x_ic, t_ic, u_ic), (x_bc_lo, x_bc_hi, t_bc)


# ===================================================================
# Gradient extraction helpers
# ===================================================================
def get_grad_vector(model, loss_fn_result):
    """Backprop a scalar loss and return the flat gradient vector."""
    model.zero_grad()
    loss_fn_result.backward(retain_graph=False)
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads)


def cosine_similarity(g1, g2):
    """Cosine similarity between two flat gradient vectors."""
    dot  = (g1 * g2).sum()
    norm = g1.norm() * g2.norm() + 1e-30
    return float((dot / norm).item())


# ===================================================================
# L2 evaluation
# ===================================================================
def compute_l2_error(model, nx=200, nt=200):
    model.eval()
    x_vals = np.linspace(0, X_MAX, nx)
    t_vals = np.linspace(0, T_MAX, nt)
    XX, TT = np.meshgrid(x_vals, t_vals, indexing="ij")

    u_exact = np.sin(XX - BETA * TT)

    with torch.no_grad():
        x_f = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        t_f = torch.tensor(TT.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u_pred = model(x_f, t_f).cpu().numpy().reshape(XX.shape)

    model.train()
    return float(np.linalg.norm(u_pred - u_exact) /
                 (np.linalg.norm(u_exact) + 1e-10))


# ===================================================================
# Training loop for a single noise level
# ===================================================================
def train_with_noise(noise_fraction, ckpt, ckpt_path):
    """
    Train with a given noise level.
    Returns dict with l2, losses, cos_sim history, pathology onset.
    """
    noise_key = f"noise_{noise_fraction:.2f}"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # Signal amplitude for sin(x) is 1.0
    noise_std = noise_fraction * 1.0
    noise_pct = int(noise_fraction * 100)

    print(f"\n{'━' * 60}")
    print(f"  Noise = {noise_pct}%  (σ = {noise_std:.4f})")
    print(f"{'━' * 60}")

    model     = AdvectionPINN(N_HIDDEN, N_NEURONS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)

    # Sample fixed collocation (noise is baked into IC targets)
    (x_pde, t_pde), (x_ic, t_ic, u_ic), (x_bc_lo, x_bc_hi, t_bc) = \
        sample_points(noise_std, rng)

    losses       = []
    cos_sim_hist = []
    cos_sim_epochs = []
    pathology_onset = None

    start_epoch = 1
    model_path  = OUTPUT_DIR / f"model_noise_{noise_pct}.pt"

    # Resume from checkpoint
    if noise_key in ckpt and "completed_epoch" in ckpt[noise_key]:
        start_epoch = ckpt[noise_key]["completed_epoch"] + 1
        losses       = ckpt[noise_key].get("losses", [])
        cos_sim_hist = ckpt[noise_key].get("cos_sim_hist", [])
        cos_sim_epochs = ckpt[noise_key].get("cos_sim_epochs", [])
        pathology_onset = ckpt[noise_key].get("pathology_onset", None)

        if model_path.exists():
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"    [Resuming from epoch {start_epoch - 1}]")

        for _ in range(start_epoch - 1):
            scheduler.step()

        if start_epoch > N_EPOCHS:
            l2 = compute_l2_error(model)
            return {
                "l2_error": l2,
                "losses": losses,
                "cos_sim_hist": cos_sim_hist,
                "cos_sim_epochs": cos_sim_epochs,
                "pathology_onset": pathology_onset,
                "final_loss": losses[-1] if losses else None,
            }

    t0 = time.time()

    for epoch in range(start_epoch, N_EPOCHS + 1):
        model.train()

        # ── Gradient conflict measurement ──────────────────────
        if epoch % TRACK_EVERY == 0:
            # PDE gradient
            model.zero_grad()
            res = pde_residual(model, x_pde, t_pde)
            l_pde = (res ** 2).mean()
            l_pde.backward()
            g_pde = torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                               for p in model.parameters()]).clone()

            # IC gradient
            model.zero_grad()
            u_ic_pred = model(x_ic, t_ic)
            l_ic = ((u_ic_pred - u_ic) ** 2).mean()
            l_ic.backward()
            g_ic = torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                              for p in model.parameters()]).clone()

            cs = cosine_similarity(g_pde, g_ic)
            cos_sim_hist.append(cs)
            cos_sim_epochs.append(epoch)

            if pathology_onset is None and cs < PATHOLOGY_THRESHOLD:
                pathology_onset = epoch

        # ── Standard training step ─────────────────────────────
        model.zero_grad()
        optimizer.zero_grad()

        res   = pde_residual(model, x_pde, t_pde)
        l_pde = (res ** 2).mean()

        u_ic_pred = model(x_ic, t_ic)
        l_ic = ((u_ic_pred - u_ic) ** 2).mean()

        l_bc = ((model(x_bc_lo, t_bc) - model(x_bc_hi, t_bc)) ** 2).mean()

        total = l_pde + 100.0 * l_ic + 10.0 * l_bc
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(float(total.item()))

        if epoch % 5000 == 0 or epoch == N_EPOCHS:
            print(f"    Epoch {epoch:>6d}: loss={total.item():.4e}  "
                  f"cos_sim={cos_sim_hist[-1]:.4f}" if cos_sim_hist else
                  f"    Epoch {epoch:>6d}: loss={total.item():.4e}")

        # Periodic checkpoint
        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS:
            torch.save(model.state_dict(), model_path)
            if noise_key not in ckpt:
                ckpt[noise_key] = {}
            ckpt[noise_key]["completed_epoch"] = epoch
            ckpt[noise_key]["losses"] = losses
            ckpt[noise_key]["cos_sim_hist"] = cos_sim_hist
            ckpt[noise_key]["cos_sim_epochs"] = cos_sim_epochs
            ckpt[noise_key]["pathology_onset"] = pathology_onset
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, default=_default)

    elapsed = time.time() - t0
    l2 = compute_l2_error(model)

    print(f"    L2 error        = {l2:.6f}")
    print(f"    Pathology onset = {pathology_onset}")
    print(f"    Time            = {elapsed:.1f}s")

    return {
        "l2_error":         l2,
        "losses":           losses,
        "cos_sim_hist":     cos_sim_hist,
        "cos_sim_epochs":   cos_sim_epochs,
        "pathology_onset":  pathology_onset,
        "final_loss":       losses[-1] if losses else None,
        "elapsed_s":        elapsed,
    }


# ===================================================================
# Plotting
# ===================================================================
def plot_gradient_conflict_grid(all_results, filepath):
    """2×2 grid of cos_sim trajectories for each noise level."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(
        f"PDE↔IC Gradient Conflict by Noise Level  (β={BETA})\n"
        f"Cosine Similarity of ∇_θ L_pde and ∇_θ L_ic",
        fontsize=13, fontweight="bold")

    colors = ["#3A7FD5", "#4CAF50", "#F5A623", "#E64040"]

    for idx, (noise_frac, color) in enumerate(zip(NOISE_LEVELS, colors)):
        ax = axes[idx // 2][idx % 2]
        r  = all_results[noise_frac]
        epochs = r["cos_sim_epochs"]
        cs     = r["cos_sim_hist"]
        pct    = int(noise_frac * 100)

        ax.plot(epochs, cs, color=color, lw=1.5, alpha=0.9)
        ax.axhline(0, color="grey", ls="-", lw=0.5)
        ax.axhline(PATHOLOGY_THRESHOLD, color="#E64040", ls="--", lw=1,
                   label=f"Threshold = {PATHOLOGY_THRESHOLD}")

        onset = r["pathology_onset"]
        if onset is not None:
            ax.axvline(onset, color="#E64040", ls=":", lw=1.2, alpha=0.7)
            ax.text(onset, PATHOLOGY_THRESHOLD + 0.05,
                    f" onset={onset}", fontsize=8, color="#E64040",
                    va="bottom")

        ax.set_title(f"Noise = {pct}%  |  L2 = {r['l2_error']:.4f}  |  "
                     f"Onset = {onset if onset else 'never'}",
                     fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("cos(∇L_pde, ∇L_ic)", fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_onset_acceleration(all_results, filepath):
    """Scatter: noise % vs pathology onset epoch."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    noise_pcts = []
    onsets     = []
    colors_pts = []
    color_map  = {0: "#3A7FD5", 1: "#4CAF50", 5: "#F5A623", 10: "#E64040"}

    for noise_frac in NOISE_LEVELS:
        pct   = int(noise_frac * 100)
        onset = all_results[noise_frac]["pathology_onset"]
        noise_pcts.append(pct)
        onsets.append(onset if onset is not None else N_EPOCHS + 5000)
        colors_pts.append(color_map.get(pct, "#999999"))

    ax.scatter(noise_pcts, onsets, s=120, c=colors_pts,
               edgecolors="black", linewidth=1, zorder=3)

    # Annotate
    for pct, onset, color in zip(noise_pcts, onsets, colors_pts):
        label = str(onset) if onset <= N_EPOCHS else "Never"
        ax.annotate(label, (pct, onset), textcoords="offset points",
                    xytext=(10, 5), fontsize=9, fontweight="bold",
                    color=color)

    ax.axhline(N_EPOCHS, color="grey", ls=":", lw=1,
               label=f"Max epochs ({N_EPOCHS:,})")
    ax.set_xlabel("Noise Level (%)", fontsize=12)
    ax.set_ylabel("Pathology Onset Epoch", fontsize=12)
    ax.set_title(
        f"Noise Accelerates Gradient Pathology  (β={BETA})\n"
        f"Onset = first epoch where cos(∇L_pde, ∇L_ic) < {PATHOLOGY_THRESHOLD}",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-1, 12)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ===================================================================
# Main experiment
# ===================================================================
def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 23: Industrial Robustness — Sensor Noise")
    print(f"  β={BETA}   Noise levels: {[f'{n*100:.0f}%' for n in NOISE_LEVELS]}")
    print(f"  Device : {DEVICE}")
    print(f"  Seed   : {SEED}")
    print(f"  Epochs : {N_EPOCHS:,}")
    print(f"  Conflict threshold: cos_sim < {PATHOLOGY_THRESHOLD}")
    print("=" * 70)

    # Checkpoint setup
    ckpt_path = OUTPUT_DIR / "exp23_checkpoint.json"
    ckpt = {}
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r") as f:
                ckpt = json.load(f)
            print(f"  [Loaded checkpoint from {ckpt_path}]")
        except Exception:
            ckpt = {}

    t0 = time.time()
    all_results = {}

    for noise_frac in NOISE_LEVELS:
        result = train_with_noise(noise_frac, ckpt, ckpt_path)
        all_results[noise_frac] = result

    total_elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 23 — SUMMARY")
    print(f"{'=' * 70}")
    for noise_frac in NOISE_LEVELS:
        r = all_results[noise_frac]
        pct = int(noise_frac * 100)
        onset_str = str(r["pathology_onset"]) if r["pathology_onset"] else "Never"
        print(f"  Noise={pct:>3d}%:  L2={r['l2_error']:.6f}  |  "
              f"Onset={onset_str}  |  FinalLoss={r['final_loss']:.4e}")

    # ── Plots ─────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_gradient_conflict_grid(
        all_results, OUTPUT_DIR / "gradient_conflict_noise.png")
    plot_onset_acceleration(
        all_results, OUTPUT_DIR / "noise_onset_acceleration.png")

    # ── JSON ──────────────────────────────────────────────────
    json_results = {
        "experiment": "Industrial Robustness — Sensor Noise",
        "version":    "v1",
        "hypothesis": (
            "In standard deep learning, noise on training targets acts as "
            "implicit regularization (label smoothing). In PINNs, noise on "
            "IC/BC targets actively shatters the loss landscape by creating "
            "an irreconcilable conflict: the PDE loss demands exact physics "
            "while the corrupted IC/BC targets demand fitting to noise. "
            "This accelerates the onset of gradient pathology."
        ),
        "config": {
            "beta":          BETA,
            "seed":          SEED,
            "n_hidden":      N_HIDDEN,
            "n_neurons":     N_NEURONS,
            "n_epochs":      N_EPOCHS,
            "lr":            LR,
            "noise_levels":  NOISE_LEVELS,
            "track_every":   TRACK_EVERY,
            "pathology_threshold": PATHOLOGY_THRESHOLD,
        },
        "per_noise_level": {
            f"{int(nf*100)}%": {
                "l2_error":        r["l2_error"],
                "pathology_onset": r["pathology_onset"],
                "final_loss":      r["final_loss"],
                "elapsed_s":       r.get("elapsed_s", None),
            } for nf, r in all_results.items()
        },
        "total_elapsed_s": total_elapsed,
    }

    out_json = OUTPUT_DIR / "exp23_noise_results.json"
    with open(out_json, "w") as f:
        json.dump(json_results, f, indent=2, default=_default)
    print(f"\nResults → {out_json}")
    print(f"Plots   → {OUTPUT_DIR}")

    return json_results


if __name__ == "__main__":
    run_experiment()
