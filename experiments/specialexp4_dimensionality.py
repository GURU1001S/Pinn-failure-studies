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
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp22] Device: {DEVICE}")
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp22"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BETA       = 30
SEED       = 42
N_HIDDEN   = 4
N_NEURONS  = 64
ACTIVATION = "tanh"
N_EPOCHS   = 30_000
LR         = 1e-3
LR_MIN     = 1e-5
T_MAX      = 1.0
X_MAX      = 2 * np.pi
FAILURE_THRESHOLD = 0.1
DIM_CONFIGS = {
    1: {"n_pde": 2_000,  "n_ic": 500,   "n_bc": 200,  "n_eval": 200},
    2: {"n_pde": 10_000, "n_ic": 2_000, "n_bc": 500,  "n_eval": 80},
    3: {"n_pde": 50_000, "n_ic": 10_000, "n_bc": 1_000, "n_eval": 30},
}
CHECKPOINT_EVERY = 5000
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
class AdvectionPINN(nn.Module):
    def __init__(self, in_dim, n_hidden=4, n_neurons=64):
        super().__init__()
        layers = [nn.Linear(in_dim, n_neurons), nn.Tanh()]
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
    def forward(self, coords):
        return self.net(coords)
def exact_solution_np(coords_np, spatial_dim):
    t = coords_np[:, -1]
    u = np.ones(len(t))
    for d in range(spatial_dim):
        u *= np.sin(coords_np[:, d] - BETA * t)
    return u
def pde_residual(model, coords, spatial_dim):
    coords = coords.requires_grad_(True)
    u = model(coords)
    grad_u = torch.autograd.grad(
        u, coords, torch.ones_like(u), create_graph=True
    )[0]
    u_t = grad_u[:, -1:]
    u_spatial = grad_u[:, :spatial_dim].sum(dim=1, keepdim=True)
    return u_t + BETA * u_spatial
def sample_pde_points(n, spatial_dim):
    coords = torch.zeros(n, spatial_dim + 1, dtype=DTYPE, device=DEVICE)
    for d in range(spatial_dim):
        coords[:, d] = torch.rand(n, dtype=DTYPE, device=DEVICE) * X_MAX
    coords[:, -1] = torch.rand(n, dtype=DTYPE, device=DEVICE) * T_MAX
    return coords
def sample_ic_points(n, spatial_dim):
    coords = torch.zeros(n, spatial_dim + 1, dtype=DTYPE, device=DEVICE)
    for d in range(spatial_dim):
        coords[:, d] = torch.rand(n, dtype=DTYPE, device=DEVICE) * X_MAX
    u_ic = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
    for d in range(spatial_dim):
        u_ic *= torch.sin(coords[:, d:d+1])
    return coords, u_ic
def sample_bc_points(n_per_face, spatial_dim):
    all_coords_lo = []
    all_coords_hi = []
    n = max(n_per_face // spatial_dim, 10)
    for d in range(spatial_dim):
        base = torch.zeros(n, spatial_dim + 1, dtype=DTYPE, device=DEVICE)
        for dd in range(spatial_dim):
            if dd != d:
                base[:, dd] = torch.rand(n, dtype=DTYPE, device=DEVICE) * X_MAX
        base[:, -1] = torch.rand(n, dtype=DTYPE, device=DEVICE) * T_MAX
        lo = base.clone()
        lo[:, d] = 0.0
        hi = base.clone()
        hi[:, d] = X_MAX
        all_coords_lo.append(lo)
        all_coords_hi.append(hi)
    return torch.cat(all_coords_lo, dim=0), torch.cat(all_coords_hi, dim=0)
def compute_loss(model, spatial_dim, n_pde, n_ic, n_bc):
    coords_pde = sample_pde_points(n_pde, spatial_dim)
    res = pde_residual(model, coords_pde, spatial_dim)
    loss_pde = (res ** 2).mean()
    coords_ic, u_ic_true = sample_ic_points(n_ic, spatial_dim)
    u_ic_pred = model(coords_ic)
    loss_ic = ((u_ic_pred - u_ic_true) ** 2).mean()
    coords_lo, coords_hi = sample_bc_points(n_bc, spatial_dim)
    loss_bc = ((model(coords_lo) - model(coords_hi)) ** 2).mean()
    return loss_pde + 100.0 * loss_ic + 10.0 * loss_bc
def compute_l2_error(model, spatial_dim, n_per_axis):
    model.eval()
    axes = [np.linspace(0, X_MAX, n_per_axis, endpoint=False)
            for _ in range(spatial_dim)]
    axes.append(np.linspace(0, T_MAX, n_per_axis))
    grids = np.meshgrid(*axes, indexing="ij")
    coords_np = np.stack([g.ravel() for g in grids], axis=1)
    u_exact = exact_solution_np(coords_np, spatial_dim)
    batch_size = 100_000
    u_preds = []
    with torch.no_grad():
        for i in range(0, len(coords_np), batch_size):
            batch = torch.tensor(
                coords_np[i:i+batch_size], dtype=DTYPE, device=DEVICE
            )
            u_preds.append(model(batch).cpu().numpy().ravel())
    u_pred = np.concatenate(u_preds)
    model.train()
    l2_err = np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10)
    return float(l2_err)
def train_dimension(spatial_dim, ckpt, ckpt_path):
    dim_key = f"dim_{spatial_dim}d"
    cfg = DIM_CONFIGS[spatial_dim]
    in_dim = spatial_dim + 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"\n{'━' * 60}")
    print(f"  {spatial_dim}D Advection  |  in_dim={in_dim}  |  "
          f"N_pde={cfg['n_pde']:,}  N_ic={cfg['n_ic']:,}")
    print(f"{'━' * 60}")
    model     = AdvectionPINN(in_dim, N_HIDDEN, N_NEURONS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)
    losses   = []
    start_epoch = 1
    model_path = OUTPUT_DIR / f"model_{spatial_dim}d.pt"
    if dim_key in ckpt and "completed_epoch" in ckpt[dim_key]:
        start_epoch = ckpt[dim_key]["completed_epoch"] + 1
        losses = ckpt[dim_key].get("losses", [])
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"    [Resuming {spatial_dim}D from epoch {start_epoch - 1}]")
        for _ in range(start_epoch - 1):
            scheduler.step()
        if start_epoch > N_EPOCHS:
            l2 = compute_l2_error(model, spatial_dim, cfg["n_eval"])
            elapsed = ckpt[dim_key].get("elapsed", 0.0)
            peak_mem = ckpt[dim_key].get("peak_mem_mb", 0.0)
            print(f"    Already complete. L2 = {l2:.6f}")
            return l2, losses, elapsed, peak_mem
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = compute_loss(
            model, spatial_dim, cfg["n_pde"], cfg["n_ic"], cfg["n_bc"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_val = float(loss.item())
        losses.append(loss_val)
        if epoch % 5000 == 0 or epoch == N_EPOCHS:
            print(f"    Epoch {epoch:>6d}: loss = {loss_val:.4e}")
        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS:
            torch.save(model.state_dict(), model_path)
            elapsed_so_far = time.time() - t0
            if dim_key not in ckpt:
                ckpt[dim_key] = {}
            ckpt[dim_key]["completed_epoch"] = epoch
            ckpt[dim_key]["losses"] = losses
            ckpt[dim_key]["elapsed"] = elapsed_so_far
            if torch.cuda.is_available():
                ckpt[dim_key]["peak_mem_mb"] = float(
                    torch.cuda.max_memory_allocated() / 1024**2
                )
            else:
                ckpt[dim_key]["peak_mem_mb"] = 0.0
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, default=_default)
    elapsed = time.time() - t0
    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = float(torch.cuda.max_memory_allocated() / 1024**2)
    l2 = compute_l2_error(model, spatial_dim, cfg["n_eval"])
    print(f"    L2 relative error = {l2:.6f}")
    print(f"    Training time     = {elapsed:.1f}s")
    print(f"    Peak GPU memory   = {peak_mem:.1f} MB")
    return l2, losses, elapsed, peak_mem
def plot_dimensionality_cliff(results_per_dim, filepath):
    dims   = [1, 2, 3]
    l2s    = [results_per_dim[d]["l2_error"] for d in dims]
    colors = ["#3A7FD5", "#F5A623", "#E64040"]
    labels = ["1D", "2D", "3D"]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    bars = ax.bar(labels, l2s, color=colors, width=0.5, edgecolor="black",
                  linewidth=0.8, zorder=3)
    for bar, l2 in zip(bars, l2s):
        ypos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, ypos * 1.05,
                f"{l2:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.axhline(FAILURE_THRESHOLD, color="#E64040", ls="--", lw=1.5,
               label=f"Failure Threshold (L2 = {FAILURE_THRESHOLD})", zorder=2)
    ax.set_xlabel("Spatial Dimensionality", fontsize=12)
    ax.set_ylabel("L2 Relative Error", fontsize=12)
    ax.set_title(f"Curse of Spectral Dimensionality  (β={BETA})\n"
                 f"Advection Equation: 1D → 2D → 3D",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(l2s) * 1.4)
    ax.grid(True, alpha=0.25, axis="y")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
def plot_loss_trajectories(results_per_dim, filepath):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = {"1D": "#3A7FD5", "2D": "#F5A623", "3D": "#E64040"}
    for dim in [1, 2, 3]:
        losses = results_per_dim[dim]["losses"]
        label  = f"{dim}D  (final loss = {losses[-1]:.3e})"
        window = min(100, len(losses) // 10)
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(losses, kernel, mode="valid")
            epochs = np.arange(window, len(losses) + 1)
        else:
            smoothed = losses
            epochs = np.arange(1, len(losses) + 1)
        ax.semilogy(epochs, smoothed, color=colors[f"{dim}D"],
                    lw=1.5, label=label, alpha=0.9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Loss (log scale)", fontsize=12)
    ax.set_title(f"Training Loss Trajectories by Dimensionality  (β={BETA})\n"
                 f"Architecture: {N_HIDDEN}×{N_NEURONS} tanh, Adam lr={LR}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, which="both")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 22: Dimensionality Cross-Validation")
    print(f"  Curse of Spectral Dimensionality  (β={BETA})")
    print(f"  Device  : {DEVICE}")
    print(f"  Seed    : {SEED}")
    print(f"  Epochs  : {N_EPOCHS:,}")
    print(f"  Dims    : 1D, 2D, 3D")
    print("=" * 70)
    ckpt_path = OUTPUT_DIR / "exp22_checkpoint.json"
    ckpt = {}
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r") as f:
                ckpt = json.load(f)
            print(f"  [Loaded checkpoint from {ckpt_path}]")
        except Exception:
            ckpt = {}
    t0_total = time.time()
    results_per_dim = {}
    for spatial_dim in [1, 2, 3]:
        l2, losses, elapsed, peak_mem = train_dimension(
            spatial_dim, ckpt, ckpt_path
        )
        results_per_dim[spatial_dim] = {
            "l2_error":    l2,
            "losses":      losses,
            "elapsed_s":   elapsed,
            "peak_mem_mb": peak_mem,
            "failed":      l2 >= FAILURE_THRESHOLD,
        }
    total_elapsed = time.time() - t0_total
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 22 — SUMMARY")
    print(f"{'=' * 70}")
    for dim in [1, 2, 3]:
        r = results_per_dim[dim]
        status = "✗ FAIL" if r["failed"] else "✓ PASS"
        print(f"  {dim}D:  L2 = {r['l2_error']:.6f}  |  "
              f"Time = {r['elapsed_s']:.1f}s  |  "
              f"Mem = {r['peak_mem_mb']:.0f} MB  |  {status}")
    print(f"\nTotal time: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    print("\nGenerating plots...")
    plot_dimensionality_cliff(
        results_per_dim,
        OUTPUT_DIR / "dimensionality_cliff.png")
    plot_loss_trajectories(
        results_per_dim,
        OUTPUT_DIR / "loss_trajectories_dim.png")
    json_results = {
        "experiment": "Dimensionality Cross-Validation",
        "version":    "v1",
        "hypothesis": (
            "Spectral bias failure worsens with spatial dimensionality. "
            "Higher-dimensional domains amplify the frequency coverage "
            "challenge: the network must represent product-of-sines "
            "across d axes, each shifted by β·t, while the NTK kernel "
            "remains biased toward low-frequency modes."
        ),
        "config": {
            "beta":         BETA,
            "seed":         SEED,
            "n_hidden":     N_HIDDEN,
            "n_neurons":    N_NEURONS,
            "activation":   ACTIVATION,
            "n_epochs":     N_EPOCHS,
            "lr":           LR,
            "t_max":        T_MAX,
            "failure_threshold": FAILURE_THRESHOLD,
        },
        "collocation_counts": {
            f"{d}D": {
                "n_pde": DIM_CONFIGS[d]["n_pde"],
                "n_ic":  DIM_CONFIGS[d]["n_ic"],
                "n_bc":  DIM_CONFIGS[d]["n_bc"],
            } for d in [1, 2, 3]
        },
        "per_dimension": {
            f"{dim}D": {
                "l2_error":       r["l2_error"],
                "failed":         r["failed"],
                "elapsed_s":      r["elapsed_s"],
                "peak_mem_mb":    r["peak_mem_mb"],
                "final_loss":     r["losses"][-1] if r["losses"] else None,
            } for dim, r in results_per_dim.items()
        },
        "dimensionality_cliff_note": (
            "If L2 error increases monotonically from 1D → 2D → 3D "
            "while all other hyperparameters are held constant, this "
            "confirms the Curse of Spectral Dimensionality: the product "
            "structure of the exact solution requires spectral coverage "
            "that scales exponentially with dimension, while the fixed-"
            "width network's representational capacity is constant."
        ),
        "total_elapsed_s": total_elapsed,
    }
    out_json = OUTPUT_DIR / "exp22_results.json"
    with open(out_json, "w") as f:
        json.dump(json_results, f, indent=2, default=_default)
    print(f"\nResults → {out_json}")
    print(f"Plots   → {OUTPUT_DIR}")
    return json_results
if __name__ == "__main__":
    run_experiment()
