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
from scipy.io import loadmat
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp24] Device: {DEVICE}")
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp24"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
SEED       = 42
NU         = 0.01 / np.pi
N_HIDDEN   = 4
N_NEURONS  = 64
N_EPOCHS   = 30_000
LR         = 1e-3
LR_MIN     = 1e-5
N_INT      = 10_000
N_IC       = 300
N_BC       = 300
TRACK_EVERY     = 1000
MASS_EVAL_TIMES = [0.5, 0.9]
N_MASS_GRID     = 1000
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
class BurgersPINN(nn.Module):
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
def burgers_residual(model, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u  = model(x, t)
    u_t  = torch.autograd.grad(u, t, torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                create_graph=True)[0]
    return u_t + u * u_x - NU * u_xx
def sample_domain():
    x_int = (torch.rand(N_INT, 1, dtype=DTYPE, device=DEVICE) * 2 - 1)
    t_int = torch.rand(N_INT, 1, dtype=DTYPE, device=DEVICE)
    x_ic = (torch.rand(N_IC, 1, dtype=DTYPE, device=DEVICE) * 2 - 1)
    t_ic = torch.zeros(N_IC, 1, dtype=DTYPE, device=DEVICE)
    u_ic = -torch.sin(np.pi * x_ic)
    n_half = N_BC // 2
    t_bc   = torch.rand(n_half, 1, dtype=DTYPE, device=DEVICE)
    x_bc_l = torch.full((n_half, 1), -1.0, dtype=DTYPE, device=DEVICE)
    x_bc_r = torch.full((n_half, 1),  1.0, dtype=DTYPE, device=DEVICE)
    x_bc   = torch.cat([x_bc_l, x_bc_r], dim=0)
    t_bc   = torch.cat([t_bc, t_bc.clone()], dim=0)
    u_bc   = torch.zeros(n_half * 2, 1, dtype=DTYPE, device=DEVICE)
    return (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc)
def compute_mass_leakage(model, t_eval, n_grid=N_MASS_GRID):
    model.eval()
    x_grid = torch.linspace(-1, 1, n_grid, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    t_grid = torch.full((n_grid, 1), t_eval, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        u_pred = model(x_grid, t_grid).squeeze(1)
    dx = 2.0 / (n_grid - 1)
    integral = torch.trapezoid(u_pred, dx=dx)
    model.train()
    return float(abs(integral.item()))
def compute_l2_vs_reference(model, x_ref, t_ref, u_ref):
    nx, nt = len(x_ref), len(t_ref)
    X, T = np.meshgrid(x_ref, t_ref, indexing="ij")
    model.eval()
    with torch.no_grad():
        x_f = torch.tensor(X.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        t_f = torch.tensor(T.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u_pred = model(x_f, t_f).cpu().numpy().reshape(nx, nt)
    model.train()
    l2 = float(np.linalg.norm(u_pred - u_ref) /
               (np.linalg.norm(u_ref) + 1e-30))
    return l2, u_pred
def train_model(ckpt, ckpt_path):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mat_path = DATASET_DIR / "burgers_shock.mat"
    data = loadmat(str(mat_path))
    x_ref = data["x"].flatten()
    t_ref = data["t"].flatten()
    u_ref = data["usol"]
    print(f"  Reference solution loaded: {mat_path}")
    print(f"  Grid: {u_ref.shape[0]} x {u_ref.shape[1]}")
    model     = BurgersPINN(N_HIDDEN, N_NEURONS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = sample_domain()
    track_epochs = []
    l2_history   = []
    mass_leakage_history = {str(t_eval): [] for t_eval in MASS_EVAL_TIMES}
    losses = []
    start_epoch = 1
    model_path  = OUTPUT_DIR / "model_burgers.pt"
    if "training" in ckpt and "completed_epoch" in ckpt["training"]:
        start_epoch = ckpt["training"]["completed_epoch"] + 1
        losses       = ckpt["training"].get("losses", [])
        track_epochs = ckpt["training"].get("track_epochs", [])
        l2_history   = ckpt["training"].get("l2_history", [])
        mass_leakage_history = ckpt["training"].get("mass_leakage_history",
                                                     mass_leakage_history)
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"    [Resuming from epoch {start_epoch - 1}]")
        for _ in range(start_epoch - 1):
            scheduler.step()
        if start_epoch > N_EPOCHS:
            l2, u_pred_final = compute_l2_vs_reference(model, x_ref, t_ref, u_ref)
            return (model, l2, losses, track_epochs, l2_history,
                    mass_leakage_history, x_ref, t_ref, u_ref, u_pred_final)
    t0 = time.time()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        res = burgers_residual(model, x_int, t_int)
        l_pde = (res ** 2).mean()
        u_ic_pred = model(x_ic, t_ic)
        l_ic = ((u_ic_pred - u_ic) ** 2).mean()
        u_bc_pred = model(x_bc, t_bc)
        l_bc = ((u_bc_pred - u_bc) ** 2).mean()
        total = l_pde + 10.0 * l_ic + l_bc
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(total.item()))
        if epoch % TRACK_EVERY == 0 or epoch == N_EPOCHS:
            l2, _ = compute_l2_vs_reference(model, x_ref, t_ref, u_ref)
            l2_history.append(l2)
            track_epochs.append(epoch)
            for t_eval in MASS_EVAL_TIMES:
                leak = compute_mass_leakage(model, t_eval)
                mass_leakage_history[str(t_eval)].append(leak)
            print(f"    Epoch {epoch:>6d}: loss={total.item():.4e}  "
                  f"L2={l2:.6f}  "
                  f"leak@0.5={mass_leakage_history['0.5'][-1]:.6f}  "
                  f"leak@0.9={mass_leakage_history['0.9'][-1]:.6f}")
        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS:
            torch.save(model.state_dict(), model_path)
            if "training" not in ckpt:
                ckpt["training"] = {}
            ckpt["training"]["completed_epoch"] = epoch
            ckpt["training"]["losses"] = losses
            ckpt["training"]["track_epochs"] = track_epochs
            ckpt["training"]["l2_history"] = l2_history
            ckpt["training"]["mass_leakage_history"] = mass_leakage_history
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, default=_default)
    elapsed = time.time() - t0
    print(f"\n    Training done in {elapsed:.1f}s")
    l2_final, u_pred_final = compute_l2_vs_reference(model, x_ref, t_ref, u_ref)
    return (model, l2_final, losses, track_epochs, l2_history,
            mass_leakage_history, x_ref, t_ref, u_ref, u_pred_final)
def plot_silent_leakage(track_epochs, l2_history, mass_leakage_history,
                        filepath):
    fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    color_l2   = "#3A7FD5"
    color_leak = "#E64040"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Relative L2 Error", fontsize=12, color=color_l2)
    ax1.semilogy(track_epochs, l2_history, color=color_l2, lw=2,
                 marker="o", markersize=3, label="L2 Error")
    ax1.tick_params(axis="y", labelcolor=color_l2)
    ax1.grid(True, alpha=0.2)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mass Leakage |∫u dx|", fontsize=12, color=color_leak)
    for t_eval in MASS_EVAL_TIMES:
        leak_vals = mass_leakage_history[str(t_eval)]
        ls = "-" if t_eval == 0.9 else "--"
        ax2.plot(track_epochs, leak_vals, color=color_leak, lw=2,
                 ls=ls, marker="s", markersize=3,
                 label=f"Mass Leakage @ t={t_eval}")
    ax2.tick_params(axis="y", labelcolor=color_leak)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", fontsize=9)
    fig.suptitle(
        "Silent Conservation Leakage\n"
        "L2 Error Decreases While Mass Conservation Degrades",
        fontsize=13, fontweight="bold")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
def plot_mass_profile(model, x_ref, t_ref, u_ref, filepath,
                      t_snapshot=0.9):
    t_idx = np.argmin(np.abs(t_ref - t_snapshot))
    u_exact_slice = u_ref[:, t_idx]
    model.eval()
    x_dense = np.linspace(-1, 1, 500)
    t_dense = np.full_like(x_dense, t_ref[t_idx])
    with torch.no_grad():
        x_t = torch.tensor(x_dense, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        t_t = torch.tensor(t_dense, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u_pred_dense = model(x_t, t_t).cpu().numpy().ravel()
    u_exact_dense = np.interp(x_dense, x_ref, u_exact_slice)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.fill_between(x_dense, 0, u_pred_dense, alpha=0.2, color="#E64040",
                    label="PINN prediction area")
    ax.fill_between(x_dense, 0, u_exact_dense, alpha=0.15, color="#3A7FD5",
                    label="Exact solution area")
    ax.plot(x_dense, u_exact_dense, color="#3A7FD5", lw=2.5,
            label=f"Exact (t={t_ref[t_idx]:.3f})")
    ax.plot(x_dense, u_pred_dense, color="#E64040", lw=2, ls="--",
            label=f"PINN  (t={t_ref[t_idx]:.3f})")
    ax.axhline(0, color="black", lw=0.5)
    dx_dense = 2.0 / (len(x_dense) - 1)
    mass_pred  = float(np.trapezoid(u_pred_dense, dx=dx_dense))
    mass_exact = float(np.trapezoid(u_exact_dense, dx=dx_dense))
    ax.text(0.02, 0.95,
            f"∫u_exact dx = {mass_exact:+.6f}\n"
            f"∫u_PINN dx  = {mass_pred:+.6f}\n"
            f"Leakage     = {abs(mass_pred):.6f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="#FFF9C4",
                      edgecolor="#F9A825", alpha=0.9))
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("u(x, t)", fontsize=12)
    ax.set_title(
        f"Mass Profile Snapshot at t = {t_ref[t_idx]:.3f}\n"
        f"Asymmetric Leakage: PINN Violates ∫u dx ≈ 0",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 24: Silent Conservation Leakage")
    print(f"  Viscous Burgers  ν = {NU:.6f}")
    print(f"  Device : {DEVICE}")
    print(f"  Seed   : {SEED}")
    print(f"  Epochs : {N_EPOCHS:,}")
    print(f"  Mass eval times: {MASS_EVAL_TIMES}")
    print("=" * 70)
    ckpt_path = OUTPUT_DIR / "exp24_checkpoint.json"
    ckpt = {}
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r") as f:
                ckpt = json.load(f)
            print(f"  [Loaded checkpoint from {ckpt_path}]")
        except Exception:
            ckpt = {}
    (model, l2_final, losses, track_epochs, l2_history,
     mass_leakage_history, x_ref, t_ref, u_ref, u_pred_final) =        train_model(ckpt, ckpt_path)
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 24 — SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Final L2 error  = {l2_final:.6f}")
    for t_eval in MASS_EVAL_TIMES:
        leak_vals = mass_leakage_history[str(t_eval)]
        if leak_vals:
            print(f"  Mass leakage @ t={t_eval}: "
                  f"initial={leak_vals[0]:.6f}  "
                  f"final={leak_vals[-1]:.6f}")
    print("\nGenerating plots...")
    plot_silent_leakage(
        track_epochs, l2_history, mass_leakage_history,
        OUTPUT_DIR / "silent_leakage_divergence.png")
    plot_mass_profile(
        model, x_ref, t_ref, u_ref,
        OUTPUT_DIR / "mass_profile_snapshot.png",
        t_snapshot=0.9)
    json_results = {
        "experiment": "Silent Conservation Leakage",
        "version":    "v1",
        "hypothesis": (
            "A PINN can achieve a passing L2 error (< 0.1) while "
            "fundamentally violating the conservation laws of the "
            "system. For Viscous Burgers with antisymmetric IC and "
            "zero BCs, the total mass ∫u dx should remain ≈ 0 at "
            "all times. The PINN's learned solution develops an "
            "asymmetric bias that accumulates 'leaked' mass, proving "
            "that L2 convergence does not imply physical correctness."
        ),
        "config": {
            "nu":          NU,
            "seed":        SEED,
            "n_hidden":    N_HIDDEN,
            "n_neurons":   N_NEURONS,
            "n_epochs":    N_EPOCHS,
            "lr":          LR,
            "track_every": TRACK_EVERY,
            "mass_eval_times": MASS_EVAL_TIMES,
        },
        "final_l2_error": l2_final,
        "tracking": {
            "epochs":           track_epochs,
            "l2_errors":        l2_history,
            "mass_leakage":     mass_leakage_history,
        },
        "final_mass_leakage": {
            str(t_eval): mass_leakage_history[str(t_eval)][-1]
            if mass_leakage_history[str(t_eval)] else None
            for t_eval in MASS_EVAL_TIMES
        },
        "conservation_note": (
            "For the Viscous Burgers equation with u(x,0) = -sin(πx) "
            "and u(-1,t) = u(1,t) = 0, the antisymmetry guarantees "
            "∫u dx = 0 at all times. Any nonzero ∫u_pred dx represents "
            "a violation of the integral conservation law, even if the "
            "pointwise L2 error is small."
        ),
    }
    out_json = OUTPUT_DIR / "exp24_conservation_results.json"
    with open(out_json, "w") as f:
        json.dump(json_results, f, indent=2, default=_default)
    print(f"\nResults → {out_json}")
    print(f"Plots   → {OUTPUT_DIR}")
    return json_results
if __name__ == "__main__":
    run_experiment()
