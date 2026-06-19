"""
exp19_burgers_failure_map.py — Failure Mode Phase Diagram for Burgers PINNs
[v2 — journal-ready fixes]

Systematically sweeps viscosity ν from 0.1 to 0.001 (15 log-spaced steps)
on the viscous Burgers equation and classifies the failure mode at each ν.

Failure mode taxonomy:
  (A) successful_convergence    — L2 < 0.05, loss stable
  (B) spectral_bias             — high-freq ERROR dominant (FIX 4: error FFT)
  (C) gradient_pathology        — |∇L_pde| / |∇L_bc| > threshold
  (D) collocation_starvation    — residual concentrated near shock
  (E) optimization_divergence   — loss NaN or > DIVERGE_THRESH × initial

Outputs (results/exp19/):
  - failure_mode_map.png
  - l2_vs_nu.png
  - diagnostic_signals.png
  - exp19_results.json

FIXES vs v1 (journal-ready):
  [FIX 1] Failure classifier uses continuous scores, not binary flags.
          v1 used float(signal > threshold) giving binary 0/1; when
          multiple conditions fired, max() picked alphabetically. v2
          computes a normalized continuous score for each mode (how far
          above threshold, scaled to [0, ∞)), picks the dominant one.
          Also adds an "unknown" fallback instead of defaulting to B.

  [FIX 2] Reference solver uses adaptive dt with CFL and diffusion
          stability checks. v1 used dt=t_vals[1]-t_vals[0]=0.01 which
          exceeds the explicit diffusion stability limit dt ≤ dx²/(2ν)
          at ν ≤ 0.001 (dt_max≈0.0015), making the reference solution
          incorrect at low ν. v2 computes dt_max = min(CFL, dx²/(2ν))
          and sub-steps within each output interval. ν sweep is also
          capped at ν_min=0.001 where reference remains reliable.
          (To study ν < 0.001, a different reference — e.g. Cole-Hopf
          series — is needed; this is documented in JSON.)

  [FIX 3] Explicit seed=42 for reproducibility. v1 ran 20 training
          sessions sequentially with no seed control; torch state at
          each training start depended on all previous runs.

  [FIX 4] Spectral ratio uses error FFT, not prediction FFT.
          v1 computed FFT(u_pred)/FFT basis — but u_pred near a shock
          correctly contains high-frequency content. This misclassified
          correct shock predictions as spectral_bias failures. v2
          computes FFT(u_pred - u_exact) normalized by ||u_exact||₂,
          same corrected metric as Exp 18 Fix 1.

  [FIX 5] Shock concentration gradient flow fixed. v1 created x,t with
          requires_grad=True, then passed .unsqueeze(1) which creates
          a new tensor disconnecting autograd. PDE residual was 0
          everywhere (all gradients zero), making D never fire. v2
          passes properly constructed tensors and validates non-zero
          residual before computing concentration.
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
from matplotlib.patches import Patch
from pathlib import Path
from scipy.fft import rfft

# ===================================================================
# Speed flags
# ===================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp19] Device: {DEVICE}")

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp19"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Config
# ===================================================================
# FIX 2: cap at nu_min=0.001 where CN reference solver is reliable
# v1 swept to 0.0001 where dt > diffusion stability limit
NU_VALUES = np.logspace(-1, -3, 15)     # 0.1 → 0.001 (15 steps)

SEED      = 42          # FIX 3
N_HIDDEN  = 4
N_NEURONS = 100
ACTIVATION = "tanh"

N_EPOCHS      = 20_000
LR            = 1e-3
LR_MIN        = 1e-5
N_COLLOCATION = 6_000
N_IC          = 200
N_BC          = 200

NX_EVAL = 256
NT_EVAL = 100

# Failure thresholds
L2_SUCCESS_THRESH     = 0.05
GRAD_RATIO_THRESH     = 50.0
SPECTRAL_RATIO_THRESH = 0.30   # FIX 4: error-based metric, lower scale
SHOCK_CONC_THRESH     = 2.5
DIVERGE_THRESH        = 10.0

FAILURE_CODES = {
    "A": "successful_convergence",
    "B": "spectral_bias",
    "C": "gradient_pathology",
    "D": "collocation_starvation",
    "E": "optimization_divergence",
    "U": "unclassified",
}
FAILURE_COLORS = {
    "A": "#2CA02C",
    "B": "#1F77B4",
    "C": "#FF7F0E",
    "D": "#9467BD",
    "E": "#D62728",
    "U": "#7F7F7F",
}


# ===================================================================
# FIX 2 — Reference solver with adaptive dt
# ===================================================================

def burgers_ic(x):
    return -np.sin(np.pi * x)


def _cn_solve_adaptive(x_vals, t_vals, nu):
    """
    Crank-Nicolson reference solver with adaptive dt sub-stepping.

    FIX 2: v1 used dt=t_vals[1]-t_vals[0] which exceeds the explicit
    diffusion stability limit dt ≤ dx²/(2ν) at small ν. v2 computes:
      dt_max = min(CFL_factor * dx / max(|u|+eps),
                   0.4 * dx² / (2*nu))
    and sub-steps within each output interval.
    """
    nx = len(x_vals)
    dx = float(x_vals[1] - x_vals[0])

    u  = burgers_ic(x_vals).copy().astype(float)
    u[0] = 0.0; u[-1] = 0.0

    output_dt      = float(t_vals[1] - t_vals[0]) if len(t_vals) > 1 else 0.01
    snapshots      = {t_vals[0]: u.copy()}
    t_current      = float(t_vals[0])
    snap_idx       = 1

    max_total_steps = 100_000

    for _ in range(max_total_steps):
        if snap_idx >= len(t_vals):
            break

        u_max  = np.abs(u).max() + 1e-8
        dt_cfl  = 0.5 * dx / u_max
        dt_diff = 0.4 * dx**2 / (2.0 * nu) if nu > 0 else 1.0
        dt      = min(dt_cfl, dt_diff, output_dt)
        dt      = min(dt, t_vals[snap_idx] - t_current + 1e-12)
        if dt <= 0:
            dt = 1e-6

        # Upwind advection (explicit)
        u_pos = np.maximum(u, 0)
        u_neg = np.minimum(u, 0)
        adv   = (u_pos * (u - np.roll(u, 1)) / dx
                 + u_neg * (np.roll(u, -1) - u) / dx)
        adv[0] = 0.0; adv[-1] = 0.0
        rhs = u - dt * adv

        # Implicit diffusion: CN (I - dt/2 * nu * D2) u_new = (I + dt/2 * nu * D2) u_old
        alpha = nu * dt / (2.0 * dx**2)
        diag  = (1 + 2 * alpha) * np.ones(nx)
        off   = -alpha * np.ones(nx - 1)
        # Add explicit diffusion to rhs
        rhs[1:-1] += alpha * (u[:-2] - 2*u[1:-1] + u[2:])
        rhs[0] = 0.0; rhs[-1] = 0.0; diag[0] = 1.0; diag[-1] = 1.0
        off_mod = off.copy()
        off_mod[0]  = 0.0  # BCs
        off_mod[-1] = 0.0

        u_new = _thomas(off_mod, diag, off_mod, rhs)
        u_new[0] = 0.0; u_new[-1] = 0.0
        u = u_new
        t_current += dt

        if (snap_idx < len(t_vals)
                and t_current >= t_vals[snap_idx] - 1e-9):
            snapshots[t_vals[snap_idx]] = u.copy()
            snap_idx += 1

    for tv in t_vals:
        if tv not in snapshots:
            snapshots[tv] = u.copy()

    return np.stack([snapshots[tv] for tv in t_vals])   # (nt, nx)


def _thomas(lower, diag, upper, rhs):
    """Thomas algorithm for tridiagonal system."""
    n = len(rhs)
    d = diag.copy(); b = rhs.copy()
    c = upper.copy(); a = lower.copy()
    for i in range(1, n):
        if abs(d[i-1]) < 1e-15: continue
        m     = a[i-1] / d[i-1]
        d[i] -= m * c[i-1]
        b[i] -= m * b[i-1]
    x = np.zeros(n)
    x[-1] = b[-1] / (d[-1] + 1e-15)
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - c[i] * x[i+1]) / (d[i] + 1e-15)
    return x


def burgers_reference(x_vals, t_vals, nu):
    return _cn_solve_adaptive(x_vals, t_vals, nu)


# ===================================================================
# Model
# ===================================================================

class BurgersPINN(nn.Module):
    def __init__(self, n_hidden=4, n_neurons=100, activation="tanh"):
        super().__init__()
        act_map = {"tanh": nn.Tanh, "silu": nn.SiLU}
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


def burgers_pde_residual(model, x, t, nu):
    """Compute Burgers PDE residual with proper autograd."""
    x_req = x.detach().requires_grad_(True)
    t_req = t.detach().requires_grad_(True)
    u    = model(x_req, t_req)
    ut   = torch.autograd.grad(u,  t_req, torch.ones_like(u),
                                create_graph=True)[0]
    ux   = torch.autograd.grad(u,  x_req, torch.ones_like(u),
                                create_graph=True)[0]
    uxx  = torch.autograd.grad(ux, x_req, torch.ones_like(ux),
                                create_graph=True)[0]
    return ut + u * ux - nu * uxx


def compute_burgers_loss(model, nu, n_col, n_ic, n_bc):
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE)
    res   = burgers_pde_residual(model, xc, tc, nu)
    l_pde = (res ** 2).mean()

    xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    ti = torch.zeros(n_ic, 1, dtype=DTYPE, device=DEVICE)
    u_ic_true = torch.tensor(
        burgers_ic(xi.detach().cpu().numpy()),
        dtype=DTYPE, device=DEVICE)
    l_ic  = ((model(xi, ti) - u_ic_true) ** 2).mean()

    tb   = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE)
    xl   = torch.full((n_bc, 1), -1.0, dtype=DTYPE, device=DEVICE)
    xr   = torch.ones(n_bc, 1, dtype=DTYPE, device=DEVICE)
    l_bc = ((model(xl, tb)) ** 2 + (model(xr, tb)) ** 2).mean()

    return l_pde + 100 * l_ic + 50 * l_bc, l_pde, l_ic, l_bc


# ===================================================================
# Diagnostics
# ===================================================================

def compute_gradient_ratio(model, nu, n_col=300, n_bc=100):
    """|∇L_pde| / |∇L_bc|."""
    model.zero_grad()
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE)
    l_pde = (burgers_pde_residual(model, xc, tc, nu) ** 2).mean()
    l_pde.backward()
    g_pde = torch.cat([p.grad.flatten() for p in model.parameters()
                       if p.grad is not None]).norm().item()

    model.zero_grad()
    tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE)
    xl = torch.full((n_bc, 1), -1.0, dtype=DTYPE, device=DEVICE)
    xr = torch.ones(n_bc, 1, dtype=DTYPE, device=DEVICE)
    l_bc = ((model(xl, tb)) ** 2 + (model(xr, tb)) ** 2).mean()
    l_bc.backward()
    g_bc = torch.cat([p.grad.flatten() for p in model.parameters()
                      if p.grad is not None]).norm().item() + 1e-30
    model.zero_grad()
    return float(g_pde / g_bc)


def compute_spectral_error_ratio(model, u_ref_t, nx=NX_EVAL, t_probe_idx=50):
    """
    FIX 4: ratio of high-freq to low-freq ERROR amplitude.
    v1 used FFT(u_pred) — this misclassified correct near-shock
    predictions (which legitimately have high-freq content).
    v2 uses FFT(u_pred - u_exact) normalized by ||u_exact||₂.
    """
    x_vals   = np.linspace(-1, 1, nx)
    t_probe  = float(t_probe_idx) / (NX_EVAL - 1)
    t_tensor = torch.full((nx, 1), t_probe, dtype=DTYPE, device=DEVICE)
    x_tensor = torch.tensor(x_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).cpu().numpy().flatten()
    model.train()

    u_exact_row = u_ref_t[t_probe_idx]   # (nx,) from reference
    err         = u_pred - u_exact_row
    global_norm = np.linalg.norm(u_exact_row) + 1e-10

    fft_err = np.abs(rfft(err))
    n_bins  = len(fft_err)
    low_err  = fft_err[:n_bins//4].mean() / global_norm + 1e-10
    high_err = fft_err[n_bins//4:].mean() / global_norm + 1e-10
    return float(high_err / low_err)


def compute_shock_concentration(model, nu, nx=NX_EVAL, nt=50):
    """
    FIX 5: residual concentration near shock region.
    v1 passed .unsqueeze(1) of a requires_grad=True tensor, which
    disconnects autograd — all residuals were zero. v2 constructs
    x_col and t_col correctly and validates non-zero residual.
    """
    x_vals = np.linspace(-1, 1, nx)
    t_vals = np.linspace(0.3, 1.0, nt)
    XX, TT = np.meshgrid(x_vals, t_vals)

    # FIX 5: construct fresh tensors correctly
    x_col = torch.tensor(XX.ravel()[:, None], dtype=DTYPE, device=DEVICE)
    t_col = torch.tensor(TT.ravel()[:, None], dtype=DTYPE, device=DEVICE)

    res    = burgers_pde_residual(model, x_col, t_col, nu)
    abs_r  = res.detach().cpu().numpy().reshape(nt, nx)
    abs_r  = np.abs(abs_r)
    mean_r = abs_r.mean() + 1e-10

    # Validate residual is non-zero
    if mean_r < 1e-9:
        return 0.0   # degenerate — don't use as classification signal

    # Shock near x=0 for IC = -sin(πx)
    shock_mask = np.abs(XX) < 0.2
    near_shock = abs_r[shock_mask].mean() + 1e-10
    return float(near_shock / mean_r)


def evaluate_l2(model, u_ref, nx=NX_EVAL, nt=NT_EVAL):
    x_vals = np.linspace(-1, 1, nx)
    t_vals = np.linspace(0, 1, nt)
    XX, TT = np.meshgrid(x_vals, t_vals)
    model.eval()
    with torch.no_grad():
        xf = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        tf = torch.tensor(TT.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u_pred = model(xf, tf).cpu().numpy().reshape(nt, nx)
    model.train()
    denom = float(np.sqrt((u_ref ** 2).mean())) + 1e-8
    return float(np.sqrt(((u_pred - u_ref) ** 2).mean()) / denom)


# ===================================================================
# FIX 1 — Continuous-score failure classifier
# ===================================================================

def classify_failure(l2_err, diverged, spectral_ratio,
                     grad_ratio, shock_conc, loss_trajectory):
    """
    FIX 1: compute continuous score for each candidate mode.
    Score = (signal - threshold) / threshold, bounded below at 0.
    Picks the highest-scoring mode. Falls back to "U" (unclassified)
    when no mode scores above zero but L2 is still high.
    """
    if diverged:
        return "E", {}

    if l2_err < L2_SUCCESS_THRESH:
        return "A", {}

    # Continuous scores: how far above threshold (normalized)
    scores = {
        "C": max(0.0, (grad_ratio   - GRAD_RATIO_THRESH)   / GRAD_RATIO_THRESH),
        "B": max(0.0, (spectral_ratio - SPECTRAL_RATIO_THRESH) / SPECTRAL_RATIO_THRESH),
        "D": max(0.0, (shock_conc   - SHOCK_CONC_THRESH)   / SHOCK_CONC_THRESH),
    }

    best  = max(scores, key=lambda k: scores[k])
    if scores[best] > 0:
        return best, scores

    # No signal dominates — report as unclassified with note
    return "U", scores


# ===================================================================
# Training
# ===================================================================

def train_one_nu(nu, u_ref, seed, nu_idx=None, full_ckpt=None, ckpt_path=None, model_path=None):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n  ν={nu:.2e}")
    model     = BurgersPINN(N_HIDDEN, N_NEURONS, ACTIVATION).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)

    loss_traj = []
    diverged  = False
    loss_init = None
    start_epoch = 0

    if full_ckpt is not None and f"nu_{nu_idx}" in full_ckpt:
        inner = full_ckpt[f"nu_{nu_idx}"]
        start_epoch = inner.get("completed_epoch", 0)
        loss_traj = inner.get("loss_traj", [])
        if model_path and model_path.exists():
            model.load_state_dict(torch.load(model_path))
            print(f"    [Resuming from epoch {start_epoch}]")
        for _ in range(start_epoch):
            scheduler.step()

    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        try:
            loss, lp, li, lb = compute_burgers_loss(
                model, nu, N_COLLOCATION, N_IC, N_BC)
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            lv = float(loss.item())
            if loss_init is None: loss_init = lv
            if lv > DIVERGE_THRESH * (loss_init + 1e-8):
                diverged = True; break
            if epoch % 2000 == 0:
                loss_traj.append(lv)
            if epoch % 5000 == 0 and epoch >= start_epoch:
                print(f"    {epoch:>6d}: loss={lv:.3e}")
                
            if (epoch + 1) % 5000 == 0 and full_ckpt is not None and ckpt_path is not None:
                if model_path:
                    torch.save(model.state_dict(), model_path)
                if f"nu_{nu_idx}" not in full_ckpt:
                    full_ckpt[f"nu_{nu_idx}"] = {}
                full_ckpt[f"nu_{nu_idx}"]["completed_epoch"] = epoch + 1
                full_ckpt[f"nu_{nu_idx}"]["loss_traj"] = loss_traj
                with open(ckpt_path, 'w') as f:
                    json.dump(full_ckpt, f)

        except Exception as e:
            print(f"    Exception at epoch {epoch}: {e}")
            diverged = True; break

    l2_err  = evaluate_l2(model, u_ref)
    grad_r  = compute_gradient_ratio(model, nu) if not diverged else 1e6
    spec_r  = compute_spectral_error_ratio(model, u_ref) if not diverged else 0.0
    shock_c = compute_shock_concentration(model, nu) if not diverged else 0.0

    code, scores = classify_failure(
        l2_err, diverged, spec_r, grad_r, shock_c, loss_traj)

    print(f"    L2={l2_err:.4f}  grad_r={grad_r:.1e}  "
          f"spec_r={spec_r:.3f}  shock={shock_c:.2f} "
          f"→ [{code}] {FAILURE_CODES[code]}")
    if scores:
        print(f"    Scores: {scores}")

    return {
        "nu":                  float(nu),
        "failure_code":        code,
        "failure_name":        FAILURE_CODES[code],
        "l2_error":            l2_err,
        "diverged":            diverged,
        "grad_ratio":          float(grad_r),
        "spectral_ratio":      float(spec_r),
        "shock_concentration": float(shock_c),
        "classifier_scores":   {k: float(v) for k, v in scores.items()},
        "loss_trajectory":     loss_traj,
    }


# ===================================================================
# Plotting
# ===================================================================

def plot_failure_mode_map(all_results, filepath):
    nu_vals = [r["nu"] for r in all_results]
    codes   = [r["failure_code"] for r in all_results]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.suptitle("Failure Mode Phase Diagram — Burgers PINN",
                 fontsize=13, fontweight="bold")

    for i in range(len(nu_vals) - 1):
        ax.axvspan(np.log10(nu_vals[i]), np.log10(nu_vals[i+1]),
                   color=FAILURE_COLORS[codes[i]], alpha=0.35)
    for nu, code in zip(nu_vals, codes):
        ax.scatter([np.log10(nu)], [0], color=FAILURE_COLORS[code],
                   s=280, marker="s", zorder=5, edgecolors="white",
                   linewidths=0.5)

    legend_elems = [Patch(facecolor=FAILURE_COLORS[k],
                          label=f"[{k}] {FAILURE_CODES[k]}")
                    for k in FAILURE_CODES]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)
    ax.set_xlabel("log₁₀(ν)  [← decreasing ν, increasing difficulty]",
                  fontsize=11)
    ax.set_yticks([])
    ax.set_title("Failure Mode vs Viscosity", fontsize=11)

    prev = codes[0]
    for i, (nu, code) in enumerate(zip(nu_vals, codes)):
        if code != prev:
            ax.axvline(np.log10(nu), color="black",
                       lw=1.5, ls="--")
            ax.text(np.log10(nu), 0.6, f"{prev}→{code}",
                    ha="center", fontsize=8,
                    transform=ax.get_xaxis_transform())
            prev = code

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_l2_vs_nu(all_results, filepath):
    nu_vals = [r["nu"] for r in all_results]
    l2_vals = [r["l2_error"] for r in all_results]
    codes   = [r["failure_code"] for r in all_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("L2 Error vs Viscosity ν — Burgers PINN\n"
                 "(colors = failure mode classification)",
                 fontsize=12, fontweight="bold")
    for nu, l2, code in zip(nu_vals, l2_vals, codes):
        ax.scatter([nu], [l2], color=FAILURE_COLORS[code],
                   s=90, zorder=5, edgecolors="white", lw=0.5)
    ax.plot(nu_vals, l2_vals, "k--", lw=1, alpha=0.4)
    ax.axhline(L2_SUCCESS_THRESH, color="green", ls=":",
               lw=1.5, label=f"Success threshold (L2={L2_SUCCESS_THRESH})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ν (viscosity)", fontsize=11)
    ax.set_ylabel("Relative L2 Error", fontsize=11)
    ax.grid(True, alpha=0.25, which="both")
    legend_elems = [Patch(facecolor=FAILURE_COLORS[k],
                          label=f"[{k}] {FAILURE_CODES[k]}")
                    for k in FAILURE_CODES]
    ax.legend(handles=legend_elems, fontsize=8, ncol=2)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_diagnostic_signals(all_results, filepath):
    nu_vals = [r["nu"] for r in all_results]
    grad_r  = [r["grad_ratio"] for r in all_results]
    spec_r  = [r["spectral_ratio"] for r in all_results]
    shock_c = [r["shock_concentration"] for r in all_results]
    codes   = [r["failure_code"] for r in all_results]
    colors_pt = [FAILURE_COLORS[c] for c in codes]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                              constrained_layout=True)
    fig.suptitle("Diagnostic Signals vs Viscosity ν\n"
                 "(FIX 4: spectral ratio uses error FFT / ||u_exact||₂)",
                 fontsize=11, fontweight="bold")

    for ax, vals, ylabel, thresh, lbl in [
        (axes[0], grad_r,  "Gradient Ratio |∇L_pde|/|∇L_bc|",
         GRAD_RATIO_THRESH, f"pathology thresh ({GRAD_RATIO_THRESH})"),
        (axes[1], spec_r,  "High/Low Error Spectral Ratio",
         SPECTRAL_RATIO_THRESH, f"spectral bias thresh ({SPECTRAL_RATIO_THRESH})"),
        (axes[2], shock_c, "Shock Concentration Ratio",
         SHOCK_CONC_THRESH, f"starvation thresh ({SHOCK_CONC_THRESH})"),
    ]:
        ax.scatter(nu_vals, vals, c=colors_pt, s=60, zorder=5,
                   edgecolors="white", lw=0.5)
        ax.plot(nu_vals, vals, "k--", lw=0.8, alpha=0.4)
        ax.axhline(thresh, color="red", ls="--", lw=1.2, label=lbl)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("ν (viscosity)", fontsize=11)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 19: Burgers Failure Mode Phase Diagram  [v2]")
    print(f"Device  : {DEVICE}")
    print(f"ν sweep : {NU_VALUES[0]:.4f} → {NU_VALUES[-1]:.6f}  "
          f"({len(NU_VALUES)} values)")
    print(f"Seed    : {SEED}  (FIX 3)")
    print(f"Spectral: error FFT / ||u_exact||₂  (FIX 4)")
    print(f"Ref solver: adaptive dt with CFL + diffusion check  (FIX 2)")
    print("=" * 70)

    x_ref = np.linspace(-1, 1, NX_EVAL)
    t_ref = np.linspace(0,  1, NT_EVAL)

    print("\nPre-computing reference solutions...")
    ref_solutions = {}
    for nu in NU_VALUES:
        print(f"  ν={nu:.2e} ... ", end="", flush=True)
        ref_solutions[nu] = burgers_reference(x_ref, t_ref, nu)
        print("done")

    ckpt_path = OUTPUT_DIR / "exp19_checkpoint.json"
    model_path = OUTPUT_DIR / "exp19_current_model.pt"
    ckpt = {}
    if ckpt_path.exists():
        try:
            with open(ckpt_path, 'r') as f:
                ckpt = json.load(f)
        except Exception:
            pass

    all_results = ckpt.get("all_results", [])
    start_idx = len(all_results)
    
    t0 = time.time() - ckpt.get("elapsed_time", 0.0)

    for idx in range(start_idx, len(NU_VALUES)):
        nu = NU_VALUES[idx]
        # FIX 3: different seed per ν but deterministic
        result = train_one_nu(nu, ref_solutions[nu], seed=SEED + idx,
                              nu_idx=idx, full_ckpt=ckpt, ckpt_path=ckpt_path, model_path=model_path)
        all_results.append(result)
        
        # Update checkpoint after each ν completes
        ckpt["all_results"] = all_results
        if f"nu_{idx}" in ckpt:
            del ckpt[f"nu_{idx}"]
        ckpt["elapsed_time"] = time.time() - t0
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt, f)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # Transition analysis
    transitions = []
    for i in range(1, len(all_results)):
        if all_results[i]["failure_code"] != all_results[i-1]["failure_code"]:
            transitions.append({
                "from_code":    all_results[i-1]["failure_code"],
                "to_code":      all_results[i]["failure_code"],
                "from_name":    all_results[i-1]["failure_name"],
                "to_name":      all_results[i]["failure_name"],
                "nu_lower":     float(all_results[i-1]["nu"]),
                "nu_upper":     float(all_results[i]["nu"]),
                "nu_midpoint":  float(np.sqrt(
                    all_results[i-1]["nu"] * all_results[i]["nu"])),
            })

    # Plots
    print("\nGenerating plots...")
    plot_failure_mode_map(all_results,     OUTPUT_DIR / "failure_mode_map.png")
    plot_l2_vs_nu(all_results,             OUTPUT_DIR / "l2_vs_nu.png")
    plot_diagnostic_signals(all_results,   OUTPUT_DIR / "diagnostic_signals.png")

    # Summary table
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 19 — SUMMARY  [v2]")
    print(f"{'=' * 70}")
    print(f"\n{'ν':>10} | {'Code':>5} | {'Failure Mode':<26} | "
          f"{'L2':>7} | {'grad_r':>8} | {'spec_r':>7}")
    print("─" * 70)
    for r in all_results:
        print(f"{r['nu']:>10.3e} | {r['failure_code']:>5} | "
              f"{r['failure_name']:<26} | {r['l2_error']:>7.4f} | "
              f"{r['grad_ratio']:>8.1e} | {r['spectral_ratio']:>7.3f}")

    print(f"\nMode Transitions:")
    for tr in transitions:
        print(f"  [{tr['from_code']}]→[{tr['to_code']}]  "
              f"near ν≈{tr['nu_midpoint']:.3e}  "
              f"({tr['from_name']} → {tr['to_name']})")

    # JSON
    results_json = {
        "experiment": "Burgers Failure Mode Phase Diagram",
        "version":    "v2-journal-ready",
        "config": {
            "nu_values":       [float(v) for v in NU_VALUES],
            "nu_min":          float(NU_VALUES[-1]),
            "nu_max":          float(NU_VALUES[0]),
            "n_nu_values":     len(NU_VALUES),
            "n_epochs":        N_EPOCHS,
            "seed":            SEED,
            "failure_codes":   FAILURE_CODES,
            "thresholds": {
                "l2_success":  L2_SUCCESS_THRESH,
                "grad_ratio":  GRAD_RATIO_THRESH,
                "spectral":    SPECTRAL_RATIO_THRESH,
                "shock_conc":  SHOCK_CONC_THRESH,
                "diverge":     DIVERGE_THRESH,
            },
        },
        "fix_notes": {
            "fix1_classifier": (
                "v1 used binary flags (0/1) per mode; max() picked "
                "alphabetically when multiple fired. v2 uses normalized "
                "continuous scores: (signal-thresh)/thresh. Also adds 'U' "
                "(unclassified) fallback instead of defaulting to 'B'."
            ),
            "fix2_reference_solver": (
                "v1 used fixed dt=output_dt=0.01, exceeding diffusion "
                "stability limit dt ≤ dx²/(2ν) at ν ≤ 0.001. "
                "v2 uses adaptive dt = min(CFL×dx/u_max, 0.4×dx²/(2ν)), "
                "sub-stepping within each output interval. ν sweep capped "
                "at 0.001 where solver remains reliable. "
                "For ν < 0.001, Cole-Hopf transform reference is recommended."
            ),
            "fix3_seed": f"Explicit seed={SEED}+idx per ν for reproducibility.",
            "fix4_spectral": (
                "v1 used FFT(u_pred) — correct near-shock predictions have "
                "high-freq content and were misclassified as spectral_bias. "
                "v2 uses FFT(u_pred - u_exact)/||u_exact||₂ — measures error "
                "frequency content, not prediction frequency content."
            ),
            "fix5_shock_gradient": (
                "v1 passed .unsqueeze(1) of requires_grad tensors which "
                "disconnected autograd — PDE residual was zero everywhere, "
                "shock_concentration always 0, mode D never fired. "
                "v2 constructs fresh x_col, t_col tensors and validates "
                "non-zero residual before computing concentration."
            ),
        },
        "per_nu": [{k: v for k, v in r.items() if k != "loss_trajectory"}
                   for r in all_results],
        "transitions": transitions,
        "elapsed_seconds": float(elapsed),
    }

    out = OUTPUT_DIR / "exp19_results.json"
    with open(out, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults → {out}")
    print(f"Plots   → {OUTPUT_DIR}")
    return results_json


if __name__ == "__main__":
    run_experiment()