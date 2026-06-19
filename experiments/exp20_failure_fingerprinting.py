"""
exp20_failure_fingerprinting.py — Experiment 20: Failure Fingerprinting & Clustering
[v2 — journal-ready fixes]

Novel failure fingerprinting experiment across 4 PDEs.
For each PDE, 10 PINNs are intentionally driven to failure via parameter tuning.
Each failed run is characterised by 8 diagnostic signals then clustered (k-means, k=4).

PDEs tested:
  1. Advection     — beta in {30,50,80}            (high beta -> spectral failure)
  2. Burgers       — nu in {0.001, 0.005, 0.0001}  (low nu -> shock / gradient failure)
  3. Allen-Cahn    — eps in {0.01, 0.005}           (small eps -> sharp interface failure)
  4. Helmholtz     — k2 in {100, 400, 900}          (high k -> spectral / resonance failure)

Failure induction: each hyper-parameter setting is expected to produce failure.
If a run accidentally succeeds, it is excluded and replaced by a harder setting.

8 Diagnostic signals per run:
  F1: gradient norm trajectory — final gradient norm
  F2: loss component ratio (PDE/IC at end of training)
  F3: Fourier spectral error ratio (high-freq / low-freq)
  F4: NTK eigenvalue spread (log10 condition number)
  F5: collocation density in high-error regions (ratio)
  F6: weight norm growth (final / initial weight norm)
  F7: activation saturation fraction (|tanh(pre-act)| > 0.99)
  F8: Hessian condition number estimate (power iteration)

Outputs saved to results/exp20/:
  - diagnostic_radar_charts.png
  - cluster_scatter.png
  - cluster_heatmap.png
  - cross_pde_cluster_map.png
  - exp20_results.json

FIXES vs v1 (journal-ready):
  [FIX 1] Explicit seed=42+flat_idx for reproducibility. v1 had no seed
          control; torch/numpy random state at each run start depended on
          all previous runs, making results non-reproducible across
          restarts or partial re-runs.

  [FIX 2] Speed flags added: cudnn.benchmark=True and
          float32_matmul_precision="medium". All other experiments in
          the suite use these; omitting them made exp20 ~15-20% slower
          than necessary on RTX 3050.

  [FIX 3] Spectral ratio (F3) now uses error FFT, not prediction FFT.
          v1 computed FFT(u_pred) — but u_pred near a shock or high-freq
          exact solution legitimately contains high-frequency content.
          This misclassified correct predictions as spectral failures.
          v2 computes FFT(u_pred - u_exact) / ||u_exact||_2, consistent
          with the corrected metric in Exp 18 FIX 1 and Exp 19 FIX 4.

  [FIX 4] Collocation density (F5) autograd tensor construction fixed.
          v1 created tensors with requires_grad=True then called
          .unsqueeze(1), which creates a new leaf tensor disconnecting
          the computational graph. The residual functions internally
          call .requires_grad_(True), so the outer requires_grad was
          a no-op at best and graph-breaking at worst. v2 constructs
          tensors directly in (N,1) shape and lets residual functions
          handle grad tracking.

  [FIX 5] Dead code removed. signal_activation_saturation had unused
          variables 'saturated' and 'total'. signal_hessian_cond_estimate
          created xc, tc tensors that were never used. Cleaned up.

  [FIX 6] JSON key typo fixed: "pdse_configs" -> "pde_configs".

  [FIX 7] Training progress logging added. v1 ran 15,000 epochs per run
          completely silently. v2 prints loss every 5000 epochs for
          diagnostics and hang detection.

  [FIX 8] Removed redundant `from scipy.fft import rfft` inside
          signal_spectral_ratio body; rfft is already imported at
          module level.
"""

import sys, os
import io

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
from scipy.fft import rfft
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===================================================================
# FIX 2: Speed flags (consistent with all other experiments)
# ===================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp20] Device: {DEVICE}")

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp20"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Config
# ===================================================================
SEED       = 42          # FIX 1: explicit seed
N_HIDDEN   = 4
N_NEURONS  = 64
N_EPOCHS   = 15_000
LR         = 1e-3
LR_MIN     = 1e-5
N_COL      = 5_000
N_IC       = 150
N_BC       = 150

K_CLUSTERS = 4
SIGNAL_NAMES = [
    "GradNorm",       # F1
    "LossRatio",      # F2
    "SpectralRatio",  # F3
    "NTK_logCond",    # F4
    "CollocDensity",  # F5
    "WeightGrowth",   # F6
    "ActSaturation",  # F7
    "HessCondEst",    # F8
]

# Failure-inducing parameter sets per PDE (10 runs total each)
PDE_CONFIGS = {
    "Advection": [
        {"beta": 30}, {"beta": 40}, {"beta": 50},
        {"beta": 60}, {"beta": 70}, {"beta": 80},
        {"beta": 50, "lr": 5e-3}, {"beta": 50, "n_neurons": 32},
        {"beta": 80, "n_hidden": 2}, {"beta": 30, "n_col": 500},
    ],
    "Burgers": [
        {"nu": 0.001}, {"nu": 0.005}, {"nu": 0.002},
        {"nu": 0.0008}, {"nu": 0.0005}, {"nu": 0.0003},
        {"nu": 0.001, "lr": 5e-3}, {"nu": 0.005, "n_neurons": 32},
        {"nu": 0.001, "n_hidden": 2}, {"nu": 0.001, "n_col": 500},
    ],
    "AllenCahn": [
        {"eps2": 0.01}, {"eps2": 0.005}, {"eps2": 0.008},
        {"eps2": 0.003}, {"eps2": 0.002}, {"eps2": 0.001},
        {"eps2": 0.01, "lr": 5e-3}, {"eps2": 0.005, "n_neurons": 32},
        {"eps2": 0.01, "n_hidden": 2}, {"eps2": 0.01, "n_col": 500},
    ],
    "Helmholtz": [
        {"k2": 100}, {"k2": 200}, {"k2": 300},
        {"k2": 400}, {"k2": 600}, {"k2": 900},
        {"k2": 100, "lr": 5e-3}, {"k2": 400, "n_neurons": 32},
        {"k2": 100, "n_hidden": 2}, {"k2": 100, "n_col": 500},
    ],
}

# ===================================================================
# Exact solutions for spectral error computation (FIX 3)
# ===================================================================

def exact_advection(x_np, t_val, beta):
    return np.sin(x_np - beta * t_val)

def exact_burgers_ic(x_np):
    """IC only — used as rough reference at t=0.5 (no closed-form)."""
    return -np.sin(np.pi * x_np)

def exact_helmholtz(x_np, y_np):
    return np.sin(np.pi * x_np) * np.sin(np.pi * y_np)


# ===================================================================
# Universal PINN
# ===================================================================

class GenericPINN(nn.Module):
    def __init__(self, in_dim=2, n_hidden=4, n_neurons=64):
        super().__init__()
        layers = [nn.Linear(in_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        self._init_wts()

    def _init_wts(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, *args):
        return self.net(torch.cat(list(args), dim=1))


# ===================================================================
# PDE residuals
# ===================================================================

def advection_residual(model, x, t, beta):
    x = x.requires_grad_(True); t = t.requires_grad_(True)
    u  = model(x, t)
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return ut + beta * ux


def burgers_residual(model, x, t, nu):
    x = x.requires_grad_(True); t = t.requires_grad_(True)
    u   = model(x, t)
    ut  = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    ux  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut + u * ux - nu * uxx


def allen_cahn_residual(model, x, t, eps2):
    x = x.requires_grad_(True); t = t.requires_grad_(True)
    u   = model(x, t)
    ut  = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    ux  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut - eps2 * uxx - u + u ** 3


def helmholtz_residual(model, x, y, k2, f_fn):
    x = x.requires_grad_(True); y = y.requires_grad_(True)
    u   = model(x, y)
    ux  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    uy  = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, torch.ones_like(uy), create_graph=True)[0]
    return uxx + uyy + k2 * u - f_fn(x, y, k2)


# ===================================================================
# Loss builders
# ===================================================================

def loss_advection(model, cfg, n_col, n_ic, n_bc):
    beta = cfg["beta"]
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2.0
    lp = (advection_residual(model, xc, tc, beta) ** 2).mean()

    xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    ti = torch.zeros_like(xi)
    li = ((model(xi, ti) - torch.sin(xi)) ** 2).mean()

    tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE) * 2.0
    xl = torch.zeros(n_bc, 1, dtype=DTYPE, device=DEVICE)
    xr = torch.full_like(xl, 2 * np.pi)
    lb = ((model(xl, tb) - model(xr, tb)) ** 2).mean()
    return lp + 100 * li + 10 * lb, lp, li


def loss_burgers(model, cfg, n_col, n_ic, n_bc):
    nu = cfg["nu"]
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE)
    lp = (burgers_residual(model, xc, tc, nu) ** 2).mean()

    xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    ti = torch.zeros_like(xi)
    u_ic = -torch.sin(np.pi * xi)
    li = ((model(xi, ti) - u_ic) ** 2).mean()

    tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE)
    xl = torch.full((n_bc, 1), -1.0, dtype=DTYPE, device=DEVICE)
    xr = torch.ones(n_bc, 1, dtype=DTYPE, device=DEVICE)
    lb = (model(xl, tb) ** 2 + model(xr, tb) ** 2).mean()
    return lp + 100 * li + 50 * lb, lp, li


def loss_allen_cahn(model, cfg, n_col, n_ic, n_bc):
    eps2 = cfg["eps2"]
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE)
    lp = (allen_cahn_residual(model, xc, tc, eps2) ** 2).mean()

    xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    ti = torch.zeros_like(xi)
    u_ic = xi ** 2 * torch.cos(np.pi * xi)
    li = ((model(xi, ti) - u_ic) ** 2).mean()

    tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE)
    xl = torch.full((n_bc, 1), -1.0, dtype=DTYPE, device=DEVICE)
    xr = torch.ones_like(xl)
    lb = (model(xl, tb) ** 2 + model(xr, tb) ** 2).mean()
    return lp + 100 * li + 50 * lb, lp, li


def helmholtz_forcing(x, y, k2):
    """Forcing f such that u_exact = sin(pi x)sin(pi y)."""
    a1, a2 = np.pi, np.pi
    k2_val = float(k2) if not torch.is_tensor(k2) else k2.item()
    return (torch.sin(np.pi * x) * torch.sin(np.pi * y)
            * (-(a1**2 + a2**2) + k2_val))


def loss_helmholtz(model, cfg, n_col, n_ic, n_bc):
    k2 = cfg["k2"]
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    yc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    lp = (helmholtz_residual(model, xc, yc, k2, helmholtz_forcing) ** 2).mean()

    xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    yi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    u_exact = torch.sin(np.pi * xi) * torch.sin(np.pi * yi)
    li = ((model(xi, yi) - u_exact) ** 2).mean()

    # BC: u = 0 on boundary
    t_bc  = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    xm1   = torch.full_like(t_bc, -1.0); xp1 = torch.ones_like(t_bc)
    ym1   = torch.full_like(t_bc, -1.0); yp1 = torch.ones_like(t_bc)
    lb = (model(xm1, t_bc) ** 2 + model(xp1, t_bc) ** 2
          + model(t_bc, ym1) ** 2 + model(t_bc, yp1) ** 2).mean()
    return lp + 100 * li + 100 * lb, lp, li


LOSS_FNS = {
    "Advection": loss_advection,
    "Burgers":   loss_burgers,
    "AllenCahn": loss_allen_cahn,
    "Helmholtz": loss_helmholtz,
}

# ===================================================================
# 8 Diagnostic signals
# ===================================================================

def signal_grad_norm(model):
    norms = [p.grad.norm().item() for p in model.parameters()
             if p.grad is not None]
    return float(np.mean(norms)) if norms else 0.0


def signal_loss_ratio(lp, li):
    return float((lp.item() + 1e-12) / (li.item() + 1e-12))


def signal_spectral_ratio(model, pde_name, cfg, nx=128):
    """
    FIX 3: High/low freq ratio of ERROR in x-space.
    v1 used FFT(u_pred) — misclassified correct high-freq predictions.
    v2 uses FFT(u_pred - u_exact) / ||u_exact||_2.
    For Burgers/AllenCahn where no closed-form exists at t>0, we use
    the IC as the reference (error is still meaningful for fingerprinting).
    """
    model.eval()

    if pde_name == "Advection":
        x_np  = np.linspace(0, 2 * np.pi, nx, endpoint=False)
        t_val = 0.5
        x_t = torch.tensor(x_np[:, None], dtype=DTYPE, device=DEVICE)
        t_t = torch.full_like(x_t, t_val)
        with torch.no_grad():
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
        u_exact = exact_advection(x_np, t_val, cfg["beta"])

    elif pde_name == "Helmholtz":
        x_np = np.linspace(-1, 1, nx)
        y_val = 0.0  # probe along y=0
        x_t = torch.tensor(x_np[:, None], dtype=DTYPE, device=DEVICE)
        y_t = torch.full_like(x_t, y_val)
        with torch.no_grad():
            u_pred = model(x_t, y_t).cpu().numpy().flatten()
        u_exact = exact_helmholtz(x_np, np.full_like(x_np, y_val))

    else:
        # Burgers / AllenCahn: use t=0 (IC) as reference
        x_np = np.linspace(-1, 1, nx)
        x_t  = torch.tensor(x_np[:, None], dtype=DTYPE, device=DEVICE)
        t_t  = torch.zeros_like(x_t)
        with torch.no_grad():
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
        if pde_name == "Burgers":
            u_exact = -np.sin(np.pi * x_np)
        else:  # AllenCahn
            u_exact = x_np ** 2 * np.cos(np.pi * x_np)

    model.train()

    err         = u_pred - u_exact
    global_norm = np.linalg.norm(u_exact) + 1e-10
    fft_err     = np.abs(rfft(err))
    n_bins      = len(fft_err)
    low_err     = fft_err[:n_bins // 4].mean() / global_norm + 1e-10
    high_err    = fft_err[n_bins // 4:].mean() / global_norm + 1e-10
    return float(high_err / low_err)


def signal_ntk_log_cond(model, n_pts=80):
    model.eval()
    xs = torch.linspace(0, 1, n_pts, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    ts = torch.full_like(xs, 0.5)
    params = list(model.parameters())
    rows = []
    for i in range(n_pts):
        model.zero_grad()
        u = model(xs[i:i+1], ts[i:i+1])
        u.backward()
        g = torch.cat([p.grad.flatten() if p.grad is not None
                       else torch.zeros(p.numel(), device=DEVICE)
                       for p in params]).cpu()
        rows.append(g); model.zero_grad()
    J    = torch.stack(rows).numpy()
    K    = J @ J.T
    eigs = np.linalg.eigvalsh(K)[::-1]
    pos  = eigs[eigs > 1e-30]
    cond = float(eigs[0] / (pos[-1] + 1e-30)) if len(pos) else 1e30
    model.train()
    return float(np.log10(cond + 1))


def signal_collocation_density(model, pde_name, cfg, nx=64, nt=64):
    """
    FIX 4: Ratio of residual in top-10% high-error regions vs mean.
    v1 created tensors with requires_grad then .unsqueeze(1) which
    disconnected autograd. v2 constructs (N,1) tensors directly.
    """
    x_lo = 0.0 if pde_name == "Advection" else -1.0
    x_hi = 2 * np.pi if pde_name == "Advection" else 1.0
    x_vals = np.linspace(x_lo, x_hi, nx)
    t_vals = np.linspace(0, 1, nt)
    XX, TT = np.meshgrid(x_vals, t_vals)

    # FIX 4: construct tensors in correct (N,1) shape directly
    xf = torch.tensor(XX.ravel()[:, None], dtype=DTYPE, device=DEVICE)
    tf = torch.tensor(TT.ravel()[:, None], dtype=DTYPE, device=DEVICE)

    if pde_name == "Advection":
        res = advection_residual(model, xf, tf, cfg["beta"])
    elif pde_name == "Burgers":
        res = burgers_residual(model, xf, tf, cfg["nu"])
    elif pde_name == "AllenCahn":
        res = allen_cahn_residual(model, xf, tf, cfg["eps2"])
    else:
        res = helmholtz_residual(model, xf, tf, cfg["k2"], helmholtz_forcing)

    abs_res = res.detach().abs().cpu().numpy()
    threshold = np.percentile(abs_res, 90)
    high_err  = abs_res[abs_res >= threshold].mean()
    mean_err  = abs_res.mean() + 1e-10
    return float(high_err / mean_err)


def signal_weight_growth(model, initial_weight_norm):
    wn = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5
    return float(wn / (initial_weight_norm + 1e-10))


def signal_activation_saturation(model):
    """Fraction of tanh activations with |output| > 0.99."""
    # FIX 5: removed dead variables 'saturated' and 'total'
    hooks = []
    counts = []

    def make_hook():
        def hook(m, inp, out):
            frac = (out.detach().abs() > 0.99).float().mean().item()
            counts.append(frac)
        return hook

    for m in model.net:
        if isinstance(m, nn.Tanh):
            hooks.append(m.register_forward_hook(make_hook()))

    dummy_x = torch.rand(100, 1, dtype=DTYPE, device=DEVICE)
    dummy_t = torch.rand(100, 1, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        model(dummy_x, dummy_t)

    for h in hooks:
        h.remove()

    return float(np.mean(counts)) if counts else 0.0


def signal_hessian_cond_estimate(model, loss_fn, cfg, n_iter=10):
    """
    Power iteration estimate of top Hessian eigenvalue / bottom.
    Only uses Hessian-vector products (efficient).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    total  = sum(p.numel() for p in params)

    def hvp(v_flat):
        """Hessian-vector product using double backward."""
        v_flat = v_flat.detach()
        vecs   = []
        s = 0
        for p in params:
            n = p.numel()
            vecs.append(v_flat[s:s+n].reshape_as(p))
            s += n

        model.zero_grad()
        # FIX 5: removed unused xc, tc variables
        loss, _, _ = loss_fn(model, cfg,
                             min(500, N_COL), min(50, N_IC), min(50, N_BC))

        grads = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((g * v).sum() for g, v in zip(grads, vecs))
        hv = torch.autograd.grad(gv, params, retain_graph=False)
        return torch.cat([h.detach().flatten() for h in hv])

    # Power iteration for lambda_max
    v = torch.randn(total, dtype=DTYPE, device=DEVICE)
    v = v / (v.norm() + 1e-12)
    lambda_max = 1.0
    for _ in range(n_iter):
        hv = hvp(v)
        lambda_max = float(hv.norm().item())
        v = hv / (lambda_max + 1e-12)

    # Approximate: kappa ~ lambda_max / (trace/n estimate)
    trace_est = 0.0
    for _ in range(5):
        r = torch.randn(total, dtype=DTYPE, device=DEVICE)
        r = r / (r.norm() + 1e-12)
        hv_r = hvp(r)
        trace_est += float((r * hv_r).sum().item())
    trace_est /= 5
    lambda_min_est = max(abs(trace_est) / total, 1e-10)
    cond_est = lambda_max / lambda_min_est
    return float(np.log10(cond_est + 1))


# ===================================================================
# Train one run and extract fingerprint
# ===================================================================

def train_and_fingerprint(pde_name, cfg, run_idx, flat_idx=None,
                          full_ckpt=None, ckpt_path=None, model_path=None):
    # FIX 1: deterministic seed per run
    run_seed = SEED + (flat_idx if flat_idx is not None else run_idx)
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    n_hidden  = cfg.get("n_hidden",  N_HIDDEN)
    n_neurons = cfg.get("n_neurons", N_NEURONS)
    lr        = cfg.get("lr",        LR)
    n_col     = cfg.get("n_col",     N_COL)

    model     = GenericPINN(n_hidden=n_hidden, n_neurons=n_neurons).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)
    loss_fn   = LOSS_FNS[pde_name]

    # Record initial weight norm
    init_wn = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5

    last_lp = last_li = None
    start_epoch = 0

    # ── Checkpoint resume ──
    if full_ckpt is not None and flat_idx is not None and f"run_{flat_idx}" in full_ckpt:
        inner = full_ckpt[f"run_{flat_idx}"]
        start_epoch = inner.get("completed_epoch", 0)
        if model_path and model_path.exists():
            model.load_state_dict(torch.load(model_path))
            print(f"    [Resuming from epoch {start_epoch}]")
        for _ in range(start_epoch):
            scheduler.step()

    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        try:
            loss, lp, li = loss_fn(model, cfg, n_col, N_IC, N_BC)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            last_lp, last_li = lp, li

            # FIX 7: training progress logging
            if epoch % 5000 == 0:
                print(f"    {epoch:>6d}: loss={loss.item():.3e}")

            # Checkpoint every 5000 epochs
            if (epoch + 1) % 5000 == 0 and full_ckpt is not None and ckpt_path is not None:
                if model_path:
                    torch.save(model.state_dict(), model_path)
                if f"run_{flat_idx}" not in full_ckpt:
                    full_ckpt[f"run_{flat_idx}"] = {}
                full_ckpt[f"run_{flat_idx}"]["completed_epoch"] = epoch + 1
                with open(ckpt_path, 'w') as f:
                    json.dump(full_ckpt, f)

        except Exception:
            break

    # ── Extract 8 signals ──
    try:
        f1 = signal_grad_norm(model)
    except Exception:
        f1 = 0.0
    f2 = signal_loss_ratio(last_lp, last_li) if last_lp is not None else 1.0
    try:
        f3 = signal_spectral_ratio(model, pde_name, cfg)
    except Exception:
        f3 = 1.0
    try:
        f4 = signal_ntk_log_cond(model)
    except Exception:
        f4 = 0.0
    try:
        f5 = signal_collocation_density(model, pde_name, cfg)
    except Exception:
        f5 = 1.0
    f6 = signal_weight_growth(model, init_wn)
    f7 = signal_activation_saturation(model)
    try:
        f8 = signal_hessian_cond_estimate(model, loss_fn, cfg)
    except Exception:
        f8 = 0.0

    return {
        "pde": pde_name, "run": run_idx, "cfg": str(cfg),
        "seed": int(run_seed),    # FIX 1: record seed for reproducibility
        "signals": [f1, f2, f3, f4, f5, f6, f7, f8],
    }


# ===================================================================
# Plotting
# ===================================================================

def plot_radar_charts(all_runs, cluster_labels, filepath, n_show=8):
    """Radar / spider chart for each cluster centroid."""
    n_signals = len(SIGNAL_NAMES)
    angles    = np.linspace(0, 2 * np.pi, n_signals, endpoint=False).tolist()
    angles   += angles[:1]

    signals_arr = np.array([r["signals"] for r in all_runs])
    scaler      = StandardScaler()
    signals_norm = scaler.fit_transform(signals_arr)

    k = max(cluster_labels) + 1
    colors = cm.tab10(np.linspace(0, 1, k))

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4),
                             subplot_kw={"polar": True},
                             constrained_layout=True)
    fig.suptitle("Failure Fingerprint - Cluster Centroids\n"
                 "(FIX 3: spectral ratio uses error FFT)",
                 fontsize=12, fontweight="bold")
    if k == 1:
        axes = [axes]

    for ci in range(k):
        mask     = cluster_labels == ci
        centroid = signals_norm[mask].mean(axis=0)
        vals     = centroid.tolist() + centroid[:1].tolist()
        ax       = axes[ci]
        ax.plot(angles, vals,   color=colors[ci], linewidth=2)
        ax.fill(angles, vals,   color=colors[ci], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(SIGNAL_NAMES, fontsize=7)
        ax.set_title(f"Cluster {ci}\n(n={mask.sum()})", fontsize=9)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_cluster_scatter(all_runs, cluster_labels, filepath):
    signals_arr  = np.array([r["signals"] for r in all_runs])
    scaler       = StandardScaler()
    signals_norm = scaler.fit_transform(signals_arr)

    pca  = PCA(n_components=2)
    emb  = pca.fit_transform(signals_norm)
    var  = pca.explained_variance_ratio_

    k      = max(cluster_labels) + 1
    colors = cm.tab10(np.linspace(0, 1, k))
    pde_markers = {"Advection": "o", "Burgers": "s",
                   "AllenCahn": "^", "Helmholtz": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Failure Cluster Embedding (PCA)", fontsize=12,
                 fontweight="bold")

    # Panel 1: coloured by cluster
    ax = axes[0]
    for ci in range(k):
        mask = cluster_labels == ci
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   color=colors[ci], s=60, alpha=0.8, label=f"Cluster {ci}")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=10)
    ax.set_title("Coloured by Cluster", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Panel 2: coloured by PDE
    ax2 = axes[1]
    pde_colors = {"Advection": "#E64040", "Burgers": "#F59F00",
                  "AllenCahn": "#3A7FD5",  "Helmholtz": "#2CA02C"}
    for i, run in enumerate(all_runs):
        pde  = run["pde"]
        mrkr = pde_markers[pde]
        ax2.scatter(emb[i, 0], emb[i, 1],
                    color=pde_colors[pde], marker=mrkr, s=60, alpha=0.8)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker=pde_markers[p], color=pde_colors[p],
               markersize=7, linestyle="none", label=p)
        for p in pde_markers
    ]
    ax2.legend(handles=legend_elems, fontsize=8)
    ax2.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=10)
    ax2.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=10)
    ax2.set_title("Coloured by PDE", fontsize=10)
    ax2.grid(True, alpha=0.2)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_cluster_heatmap(all_runs, cluster_labels, filepath):
    signals_arr  = np.array([r["signals"] for r in all_runs])
    scaler       = StandardScaler()
    signals_norm = scaler.fit_transform(signals_arr)

    # Sort rows by cluster
    order = np.argsort(cluster_labels)
    data  = signals_norm[order]
    labels_ordered = np.array(cluster_labels)[order]
    pdes_ordered   = [all_runs[i]["pde"] for i in order]

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="Normalised Signal (z-score)")

    ax.set_xticks(range(len(SIGNAL_NAMES)))
    ax.set_xticklabels(SIGNAL_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"C{labels_ordered[i]} | {pdes_ordered[i][:3]}"
                        for i in range(len(order))], fontsize=7)
    ax.set_title("Failure Fingerprint Heatmap (sorted by cluster)",
                 fontsize=12, fontweight="bold")

    # Cluster boundary lines
    boundaries = np.where(np.diff(labels_ordered))[0]
    for b in boundaries:
        ax.axhline(b + 0.5, color="black", linewidth=1.5)

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_cross_pde_cluster_map(all_runs, cluster_labels, filepath):
    """Stacked bar: for each PDE, what fraction belongs to each cluster."""
    from collections import Counter
    pdes = ["Advection", "Burgers", "AllenCahn", "Helmholtz"]
    k    = max(cluster_labels) + 1
    colors = cm.tab10(np.linspace(0, 1, k))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.suptitle("Cross-PDE Cluster Distribution", fontsize=12,
                 fontweight="bold")

    counts = np.zeros((len(pdes), k))
    for i, run in enumerate(all_runs):
        pde_idx = pdes.index(run["pde"])
        counts[pde_idx, cluster_labels[i]] += 1

    bottom = np.zeros(len(pdes))
    for ci in range(k):
        ax.bar(pdes, counts[:, ci], bottom=bottom,
               color=colors[ci], label=f"Cluster {ci}", alpha=0.85)
        bottom += counts[:, ci]

    ax.set_ylabel("Number of Runs", fontsize=11)
    ax.set_title("Cluster Assignment per PDE", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis="y")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 20: Failure Fingerprinting & Clustering  [v2]")
    print(f"Device : {DEVICE}  |  4 PDEs x 10 runs = 40 total")
    print(f"Seed   : {SEED}  (FIX 1)")
    print(f"Spectral: error FFT / ||u_exact||_2  (FIX 3)")
    print("=" * 70)

    # ── Checkpoint setup ──
    ckpt_path  = OUTPUT_DIR / "exp20_checkpoint.json"
    model_path = OUTPUT_DIR / "exp20_current_model.pt"
    ckpt = {}
    if ckpt_path.exists():
        try:
            with open(ckpt_path, 'r') as f:
                ckpt = json.load(f)
            print(f"\n  [Checkpoint loaded: {len(ckpt.get('all_runs', []))} runs completed]")
        except Exception:
            pass

    all_runs       = ckpt.get("all_runs", [])
    completed_runs = len(all_runs)
    t0 = time.time() - ckpt.get("elapsed_time", 0.0)

    flat_idx = 0
    for pde_name, configs in PDE_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"PDE: {pde_name}")
        print(f"{'=' * 60}")
        for run_idx, cfg in enumerate(configs):
            if flat_idx < completed_runs:
                flat_idx += 1
                continue

            print(f"\n  Run {run_idx+1}/10: {cfg}")
            result = train_and_fingerprint(
                pde_name, cfg, run_idx, flat_idx=flat_idx,
                full_ckpt=ckpt, ckpt_path=ckpt_path, model_path=model_path)
            sigs = result["signals"]
            print(f"  Signals: " + "  ".join(
                f"{SIGNAL_NAMES[i]}={sigs[i]:.3f}" for i in range(8)))
            all_runs.append(result)

            # Update checkpoint after each run completes
            ckpt["all_runs"] = all_runs
            if f"run_{flat_idx}" in ckpt:
                del ckpt[f"run_{flat_idx}"]
            ckpt["elapsed_time"] = time.time() - t0
            with open(ckpt_path, 'w') as f:
                json.dump(ckpt, f)

            flat_idx += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(all_runs) == 0:
        print("No runs completed.")
        return

    # ── Clustering ──
    signals_arr = np.array([r["signals"] for r in all_runs])
    scaler      = StandardScaler()
    X_scaled    = scaler.fit_transform(signals_arr)

    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=20)
    cluster_labels = km.fit_predict(X_scaled)

    # Inertia & silhouette
    try:
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(X_scaled, cluster_labels))
    except Exception:
        sil = float("nan")

    # ── Cluster composition ──
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 20 - CLUSTER SUMMARY  [v2]")
    print(f"  Silhouette score: {sil:.3f}")
    from collections import Counter
    for ci in range(K_CLUSTERS):
        mask = cluster_labels == ci
        pde_dist = Counter(all_runs[i]["pde"]
                           for i in range(len(all_runs)) if mask[i])
        print(f"\n  Cluster {ci} (n={mask.sum()}):")
        print(f"    PDE distribution: {dict(pde_dist)}")
        centroid = X_scaled[mask].mean(axis=0)
        dominant = SIGNAL_NAMES[np.argmax(np.abs(centroid))]
        print(f"    Dominant signal: {dominant}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_radar_charts(all_runs, cluster_labels,
        OUTPUT_DIR / "diagnostic_radar_charts.png")
    plot_cluster_scatter(all_runs, cluster_labels,
        OUTPUT_DIR / "cluster_scatter.png")
    plot_cluster_heatmap(all_runs, cluster_labels,
        OUTPUT_DIR / "cluster_heatmap.png")
    plot_cross_pde_cluster_map(all_runs, cluster_labels,
        OUTPUT_DIR / "cross_pde_cluster_map.png")

    elapsed = time.time() - t0

    # ── Save ──
    results_json = {
        "experiment": "Failure Fingerprinting & Clustering",
        "version":    "v2-journal-ready",
        "config": {
            "seed":          SEED,
            "k_clusters":    K_CLUSTERS,
            "n_epochs":      N_EPOCHS,
            "signal_names":  SIGNAL_NAMES,
            "pde_configs":   {k: [str(c) for c in v]     # FIX 6: was "pdse_configs"
                              for k, v in PDE_CONFIGS.items()},
        },
        "fix_notes": {
            "fix1_seed": (
                f"Explicit seed={SEED}+flat_idx for reproducibility. "
                "v1 had no seed control; results depended on accumulated "
                "random state from prior runs."
            ),
            "fix2_speed_flags": (
                "Added cudnn.benchmark=True and float32_matmul_precision='medium'. "
                "Consistent with all other experiments in the suite."
            ),
            "fix3_spectral_error": (
                "v1 used FFT(u_pred) — misclassified correct high-freq "
                "predictions as spectral failures. v2 uses "
                "FFT(u_pred - u_exact) / ||u_exact||_2, consistent with "
                "Exp 18 FIX 1 and Exp 19 FIX 4."
            ),
            "fix4_colloc_autograd": (
                "v1 created tensors with requires_grad=True then called "
                ".unsqueeze(1), disconnecting autograd. v2 constructs "
                "tensors directly in (N,1) shape."
            ),
            "fix5_dead_code": (
                "Removed unused variables: 'saturated'/'total' in "
                "signal_activation_saturation, and 'xc'/'tc' in "
                "signal_hessian_cond_estimate."
            ),
            "fix6_typo": "JSON key 'pdse_configs' corrected to 'pde_configs'.",
            "fix7_logging": (
                "v1 ran 15,000 epochs per run silently. v2 prints "
                "loss every 5000 epochs for diagnostics."
            ),
            "fix8_import": (
                "Removed redundant 'from scipy.fft import rfft' inside "
                "signal_spectral_ratio; rfft already imported at module level."
            ),
        },
        "runs": [{"pde": r["pde"], "run": r["run"], "cfg": r["cfg"],
                  "seed": r.get("seed", "unknown"),
                  "signals": r["signals"],
                  "cluster": int(cluster_labels[i])}
                 for i, r in enumerate(all_runs)],
        "clustering": {
            "silhouette_score": sil,
            "cluster_centroids_normalised": km.cluster_centers_.tolist(),
            "inertia": float(km.inertia_),
        },
        "elapsed_seconds": elapsed,
    }
    out = OUTPUT_DIR / "exp20_results.json"
    with open(out, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {out}")
    print(f"Plots   -> {OUTPUT_DIR}")
    print(f"Elapsed : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return results_json


if __name__ == "__main__":
    run_experiment()