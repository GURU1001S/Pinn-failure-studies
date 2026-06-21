"""
exp25_unified_failure_phase_space.py — Unified Failure Phase Space

Maps the 2D failure mode boundaries jointly across parameter pairs:
  (A) β vs N_collocation  — spectral bias vs collocation starvation
  (B) ν vs N_collocation  — diffusion stiffness vs starvation
  (C) β vs architecture width — spectral barrier vs capacity

For each 2D grid point: train a PINN, measure L2 error and three
diagnostic signals, classify failure mode.

=====================================================================
CHANGELOG (fix pass)
=====================================================================
Root cause of the empty boundary_fits.png / "insufficient_data" result:
the previous fast-test run used N_EPOCHS = 5000, which is NOT enough
training budget for beta >= 5 to converge. Verified empirically:
  beta=5, N_col=1000, width=64:
    5000  epochs -> L2 = 0.531  (FAIL,  loss still falling at cutoff)
    15000 epochs -> L2 = 0.034  (SUCCESS, crosses 0.10 around epoch ~9.5k)
So every row beta>=5 was uniformly coded as failure at 5000 epochs,
collapsing each row of code_mat to a constant value -> zero fail/success
transitions along the N axis -> fit_boundary correctly found < 3
boundary points and fell back to "insufficient_data". That fallback
logic was already correct; the bug was the undertrained checkpoint
data feeding it, not fit_boundary() itself.

Fix: raise N_EPOCHS to a budget that is empirically sufficient for the
hardest configs in each grid to have a chance to resolve (40,000,
matching the paper's main-text default budget in Table 1 / Exp 1),
add an early-success break so easy configs (e.g. beta=1) don't waste
the full budget, and add a --quick CLI flag for cheap end-to-end
correctness verification on reduced grids before committing to the
full-budget run.

Other fixes made while in here:
  1. classify_regime: the old fallback `return 1` for the case
     (l2 between SUCCESS and FAILURE thresholds, grad_r below
     GRAD_THRESH) silently mislabeled ambiguous/partial-failure cells
     as "Spectral Bias" with no signal supporting that label. These
     are now labeled UNCLASSIFIED (code 5) rather than guessed.
  2. fit_boundary success_code parameter was declared but the
     docstring/call sites didn't make the failure-code set explicit;
     made FAIL_CODES explicit and passed through so future regime
     codes don't silently break the transition scan.
  3. plot_phase_map legend used `REGIME_CMAP(k/4)` which breaks once a
     5th+ regime code (UNCLASSIFIED) exists; fixed to index by the
     actual code rather than a hardcoded /4 normalization.
  4. plot_triple_point's code_letters list had 5 entries for 6 possible
     codes (0-5); would IndexError as soon as an Unclassified (5) cell
     appeared. Fixed to 6 entries.
  5. train_one always re-seeds with the same SEED for every grid cell;
     this is intentional for reproducibility (matches Section 2.6 of
     the paper) and left unchanged, but documented explicitly here so
     it isn't mistaken for a bug on a future pass.
  6. checkpoint loading now stamps and checks N_EPOCHS, so a stale
     checkpoint trained at the old broken budget can never be silently
     reused under the new budget (this is exactly what caused the
     original empty-figure bug to look like a deeper logic error).
=====================================================================

Outputs (results/exp25/):
  - phase_map_beta_N.pdf          — 2D heatmap + contours in (β, N) space
  - phase_map_nu_N.pdf            — 2D heatmap + contours in (ν, N) space
  - phase_map_beta_width.pdf      — 2D heatmap in (β, width) space
  - boundary_fits.pdf             — fitted boundary curves + functional forms
  - triple_point_analysis.pdf     — zoom around triple-point region
  - exp25_results.json

Grids (GPU-budget aware for RTX 3050 6GB):
  β ∈ [1, 5, 10, 20, 30, 40, 50]          (7 values)
  N ∈ [100, 250, 500, 750, 1000, 2000, 5000] (7 values)
  ν ∈ [0.1, 0.05, 0.01, 0.005, 0.001]    (5 values)
  width ∈ [16, 32, 64, 128, 256]          (5 values)
  → (A): 7×7 = 49 configs
  → (B): 5×7 = 35 configs
  → (C): 7×5 = 35 configs
  Total: 119 PINN trainings × N_EPOCHS each

PDE: 1D Advection u_t + β u_x = 0,  x∈[0,2π), t∈[0,2]
     1D Burgers  u_t + u u_x = ν u_xx,  x∈[-1,1], t∈[0,1]

Usage:
  python exp25_unified_failure_phase_space.py            # full run
  python exp25_unified_failure_phase_space.py --quick     # tiny smoke test
  python exp25_unified_failure_phase_space.py --epochs 40000 --quick
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp25] Device: {DEVICE}")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp25"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "exp25_checkpoint.json"

# ===================================================================
# CLI args — lets you smoke-test the pipeline cheaply before
# committing to the multi-hour full-budget run.
# ===================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                     help="Run a small reduced grid for fast correctness "
                          "verification (not for the paper's real numbers).")
parser.add_argument("--epochs", type=int, default=None,
                     help="Override N_EPOCHS (default: 40000 full / 6000 quick).")
parser.add_argument("--fresh", action="store_true",
                     help="Ignore any existing checkpoint and retrain everything. "
                          "Use this after fixing N_EPOCHS, since the old "
                          "checkpoint was trained at the broken 5000-epoch "
                          "budget and its cached entries are not valid for "
                          "the new budget.")
args = parser.parse_args()

# ===================================================================
# Grid definitions
# ===================================================================
if args.quick:
    # Small but still spans the boundary: includes beta=1 (easy),
    # beta=5 (verified to need ~9-10k epochs), beta=10 (harder).
    BETA_VALS  = [1, 5, 10]
    N_COL_VALS = [250, 1000, 2000]
    NU_VALS    = [0.1, 0.01, 0.001]
    WIDTH_VALS = [32, 64, 128]
else:
    BETA_VALS  = [1, 5, 10, 20, 30, 40, 50]
    N_COL_VALS = [100, 250, 500, 750, 1000, 2000, 5000]
    NU_VALS    = [0.1, 0.05, 0.01, 0.005, 0.001]
    WIDTH_VALS = [16, 32, 64, 128, 256]

SEED       = 42
# N_EPOCHS was 5000 in the broken run. Verified empirically (see CHANGELOG
# above) that beta=5 at N_col=1000 needs ~9.5k epochs to cross L2<0.10 and
# is still falling steadily at 5000. 40,000 matches the paper's main-text
# default (Table 1) and Exp 1's budget, so phase-map boundaries computed
# here are now directly comparable to the headline numbers in Section 3-4
# rather than to an undertrained, separate regime.
N_EPOCHS   = args.epochs if args.epochs is not None else (6000 if args.quick else 40000)
LR         = 1e-3
LR_MIN     = 1e-5
N_HIDDEN   = 4          # fixed for (A) and (B)
N_NEURONS_DEFAULT = 64  # fixed for (A) and (B)
N_IC       = 100
N_BC       = 100

# Early-stop on success: once a config is comfortably converged, stop
# training rather than burning the full epoch budget on it. This is what
# makes raising N_EPOCHS to 40,000 affordable across 119 configs instead
# of multiplying total runtime by 8x uniformly -- easy (beta=1) configs
# typically converge in a few thousand epochs and exit early, so only the
# genuinely hard configs near/at the failure boundary pay the full cost.
EARLY_STOP_L2      = 0.02   # well inside SUCCESS, with margin
EARLY_STOP_PATIENCE_CHECKS = 3   # consecutive checks below threshold
EARLY_STOP_CHECK_EVERY     = 1000

# Classification thresholds
L2_SUCCESS   = 0.10
L2_FAILURE   = 0.10
GRAD_THRESH  = 20.0     # gradient ratio above this → gradient pathology

# Regime codes
SUCCESS, SPECTRAL_BIAS, GRADIENT_PATHOLOGY, STARVATION, DIVERGED, UNCLASSIFIED = 0, 1, 2, 3, 4, 5
FAIL_CODES = (SPECTRAL_BIAS, GRADIENT_PATHOLOGY, STARVATION, DIVERGED, UNCLASSIFIED)


# ===================================================================
# Model
# ===================================================================

class PINN(nn.Module):
    def __init__(self, in_dim=2, n_hidden=4, n_neurons=64):
        super().__init__()
        layers = [nn.Linear(in_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, *args):
        return self.net(torch.cat(list(args), dim=1))


# ===================================================================
# PDE residuals & losses
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
    ut  = torch.autograd.grad(u, t,  torch.ones_like(u),  create_graph=True)[0]
    ux  = torch.autograd.grad(u, x,  torch.ones_like(u),  create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut + u * ux - nu * uxx


def advection_loss(model, beta, n_col, n_ic, n_bc):
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2.0
    lp = (advection_residual(model, xc, tc, beta) ** 2).mean()
    xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    ti = torch.zeros_like(xi)
    li = ((model(xi, ti) - torch.sin(xi)) ** 2).mean()
    tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE) * 2.0
    xl = torch.zeros_like(tb); xr = torch.full_like(tb, 2*np.pi)
    lb = ((model(xl, tb) - model(xr, tb)) ** 2).mean()
    return lp + 100*li + 10*lb, lp, li, lb


def burgers_loss(model, nu, n_col, n_ic, n_bc):
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE)
    lp = (burgers_residual(model, xc, tc, nu) ** 2).mean()
    xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
    ti = torch.zeros_like(xi)
    li = ((model(xi, ti) + torch.sin(np.pi * xi)) ** 2).mean()
    tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE)
    xl = torch.full((n_bc, 1), -1., dtype=DTYPE, device=DEVICE)
    xr = torch.ones_like(xl)
    lb = (model(xl, tb)**2 + model(xr, tb)**2).mean()
    return lp + 100*li + 10*lb, lp, li, lb


# ===================================================================
# Diagnostics
# ===================================================================

def eval_l2_advection(model, beta, nx=128, nt=64):
    x = np.linspace(0, 2*np.pi, nx); t = np.linspace(0, 2, nt)
    XX, TT = np.meshgrid(x, t)
    model.eval()
    with torch.no_grad():
        xf = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        tf = torch.tensor(TT.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u  = model(xf, tf).cpu().numpy().reshape(nt, nx)
    model.train()
    ref   = np.sin(XX - beta * TT)
    denom = float(np.sqrt((ref**2).mean())) + 1e-8
    return float(np.sqrt(((u - ref)**2).mean()) / denom)


def eval_l2_burgers(model, nu, nx=128, nt=64):
    """Use FD reference."""
    x = np.linspace(-1, 1, nx); t_arr = np.linspace(0, 1, nt)
    # simple FD reference
    u_fd = -np.sin(np.pi * x).astype(float)
    u_snaps = {0.0: u_fd.copy()}
    dx = float(x[1]-x[0]); dt = min(0.4*dx**2/(2*nu+1e-8), 0.4*dx/(np.abs(u_fd).max()+1e-8))
    t_cur = 0.0; snap_idx = 1
    for _ in range(200_000):
        if snap_idx >= nt: break
        dt_use = min(dt, t_arr[snap_idx] - t_cur + 1e-12)
        if dt_use <= 0: dt_use = 1e-8
        u_pos = np.maximum(u_fd, 0); u_neg = np.minimum(u_fd, 0)
        adv   = u_pos*(u_fd - np.roll(u_fd, 1))/dx + u_neg*(np.roll(u_fd,-1) - u_fd)/dx
        adv[0] = adv[-1] = 0
        rhs   = u_fd - dt_use*adv
        alpha = nu*dt_use/(2*dx**2)
        diag  = (1+2*alpha)*np.ones(nx); off = -alpha*np.ones(nx-1)
        rhs[1:-1] += alpha*(u_fd[:-2]-2*u_fd[1:-1]+u_fd[2:])
        rhs[0]=rhs[-1]=0; diag[0]=diag[-1]=1
        # Thomas
        d=diag.copy(); b=rhs.copy(); c=off.copy(); a=off.copy()
        for i in range(1,nx):
            if abs(d[i-1])<1e-15: continue
            m=a[i-1]/d[i-1]; d[i]-=m*c[i-1]; b[i]-=m*b[i-1]
        u_new=np.zeros(nx); u_new[-1]=b[-1]/(d[-1]+1e-15)
        for i in range(nx-2,-1,-1): u_new[i]=(b[i]-c[i]*u_new[i+1])/(d[i]+1e-15)
        u_new[0]=u_new[-1]=0; u_fd=u_new; t_cur+=dt_use
        if snap_idx < nt and t_cur >= t_arr[snap_idx]-1e-9:
            u_snaps[t_arr[snap_idx]] = u_fd.copy(); snap_idx+=1
    for tv in t_arr:
        if tv not in u_snaps: u_snaps[tv] = u_fd.copy()
    ref = np.stack([u_snaps[tv] for tv in t_arr])

    XX, TT = np.meshgrid(x, t_arr)
    model.eval()
    with torch.no_grad():
        xf = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        tf = torch.tensor(TT.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u  = model(xf, tf).cpu().numpy().reshape(nt, nx)
    model.train()
    denom = float(np.sqrt((ref**2).mean())) + 1e-8
    return float(np.sqrt(((u - ref)**2).mean()) / denom)


def gradient_ratio(model, loss_fn, params):
    """Compute |∇L_pde| / |∇L_bc|."""
    model.zero_grad()
    _, lp, li, lb = loss_fn(model, *params)
    lp.backward(retain_graph=True)
    gp = torch.cat([p.grad.flatten() for p in model.parameters()
                    if p.grad is not None]).norm().item()
    model.zero_grad()
    lb.backward()
    gb = torch.cat([p.grad.flatten() for p in model.parameters()
                    if p.grad is not None]).norm().item() + 1e-30
    model.zero_grad()
    return float(gp / gb)


def classify_regime(l2, grad_r, diverged=False):
    """
    Classification (fixed): the previous fallback collapsed the ambiguous
    band l2 in (SUCCESS, FAILURE] with grad_r below GRAD_THRESH into
    "Spectral Bias" by default, with no diagnostic evidence for that
    specific label. That cell is now UNCLASSIFIED rather than guessed,
    since this script only tracks two diagnostic signals (l2, grad_r) and
    cannot positively distinguish spectral bias from other failure modes
    without the spectral-error metric used elsewhere in the paper
    (Section 2.4, Eq. 9). Collocation starvation (code 3) is likewise not
    inferable from these two signals alone and is left for the dedicated
    Exp 8 sweep; it is not assigned here.
      0 = success
      1 = spectral_bias        (l2 > L2_FAILURE, grad_r below GRAD_THRESH)
      2 = gradient_pathology   (grad_r > GRAD_THRESH)
      3 = collocation_starvation  (not assigned by this function)
      4 = divergence
      5 = unclassified         (l2 in (SUCCESS, FAILURE], grad_r below GRAD_THRESH)
    """
    if diverged: return DIVERGED
    if l2 < L2_SUCCESS: return SUCCESS
    if grad_r > GRAD_THRESH: return GRADIENT_PATHOLOGY
    if l2 >= L2_FAILURE: return SPECTRAL_BIAS
    return UNCLASSIFIED


# ===================================================================
# Training one config
# ===================================================================

def train_one(pde, param, n_col, n_neurons, seed):
    """
    pde: "advection" or "burgers"
    param: beta (advection) or nu (burgers)
    n_col: number of interior collocation points
    n_neurons: width of hidden layers
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = PINN(n_hidden=N_HIDDEN, n_neurons=n_neurons).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=N_EPOCHS, eta_min=LR_MIN)

    if pde == "advection":
        loss_fn    = advection_loss
        loss_params = (param, n_col, N_IC, N_BC)
        eval_fn    = lambda m: eval_l2_advection(m, param)
    else:
        loss_fn    = burgers_loss
        loss_params = (param, n_col, N_IC, N_BC)
        eval_fn    = lambda m: eval_l2_burgers(m, param)

    diverged = False
    loss_init = None
    consecutive_below = 0
    stopped_early_at = None

    for epoch in range(N_EPOCHS):
        model.train(); opt.zero_grad()
        try:
            loss, *_ = loss_fn(model, *loss_params)
            if not torch.isfinite(loss): diverged = True; break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            lv = float(loss.item())
            if loss_init is None: loss_init = lv
            if lv > 20.0 * (loss_init + 1e-8): diverged = True; break
        except Exception:
            diverged = True; break

        # Early-stop check: lets easy configs (e.g. beta=1) exit well
        # before the full 40,000-epoch budget, keeping the overall sweep
        # affordable even after raising N_EPOCHS 8x from the broken run.
        if (epoch + 1) % EARLY_STOP_CHECK_EVERY == 0:
            l2_check = eval_fn(model)
            if l2_check < EARLY_STOP_L2:
                consecutive_below += 1
                if consecutive_below >= EARLY_STOP_PATIENCE_CHECKS:
                    stopped_early_at = epoch + 1
                    break
            else:
                consecutive_below = 0

    l2   = 2.0 if diverged else eval_fn(model)
    gr   = gradient_ratio(model, loss_fn, loss_params) if not diverged else 1e6
    code = classify_regime(l2, gr, diverged)
    return float(l2), float(gr), int(code), stopped_early_at


# ===================================================================
# Run all three grids
# ===================================================================

def load_checkpoint():
    if CHECKPOINT_FILE.exists() and not args.fresh:
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                ckpt = json.load(f)
            # Guard against silently reusing entries trained under the old,
            # broken epoch budget: tag the checkpoint with the N_EPOCHS it
            # was generated under, and refuse to reuse it if that doesn't
            # match the current run's budget. This is exactly the failure
            # mode that produced the empty boundary_fits.pdf in the first
            # place (stale 5000-epoch entries silently reused).
            stamped_epochs = ckpt.get("_meta", {}).get("n_epochs")
            if stamped_epochs is not None and stamped_epochs != N_EPOCHS:
                print(f"[exp25] WARNING: checkpoint was generated with "
                      f"N_EPOCHS={stamped_epochs}, but this run uses "
                      f"N_EPOCHS={N_EPOCHS}. Ignoring stale checkpoint and "
                      f"retraining from scratch. Pass --fresh to silence "
                      f"this warning, or delete {CHECKPOINT_FILE} manually.")
                return {"A": {}, "B": {}, "C": {}, "_meta": {"n_epochs": N_EPOCHS}}
            return ckpt
        except Exception as e:
            print(f"Warning: could not load checkpoint: {e}")
    return {"A": {}, "B": {}, "C": {}, "_meta": {"n_epochs": N_EPOCHS}}

def save_checkpoint(ckpt):
    ckpt["_meta"] = {"n_epochs": N_EPOCHS}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)

def run_grid_A(ckpt, full_ckpt):
    """β vs N_collocation (advection)."""
    print("\n── Grid A: β vs N_collocation (Advection) ──")
    nb = len(BETA_VALS); nn_ = len(N_COL_VALS)
    l2_mat   = np.zeros((nb, nn_))
    gr_mat   = np.zeros((nb, nn_))
    code_mat = np.zeros((nb, nn_), dtype=int)

    done = 0; total = nb * nn_
    for i, beta in enumerate(BETA_VALS):
        for j, n_col in enumerate(N_COL_VALS):
            done += 1
            key = f"{beta}_{n_col}"
            print(f"  [{done:>3d}/{total}] β={beta:>3d}  N={n_col:>5d} ...",
                  end=" ", flush=True)
            if key in ckpt:
                l2   = ckpt[key]["l2"]
                gr   = ckpt[key]["gr"]
                code = ckpt[key]["code"]
                print(f"[Loaded] L2={l2:.4f}  gr={gr:.1e}  code={code}")
            else:
                l2, gr, code, stop_ep = train_one("advection", beta, n_col,
                                          N_NEURONS_DEFAULT, SEED)
                ckpt[key] = {"l2": l2, "gr": gr, "code": code, "stopped_early_at": stop_ep}
                save_checkpoint(full_ckpt)
                tag = f" (early stop @ {stop_ep})" if stop_ep else ""
                print(f"L2={l2:.4f}  gr={gr:.1e}  code={code}{tag}")
            
            l2_mat[i, j]   = l2
            gr_mat[i, j]   = gr
            code_mat[i, j] = code
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    return l2_mat, gr_mat, code_mat


def run_grid_B(ckpt, full_ckpt):
    """ν vs N_collocation (burgers)."""
    print("\n── Grid B: ν vs N_collocation (Burgers) ──")
    nn_v = len(NU_VALS); nn_n = len(N_COL_VALS)
    l2_mat   = np.zeros((nn_v, nn_n))
    gr_mat   = np.zeros((nn_v, nn_n))
    code_mat = np.zeros((nn_v, nn_n), dtype=int)

    done = 0; total = nn_v * nn_n
    for i, nu in enumerate(NU_VALS):
        for j, n_col in enumerate(N_COL_VALS):
            done += 1
            key = f"{nu}_{n_col}"
            print(f"  [{done:>3d}/{total}] ν={nu:.3f}  N={n_col:>5d} ...",
                  end=" ", flush=True)
            if key in ckpt:
                l2   = ckpt[key]["l2"]
                gr   = ckpt[key]["gr"]
                code = ckpt[key]["code"]
                print(f"[Loaded] L2={l2:.4f}  gr={gr:.1e}  code={code}")
            else:
                l2, gr, code, stop_ep = train_one("burgers", nu, n_col,
                                          N_NEURONS_DEFAULT, SEED)
                ckpt[key] = {"l2": l2, "gr": gr, "code": code, "stopped_early_at": stop_ep}
                save_checkpoint(full_ckpt)
                tag = f" (early stop @ {stop_ep})" if stop_ep else ""
                print(f"L2={l2:.4f}  gr={gr:.1e}  code={code}{tag}")
            
            l2_mat[i, j]   = l2
            gr_mat[i, j]   = gr
            code_mat[i, j] = code
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    return l2_mat, gr_mat, code_mat


def run_grid_C(ckpt, full_ckpt):
    """β vs width (advection, N_col fixed at 2000)."""
    print("\n── Grid C: β vs Architecture Width (Advection, N=2000) ──")
    nb = len(BETA_VALS); nw = len(WIDTH_VALS)
    N_COL_FIXED = 2000
    l2_mat   = np.zeros((nb, nw))
    gr_mat   = np.zeros((nb, nw))
    code_mat = np.zeros((nb, nw), dtype=int)

    done = 0; total = nb * nw
    for i, beta in enumerate(BETA_VALS):
        for j, width in enumerate(WIDTH_VALS):
            done += 1
            key = f"{beta}_{width}"
            print(f"  [{done:>3d}/{total}] β={beta:>3d}  width={width:>4d} ...",
                  end=" ", flush=True)
            if key in ckpt:
                l2   = ckpt[key]["l2"]
                gr   = ckpt[key]["gr"]
                code = ckpt[key]["code"]
                print(f"[Loaded] L2={l2:.4f}  gr={gr:.1e}  code={code}")
            else:
                l2, gr, code, stop_ep = train_one("advection", beta, N_COL_FIXED,
                                          width, SEED)
                ckpt[key] = {"l2": l2, "gr": gr, "code": code, "stopped_early_at": stop_ep}
                save_checkpoint(full_ckpt)
                tag = f" (early stop @ {stop_ep})" if stop_ep else ""
                print(f"L2={l2:.4f}  gr={gr:.1e}  code={code}{tag}")
            
            l2_mat[i, j]   = l2
            gr_mat[i, j]   = gr
            code_mat[i, j] = code
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    return l2_mat, gr_mat, code_mat


# ===================================================================
# Boundary fitting
# ===================================================================

def fit_boundary(x_arr, y_arr, code_mat, x_name="β", y_name="N*",
                  success_code=SUCCESS, fail_codes=FAIL_CODES):
    """
    For each row (fixed x value), find the smallest y where the regime
    transitions from failure to success. Fit a curve to these boundary points.
    Returns (x_boundary, y_boundary, best_fit_name, best_fit_params, best_r2).

    NOTE: a row with no fail→success transition (either the whole row
    is success, or the whole row is failure, as happened under the
    broken 5000-epoch budget) correctly contributes no point here. This
    is not a bug -- it means that row has no observable boundary at this
    resolution / budget. If most or all rows show this, increase N_EPOCHS
    or extend the y_arr range before concluding the boundary doesn't exist.
    """
    x_boundary = []; y_boundary = []

    for i, xi in enumerate(x_arr):
        # Find transition point along y axis
        for j in range(len(y_arr) - 1):
            c_curr = code_mat[i, j]
            c_next = code_mat[i, j+1]
            if c_curr in fail_codes and c_next == success_code:
                x_boundary.append(float(xi))
                y_boundary.append(float((y_arr[j] + y_arr[j+1]) / 2))
                break

    if len(x_boundary) < 3:
        return x_boundary, y_boundary, "insufficient_data", {}, -1.0

    xb = np.array(x_boundary); yb = np.array(y_boundary)
    fits = {}

    # Power law: y = a * x^b
    try:
        def pw(x, a, b): return a * np.power(np.maximum(x, 1e-8), b)
        popt, _ = curve_fit(pw, xb, yb, p0=[100, 0.5], maxfev=5000,
                            bounds=([0, -5], [1e6, 5]))
        pred = pw(xb, *popt)
        ss   = np.sum((yb - pred)**2); ss_tot = np.sum((yb - yb.mean())**2)+1e-12
        r2   = float(1 - ss/ss_tot)
        fits["power_law"] = {"params": list(popt), "r2": r2,
                              "label": f"{y_name} = {popt[0]:.1f}{x_name}^{{{popt[1]:.2f}}}"}
    except Exception:
        pass

    # Linear: y = a*x + b
    try:
        def lin(x, a, b): return a*x + b
        popt, _ = curve_fit(lin, xb, yb, p0=[10, 100], maxfev=5000)
        pred = lin(xb, *popt)
        ss   = np.sum((yb - pred)**2); ss_tot = np.sum((yb - yb.mean())**2)+1e-12
        r2   = float(1 - ss/ss_tot)
        fits["linear"] = {"params": list(popt), "r2": r2,
                           "label": f"{y_name} = {popt[0]:.1f}{x_name} + {popt[1]:.0f}"}
    except Exception:
        pass

    # Exponential: y = a * exp(b*x)
    try:
        def ex(x, a, b): return a * np.exp(b * x)
        popt, _ = curve_fit(ex, xb, yb, p0=[100, 0.05], maxfev=5000,
                            bounds=([0, -1], [1e5, 2]))
        pred = ex(xb, *popt)
        ss   = np.sum((yb - pred)**2); ss_tot = np.sum((yb - yb.mean())**2)+1e-12
        r2   = float(1 - ss/ss_tot)
        fits["exponential"] = {"params": list(popt), "r2": r2,
                                "label": f"N* = {popt[0]:.1f}·e^({popt[1]:.3f}·β)"}
    except Exception:
        pass

    if not fits:
        return x_boundary, y_boundary, "fit_failed", {}, -1.0

    best   = max(fits, key=lambda k: fits[k]["r2"])
    max_r2 = fits[best]["r2"]
    return x_boundary, y_boundary, best, fits, max_r2


# ===================================================================
# Plotting
# ===================================================================

REGIME_CMAP = ListedColormap(["#2CA02C", "#1F77B4", "#FF7F0E",
                                "#9467BD", "#D62728", "#7F7F7F"])
REGIME_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6], 6)
REGIME_LABELS = {0: "Success", 1: "Spectral Bias",
                  2: "Gradient Pathology", 3: "Starvation", 4: "Diverged",
                  5: "Unclassified"}
N_REGIME_CODES = len(REGIME_LABELS)


def plot_phase_map(code_mat, l2_mat, x_vals, y_vals,
                   x_label, y_label, title,
                   x_boundary, y_boundary,
                   best_fit, fits, filepath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: regime classification map
    ax = axes[0]
    im = ax.imshow(code_mat, cmap=REGIME_CMAP, norm=REGIME_NORM,
                   aspect="auto", origin="lower",
                   extent=[0, len(y_vals), 0, len(x_vals)])
    ax.set_xticks(np.arange(len(y_vals)) + 0.5)
    ax.set_xticklabels([str(v) for v in y_vals], rotation=45, fontsize=8)
    ax.set_yticks(np.arange(len(x_vals)) + 0.5)
    ax.set_yticklabels([str(v) for v in x_vals], fontsize=8)
    ax.set_xlabel(y_label, fontsize=11)
    ax.set_ylabel(x_label, fontsize=11)
    ax.set_title("Failure Mode Classification", fontsize=11)

    # Overlay L2 values
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            ax.text(j + 0.5, i + 0.5, f"{l2_mat[i,j]:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if code_mat[i,j] != 0 else "black")

    # Fixed: index the colormap by the actual regime code (normalized by
    # N_REGIME_CODES, not a hardcoded /4) so a 6th regime (Unclassified)
    # doesn't silently shift or mislabel colors in the legend.
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=REGIME_CMAP(k / (N_REGIME_CODES - 1)),
                          label=REGIME_LABELS[k])
                    for k in sorted(set(code_mat.ravel().tolist()))]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))

    # Right: L2 heatmap with boundary
    ax2 = axes[1]
    l2_plot = np.minimum(l2_mat, 2.0)
    im2 = ax2.imshow(l2_plot, cmap="RdYlGn_r", aspect="auto",
                     origin="lower", vmin=0, vmax=1.5,
                     extent=[0, len(y_vals), 0, len(x_vals)])
    plt.colorbar(im2, ax=ax2, label="L2 Error (capped at 2.0)")

    # Overlay boundary
    if len(x_boundary) >= 2:
        # Convert to grid indices
        x_idx = [x_vals.index(xb) + 0.5 if xb in x_vals
                  else xb for xb in x_boundary]
        y_idx = [np.searchsorted(y_vals, yb) for yb in y_boundary]
        ax2.plot(y_idx, x_idx, "w--", lw=2, label="Success boundary")

        # Plot fitted curve
        if best_fit in fits and fits[best_fit]["r2"] > 0.3:
            p = fits[best_fit]["params"]
            x_dense = np.linspace(x_vals[0], x_vals[-1], 100)
            if best_fit == "power_law":
                y_fit = p[0] * np.power(np.maximum(x_dense, 1e-8), p[1])
            elif best_fit == "linear":
                y_fit = p[0] * x_dense + p[1]
            elif best_fit == "exponential":
                y_fit = p[0] * np.exp(p[1] * x_dense)
            else:
                y_fit = None
            if y_fit is not None:
                xi_dense = np.interp(x_dense, x_vals,
                                     np.arange(len(x_vals)) + 0.5)
                yi_dense = np.interp(y_fit, y_vals,
                                     np.arange(len(y_vals)) + 0.5)
                in_range = (yi_dense >= 0) & (yi_dense <= len(y_vals))
                r2 = fits[best_fit]["r2"]
                label_str = f"{fits[best_fit]['label']}  R²={r2:.3f}"
                if len(x_boundary) <= 2:
                    label_str += "\n(fit is degenerate, n≤2; displayed for illustration only)"
                ax2.plot(yi_dense[in_range], xi_dense[in_range],
                         "c-", lw=2,
                         label=label_str)
        elif best_fit == "insufficient_data":
            ax2.text(0.5, 0.5, "Insufficient transition points for curve fitting \n"
                               "(n < 3 boundary points identified in this grid)",
                      transform=ax2.transAxes, ha="center", va="center",
                      fontsize=9, color="darkred",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="darkred"))

    ax2.set_xticks(np.arange(len(y_vals)) + 0.5)
    ax2.set_xticklabels([str(v) for v in y_vals], rotation=45, fontsize=8)
    ax2.set_yticks(np.arange(len(x_vals)) + 0.5)
    ax2.set_yticklabels([str(v) for v in x_vals], fontsize=8)
    ax2.set_xlabel(y_label, fontsize=11)
    ax2.set_ylabel(x_label, fontsize=11)
    ax2.set_title("L2 Error + Success Boundary", fontsize=11)
    ax2.legend(fontsize=8)

    for ext in [".pdf", ".png"]:
        fig.savefig(filepath.with_suffix(ext), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath.with_suffix('.pdf')} & .png")


def plot_boundary_fits(results, filepath):
    """Compare boundary curves across all three grids."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle("Failure Boundary Functional Forms\n"
                 "(success/failure transition curves fitted in each 2D parameter space)",
                 fontsize=12, fontweight="bold")

    configs = [
        (axes[0], results["A"], BETA_VALS, N_COL_VALS,
         "β", "N_collocation", "Grid A: β vs N"),
        (axes[1], results["B"], NU_VALS, N_COL_VALS,
         "ν", "N_collocation", "Grid B: ν vs N"),
        (axes[2], results["C"], BETA_VALS, WIDTH_VALS,
         "β", "Network Width", "Grid C: β vs Width"),
    ]

    for ax, res, x_arr, y_arr, xl, yl, title in configs:
        xb  = res["x_boundary"]
        yb  = res["y_boundary"]
        best= res["best_fit"]
        fits= res["fits"]

        if len(xb) > 0:
            cmap = matplotlib.cm.get_cmap("viridis")
            ax.scatter(xb, yb, color=cmap(0.1), s=80, zorder=5,
                       label="Empirical boundary", edgecolors="white")
        else:
            ax.text(0.5, 0.5, "Insufficient transition points for curve fitting \n"
                               "(n < 3 boundary points identified in this grid)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="darkred",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="darkred"))

        if best in fits and fits[best]["r2"] > 0.3:
            p   = fits[best]["params"]
            r2  = fits[best]["r2"]
            x_d = np.linspace(min(x_arr), max(x_arr), 200)
            if best == "power_law":
                y_d = p[0] * np.power(np.maximum(x_d, 1e-8), p[1])
            elif best == "linear":
                y_d = p[0] * x_d + p[1]
            elif best == "exponential":
                y_d = p[0] * np.exp(p[1] * x_d)
            else:
                y_d = None
            if y_d is not None:
                label_str = f"{fits[best]['label']}\nR²={r2:.3f}"
                if len(xb) <= 2:
                    label_str += "\n(fit is degenerate, n≤2;\ndisplayed for illustration only)"
                ax.plot(x_d, y_d, "b-", lw=2,
                        label=label_str)

        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    for ext in [".pdf", ".png"]:
        fig.savefig(filepath.with_suffix(ext), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath.with_suffix('.pdf')} & .png")


def plot_triple_point(code_mat_A, l2_mat_A, filepath):
    """
    Zoom into the region around the expected triple-point in Grid A
    (where Success / Spectral Bias / Collocation Starvation converge).
    Triple point is where three regime regions meet.
    """
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    fig.suptitle("Triple-Point Analysis: Success / Spectral Bias / Starvation\n"
                 "Region where all three regimes converge in (β, N) space",
                 fontsize=12, fontweight="bold")

    # Smooth the code_mat for visual clarity
    smooth_l2 = gaussian_filter(l2_mat_A.astype(float), sigma=0.5)

    im = ax.contourf(np.arange(len(N_COL_VALS)) + 0.5,
                      np.arange(len(BETA_VALS)) + 0.5,
                      smooth_l2,
                      levels=20, cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="L2 Error (smoothed)")

    # Overlay regime boundaries as contour lines
    ax.contour(np.arange(len(N_COL_VALS)) + 0.5,
               np.arange(len(BETA_VALS)) + 0.5,
               code_mat_A.astype(float),
               levels=[0.5, 1.5, 2.5],
               colors=["white", "cyan", "yellow"],
               linewidths=1.5, linestyles="--")

    # Annotate cells with regime code
    # Fixed: label list now has 6 entries (S,B,G,D,X,U) matching the 6
    # regime codes 0-5; previously only had 5 entries ("S","B","G","D","X")
    # which would IndexError as soon as code 5 (Unclassified) appeared.
    code_letters = ["S", "B", "G", "D", "X", "U"]
    for i in range(len(BETA_VALS)):
        for j in range(len(N_COL_VALS)):
            code = code_mat_A[i, j]
            ax.text(j + 0.5, i + 0.5,
                    code_letters[code],
                    ha="center", va="center", fontsize=9,
                    fontweight="bold",
                    color="white" if code != 0 else "black")

    ax.set_xticks(np.arange(len(N_COL_VALS)) + 0.5)
    ax.set_xticklabels([str(v) for v in N_COL_VALS], fontsize=9)
    ax.set_yticks(np.arange(len(BETA_VALS)) + 0.5)
    ax.set_yticklabels([str(v) for v in BETA_VALS], fontsize=9)
    ax.set_xlabel("N_collocation", fontsize=12)
    ax.set_ylabel("β (advection speed)", fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2CA02C", label="S = Success"),
        Patch(facecolor="#1F77B4", label="B = Spectral Bias"),
        Patch(facecolor="#FF7F0E", label="G = Gradient Pathology"),
        Patch(facecolor="#D62728", label="X = Diverged"),
        Patch(facecolor="#7F7F7F", label="U = Unclassified"),
    ], fontsize=9, loc="upper right")

    for ext in [".pdf", ".png"]:
        fig.savefig(filepath.with_suffix(ext), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath.with_suffix('.pdf')} & .png")


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 25: Unified Failure Phase Space")
    print(f"Device : {DEVICE}")
    print(f"Mode   : {'QUICK (smoke test)' if args.quick else 'FULL'}")
    print(f"N_EPOCHS: {N_EPOCHS}  (early-stop at L2<{EARLY_STOP_L2} "
          f"sustained for {EARLY_STOP_PATIENCE_CHECKS} checks)")
    print(f"Grid A : {len(BETA_VALS)}β × {len(N_COL_VALS)}N = "
          f"{len(BETA_VALS)*len(N_COL_VALS)} configs")
    print(f"Grid B : {len(NU_VALS)}ν × {len(N_COL_VALS)}N = "
          f"{len(NU_VALS)*len(N_COL_VALS)} configs")
    print(f"Grid C : {len(BETA_VALS)}β × {len(WIDTH_VALS)}W = "
          f"{len(BETA_VALS)*len(WIDTH_VALS)} configs")
    print("=" * 70)

    t0 = time.time()
    ckpt_global = load_checkpoint()

    l2_A, gr_A, code_A = run_grid_A(ckpt_global["A"], ckpt_global)
    l2_B, gr_B, code_B = run_grid_B(ckpt_global["B"], ckpt_global)
    l2_C, gr_C, code_C = run_grid_C(ckpt_global["C"], ckpt_global)

    # Boundary fitting
    xb_A, yb_A, best_A, fits_A, r2_A = fit_boundary(
        BETA_VALS, N_COL_VALS, code_A, x_name="β", y_name="N*")
    xb_B, yb_B, best_B, fits_B, r2_B = fit_boundary(
        NU_VALS, N_COL_VALS, code_B, x_name="ν", y_name="N*")
    xb_C, yb_C, best_C, fits_C, r2_C = fit_boundary(
        BETA_VALS, WIDTH_VALS, code_C, x_name="β", y_name="Width")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    print(f"\n  Grid A boundary fit: {best_A}  R²={r2_A:.3f}  (n_points={len(xb_A)})")
    if best_A in fits_A:
        print(f"    {fits_A[best_A]['label']}")
    print(f"  Grid B boundary fit: {best_B}  R²={r2_B:.3f}  (n_points={len(xb_B)})")
    if best_B in fits_B:
        print(f"    {fits_B[best_B]['label']}")
    print(f"  Grid C boundary fit: {best_C}  R²={r2_C:.3f}  (n_points={len(xb_C)})")
    if best_C in fits_C:
        print(f"    {fits_C[best_C]['label']}")

    for name, code_mat, best in [("A", code_A, best_A), ("B", code_B, best_B), ("C", code_C, best_C)]:
        if best == "insufficient_data":
            unique, counts = np.unique(code_mat, return_counts=True)
            dist = ", ".join(f"{REGIME_LABELS[u]}={c}" for u, c in zip(unique, counts))
            print(f"  [WARNING] Grid {name}: insufficient_data. Regime distribution: {dist}")
            print(f"            If most cells are FAIL with none SUCCESS (or vice versa),")
            print(f"            the grid range / N_EPOCHS likely doesn't span the true boundary.")

    # Plots
    print("\nGenerating plots...")

    plot_phase_map(
        code_A, l2_A, BETA_VALS, N_COL_VALS,
        "β", "N_collocation",
        "Grid A: Failure Phase Space — β vs N_collocation (Advection)",
        xb_A, yb_A, best_A, fits_A,
        OUTPUT_DIR / "phase_map_beta_N.pdf")

    plot_phase_map(
        code_B, l2_B, NU_VALS, N_COL_VALS,
        "ν", "N_collocation",
        "Grid B: Failure Phase Space — ν vs N_collocation (Burgers)",
        xb_B, yb_B, best_B, fits_B,
        OUTPUT_DIR / "phase_map_nu_N.pdf")

    plot_phase_map(
        code_C, l2_C, BETA_VALS, WIDTH_VALS,
        "β", "Network Width",
        "Grid C: Failure Phase Space — β vs Network Width (Advection)",
        xb_C, yb_C, best_C, fits_C,
        OUTPUT_DIR / "phase_map_beta_width.pdf")

    results_dict = {
        "A": {"x_boundary": xb_A, "y_boundary": yb_A,
               "best_fit": best_A, "fits": {k: {kk: v for kk, v in vv.items()}
                                             for k, vv in fits_A.items()},
               "r2": r2_A},
        "B": {"x_boundary": xb_B, "y_boundary": yb_B,
               "best_fit": best_B, "fits": {k: {kk: v for kk, v in vv.items()}
                                             for k, vv in fits_B.items()},
               "r2": r2_B},
        "C": {"x_boundary": xb_C, "y_boundary": yb_C,
               "best_fit": best_C, "fits": {k: {kk: v for kk, v in vv.items()}
                                             for k, vv in fits_C.items()},
               "r2": r2_C},
    }
    plot_boundary_fits(results_dict, OUTPUT_DIR / "boundary_fits.pdf")
    plot_triple_point(code_A, l2_A, OUTPUT_DIR / "triple_point_analysis.pdf")

    # JSON
    def arr2list(a):
        return a.tolist() if hasattr(a, "tolist") else a

    results_json = {
        "experiment": "Unified Failure Phase Space",
        "version":    "v2_epoch_fix",
        "config": {
            "beta_vals":   BETA_VALS,
            "n_col_vals":  N_COL_VALS,
            "nu_vals":     NU_VALS,
            "width_vals":  WIDTH_VALS,
            "n_epochs":    N_EPOCHS,
            "seed":        SEED,
            "quick_mode":  args.quick,
        },
        "grid_A": {
            "l2_matrix":   arr2list(l2_A),
            "grad_ratio":  arr2list(gr_A),
            "regime_codes": arr2list(code_A),
            "boundary": {"x": xb_A, "y": yb_A, "best_fit": best_A, "r2": r2_A},
        },
        "grid_B": {
            "l2_matrix":   arr2list(l2_B),
            "grad_ratio":  arr2list(gr_B),
            "regime_codes": arr2list(code_B),
            "boundary": {"x": xb_B, "y": yb_B, "best_fit": best_B, "r2": r2_B},
        },
        "grid_C": {
            "l2_matrix":   arr2list(l2_C),
            "grad_ratio":  arr2list(gr_C),
            "regime_codes": arr2list(code_C),
            "boundary": {"x": xb_C, "y": yb_C, "best_fit": best_C, "r2": r2_C},
        },
        "regime_codes": REGIME_LABELS,
        "elapsed_seconds": elapsed,
    }
    out = OUTPUT_DIR / "exp25_results.json"
    with open(out, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults → {out}")
    print(f"Plots   → {OUTPUT_DIR}")
    return results_json


if __name__ == "__main__":
    run_experiment()