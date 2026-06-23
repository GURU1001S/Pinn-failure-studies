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
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp18] Device: {DEVICE}")
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp18"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BETA         = 30
SEED         = 42
N_HIDDEN     = 4
N_NEURONS    = 128
ACTIVATION   = "tanh"
N_EPOCHS     = 50_000
LR           = 1e-3
LR_MIN       = 1e-5
N_COLLOCATION = 8_000
N_IC         = 300
N_BC         = 300
TRACK_EVERY  = 200
NTK_CHECKPOINTS  = [0, 10_000, 30_000, 50_000]
NTK_N_POINTS     = 150
NTK_MAX_PARAMS   = 30_000
N_FREQ_BANDS = 16
NX_EVAL      = 256
NT_EVAL      = 128
class AdvectionPINN(nn.Module):
    def __init__(self, n_hidden=4, n_neurons=128, activation="tanh"):
        super().__init__()
        act_map = {"tanh": nn.Tanh, "silu": nn.SiLU, "relu": nn.ReLU}
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
def exact(x_np, t_np):
    return np.sin(x_np - BETA * t_np)
def pde_res(model, x, t):
    x = x.requires_grad_(True); t = t.requires_grad_(True)
    u  = model(x, t)
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return ut + BETA * ux
def compute_losses(model, n_col, n_ic, n_bc):
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2.0
    l_pde = (pde_res(model, xc, tc) ** 2).mean()
    xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    ti = torch.zeros(n_ic, 1, dtype=DTYPE, device=DEVICE)
    l_ic = ((model(xi, ti) - torch.sin(xi)) ** 2).mean()
    tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE) * 2.0
    xl = torch.zeros(n_bc, 1, dtype=DTYPE, device=DEVICE)
    xr = torch.full_like(xl, 2 * np.pi)
    l_bc = ((model(xl, tb) - model(xr, tb)) ** 2).mean()
    return l_pde, l_ic, l_bc
def spectral_error_per_band(model, n_freq=N_FREQ_BANDS,
                             nx=NX_EVAL, nt=NT_EVAL):
    x_vals = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    t_vals = np.linspace(0, 2.0, nt)
    XX, TT = np.meshgrid(x_vals, t_vals)
    model.eval()
    with torch.no_grad():
        xf = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        tf = torch.tensor(TT.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        u_pred = model(xf, tf).cpu().numpy().reshape(XX.shape)
    model.train()
    u_ex      = exact(XX, TT)
    err_field = u_pred - u_ex
    global_norm = np.linalg.norm(u_ex) + 1e-10
    err_fft = np.fft.rfft(err_field, axis=1)
    band_errors = []
    for k in range(1, n_freq + 1):
        masked       = np.zeros_like(err_fft)
        masked[:, k] = err_fft[:, k]
        err_k_spatial = np.fft.irfft(masked, n=nx, axis=1)
        band_err      = float(np.linalg.norm(err_k_spatial) / global_norm)
        band_errors.append(band_err)
    return np.array(band_errors)
def gradient_ratio(model, n_col=500, n_bc=200):
    model.zero_grad()
    xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
    tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2.0
    l_pde = (pde_res(model, xc, tc) ** 2).mean()
    l_pde.backward()
    g_pde = torch.cat([p.grad.detach().flatten()
                       for p in model.parameters() if p.grad is not None])
    norm_pde = g_pde.norm().item()
    model.zero_grad()
    tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE) * 2.0
    xl = torch.zeros(n_bc, 1, dtype=DTYPE, device=DEVICE)
    xr = torch.full_like(xl, 2 * np.pi)
    l_bc = ((model(xl, tb) - model(xr, tb)) ** 2).mean()
    l_bc.backward()
    g_bc = torch.cat([p.grad.detach().flatten()
                      for p in model.parameters() if p.grad is not None])
    norm_bc = g_bc.norm().item() + 1e-30
    model.zero_grad()
    return float(norm_pde / norm_bc)
def ntk_eigenvalues(model, n_pts=NTK_N_POINTS, max_params=NTK_MAX_PARAMS):
    model.eval()
    xs = torch.linspace(0, 2 * np.pi, n_pts, dtype=DTYPE,
                        device=DEVICE).unsqueeze(1)
    ts = torch.ones(n_pts, 1, dtype=DTYPE, device=DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    total  = sum(p.numel() for p in params)
    rows = []
    for i in range(n_pts):
        model.zero_grad()
        u = model(xs[i:i+1], ts[i:i+1]); u.backward()
        g = torch.cat([p.grad.detach().flatten() if p.grad is not None
                       else torch.zeros(p.numel(), device=DEVICE)
                       for p in params])
        rows.append(g.cpu()); model.zero_grad()
    J = torch.stack(rows).numpy()
    if total > max_params:
        idx = np.random.choice(total, max_params, replace=False)
        J   = J[:, idx]
    K    = J @ J.T
    eigs = np.linalg.eigvalsh(K)[::-1]
    eigs = np.maximum(eigs, 0)
    pos  = eigs[eigs > 1e-30]
    cond = float(eigs[0] / (pos[-1] if len(pos) else 1e-30))
    log_k  = np.log(np.arange(1, len(pos) + 1, dtype=float))
    log_e  = np.log(pos)
    slope  = np.linalg.lstsq(
        np.vstack([log_k, np.ones_like(log_k)]).T, log_e, rcond=None)[0][0]
    model.train()
    return eigs, cond, float(-slope)
def pearson_r(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
def bootstrap_pvalue(a, b, n_boot=1000, seed=0):
    rng    = np.random.RandomState(seed)
    r_obs  = abs(pearson_r(a, b))
    count  = 0
    for _ in range(n_boot):
        b_perm = rng.permutation(b)
        if abs(pearson_r(a, b_perm)) >= r_obs:
            count += 1
    return float(count / n_boot)
def lag_xcorr(a, b, max_lag=50):
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    lags  = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag >= 0:
            c = np.dot(a[lag:], b[:n - lag]) / n
        else:
            c = np.dot(a[:n + lag], b[-lag:]) / n
        corrs.append(c)
    best = int(np.argmax(np.abs(corrs)))
    return list(lags)[best], float(np.max(np.abs(corrs)))
def plot_spectral_trajectories(history, filepath):
    steps_rec = history["steps_recorded"]
    spec_arr  = np.array(history["spectral_errors"])
    freqs     = np.arange(1, N_FREQ_BANDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(f"Spectral Bias Trajectory (β={BETA})", fontsize=13,
                 fontweight="bold")
    ax = axes[0]
    vmax = float(np.percentile(spec_arr, 99))
    im   = ax.pcolormesh(steps_rec, freqs, spec_arr.T,
                         cmap="inferno", shading="auto",
                         vmin=0, vmax=vmax)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Frequency k", fontsize=11)
    ax.set_title("Relative Spectral Error vs Training\n"
                 "(clipped at 99th pct for visibility)", fontsize=10)
    plt.colorbar(im, ax=ax, label=f"Rel. Error (vmax={vmax:.3f})")
    ax2 = axes[1]
    colors = cm.viridis(np.linspace(0, 1, N_FREQ_BANDS))
    for k in range(N_FREQ_BANDS):
        ax2.semilogy(steps_rec, spec_arr[:, k] + 1e-8,
                     color=colors[k], lw=1.2,
                     label=f"k={k+1}" if k in [0, 3, 7, 15] else None)
    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Relative Spectral Error", fontsize=11)
    ax2.set_title("Per-Frequency Convergence\n"
                  "FIX 1: normalized by ||u_exact||₂ (bounded in [0,~2])",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
def plot_gradient_ratio(history, filepath):
    steps = history["steps_recorded"]
    ratio = history["grad_ratios"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(f"Gradient Magnitude Ratio |∇L_pde|/|∇L_bc|  (β={BETA})",
                 fontsize=12, fontweight="bold")
    ax = axes[0]
    ax.semilogy(steps, np.array(ratio) + 1e-12, color="#E64040", lw=1.5)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="ratio=1")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Gradient Ratio", fontsize=11)
    ax.set_title("Gradient Ratio Over Training\n"
                 "(sustained >1 = PDE dominates IC/BC gradients)", fontsize=10)
    ax.legend(); ax.grid(True, alpha=0.25)
    ax2 = axes[1]
    ax2.semilogy(steps, history["loss_pde"],  label="PDE",  color="#E64040", lw=1.5)
    ax2.semilogy(steps, history["loss_ic"],   label="IC",   color="#3A7FD5", lw=1.5)
    ax2.semilogy(steps, history["loss_bc"],   label="BC",   color="#2CA02C", lw=1.5)
    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("Loss Component Trajectories", fontsize=11)
    ax2.legend(); ax2.grid(True, alpha=0.25)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
def plot_ntk_spectra(ntk_data, filepath):
    ck_list = sorted(ntk_data.keys())
    n       = len(ck_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    fig.suptitle(f"NTK Eigenvalue Spectra at Checkpoints (β={BETA})",
                 fontsize=12, fontweight="bold")
    if n == 1: axes = [axes]
    colors = cm.plasma(np.linspace(0.2, 0.9, n))
    conds = [ntk_data[ck]["condition_number"] for ck in ck_list]
    monotone = all(conds[i] >= conds[i+1] for i in range(len(conds)-1))
    for i, ck in enumerate(ck_list):
        eigs  = ntk_data[ck]["eigenvalues"]
        cond  = ntk_data[ck]["condition_number"]
        alpha = ntk_data[ck]["decay_exponent"]
        k_idx = np.arange(1, len(eigs) + 1)
        ax    = axes[i]
        ax.loglog(k_idx, np.array(eigs) + 1e-30,
                  color=colors[i], lw=1.5)
        title_str = f"Iter {ck}\nκ={cond:.1e}  α={alpha:.2f}"
        if i > 0 and cond > conds[i-1] * 3:
            title_str += "\n⚠ κ jump — potential instability"
            ax.set_facecolor("#FFF9C4")
        ax.set_title(title_str, fontsize=9)
        ax.set_xlabel("Index k"); ax.set_ylabel("λ_k")
        ax.grid(True, alpha=0.2, which="both")
    if not monotone:
        pass
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
def plot_joint_phase_portrait(history, filepath):
    steps     = np.array(history["steps_recorded"])
    spec_mean = np.array(history["spectral_errors"]).mean(axis=1)
    spec_high = np.array(history["spectral_errors"])[:, -1]
    grad_log  = np.log10(np.array(history["grad_ratios"]) + 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        f"Joint Phase Portrait: Spectral Failure × Gradient Pathology (β={BETA})",
        fontsize=12, fontweight="bold")
    for ax, spec_x, lbl in [
        (axes[0], spec_mean, "Mean Spectral Error (all k)"),
        (axes[1], spec_high, f"Spectral Error (k={N_FREQ_BANDS})"),
    ]:
        sc = ax.scatter(spec_x, grad_log, c=steps, cmap="viridis",
                        s=18, alpha=0.8)
        n    = len(spec_x)
        step = max(1, n // 20)
        for i in range(0, n - step, step):
            ax.annotate("",
                        xy=(spec_x[i + step], grad_log[i + step]),
                        xytext=(spec_x[i], grad_log[i]),
                        arrowprops=dict(arrowstyle="->",
                                        color="grey", lw=0.8))
        plt.colorbar(sc, ax=ax, label="Training Step")
        ax.set_xlabel(lbl, fontsize=10)
        ax.set_ylabel("log₁₀(Gradient Ratio)", fontsize=10)
        ax.set_title(lbl, fontsize=10)
        ax.grid(True, alpha=0.25)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
def plot_correlation_analysis(history, r_sg, pval_sg,
                               r_sp, pval_sp,
                               lag_sg, xcorr_peak,
                               filepath):
    steps     = np.array(history["steps_recorded"])
    spec_mean = np.array(history["spectral_errors"]).mean(axis=1)
    grad_rat  = np.array(history["grad_ratios"])
    l_pde     = np.array(history["loss_pde"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        f"Correlation & Causal Link Analysis (β={BETA})",
        fontsize=11, fontweight="bold")
    ax = axes[0][0]
    ax.scatter(spec_mean, np.log10(grad_rat + 1e-12),
               c=steps, cmap="viridis", s=14, alpha=0.7)
    sig_str = f"p={pval_sg:.3f}"
    ax.set_xlabel("Mean Spectral Error"); ax.set_ylabel("log₁₀(Grad Ratio)")
    ax.set_title(f"Spectral vs Gradient Ratio\nr={r_sg:.3f}  {sig_str}",
                 fontsize=10)
    ax.grid(True, alpha=0.2)
    ax2 = axes[0][1]
    ax2.scatter(np.log10(l_pde + 1e-12), spec_mean,
                c=steps, cmap="plasma", s=14, alpha=0.7)
    sig_str2 = f"p={pval_sp:.3f}"
    ax2.set_xlabel("log₁₀(PDE Loss)"); ax2.set_ylabel("Mean Spectral Error")
    ax2.set_title(f"PDE Loss vs Spectral Error\nr={r_sp:.3f}  {sig_str2}",
                  fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax3 = axes[1][0]
    n     = len(spec_mean)
    lags  = range(-50, 51)
    a_n   = (spec_mean - spec_mean.mean()) / (spec_mean.std() + 1e-12)
    b_n   = (grad_rat  - grad_rat.mean())  / (grad_rat.std()  + 1e-12)
    xcorrs = []
    for lag in lags:
        if lag >= 0:
            c = np.dot(a_n[lag:], b_n[:n - lag]) / n
        else:
            c = np.dot(a_n[:n + lag], b_n[-lag:]) / n
        xcorrs.append(c)
    ax3.plot(list(lags), xcorrs, color="#E64040", lw=1.5)
    ax3.axvline(lag_sg, color="black", ls="--", lw=1,
                label=f"max lag={lag_sg}")
    ax3.axhline(0, color="grey", lw=0.8)
    ax3.set_xlabel("Lag (recording steps)"); ax3.set_ylabel("Cross-correlation")
    ax3.set_title(f"Spectral ↔ Gradient Cross-correlation\n"
                  f"Peak lag={lag_sg}×{TRACK_EVERY} steps  "
                  f"max_xcorr={xcorr_peak:.3f}", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.2)
    ax4 = axes[1][1]
    labels = ["Spec Error", "log Grad Ratio", "log PDE Loss"]
    arr    = np.column_stack([
        spec_mean,
        np.log(grad_rat + 1e-12),
        np.log(l_pde + 1e-12),
    ])
    C  = np.corrcoef(arr.T)
    im = ax4.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax4.set_xticks(range(3)); ax4.set_yticks(range(3))
    ax4.set_xticklabels(labels, rotation=30, fontsize=8)
    ax4.set_yticklabels(labels, fontsize=8)
    for ii in range(3):
        for jj in range(3):
            ax4.text(jj, ii, f"{C[ii,jj]:.2f}", ha="center", va="center",
                     fontsize=9)
    plt.colorbar(im, ax=ax4)
    ax4.set_title("Failure Mode Correlation Matrix", fontsize=10)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
def run_experiment():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 70)
    print("EXPERIMENT 18: Compound Failure Interaction Map  [v2]")
    print(f"Device : {DEVICE}   β={BETA}   epochs={N_EPOCHS}")
    print(f"Seed   : {SEED}   (FIX 4)")
    print(f"Spectral metric: band-filtered L2 / ||u_exact||₂  (FIX 1)")
    print("=" * 70)
    model     = AdvectionPINN(N_HIDDEN, N_NEURONS, ACTIVATION).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)
    ckpt_path = OUTPUT_DIR / "exp18_checkpoint.json"
    model_ckpt_path = OUTPUT_DIR / "exp18_model_checkpoint.pt"
    ckpt = {}
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r") as f:
                ckpt = json.load(f)
            print(f"  [Loaded checkpoint from {ckpt_path.name}]")
        except Exception:
            ckpt = {}
    history = {
        "steps_recorded": [],
        "spectral_errors": [],
        "grad_ratios":     [],
        "loss_pde":        [],
        "loss_ic":         [],
        "loss_bc":         [],
    }
    ntk_data = {}
    start_epoch = 0
    elapsed_so_far = 0.0
    if ckpt and "completed_epoch" in ckpt:
        start_epoch = ckpt["completed_epoch"] + 1
        history = ckpt.get("history", history)
        ntk_data = {int(k): v for k, v in ckpt.get("ntk_data", {}).items()}
        elapsed_so_far = ckpt.get("elapsed_seconds", 0.0)
        if model_ckpt_path.exists():
            model.load_state_dict(torch.load(model_ckpt_path, map_location=DEVICE))
            print(f"  [Resuming from epoch {start_epoch - 1}]")
        for _ in range(start_epoch - 1):
            scheduler.step()
    if start_epoch <= N_EPOCHS:
        for cp in NTK_CHECKPOINTS:
            if cp < start_epoch and cp not in ntk_data:
                print(f"\n  NTK @ epoch {cp} (computed on resume)...")
                eigs, cond, alpha = ntk_eigenvalues(model)
                ntk_data[cp] = {
                    "eigenvalues":      eigs[:100].tolist(),
                    "condition_number": cond,
                    "decay_exponent":   alpha,
                }
                print(f"    κ={cond:.3e}  α={alpha:.3f}")
        t0 = time.time()
        for epoch in range(start_epoch, N_EPOCHS + 1):
            if epoch in NTK_CHECKPOINTS:
                if epoch not in ntk_data:
                    print(f"\n  NTK @ epoch {epoch}...")
                    eigs, cond, alpha = ntk_eigenvalues(model)
                    ntk_data[epoch] = {
                        "eigenvalues":      eigs[:100].tolist(),
                        "condition_number": cond,
                        "decay_exponent":   alpha,
                    }
                    print(f"    κ={cond:.3e}  α={alpha:.3f}")
                else:
                    print(f"\n  NTK @ epoch {epoch} already computed. Skipping.")
            if epoch == N_EPOCHS:
                break
            model.train()
            optimizer.zero_grad()
            l_pde, l_ic, l_bc = compute_losses(model, N_COLLOCATION, N_IC, N_BC)
            total = l_pde + 100 * l_ic + 10 * l_bc
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            if epoch % TRACK_EVERY == 0:
                spec_err = spectral_error_per_band(model)
                grad_r   = gradient_ratio(model)
                history["steps_recorded"].append(epoch)
                history["spectral_errors"].append(spec_err.tolist())
                history["grad_ratios"].append(grad_r)
                history["loss_pde"].append(float(l_pde.item()))
                history["loss_ic"].append(float(l_ic.item()))
                history["loss_bc"].append(float(l_bc.item()))
                if epoch % 5000 == 0:
                    print(f"  Epoch {epoch:>6d}: pde={l_pde.item():.3e}  "
                          f"grad_r={grad_r:.2e}  "
                          f"spec_mean={spec_err.mean():.4f}")
            if epoch % 5000 == 0 or epoch == N_EPOCHS:
                torch.save(model.state_dict(), model_ckpt_path)
                ckpt["completed_epoch"] = epoch
                ckpt["history"] = history
                ckpt["ntk_data"] = {str(k): v for k, v in ntk_data.items()}
                ckpt["elapsed_seconds"] = (time.time() - t0) + elapsed_so_far
                with open(ckpt_path, "w") as f:
                    json.dump(ckpt, f, indent=2)
        elapsed = (time.time() - t0) + elapsed_so_far
    else:
        elapsed = elapsed_so_far
    print(f"\nTraining done in {elapsed:.1f}s")
    spec_mean = np.array(history["spectral_errors"]).mean(axis=1)
    grad_rat  = np.array(history["grad_ratios"])
    l_pde_arr = np.array(history["loss_pde"])
    r_sg      = pearson_r(spec_mean, np.log(grad_rat + 1e-12))
    pval_sg   = bootstrap_pvalue(spec_mean, np.log(grad_rat + 1e-12))
    r_sp      = pearson_r(spec_mean, np.log(l_pde_arr + 1e-12))
    pval_sp   = bootstrap_pvalue(spec_mean, np.log(l_pde_arr + 1e-12))
    lag_sg, xcorr_peak = lag_xcorr(spec_mean, grad_rat)
    if abs(lag_sg) <= 2:
        causal_link = "correlated_simultaneous"
    elif lag_sg > 2:
        causal_link = "spectral_leads_gradient"
    else:
        causal_link = "gradient_leads_spectral"
    sg_significant = pval_sg < 0.05
    if not sg_significant:
        causal_link = "not_significant"
    conds    = [ntk_data[ck]["condition_number"] for ck in sorted(ntk_data)]
    ntk_jump = not all(conds[i] >= conds[i+1] for i in range(len(conds)-1))
    print("\nGenerating plots...")
    plot_spectral_trajectories(history,
        OUTPUT_DIR / "spectral_error_trajectories.png")
    plot_gradient_ratio(history,
        OUTPUT_DIR / "gradient_ratio_trajectory.png")
    plot_ntk_spectra(ntk_data,
        OUTPUT_DIR / "ntk_spectra_checkpoints.png")
    plot_joint_phase_portrait(history,
        OUTPUT_DIR / "joint_phase_portrait.png")
    plot_correlation_analysis(
        history, r_sg, pval_sg, r_sp, pval_sp, lag_sg, xcorr_peak,
        OUTPUT_DIR / "correlation_analysis.png")
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 18 — SUMMARY  [v2]")
    print(f"  Spectral error range (k=1): "
          f"{np.array(history['spectral_errors'])[:,0].mean():.4f}")
    print(f"  Spectral error range (k=16): "
          f"{np.array(history['spectral_errors'])[:,15].mean():.4f}")
    print(f"  Pearson r (spec vs grad): {r_sg:.3f}  p={pval_sg:.3f} "
          f"({'sig.' if sg_significant else 'n.s.'})")
    print(f"  Causal link: {causal_link}")
    print(f"  NTK κ: {' → '.join(f'{c:.2e}' for c in conds)}")
    print(f"  NTK non-monotone jump: {ntk_jump}")
    results = {
        "experiment": "Compound Failure Interaction Map",
        "version":    "v2-journal-ready",
        "config":     {
            "beta":          BETA,
            "seed":          SEED,
            "n_epochs":      N_EPOCHS,
            "track_every":   TRACK_EVERY,
            "spectral_metric": "band_filtered_L2_over_global_norm",
        },
        "fix_notes": {
            "fix1_spectral": (
                "v1 computed err_fft[:,k]/(exact_fft[:,k]+1e-10). "
                "For sin(x-βt), exact_fft[:,k]≈0 for k≥2, so the denominator "
                "collapsed to 1e-10 producing values of ~1e10. "
                "v2 uses ||IFFT(masked_err_fft_k)||₂/||u_exact||₂, "
                "which is bounded in [0,~2]."
            ),
            "fix2_correlation": (
                "v1 reported r=0.082 but xcorr_peak=0.976 without reconciling. "
                "With broken spectral metric, both were invalid. "
                "v2 reports corrected Pearson r on corrected spectral errors, "
                "adds bootstrap p-values, and explicitly flags non-significance."
            ),
            "fix3_ntk_jump": (
                "NTK κ increased 10× at final checkpoint (iter=50k). "
                "v2 detects and annotates this as potential numerical instability."
            ),
            "fix4_seed":  f"Explicit seed={SEED} added.",
            "fix5_colormap": (
                "v1 heatmap vmax dominated by 1e10 noise — entirely black. "
                "v2 clips at 99th percentile for visibility."
            ),
        },
        "ntk_data": {
            str(k): {kk: v for kk, v in vv.items() if kk != "eigenvalues"}
            for k, vv in ntk_data.items()
        },
        "ntk_non_monotone_jump":   ntk_jump,
        "correlation": {
            "pearson_spectral_vs_grad":  r_sg,
            "pvalue_spectral_vs_grad":   pval_sg,
            "significant_p05":           sg_significant,
            "pearson_spectral_vs_pde":   r_sp,
            "pvalue_spectral_vs_pde":    pval_sp,
            "peak_xcorr_lag_steps":      lag_sg,
            "peak_xcorr_value":          float(xcorr_peak),
            "causal_link_assessment":    causal_link,
        },
        "final_spectral_error_per_freq": history["spectral_errors"][-1],
        "final_grad_ratio":    history["grad_ratios"][-1],
        "elapsed_seconds":     elapsed,
    }
    out = OUTPUT_DIR / "exp18_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out}")
    return results
if __name__ == "__main__":
    run_experiment()
