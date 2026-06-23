import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json, time, copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[exp22_2] Device: {DEVICE}")
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "exp22_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N_SEEDS        = 4
N_INTERVENTIONS = 5
N_FAILED_RUNS   = 20
FAIL_EPOCHS    = 10_000
FAIL_LR        = 1e-3
FAIL_LR_MIN    = 1e-5
FAIL_N_COL     = 3_000
FAIL_N_IC      = 150
FAIL_N_BC      = 150
RECOVER_EPOCHS = 10_000
RECOVER_LR     = 5e-4
RECOVER_LR_MIN = 1e-6
RECOVER_N_COL  = 5_000
RECOVERED_THRESH = 0.10
PARTIAL_THRESH   = 0.40
GLOBAL_SEED = 42
FAILURE_MODES = [
    "SpectralBias",
    "GradientPathology",
    "SharpInterface",
    "HighFreqHelmholtz",
    "OptimStagnation",
]
FAILURE_COLORS = {
    "SpectralBias":      "#1F77B4",
    "GradientPathology": "#FF7F0E",
    "SharpInterface":    "#2CA02C",
    "HighFreqHelmholtz": "#9467BD",
    "OptimStagnation":   "#D62728",
}
INTERVENTION_LABELS = [
    "I1: 10× Collocation",
    "I2: Fourier Features",
    "I3: L-BFGS",
    "I4: Loss Reweighting",
    "I5: Domain Decomp.",
]
OUTCOME_COLORS = {
    "RECOVERED": "#2CA02C",
    "PARTIAL":   "#F59F00",
    "FAILED":    "#D62728",
}
FAILURE_CONFIGS = []
for seed in range(N_SEEDS):
    FAILURE_CONFIGS.append(
        {"mode": "SpectralBias",      "pde": "Advection",
         "beta": 50,    "seed": seed})
    FAILURE_CONFIGS.append(
        {"mode": "GradientPathology", "pde": "Burgers",
         "nu": 0.001,   "seed": seed})
    FAILURE_CONFIGS.append(
        {"mode": "SharpInterface",    "pde": "AllenCahn",
         "eps2": 0.005, "seed": seed})
    FAILURE_CONFIGS.append(
        {"mode": "HighFreqHelmholtz", "pde": "Helmholtz",
         "k2": 400,     "seed": seed})
    FAILURE_CONFIGS.append(
        {"mode": "OptimStagnation",   "pde": "Advection",
         "beta": 10, "n_hidden": 1, "n_neurons": 16, "seed": seed})
assert len(FAILURE_CONFIGS) == N_FAILED_RUNS,    f"Expected {N_FAILED_RUNS} configs, got {len(FAILURE_CONFIGS)}"
class StandardPINN(nn.Module):
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
class FourierFeaturePINN(nn.Module):
    def __init__(self, in_dim=2, n_hidden=4, n_neurons=64,
                 n_fourier=128, sigma=1.0):
        super().__init__()
        self.register_buffer(
            "B", torch.randn(in_dim, n_fourier, dtype=DTYPE) * sigma)
        feat_dim = 2 * n_fourier
        layers   = [nn.Linear(feat_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, *args):
        x    = torch.cat(list(args), dim=1)
        proj = x @ self.B
        feat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)
        return self.net(feat)
def res_advection(model, x, t, beta):
    x = x.requires_grad_(True); t = t.requires_grad_(True)
    u  = model(x, t)
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return ut + beta * ux
def res_burgers(model, x, t, nu):
    x = x.requires_grad_(True); t = t.requires_grad_(True)
    u   = model(x, t)
    ut  = torch.autograd.grad(u, t,  torch.ones_like(u),  create_graph=True)[0]
    ux  = torch.autograd.grad(u, x,  torch.ones_like(u),  create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut + u * ux - nu * uxx
def res_allen_cahn(model, x, t, eps2):
    x = x.requires_grad_(True); t = t.requires_grad_(True)
    u   = model(x, t)
    ut  = torch.autograd.grad(u, t,  torch.ones_like(u),  create_graph=True)[0]
    ux  = torch.autograd.grad(u, x,  torch.ones_like(u),  create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut - eps2 * uxx - u + u ** 3
def res_helmholtz(model, x, y, k2):
    x  = x.requires_grad_(True); y = y.requires_grad_(True)
    u  = model(x, y)
    ux = torch.autograd.grad(u,  x, torch.ones_like(u),  create_graph=True)[0]
    uxx= torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    uy = torch.autograd.grad(u,  y, torch.ones_like(u),  create_graph=True)[0]
    uyy= torch.autograd.grad(uy, y, torch.ones_like(uy), create_graph=True)[0]
    a  = np.pi
    f  = (k2 - 2*a**2) * torch.sin(a*x) * torch.sin(a*y)
    return uxx + uyy + k2 * u - f
def build_loss(model, cfg, n_col, n_ic, n_bc,
               w_pde=1.0, w_ic=100.0, w_bc=10.0):
    pde = cfg["pde"]
    if pde == "Advection":
        xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
        tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2.0
        lp = (res_advection(model, xc, tc, cfg["beta"]) ** 2).mean()
        xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 * np.pi
        ti = torch.zeros_like(xi)
        li = ((model(xi, ti) - torch.sin(xi)) ** 2).mean()
        tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE) * 2.0
        xl = torch.zeros(n_bc, 1, dtype=DTYPE, device=DEVICE)
        xr = torch.full_like(xl, 2*np.pi)
        lb = ((model(xl, tb) - model(xr, tb)) ** 2).mean()
    elif pde == "Burgers":
        xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
        tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE)
        lp = (res_burgers(model, xc, tc, cfg["nu"]) ** 2).mean()
        xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
        ti = torch.zeros_like(xi)
        li = ((model(xi, ti) + torch.sin(np.pi * xi)) ** 2).mean()
        tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE)
        xl = torch.full((n_bc, 1), -1.0, dtype=DTYPE, device=DEVICE)
        xr = torch.ones_like(xl)
        lb = (model(xl, tb)**2 + model(xr, tb)**2).mean()
    elif pde == "AllenCahn":
        xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
        tc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE)
        lp = (res_allen_cahn(model, xc, tc, cfg["eps2"]) ** 2).mean()
        xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
        ti = torch.zeros_like(xi)
        u_ic = xi**2 * torch.cos(np.pi * xi)
        li = ((model(xi, ti) - u_ic) ** 2).mean()
        tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE)
        xl = torch.full((n_bc, 1), -1.0, dtype=DTYPE, device=DEVICE)
        xr = torch.ones_like(xl)
        lb = (model(xl, tb)**2 + model(xr, tb)**2).mean()
    elif pde == "Helmholtz":
        xc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
        yc = torch.rand(n_col, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
        lp = (res_helmholtz(model, xc, yc, cfg["k2"]) ** 2).mean()
        xi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
        yi = torch.rand(n_ic, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
        u_e = torch.sin(np.pi * xi) * torch.sin(np.pi * yi)
        li = ((model(xi, yi) - u_e) ** 2).mean()
        tb = torch.rand(n_bc, 1, dtype=DTYPE, device=DEVICE) * 2 - 1
        wall = torch.ones(n_bc, 1, dtype=DTYPE, device=DEVICE)
        lb = (model( wall, tb)**2 + model(-wall, tb)**2
              + model(tb,  wall)**2 + model(tb, -wall)**2).mean()
    else:
        raise ValueError(f"Unknown PDE: {pde}")
    return w_pde*lp + w_ic*li + w_bc*lb, lp, li, lb
def _thomas(lower, diag, upper, rhs):
    n = len(rhs); d = diag.copy(); b = rhs.copy()
    c = upper.copy(); a = lower.copy()
    for i in range(1, n):
        if abs(d[i-1]) < 1e-15: continue
        m = a[i-1]/d[i-1]; d[i] -= m*c[i-1]; b[i] -= m*b[i-1]
    x = np.zeros(n); x[-1] = b[-1]/(d[-1]+1e-15)
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - c[i]*x[i+1])/(d[i]+1e-15)
    return x
def burgers_fd_reference(x_vals, t_vals, nu):
    nx = len(x_vals); dx = float(x_vals[1]-x_vals[0])
    u  = -np.sin(np.pi * x_vals).astype(float)
    u[0] = 0.0; u[-1] = 0.0
    snaps = {t_vals[0]: u.copy()}
    t_cur = float(t_vals[0]); snap_idx = 1
    for _ in range(500_000):
        if snap_idx >= len(t_vals): break
        u_max  = np.abs(u).max() + 1e-8
        dt_cfl = 0.4 * dx / u_max
        dt_diff= 0.4 * dx**2 / (2*nu) if nu > 0 else 1.0
        dt     = min(dt_cfl, dt_diff, t_vals[snap_idx] - t_cur + 1e-12)
        if dt <= 0: dt = 1e-8
        u_pos = np.maximum(u, 0); u_neg = np.minimum(u, 0)
        adv   = u_pos*(u - np.roll(u,1))/dx + u_neg*(np.roll(u,-1) - u)/dx
        adv[0] = 0.0; adv[-1] = 0.0
        rhs   = u - dt*adv
        alpha = nu*dt/(2*dx**2)
        diag  = (1+2*alpha)*np.ones(nx)
        off   = -alpha*np.ones(nx-1)
        rhs[1:-1] += alpha*(u[:-2]-2*u[1:-1]+u[2:])
        rhs[0] = 0.0; rhs[-1] = 0.0; diag[0] = 1.0; diag[-1] = 1.0
        u = _thomas(off, diag, off, rhs)
        u[0] = 0.0; u[-1] = 0.0; t_cur += dt
        if snap_idx < len(t_vals) and t_cur >= t_vals[snap_idx]-1e-9:
            snaps[t_vals[snap_idx]] = u.copy(); snap_idx += 1
    for tv in t_vals:
        if tv not in snaps: snaps[tv] = u.copy()
    return np.stack([snaps[tv] for tv in t_vals])
def evaluate_l2(model, cfg, nx=64, nt=64):
    pde = cfg["pde"]
    model.eval()
    if pde in ("Advection", "Burgers", "AllenCahn"):
        x_lo = 0.0 if pde == "Advection" else -1.0
        x_hi = 2*np.pi if pde == "Advection" else 1.0
        t_hi = 2.0 if pde == "Advection" else 1.0
        x_vals = np.linspace(x_lo, x_hi, nx)
        t_vals = np.linspace(0, t_hi, nt)
        XX, TT = np.meshgrid(x_vals, t_vals)
        xf = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        tf = torch.tensor(TT.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            u_pred = model(xf, tf).cpu().numpy().reshape(nt, nx)
        if pde == "Advection":
            u_ref = np.sin(XX - cfg["beta"] * TT)
        elif pde == "Burgers":
            u_ref = burgers_fd_reference(x_vals, t_vals, cfg["nu"])
        else:
            u = (x_vals**2 * np.cos(np.pi * x_vals)).astype(float)
            snaps = {0.0: u.copy()}
            dx_ac = float(x_vals[1]-x_vals[0]); t_cur = 0.0
            dt_ac = min(0.4*dx_ac**2 / (2*cfg["eps2"]+1e-8), 0.01)
            for _ in range(int(t_hi/dt_ac)+2):
                if t_cur >= t_hi: break
                uxx = (np.roll(u,-1) - 2*u + np.roll(u,1))/dx_ac**2
                u   = u + dt_ac*(cfg["eps2"]*uxx + u - u**3)
                u[0] = 0.0; u[-1] = 0.0; t_cur += dt_ac
                nearest_idx = np.argmin(np.abs(t_vals - t_cur))
                if abs(t_cur - t_vals[nearest_idx]) < dt_ac/2:
                    snaps[t_vals[nearest_idx]] = u.copy()
            for tv in t_vals:
                if tv not in snaps: snaps[tv] = u.copy()
            u_ref = np.stack([snaps.get(tv, u) for tv in t_vals])
    elif pde == "Helmholtz":
        x_vals = np.linspace(-1, 1, nx)
        y_vals = np.linspace(-1, 1, nx)
        XX, YY = np.meshgrid(x_vals, y_vals)
        xf = torch.tensor(XX.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        yf = torch.tensor(YY.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            u_pred = model(xf, yf).cpu().numpy().reshape(nx, nx)
        u_ref  = np.sin(np.pi*XX) * np.sin(np.pi*YY)
    model.train()
    denom = float(np.sqrt((u_ref**2).mean())) + 1e-8
    return float(np.sqrt(((u_pred - u_ref)**2).mean()) / denom)
def classify_outcome(l2):
    if l2 < RECOVERED_THRESH:  return "RECOVERED"
    if l2 < PARTIAL_THRESH:    return "PARTIAL"
    return "FAILED"
def produce_failed_model(cfg):
    torch.manual_seed(GLOBAL_SEED + cfg["seed"] * 100)
    np.random.seed(GLOBAL_SEED + cfg["seed"] * 100)
    n_h = cfg.get("n_hidden", 4); n_n = cfg.get("n_neurons", 64)
    model = StandardPINN(n_hidden=n_h, n_neurons=n_n).to(DEVICE)
    if cfg["mode"] == "OptimStagnation":
        for m in model.net:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -3.0, 3.0)
                nn.init.constant_(m.bias, 0.5)
    opt = torch.optim.Adam(model.parameters(), lr=FAIL_LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=FAIL_EPOCHS, eta_min=FAIL_LR_MIN)
    loss_init = None
    for epoch in range(FAIL_EPOCHS):
        model.train(); opt.zero_grad()
        try:
            loss, *_ = build_loss(model, cfg, FAIL_N_COL, FAIL_N_IC, FAIL_N_BC)
            if not torch.isfinite(loss): break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            lv = float(loss.item())
            if loss_init is None: loss_init = lv
        except Exception:
            break
    return model, evaluate_l2(model, cfg)
def _train_adam_recovery(model, cfg, n_col, n_ic, n_bc,
                          seed_offset=0,
                          w_pde=1.0, w_ic=100.0, w_bc=10.0):
    torch.manual_seed(GLOBAL_SEED + seed_offset)
    opt = torch.optim.Adam(model.parameters(), lr=RECOVER_LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=RECOVER_EPOCHS, eta_min=RECOVER_LR_MIN)
    traj = []
    for epoch in range(RECOVER_EPOCHS):
        model.train(); opt.zero_grad()
        try:
            loss, *_ = build_loss(model, cfg, n_col, n_ic, n_bc,
                                   w_pde, w_ic, w_bc)
            if not torch.isfinite(loss): break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            if epoch % 500 == 0:
                traj.append(float(loss.item()))
        except Exception:
            break
    return traj
def recover_i1(model_ckpt, cfg, run_idx):
    model = copy.deepcopy(model_ckpt)
    traj  = _train_adam_recovery(
        model, cfg,
        n_col=RECOVER_N_COL * 10,
        n_ic=FAIL_N_IC * 5, n_bc=FAIL_N_BC * 5,
        seed_offset=run_idx * 10 + 1)
    return model, traj
def recover_i2(model_ckpt, cfg, run_idx):
    torch.manual_seed(GLOBAL_SEED + run_idx * 10 + 2)
    n_h    = cfg.get("n_hidden", 4); n_n = cfg.get("n_neurons", 64)
    sigma  = max(1.0, float(cfg.get("beta", cfg.get("k2", 10))) / 10)
    ff_mod = FourierFeaturePINN(n_hidden=n_h, n_neurons=n_n,
                                 n_fourier=128, sigma=sigma).to(DEVICE)
    traj   = _train_adam_recovery(
        ff_mod, cfg,
        n_col=RECOVER_N_COL, n_ic=FAIL_N_IC*3, n_bc=FAIL_N_BC*3,
        seed_offset=run_idx * 10 + 2)
    return ff_mod, traj
def recover_i3(model_ckpt, cfg, run_idx):
    torch.manual_seed(GLOBAL_SEED + run_idx * 10 + 3)
    model = copy.deepcopy(model_ckpt)
    opt   = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe")
    traj  = []
    n_lbfgs = RECOVER_EPOCHS // 20
    for step in range(n_lbfgs):
        xc_fix = torch.rand(RECOVER_N_COL, 1, dtype=DTYPE, device=DEVICE)
        tc_fix = torch.rand(RECOVER_N_COL, 1, dtype=DTYPE, device=DEVICE)
        xi_fix = torch.rand(FAIL_N_IC*2, 1, dtype=DTYPE, device=DEVICE)
        ti_fix = torch.zeros_like(xi_fix)
        tb_fix = torch.rand(FAIL_N_BC*2, 1, dtype=DTYPE, device=DEVICE)
        loss_container = [None]
        def closure():
            opt.zero_grad()
            try:
                pde = cfg["pde"]
                if pde == "Advection":
                    xc_ = xc_fix * 2*np.pi; tc_ = tc_fix * 2.0
                    lp  = (res_advection(model, xc_, tc_, cfg["beta"])**2).mean()
                    xl  = torch.zeros_like(tb_fix)
                    xr  = torch.full_like(tb_fix, 2*np.pi)
                    lb  = ((model(xl, tb_fix) - model(xr, tb_fix))**2).mean()
                    li  = ((model(xi_fix*2*np.pi, ti_fix) - torch.sin(xi_fix*2*np.pi))**2).mean()
                elif pde == "Burgers":
                    xc_ = xc_fix*2-1; tc_ = tc_fix
                    lp  = (res_burgers(model, xc_, tc_, cfg["nu"])**2).mean()
                    xl  = torch.full_like(tb_fix, -1.0)
                    xr  = torch.ones_like(tb_fix)
                    lb  = (model(xl, tb_fix)**2 + model(xr, tb_fix)**2).mean()
                    xi_ = xi_fix*2-1
                    li  = ((model(xi_, ti_fix) + torch.sin(np.pi*xi_))**2).mean()
                elif pde == "AllenCahn":
                    xc_ = xc_fix*2-1; tc_ = tc_fix
                    lp  = (res_allen_cahn(model, xc_, tc_, cfg["eps2"])**2).mean()
                    xl  = torch.full_like(tb_fix, -1.0)
                    xr  = torch.ones_like(tb_fix)
                    lb  = (model(xl, tb_fix)**2 + model(xr, tb_fix)**2).mean()
                    xi_ = xi_fix*2-1
                    li  = ((model(xi_, ti_fix) - xi_**2*torch.cos(np.pi*xi_))**2).mean()
                elif pde == "Helmholtz":
                    xc_ = xc_fix*2-1; yc_ = tc_fix*2-1
                    lp  = (res_helmholtz(model, xc_, yc_, cfg["k2"])**2).mean()
                    tb_ = tb_fix*2-1; w_ = torch.ones_like(tb_)
                    lb  = (model(w_, tb_)**2 + model(-w_, tb_)**2
                           + model(tb_, w_)**2 + model(tb_, -w_)**2).mean()
                    xi_ = xi_fix*2-1; yi_ = torch.rand_like(xi_)*2-1
                    li  = ((model(xi_, yi_) - torch.sin(np.pi*xi_)*torch.sin(np.pi*yi_))**2).mean()
                else:
                    return torch.tensor(0.0, requires_grad=True)
                loss = lp + 100*li + 10*lb
                if torch.isfinite(loss):
                    loss.backward()
                    loss_container[0] = loss
                    return loss
            except Exception:
                pass
            return torch.tensor(0.0, requires_grad=True)
        try:
            opt.step(closure)
            if loss_container[0] is not None and step % 25 == 0:
                traj.append(float(loss_container[0].item()))
        except Exception:
            break
    return model, traj
def recover_i4(model_ckpt, cfg, run_idx):
    torch.manual_seed(GLOBAL_SEED + run_idx * 10 + 4)
    model = copy.deepcopy(model_ckpt)
    opt   = torch.optim.Adam(model.parameters(), lr=RECOVER_LR)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=RECOVER_EPOCHS, eta_min=RECOVER_LR_MIN)
    traj  = []
    w_pde, w_ic, w_bc = 1.0, 100.0, 10.0
    REWEIGHT_FREQ = 1000
    for epoch in range(RECOVER_EPOCHS):
        if epoch % REWEIGHT_FREQ == 0:
            model.zero_grad()
            try:
                _, lp, li, lb = build_loss(model, cfg,
                                            RECOVER_N_COL, FAIL_N_IC*3, FAIL_N_BC*3,
                                            1.0, 1.0, 1.0)
                lp.backward(retain_graph=True)
                gp = torch.cat([p.grad.flatten() for p in model.parameters()
                                if p.grad is not None]).norm().item()
                model.zero_grad()
                li.backward(retain_graph=True)
                gi = torch.cat([p.grad.flatten() for p in model.parameters()
                                if p.grad is not None]).norm().item()
                model.zero_grad()
                lb.backward()
                gb = torch.cat([p.grad.flatten() for p in model.parameters()
                                if p.grad is not None]).norm().item()
                model.zero_grad()
                ref   = max(gp, gi, gb) + 1e-12
                w_pde = float(ref / (gp + 1e-12))
                w_ic  = float(ref / (gi + 1e-12))
                w_bc  = float(ref / (gb + 1e-12))
                wmax  = max(w_pde, w_ic, w_bc)
                w_pde = 100.0 * w_pde / wmax
                w_ic  = 100.0 * w_ic  / wmax
                w_bc  = 100.0 * w_bc  / wmax
            except Exception:
                pass
        model.train(); opt.zero_grad()
        try:
            loss, *_ = build_loss(model, cfg, RECOVER_N_COL,
                                   FAIL_N_IC*3, FAIL_N_BC*3,
                                   w_pde, w_ic, w_bc)
            if not torch.isfinite(loss): break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            if epoch % 500 == 0:
                traj.append(float(loss.item()))
        except Exception:
            break
    return model, traj
def recover_i5(model_ckpt, cfg, run_idx):
    torch.manual_seed(GLOBAL_SEED + run_idx * 10 + 5)
    pde = cfg["pde"]
    if pde == "Helmholtz":
        x_lo, x_hi = -1.0, 1.0
    elif pde == "Advection":
        x_lo, x_hi = 0.0, 2 * np.pi
    else:
        x_lo, x_hi = -1.0, 1.0
    x_mid = (x_lo + x_hi) / 2.0
    n_h = cfg.get("n_hidden", 4); n_n = cfg.get("n_neurons", 64)
    net_l = StandardPINN(n_hidden=n_h, n_neurons=n_n).to(DEVICE)
    net_r = StandardPINN(n_hidden=n_h, n_neurons=n_n).to(DEVICE)
    opt   = torch.optim.Adam(
        list(net_l.parameters()) + list(net_r.parameters()),
        lr=RECOVER_LR)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=RECOVER_EPOCHS, eta_min=RECOVER_LR_MIN)
    traj  = []
    for epoch in range(RECOVER_EPOCHS):
        opt.zero_grad()
        try:
            xc_l = (torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)
                    * (x_mid - x_lo) + x_lo)
            xc_r = (torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)
                    * (x_hi - x_mid) + x_mid)
            xc_l_n = 2*(xc_l - x_lo)/(x_mid - x_lo) - 1
            xc_r_n = 2*(xc_r - x_mid)/(x_hi - x_mid) - 1
            if pde == "Advection":
                tc_l = torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)*2.0
                tc_r = torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)*2.0
                res_l = res_advection(net_l, xc_l_n, tc_l, cfg["beta"])
                res_r = res_advection(net_r, xc_r_n, tc_r, cfg["beta"])
                t_if  = torch.rand(200, 1, dtype=DTYPE, device=DEVICE)*2.0
                coord2_l = tc_l; coord2_r = tc_r
            elif pde == "Burgers":
                tc_l = torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)
                tc_r = torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)
                res_l = res_burgers(net_l, xc_l_n, tc_l, cfg["nu"])
                res_r = res_burgers(net_r, xc_r_n, tc_r, cfg["nu"])
                t_if  = torch.rand(200, 1, dtype=DTYPE, device=DEVICE)
                coord2_l = tc_l; coord2_r = tc_r
            elif pde == "AllenCahn":
                tc_l = torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)
                tc_r = torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)
                res_l = res_allen_cahn(net_l, xc_l_n, tc_l, cfg["eps2"])
                res_r = res_allen_cahn(net_r, xc_r_n, tc_r, cfg["eps2"])
                t_if  = torch.rand(200, 1, dtype=DTYPE, device=DEVICE)
                coord2_l = tc_l; coord2_r = tc_r
            else:
                yc_l = torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)*2-1
                yc_r = torch.rand(RECOVER_N_COL//2, 1, dtype=DTYPE, device=DEVICE)*2-1
                res_l = res_helmholtz(net_l, xc_l_n, yc_l, cfg["k2"])
                res_r = res_helmholtz(net_r, xc_r_n, yc_r, cfg["k2"])
                t_if  = torch.rand(200, 1, dtype=DTYPE, device=DEVICE)*2-1
                coord2_l = yc_l; coord2_r = yc_r
            l_pde  = (res_l**2).mean() + (res_r**2).mean()
            xl_at_mid = torch.ones(200, 1, dtype=DTYPE, device=DEVICE)
            xr_at_mid = torch.full((200, 1), -1.0, dtype=DTYPE, device=DEVICE)
            l_iface   = ((net_l(xl_at_mid, t_if) - net_r(xr_at_mid, t_if))**2).mean()
            loss = l_pde + 200*l_iface
            if not torch.isfinite(loss): break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(net_l.parameters())+list(net_r.parameters()), 1.0)
            opt.step(); sch.step()
            if epoch % 500 == 0:
                traj.append(float(loss.item()))
        except Exception:
            break
    class StitchedModel(nn.Module):
        def __init__(self, nl, nr, xl, xm, xh):
            super().__init__()
            self.nl = nl; self.nr = nr
            self.xl = xl; self.xm = xm; self.xh = xh
        def forward(self, x, y):
            out = torch.zeros(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
            ml  = (x.squeeze() <= self.xm)
            mr  = ~ml
            if ml.any():
                xn = 2*(x[ml]   - self.xl)/(self.xm - self.xl) - 1
                out[ml] = self.nl(xn, y[ml] if y.shape[0] == x.shape[0]
                                  else y)
            if mr.any():
                xn = 2*(x[mr]   - self.xm)/(self.xh - self.xm) - 1
                out[mr] = self.nr(xn, y[mr] if y.shape[0] == x.shape[0]
                                  else y)
            return out
    stitched = StitchedModel(net_l, net_r, x_lo, x_mid, x_hi).to(DEVICE)
    return stitched, traj
INTERVENTIONS = [
    ("I1: 10× Collocation", recover_i1),
    ("I2: Fourier Features", recover_i2),
    ("I3: L-BFGS",           recover_i3),
    ("I4: Loss Reweighting", recover_i4),
    ("I5: Domain Decomp.",   recover_i5),
]
def plot_recovery_matrix(recovery_matrix, cfg_list, filepath):
    o2i    = {"RECOVERED": 2, "PARTIAL": 1, "FAILED": 0}
    cmap   = mcolors.ListedColormap(
        [OUTCOME_COLORS["FAILED"], OUTCOME_COLORS["PARTIAL"],
         OUTCOME_COLORS["RECOVERED"]])
    norm   = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)
    n_runs = len(cfg_list)
    data   = np.array([[o2i.get(recovery_matrix[ri][ii]["outcome"], 0)
                        for ii in range(N_INTERVENTIONS)]
                       for ri in range(n_runs)], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(N_INTERVENTIONS))
    ax.set_xticklabels(INTERVENTION_LABELS, rotation=25, ha="right", fontsize=9)
    row_labels = [f"{c['mode'][:12]}\nseed={c['seed']}" for c in cfg_list]
    ax.set_yticks(range(n_runs)); ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("Recovery Outcome Matrix\n(20 failed runs × 5 interventions)",
                 fontsize=12, fontweight="bold")
    for ri in range(n_runs):
        for ii in range(N_INTERVENTIONS):
            l2 = recovery_matrix[ri][ii]["post_l2"]
            ax.text(ii, ri, f"{l2:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if data[ri, ii] < 1.5 else "black")
    modes = [c["mode"] for c in cfg_list]
    for i in range(1, n_runs):
        if modes[i] != modes[i-1]:
            ax.axhline(i-0.5, color="black", lw=1.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=OUTCOME_COLORS[k], label=k)
                        for k in ["RECOVERED","PARTIAL","FAILED"]],
              loc="upper right", fontsize=9, bbox_to_anchor=(1.25, 1.0))
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: {filepath}")
def plot_recovery_by_failure_mode(recovery_matrix, cfg_list, filepath):
    modes = list(dict.fromkeys(c["mode"] for c in cfg_list))
    n_m   = len(modes)
    fig, axes = plt.subplots(1, n_m, figsize=(4*n_m, 5),
                              constrained_layout=True, sharey=True)
    fig.suptitle("Recovery Rate by Failure Mode", fontsize=13,
                 fontweight="bold")
    if n_m == 1: axes = [axes]
    for ax, mode in zip(axes, modes):
        runs = [i for i, c in enumerate(cfg_list) if c["mode"] == mode]
        rates= [sum(recovery_matrix[ri][ii]["outcome"] == "RECOVERED"
                    for ri in runs) / len(runs)
                for ii in range(N_INTERVENTIONS)]
        bars = ax.bar(range(N_INTERVENTIONS), rates,
                      color=[cm.RdYlGn(r) for r in rates],
                      edgecolor="black", lw=0.8)
        ax.set_xticks(range(N_INTERVENTIONS))
        ax.set_xticklabels([f"I{i+1}" for i in range(N_INTERVENTIONS)],
                           fontsize=9)
        ax.set_title(mode, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recovery Rate" if ax is axes[0] else "")
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.grid(True, axis="y", alpha=0.25)
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x()+bar.get_width()/2, r+0.02,
                    f"{r:.0%}", ha="center", fontsize=8)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: {filepath}")
def plot_intervention_profiles(recovery_matrix, cfg_list, filepath):
    fig, axes = plt.subplots(1, N_INTERVENTIONS,
                              figsize=(4*N_INTERVENTIONS, 4),
                              constrained_layout=True)
    fig.suptitle("Loss Trajectories During Recovery", fontsize=12,
                 fontweight="bold")
    colors_runs = cm.tab20(np.linspace(0, 1, len(cfg_list)))
    for ii, (lbl, _) in enumerate(INTERVENTIONS):
        ax = axes[ii]
        for ri in range(len(cfg_list)):
            traj = recovery_matrix[ri][ii]["traj"]
            if len(traj) > 1:
                ax.semilogy(range(len(traj)), traj,
                            color=colors_runs[ri], lw=1, alpha=0.6)
        ax.set_title(lbl.split(":")[0], fontsize=10)
        ax.set_xlabel("Checkpoint (×500 steps)", fontsize=8)
        ax.set_ylabel("Loss" if ii == 0 else "")
        ax.grid(True, alpha=0.2, which="both")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: {filepath}")
def plot_unrecoverable(recovery_matrix, cfg_list, filepath):
    unrecoverable = []
    for ri, cfg in enumerate(cfg_list):
        if all(recovery_matrix[ri][ii]["outcome"] == "FAILED"
               for ii in range(N_INTERVENTIONS)):
            best = min(recovery_matrix[ri][ii]["post_l2"]
                       for ii in range(N_INTERVENTIONS))
            unrecoverable.append({
                "run": ri, "mode": cfg["mode"], "seed": cfg["seed"],
                "pre_l2": recovery_matrix[ri][0]["pre_l2"],
                "best_post_l2": best,
            })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Unrecoverable Failure Analysis", fontsize=13,
                 fontweight="bold")
    ax = axes[0]
    s_c = [sum(recovery_matrix[ri][ii]["outcome"] == "RECOVERED"
               for ri in range(len(cfg_list))) for ii in range(N_INTERVENTIONS)]
    p_c = [sum(recovery_matrix[ri][ii]["outcome"] == "PARTIAL"
               for ri in range(len(cfg_list))) for ii in range(N_INTERVENTIONS)]
    f_c = [sum(recovery_matrix[ri][ii]["outcome"] == "FAILED"
               for ri in range(len(cfg_list))) for ii in range(N_INTERVENTIONS)]
    xp  = np.arange(N_INTERVENTIONS); w = 0.25
    ax.bar(xp-w, s_c, w, color=OUTCOME_COLORS["RECOVERED"], label="RECOVERED")
    ax.bar(xp,   p_c, w, color=OUTCOME_COLORS["PARTIAL"],   label="PARTIAL")
    ax.bar(xp+w, f_c, w, color=OUTCOME_COLORS["FAILED"],    label="FAILED")
    ax.set_xticks(xp)
    ax.set_xticklabels([f"I{i+1}" for i in range(N_INTERVENTIONS)], fontsize=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Outcome Distribution per Intervention", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.25)
    ax2 = axes[1]
    pre  = [recovery_matrix[ri][0]["pre_l2"] for ri in range(len(cfg_list))]
    best = [min(recovery_matrix[ri][ii]["post_l2"]
                for ii in range(N_INTERVENTIONS))
            for ri in range(len(cfg_list))]
    mc   = [FAILURE_COLORS.get(cfg_list[ri]["mode"], "grey")
            for ri in range(len(cfg_list))]
    ax2.scatter(pre, best, c=mc, s=80, alpha=0.8, zorder=5)
    d_max = max(max(pre), max(best)) * 1.1
    ax2.plot([0, d_max], [0, d_max], "k--", lw=1, label="no change")
    ax2.axhline(RECOVERED_THRESH, color="green", ls=":", lw=1,
                label=f"Recovery thresh ({RECOVERED_THRESH})")
    for u in unrecoverable:
        ax2.scatter([u["pre_l2"]], [u["best_post_l2"]],
                    s=200, facecolors="none",
                    edgecolors="red", lw=2, zorder=6)
    ax2.set_xlabel("Pre-intervention L2", fontsize=11)
    ax2.set_ylabel("Best Post-intervention L2", fontsize=11)
    ax2.set_title("Pre vs Post L2 (best intervention)\nRed circles = unrecoverable",
                  fontsize=11)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=FAILURE_COLORS[m], label=m[:14])
                         for m in FAILURE_MODES],
               fontsize=7, loc="lower right")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: {filepath}")
    return unrecoverable
def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 22_2: Failure Recovery Boundary  [v2]")
    print(f"Device: {DEVICE}")
    print(f"Failed runs: {N_FAILED_RUNS}  ×  Interventions: {N_INTERVENTIONS}")
    print(f"Config: {N_SEEDS} seeds × {len(FAILURE_MODES)} modes = {N_FAILED_RUNS}")
    print("=" * 70)
    ckpt_path = OUTPUT_DIR / "exp22_2_checkpoint.json"
    ckpt_models_dir = OUTPUT_DIR / "checkpoints"
    ckpt_models_dir.mkdir(exist_ok=True)
    ckpt = {"phase1_l2s": [], "recovery_matrix": [[None]*N_INTERVENTIONS for _ in range(N_FAILED_RUNS)]}
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r") as f:
                ckpt = json.load(f)
            print("  [Loaded checkpoint]")
        except Exception:
            pass
    print(f"\n{'─'*60}\nPhase 1: Producing failed models\n{'─'*60}")
    failed_models = []; failed_l2s = ckpt.get("phase1_l2s", [])
    for i, cfg in enumerate(FAILURE_CONFIGS):
        print(f"\n  [{i+1:>2d}/{N_FAILED_RUNS}] {cfg['mode']:<22} "
              f"{cfg['pde']:<12} seed={cfg['seed']}", end="  ")
        model_path = ckpt_models_dir / f"failed_model_{i}.pt"
        if i < len(failed_l2s) and model_path.exists():
            n_h = cfg.get("n_hidden", 4); n_n = cfg.get("n_neurons", 64)
            model = StandardPINN(n_hidden=n_h, n_neurons=n_n).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            l2 = failed_l2s[i]
            print(f"L2={l2:.4f}  [loaded from checkpoint]")
        else:
            model, l2 = produce_failed_model(cfg)
            print(f"L2={l2:.4f}  "
                  + ("← succeeded anyway" if l2 < 0.05 else "← FAILED ✓"))
            torch.save(model.state_dict(), model_path)
            if i >= len(failed_l2s):
                failed_l2s.append(l2)
            else:
                failed_l2s[i] = l2
            ckpt["phase1_l2s"] = failed_l2s
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f)
        failed_models.append(model)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"\n  Mean pre-intervention L2 : {np.mean(failed_l2s):.4f}")
    print(f"  Confirmed failed (L2>0.05): "
          f"{sum(l>0.05 for l in failed_l2s)}/{N_FAILED_RUNS}")
    print(f"\n{'─'*60}\nPhase 2: Recovery interventions\n{'─'*60}")
    recovery_matrix = ckpt.get("recovery_matrix", [[None]*N_INTERVENTIONS for _ in range(N_FAILED_RUNS)])
    if len(recovery_matrix) < N_FAILED_RUNS:
        recovery_matrix.extend([[None]*N_INTERVENTIONS for _ in range(N_FAILED_RUNS - len(recovery_matrix))])
    t0 = time.time()
    for ri, (cfg, ckpt_model, pre_l2) in enumerate(
            zip(FAILURE_CONFIGS, failed_models, failed_l2s)):
        print(f"\n  Run {ri+1:>2d}/{N_FAILED_RUNS}: {cfg['mode']}  "
              f"(pre_l2={pre_l2:.4f})")
        for ii, (lbl, fn) in enumerate(INTERVENTIONS):
            print(f"    {lbl} ...", end=" ", flush=True)
            if recovery_matrix[ri][ii] is not None:
                outcome = recovery_matrix[ri][ii]["outcome"]
                post_l2 = recovery_matrix[ri][ii]["post_l2"]
                print(f"post_l2={post_l2:.4f}  [{outcome}] [loaded]")
                continue
            try:
                rec_model, traj = fn(ckpt_model, cfg, ri)
                post_l2 = evaluate_l2(rec_model, cfg)
                del rec_model
            except Exception as e:
                print(f"[ERROR: {e}]", end=" ")
                post_l2 = pre_l2; traj = []
            outcome = classify_outcome(post_l2)
            recovery_matrix[ri][ii] = {
                "outcome":     outcome,
                "pre_l2":      float(pre_l2),
                "post_l2":     float(post_l2),
                "improvement": float(pre_l2 - post_l2),
                "traj":        traj[:20],
            }
            ckpt["recovery_matrix"] = recovery_matrix
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f)
            print(f"post_l2={post_l2:.4f}  [{outcome}]")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    elapsed = time.time() - t0
    print(f"\n{'─'*60}\nGenerating plots\n{'─'*60}")
    plot_recovery_matrix(recovery_matrix, FAILURE_CONFIGS,
        OUTPUT_DIR / "recovery_matrix_heatmap.png")
    plot_recovery_by_failure_mode(recovery_matrix, FAILURE_CONFIGS,
        OUTPUT_DIR / "recovery_by_failure_mode.png")
    plot_intervention_profiles(recovery_matrix, FAILURE_CONFIGS,
        OUTPUT_DIR / "intervention_profiles.png")
    unrecoverable = plot_unrecoverable(recovery_matrix, FAILURE_CONFIGS,
        OUTPUT_DIR / "unrecoverable_analysis.png")
    int_rates = [
        sum(recovery_matrix[ri][ii]["outcome"] == "RECOVERED"
            for ri in range(N_FAILED_RUNS)) / N_FAILED_RUNS
        for ii in range(N_INTERVENTIONS)
    ]
    best_int = int(np.argmax(int_rates))
    print(f"\n{'='*70}")
    print("EXPERIMENT 22_2 — RECOVERY MAPPING TABLE")
    print(f"{'='*70}")
    hdr = f"{'Failure Mode':<22} | " + " | ".join(f"I{i+1}" for i in range(N_INTERVENTIONS))
    print(hdr); print("─"*len(hdr))
    for mode in FAILURE_MODES:
        runs = [i for i, c in enumerate(FAILURE_CONFIGS) if c["mode"] == mode]
        row  = f"{mode:<22} | "
        for ii in range(N_INTERVENTIONS):
            nr   = sum(recovery_matrix[ri][ii]["outcome"] == "RECOVERED"
                       for ri in runs)
            row += f"{nr}/{len(runs)}  | "
        print(row)
    print(f"\n  Intervention recovery rates:")
    for ii, (lbl, _) in enumerate(INTERVENTIONS):
        print(f"    {lbl:<30}: {int_rates[ii]*100:.0f}%")
    print(f"  Best: [{best_int+1}] {INTERVENTION_LABELS[best_int]} "
          f"({int_rates[best_int]*100:.0f}%)")
    print(f"  Unrecoverable: {len(unrecoverable)}/{N_FAILED_RUNS}")
    results_json = {
        "experiment":  "Failure Recovery Boundary",
        "experiment_id": "22_2",
        "version":     "v2-journal-ready",
        "output_dir":  str(OUTPUT_DIR),
        "config": {
            "n_failed_runs":     N_FAILED_RUNS,
            "n_interventions":   N_INTERVENTIONS,
            "n_seeds":           N_SEEDS,
            "failure_modes":     FAILURE_MODES,
            "intervention_labels": INTERVENTION_LABELS,
            "recovered_thresh":  RECOVERED_THRESH,
            "partial_thresh":    PARTIAL_THRESH,
            "global_seed":       GLOBAL_SEED,
        },
        "fix_notes": {
            "fix1_config_count": (
                f"v1 generated 4×5+5=25 configs then trimmed with [:20], "
                f"dropping all OptimStagnation. v2 generates exactly "
                f"{N_SEEDS} seeds × {len(FAILURE_MODES)} modes = {N_FAILED_RUNS}."
            ),
            "fix2_burgers_ref": (
                "v1 used linear heat solution -sin(πx)exp(-νπ²t) as Burgers "
                "reference. For ν=0.001 this is invalid past t≈0.3. "
                "v2 uses Crank-Nicolson FD reference."
            ),
            "fix3_allencahn_ref": (
                "v1 used x²cos(πx)exp(-t) which satisfies the IC but not "
                "the Allen-Cahn PDE. v2 uses forward-Euler FD reference."
            ),
            "fix4_helmholtz_decomp": (
                "v1 passed tc_l (time) as second coord to Helmholtz "
                "sub-nets (which take y not t). v2 samples yc_l, yc_r "
                "and uses them consistently for both residual and network."
            ),
            "fix5_lbfgs_closure": (
                "v1 re-sampled collocation data on each closure call "
                "(up to 20 per step), violating Wolfe conditions. "
                "v2 samples once per outer step and captures in closure."
            ),
            "fix6_seeds": (
                f"Recovery training seeded by (GLOBAL_SEED + run*10 + intervention)."
            ),
        },
        "recovery_matrix": [
            [{k: v for k, v in recovery_matrix[ri][ii].items()
              if k != "traj"}
             for ii in range(N_INTERVENTIONS)]
            for ri in range(N_FAILED_RUNS)
        ],
        "intervention_recovery_rates": {
            INTERVENTION_LABELS[ii]: float(int_rates[ii])
            for ii in range(N_INTERVENTIONS)
        },
        "best_intervention": INTERVENTION_LABELS[best_int],
        "unrecoverable_runs": unrecoverable,
        "elapsed_seconds": float(elapsed),
    }
    out = OUTPUT_DIR / "exp22_2_results.json"
    with open(out, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults → {out}")
    print(f"Plots   → {OUTPUT_DIR}")
    return results_json
if __name__ == "__main__":
    run_experiment()
