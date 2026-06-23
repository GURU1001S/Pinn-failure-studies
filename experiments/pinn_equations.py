import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from scipy.io import loadmat
from pinn_core import DEVICE, DTYPE, get_activation, save_results
class GenericPINN(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, n_hidden=4, n_neurons=64,
                 activation="tanh"):
        super().__init__()
        self.config = dict(in_dim=in_dim, out_dim=out_dim,
                           n_hidden=n_hidden, n_neurons=n_neurons,
                           activation=activation)
        act_cls = get_activation(activation)
        layers = [nn.Linear(in_dim, n_neurons), act_cls()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), act_cls()]
        layers.append(nn.Linear(n_neurons, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, *inputs):
        if len(inputs) == 1:
            return self.net(inputs[0])
        return self.net(torch.cat(inputs, dim=1))
BURGERS_NU = 0.01 / np.pi
BURGERS_X_RANGE = (-1.0, 1.0)
BURGERS_T_RANGE = (0.0, 1.0)
def burgers_residual(model, x, t, nu=BURGERS_NU):
    u = model(x, t)
    grads = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, grads, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grads, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    return u_t + u * u_x - nu * u_xx
def burgers_ic(x):
    return -torch.sin(np.pi * x)
def load_burgers_reference(dataset_path="datasets/burgers_shock.mat"):
    data = loadmat(dataset_path)
    x = data["x"].flatten()
    t = data["t"].flatten()
    u = data["usol"]
    return x, t, u
def sample_burgers_domain(n_int, n_ic, n_bc):
    x_int = torch.rand(n_int, 1, device=DEVICE, dtype=DTYPE) * 2 - 1
    t_int = torch.rand(n_int, 1, device=DEVICE, dtype=DTYPE)
    x_int.requires_grad_(True)
    t_int.requires_grad_(True)
    x_ic = (torch.rand(n_ic, 1, device=DEVICE, dtype=DTYPE) * 2 - 1).requires_grad_(True)
    t_ic = torch.zeros(n_ic, 1, device=DEVICE, dtype=DTYPE).requires_grad_(True)
    u_ic = burgers_ic(x_ic).detach()
    n_bc_half = n_bc // 2
    t_bc = torch.rand(n_bc_half, 1, device=DEVICE, dtype=DTYPE)
    x_bc_left = torch.full((n_bc_half, 1), -1.0, device=DEVICE, dtype=DTYPE)
    x_bc_right = torch.full((n_bc_half, 1), 1.0, device=DEVICE, dtype=DTYPE)
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0).requires_grad_(True)
    t_bc = torch.cat([t_bc, t_bc.clone()], dim=0).requires_grad_(True)
    u_bc = torch.zeros(n_bc_half * 2, 1, device=DEVICE, dtype=DTYPE)
    return (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc)
def train_burgers_pinn(model, n_epochs=20000, lr=1e-3, lr_min=1e-5,
                       n_int=10000, n_ic=200, n_bc=200,
                       nu=BURGERS_NU, w_pde=1.0, w_ic=10.0, w_bc=1.0,
                       log_every=1000, verbose=True,
                       gradient_tracking=False, track_every=1000,
                       lambda_bc=1.0):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min)
    loss_hist = []
    pde_hist = []
    ic_hist = []
    bc_hist = []
    grad_records = [] if gradient_tracking else None
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) =        sample_burgers_domain(n_int, n_ic, n_bc)
    start = time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        res = burgers_residual(model, x_int, t_int, nu)
        loss_pde = torch.mean(res ** 2)
        u_ic_pred = model(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)
        u_bc_pred = model(x_bc, t_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)
        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * lambda_bc * loss_bc
        loss.backward()
        if gradient_tracking and epoch % track_every == 0:
            record = {"epoch": epoch}
            optimizer.zero_grad()
            res2 = burgers_residual(model, x_int, t_int, nu)
            l_pde = torch.mean(res2 ** 2)
            l_pde.backward()
            pde_grad = torch.cat([p.grad.flatten() for p in model.parameters()
                                  if p.grad is not None])
            record["pde_grad_norm"] = pde_grad.norm().item()
            record["pde_grad_vec"] = pde_grad.detach().cpu().clone()
            optimizer.zero_grad()
            u_bc2 = model(x_bc, t_bc)
            l_bc = torch.mean((u_bc2 - u_bc) ** 2)
            (lambda_bc * l_bc).backward()
            bc_grad = torch.cat([p.grad.flatten() for p in model.parameters()
                                 if p.grad is not None])
            record["bc_grad_norm"] = bc_grad.norm().item()
            record["bc_grad_vec"] = bc_grad.detach().cpu().clone()
            optimizer.zero_grad()
            u_ic2 = model(x_ic, t_ic)
            l_ic = torch.mean((u_ic2 - u_ic) ** 2)
            l_ic.backward()
            ic_grad = torch.cat([p.grad.flatten() for p in model.parameters()
                                 if p.grad is not None])
            record["ic_grad_norm"] = ic_grad.norm().item()
            record["ic_grad_vec"] = ic_grad.detach().cpu().clone()
            grad_records.append(record)
            optimizer.zero_grad()
            res3 = burgers_residual(model, x_int, t_int, nu)
            l_pde3 = torch.mean(res3 ** 2)
            u_ic3 = model(x_ic, t_ic)
            l_ic3 = torch.mean((u_ic3 - u_ic) ** 2)
            u_bc3 = model(x_bc, t_bc)
            l_bc3 = torch.mean((u_bc3 - u_bc) ** 2)
            loss_total = w_pde * l_pde3 + w_ic * l_ic3 + w_bc * lambda_bc * l_bc3
            loss_total.backward()
        optimizer.step()
        scheduler.step()
        loss_hist.append(loss.item())
        pde_hist.append(loss_pde.item())
        ic_hist.append(loss_ic.item())
        bc_hist.append(loss_bc.item())
        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  [{epoch:6d}/{n_epochs}] Loss={loss.item():.4e} "
                  f"PDE={loss_pde.item():.4e} IC={loss_ic.item():.4e} "
                  f"BC={loss_bc.item():.4e}")
    return {
        "model": model,
        "loss_history": loss_hist,
        "pde_loss_history": pde_hist,
        "ic_loss_history": ic_hist,
        "bc_loss_history": bc_hist,
        "training_time": time.time() - start,
        "gradient_records": grad_records,
    }
def evaluate_burgers(model, x_ref, t_ref, u_ref):
    nx, nt = len(x_ref), len(t_ref)
    X, T = np.meshgrid(x_ref, t_ref, indexing="ij")
    x_flat = torch.tensor(X.flatten(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
    t_flat = torch.tensor(T.flatten(), dtype=DTYPE, device=DEVICE).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(nx, nt)
    l2_err = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-30)
    return u_pred, l2_err
HELMHOLTZ_A1 = 1
HELMHOLTZ_A2 = 4
HELMHOLTZ_K_SQ = 1.0
def helmholtz_source(x1, x2, a1=HELMHOLTZ_A1, a2=HELMHOLTZ_A2, k_sq=HELMHOLTZ_K_SQ):
    u = torch.sin(a1 * np.pi * x1) * torch.sin(a2 * np.pi * x2)
    coeff = k_sq - (a1**2 + a2**2) * np.pi**2
    return coeff * u
def helmholtz_exact(x1, x2, a1=HELMHOLTZ_A1, a2=HELMHOLTZ_A2):
    return np.sin(a1 * np.pi * x1) * np.sin(a2 * np.pi * x2)        if isinstance(x1, np.ndarray)        else torch.sin(a1 * np.pi * x1) * torch.sin(a2 * np.pi * x2)
def helmholtz_residual(model, x1, x2, k_sq=HELMHOLTZ_K_SQ):
    u = model(x1, x2)
    ones = torch.ones_like(u)
    u_x1 = torch.autograd.grad(u, x1, ones, create_graph=True, retain_graph=True)[0]
    u_x1x1 = torch.autograd.grad(u_x1, x1, torch.ones_like(u_x1),
                                   create_graph=True, retain_graph=True)[0]
    u_x2 = torch.autograd.grad(u, x2, ones, create_graph=True, retain_graph=True)[0]
    u_x2x2 = torch.autograd.grad(u_x2, x2, torch.ones_like(u_x2),
                                   create_graph=True, retain_graph=True)[0]
    q = helmholtz_source(x1, x2, k_sq=k_sq)
    return u_x1x1 + u_x2x2 + k_sq * u - q
def sample_helmholtz_domain(n_int, n_bc, method="random"):
    if method == "random":
        pts = np.random.rand(n_int, 2) * 2 - 1
    elif method == "lhs":
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=2)
        pts = sampler.random(n=n_int) * 2 - 1
    elif method == "sobol":
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=2, scramble=True)
        pts = sampler.random(n=n_int) * 2 - 1
    elif method == "halton":
        from scipy.stats.qmc import Halton
        sampler = Halton(d=2, scramble=True)
        pts = sampler.random(n=n_int) * 2 - 1
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    x1_int = torch.tensor(pts[:, 0:1], dtype=DTYPE, device=DEVICE).requires_grad_(True)
    x2_int = torch.tensor(pts[:, 1:2], dtype=DTYPE, device=DEVICE).requires_grad_(True)
    n_per_edge = n_bc // 4
    edges = []
    e = np.column_stack([np.linspace(-1, 1, n_per_edge), -np.ones(n_per_edge)])
    edges.append(e)
    e = np.column_stack([np.linspace(-1, 1, n_per_edge), np.ones(n_per_edge)])
    edges.append(e)
    e = np.column_stack([-np.ones(n_per_edge), np.linspace(-1, 1, n_per_edge)])
    edges.append(e)
    e = np.column_stack([np.ones(n_per_edge), np.linspace(-1, 1, n_per_edge)])
    edges.append(e)
    bc_pts = np.vstack(edges)
    x1_bc = torch.tensor(bc_pts[:, 0:1], dtype=DTYPE, device=DEVICE).requires_grad_(True)
    x2_bc = torch.tensor(bc_pts[:, 1:2], dtype=DTYPE, device=DEVICE).requires_grad_(True)
    u_bc = torch.zeros(len(bc_pts), 1, dtype=DTYPE, device=DEVICE)
    return (x1_int, x2_int), (x1_bc, x2_bc, u_bc)
def train_helmholtz_pinn(model, n_epochs=15000, lr=1e-3, lr_min=1e-5,
                         n_int=2000, n_bc=400, k_sq=HELMHOLTZ_K_SQ,
                         w_pde=1.0, w_bc=10.0, sampling_method="random",
                         log_every=1000, verbose=True):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min)
    (x1_int, x2_int), (x1_bc, x2_bc, u_bc) =        sample_helmholtz_domain(n_int, n_bc, method=sampling_method)
    loss_hist, pde_hist, bc_hist = [], [], []
    start = time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        res = helmholtz_residual(model, x1_int, x2_int, k_sq)
        loss_pde = torch.mean(res ** 2)
        u_bc_pred = model(x1_bc, x2_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)
        loss = w_pde * loss_pde + w_bc * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_hist.append(loss.item())
        pde_hist.append(loss_pde.item())
        bc_hist.append(loss_bc.item())
        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  [{epoch:6d}/{n_epochs}] Loss={loss.item():.4e} "
                  f"PDE={loss_pde.item():.4e} BC={loss_bc.item():.4e}")
    return {
        "model": model,
        "loss_history": loss_hist,
        "pde_loss_history": pde_hist,
        "bc_loss_history": bc_hist,
        "training_time": time.time() - start,
    }
def evaluate_helmholtz(model, nx=100, ny=100):
    x1 = np.linspace(-1, 1, nx)
    x2 = np.linspace(-1, 1, ny)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    x1_t = torch.tensor(X1.flatten()[:, None], dtype=DTYPE, device=DEVICE)
    x2_t = torch.tensor(X2.flatten()[:, None], dtype=DTYPE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        u_pred = model(x1_t, x2_t).cpu().numpy().reshape(nx, ny)
    u_exact = helmholtz_exact(X1, X2)
    l2 = np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-30)
    return {"x1": x1, "x2": x2, "u_pred": u_pred, "u_exact": u_exact,
            "l2_error": l2, "pointwise_error": np.abs(u_pred - u_exact)}
HEAT_ALPHA = 0.01
HEAT_X_RANGE = (0.0, 1.0)
HEAT_T_RANGE = (0.0, 1.0)
def heat_residual(model, x, t, alpha=HEAT_ALPHA):
    u = model(x, t)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    return u_t - alpha * u_xx
def heat_exact(x, t, alpha=HEAT_ALPHA):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)
def heat_exact_torch(x, t, alpha=HEAT_ALPHA):
    return torch.exp(-alpha * np.pi**2 * t) * torch.sin(np.pi * x)
def sample_heat_domain(n_int, n_ic, n_bc, x_range=HEAT_X_RANGE,
                       t_range=HEAT_T_RANGE):
    x_int = (torch.rand(n_int, 1, device=DEVICE, dtype=DTYPE)
             * (x_range[1] - x_range[0]) + x_range[0]).requires_grad_(True)
    t_int = (torch.rand(n_int, 1, device=DEVICE, dtype=DTYPE)
             * (t_range[1] - t_range[0]) + t_range[0]).requires_grad_(True)
    x_ic = torch.linspace(x_range[0], x_range[1], n_ic, device=DEVICE,
                           dtype=DTYPE).unsqueeze(1).requires_grad_(True)
    t_ic = torch.full((n_ic, 1), t_range[0], device=DEVICE,
                       dtype=DTYPE).requires_grad_(True)
    u_ic = torch.sin(np.pi * x_ic).detach()
    n_half = n_bc // 2
    t_bc_vals = torch.linspace(t_range[0], t_range[1], n_half,
                                device=DEVICE, dtype=DTYPE).unsqueeze(1)
    x_bc_left = torch.full((n_half, 1), x_range[0], device=DEVICE, dtype=DTYPE)
    x_bc_right = torch.full((n_half, 1), x_range[1], device=DEVICE, dtype=DTYPE)
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0).requires_grad_(True)
    t_bc = torch.cat([t_bc_vals, t_bc_vals.clone()], dim=0).requires_grad_(True)
    u_bc = torch.zeros(n_half * 2, 1, device=DEVICE, dtype=DTYPE)
    return (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc)
def train_heat_pinn(model, alpha=HEAT_ALPHA, n_epochs=20000, lr=1e-3,
                    lr_min=1e-5, n_int=5000, n_ic=200, n_bc=200,
                    x_range=HEAT_X_RANGE, t_range=HEAT_T_RANGE,
                    w_pde=1.0, w_ic=10.0, w_bc=1.0,
                    log_every=2000, verbose=True):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min)
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) =        sample_heat_domain(n_int, n_ic, n_bc, x_range, t_range)
    loss_hist, pde_hist, ic_hist, bc_hist = [], [], [], []
    start = time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        res = heat_residual(model, x_int, t_int, alpha)
        loss_pde = torch.mean(res ** 2)
        u_ic_pred = model(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)
        u_bc_pred = model(x_bc, t_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)
        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_hist.append(loss.item())
        pde_hist.append(loss_pde.item())
        ic_hist.append(loss_ic.item())
        bc_hist.append(loss_bc.item())
        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  [{epoch:6d}/{n_epochs}] Loss={loss.item():.4e} "
                  f"PDE={loss_pde.item():.4e} IC={loss_ic.item():.4e} "
                  f"BC={loss_bc.item():.4e}")
    return {
        "model": model, "loss_history": loss_hist,
        "pde_loss_history": pde_hist, "ic_loss_history": ic_hist,
        "bc_loss_history": bc_hist, "training_time": time.time() - start,
    }
def evaluate_heat(model, alpha=HEAT_ALPHA, nx=200, nt=100,
                  x_range=HEAT_X_RANGE, t_range=HEAT_T_RANGE):
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    x_t = torch.tensor(X.flatten()[:, None], dtype=DTYPE, device=DEVICE)
    t_t = torch.tensor(T.flatten()[:, None], dtype=DTYPE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        u_pred = model(x_t, t_t).cpu().numpy().reshape(nx, nt)
    u_exact = heat_exact(X, T, alpha)
    l2 = np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-30)
    return {"x": x, "t": t, "u_pred": u_pred, "u_exact": u_exact, "l2_error": l2}
def allen_cahn_residual(model, x, t, epsilon):
    u = model(x, t)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    return u_t - epsilon**2 * u_xx - u + u**3
def allen_cahn_ic(x):
    return x**2 * torch.cos(np.pi * x)
def solve_allen_cahn_reference(epsilon, nx=512, nt=201, t_end=1.0):
    from scipy.integrate import solve_ivp
    x = np.linspace(-1, 1, nx)
    dx = x[1] - x[0]
    u0 = x**2 * np.cos(np.pi * x)
    def rhs(t_val, u):
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        u_xx[0] = (u[1] - 2 * u[0] + (-1)) / dx**2
        u_xx[-1] = ((-1) - 2 * u[-1] + u[-2]) / dx**2
        return epsilon**2 * u_xx + u - u**3
    t_eval = np.linspace(0, t_end, nt)
    sol = solve_ivp(rhs, [0, t_end], u0, t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-10, max_step=dx**2 / (2 * epsilon**2 + 0.01))
    if sol.success:
        return x, sol.t, sol.y
    else:
        print(f"  Warning: Allen-Cahn solver did not converge for ε={epsilon}")
        return x, sol.t, sol.y
def sample_allen_cahn_domain(n_int, n_ic, n_bc):
    x_int = (torch.rand(n_int, 1, device=DEVICE, dtype=DTYPE) * 2 - 1).requires_grad_(True)
    t_int = torch.rand(n_int, 1, device=DEVICE, dtype=DTYPE).requires_grad_(True)
    x_ic = torch.linspace(-1, 1, n_ic, device=DEVICE, dtype=DTYPE).unsqueeze(1).requires_grad_(True)
    t_ic = torch.zeros(n_ic, 1, device=DEVICE, dtype=DTYPE).requires_grad_(True)
    u_ic = allen_cahn_ic(x_ic).detach()
    n_half = n_bc // 2
    t_bc_vals = torch.linspace(0, 1, n_half, device=DEVICE, dtype=DTYPE).unsqueeze(1)
    x_bc = torch.cat([
        torch.full((n_half, 1), -1.0, device=DEVICE, dtype=DTYPE),
        torch.full((n_half, 1), 1.0, device=DEVICE, dtype=DTYPE)
    ], dim=0).requires_grad_(True)
    t_bc = torch.cat([t_bc_vals, t_bc_vals.clone()], dim=0).requires_grad_(True)
    u_bc = torch.full((n_half * 2, 1), -1.0, device=DEVICE, dtype=DTYPE)
    return (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc)
def train_allen_cahn_pinn(model, epsilon, n_epochs=20000, lr=1e-3, lr_min=1e-5,
                          n_int=10000, n_ic=200, n_bc=200,
                          w_pde=1.0, w_ic=10.0, w_bc=1.0,
                          log_every=2000, verbose=True):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min)
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) =        sample_allen_cahn_domain(n_int, n_ic, n_bc)
    loss_hist = []
    start = time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        res = allen_cahn_residual(model, x_int, t_int, epsilon)
        loss_pde = torch.mean(res ** 2)
        u_ic_pred = model(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)
        u_bc_pred = model(x_bc, t_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)
        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_hist.append(loss.item())
        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  [{epoch:6d}/{n_epochs}] Loss={loss.item():.4e} "
                  f"PDE={loss_pde.item():.4e} IC={loss_ic.item():.4e} "
                  f"BC={loss_bc.item():.4e}")
    return {
        "model": model, "loss_history": loss_hist,
        "training_time": time.time() - start,
    }
def evaluate_allen_cahn(model, x_ref, t_ref, u_ref):
    nx = len(x_ref)
    nt = len(t_ref)
    X, T = np.meshgrid(x_ref, t_ref, indexing="ij")
    x_t = torch.tensor(X.flatten()[:, None], dtype=DTYPE, device=DEVICE)
    t_t = torch.tensor(T.flatten()[:, None], dtype=DTYPE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        u_pred = model(x_t, t_t).cpu().numpy().reshape(nx, nt)
    l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-30)
    return {"u_pred": u_pred, "l2_error": l2}
