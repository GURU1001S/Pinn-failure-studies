import sys
import io
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
class FourierFeaturesLayer(nn.Module):
    def __init__(self, in_dim: int = 2, n_features: int = 64, sigma: float = 1.0):
        super().__init__()
        self.n_features = n_features
        B = torch.randn(n_features, in_dim) * sigma
        self.register_buffer("B", B)
    @property
    def out_dim(self):
        return 2 * self.n_features
    def forward(self, x):
        proj = x @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
def get_activation(name: str):
    activations = {
        "tanh": nn.Tanh,
        "sin": SinActivation,
        "swish": nn.SiLU,
        "gelu": nn.GELU,
    }
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation '{name}'. "
                         f"Choose from {list(activations.keys())} or use FourierFeatures.")
    return activations[name_lower]
class AdvectionPINN(nn.Module):
    def __init__(
        self,
        n_hidden: int = 4,
        n_neurons: int = 64,
        activation: str = "tanh",
        fourier_features: bool = False,
        fourier_sigma: float = 1.0,
        fourier_n_features: int = 64,
    ):
        super().__init__()
        self.config = {
            "n_hidden": n_hidden,
            "n_neurons": n_neurons,
            "activation": activation,
            "fourier_features": fourier_features,
            "fourier_sigma": fourier_sigma,
            "fourier_n_features": fourier_n_features,
        }
        act_cls = get_activation(activation)
        layers = []
        if fourier_features:
            self.ff_layer = FourierFeaturesLayer(
                in_dim=2,
                n_features=fourier_n_features,
                sigma=fourier_sigma,
            )
            in_dim = self.ff_layer.out_dim
        else:
            self.ff_layer = None
            in_dim = 2
        layers.append(nn.Linear(in_dim, n_neurons))
        layers.append(act_cls())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(act_cls())
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        if self.ff_layer is not None:
            inp = self.ff_layer(inp)
        return self.net(inp)
def advection_residual(model, x, t, beta):
    u = model(x, t)
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    return u_t + beta * u_x
def exact_solution(x, t, beta):
    return np.sin(x - beta * t)
def exact_solution_torch(x, t, beta):
    return torch.sin(x - beta * t)
def sample_collocation(n_points, x_range=(0, 2 * np.pi), t_range=(0, 2),
                       method="lhs"):
    if method == "lhs":
        n = n_points
        perms = np.random.permutation(n)
        x_vals = (perms + np.random.rand(n)) / n
        perms = np.random.permutation(n)
        t_vals = (perms + np.random.rand(n)) / n
        x_vals = x_range[0] + x_vals * (x_range[1] - x_range[0])
        t_vals = t_range[0] + t_vals * (t_range[1] - t_range[0])
    else:
        x_vals = np.random.uniform(*x_range, n_points)
        t_vals = np.random.uniform(*t_range, n_points)
    x = torch.tensor(x_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    t = torch.tensor(t_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    return x, t
def sample_initial_condition(n_points, x_range=(0, 2 * np.pi)):
    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    x = torch.tensor(x_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    t = torch.zeros(n_points, 1, dtype=DTYPE, device=DEVICE).requires_grad_(True)
    return x, t
def sample_boundary_condition(n_points, x_range=(0, 2 * np.pi), t_range=(0, 2)):
    t_vals = np.linspace(t_range[0], t_range[1], n_points)
    t_left = torch.tensor(t_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    x_left = torch.full((n_points, 1), x_range[0], dtype=DTYPE, device=DEVICE).requires_grad_(True)
    t_right = torch.tensor(t_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    x_right = torch.full((n_points, 1), x_range[1], dtype=DTYPE, device=DEVICE).requires_grad_(True)
    return x_left, t_left, x_right, t_right
def train_pinn(
    model,
    beta,
    n_epochs=15000,
    lr=1e-3,
    lr_min=1e-5,
    n_collocation=10000,
    n_ic=200,
    n_bc=200,
    resample_every=1000,
    w_pde=1.0,
    w_ic=10.0,
    w_bc=1.0,
    log_every=500,
    verbose=True,
):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min
    )
    loss_history = []
    pde_loss_history = []
    ic_loss_history = []
    bc_loss_history = []
    x_col, t_col = sample_collocation(n_collocation)
    x_ic, t_ic = sample_initial_condition(n_ic)
    x_bl, t_bl, x_br, t_br = sample_boundary_condition(n_bc)
    u_ic_target = torch.sin(x_ic).detach()
    start_time = time.time()
    for epoch in range(n_epochs):
        if epoch > 0 and epoch % resample_every == 0:
            x_col, t_col = sample_collocation(n_collocation)
        optimizer.zero_grad()
        residual = advection_residual(model, x_col, t_col, beta)
        loss_pde = torch.mean(residual ** 2)
        u_ic_pred = model(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
        u_left = model(x_bl, t_bl)
        u_right = model(x_br, t_br)
        loss_bc = torch.mean((u_left - u_right) ** 2)
        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())
        pde_loss_history.append(loss_pde.item())
        ic_loss_history.append(loss_ic.item())
        bc_loss_history.append(loss_bc.item())
        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:5d}/{n_epochs} | "
                  f"Loss: {loss.item():.6e} | "
                  f"PDE: {loss_pde.item():.6e} | "
                  f"IC: {loss_ic.item():.6e} | "
                  f"BC: {loss_bc.item():.6e} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
    training_time = time.time() - start_time
    if verbose:
        print(f"  Training complete in {training_time:.1f}s")
    return {
        "model": model,
        "loss_history": loss_history,
        "pde_loss_history": pde_loss_history,
        "ic_loss_history": ic_loss_history,
        "bc_loss_history": bc_loss_history,
        "training_time": training_time,
    }
def evaluate_on_grid(model, beta, nx=256, nt=100,
                     x_range=(0, 2 * np.pi), t_range=(0, 2)):
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    x_flat = X.flatten()
    t_flat = T.flatten()
    x_t = torch.tensor(x_flat, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    t_t = torch.tensor(t_flat, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        u_pred = model(x_t, t_t).cpu().numpy().reshape(nx, nt)
    u_exact = exact_solution(X, T, beta)
    l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    return {
        "x": x,
        "t": t,
        "u_pred": u_pred,
        "u_exact": u_exact,
        "l2_error": l2_error,
    }
def compute_spectrum(signal):
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    power = (np.abs(fft_vals) ** 2) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / N)
    return freqs, power
def spectral_residual(pred, exact):
    residual = pred - exact
    return compute_spectrum(residual)
def find_cutoff_frequency(power_pred, power_exact, threshold=0.01):
    max_exact = np.max(power_exact)
    if max_exact == 0:
        return len(power_pred) - 1
    for i in range(len(power_pred)):
        if power_exact[i] > threshold * max_exact:
            if power_pred[i] < threshold * power_exact[i]:
                return i
    return len(power_pred) - 1
def dominant_frequency(beta):
    return beta / (2 * np.pi)
def save_results(results_dict, filepath):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(convert(results_dict), f, indent=2)
    print(f"  Results saved to {filepath}")
def save_model(model, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": model.config,
    }, filepath)
