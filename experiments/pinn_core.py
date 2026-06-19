"""
pinn_core.py — Core PINN framework for spectral bias experiments.

Provides:
  - Configurable MLP with pluggable activation functions
  - FourierFeatures input embedding layer
  - Advection PDE residual (u_t + beta * u_x = 0) via autograd
  - Training loop with Adam + cosine annealing
  - Exact solution u(x,t) = sin(x - beta*t)
  - Fourier analysis utilities (power spectrum, spectral residual, cutoff)
  - L2 relative error metric
"""

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


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# ===================================================================
# Activation registry
# ===================================================================

class SinActivation(nn.Module):
    """Sinusoidal activation (SIREN-style)."""
    def forward(self, x):
        return torch.sin(x)


class FourierFeaturesLayer(nn.Module):
    """
    Random Fourier Feature embedding.

    Maps input (x, t) ∈ R^2 → R^(2*n_features) via:
        [cos(B @ input), sin(B @ input)]
    where B ~ N(0, sigma^2) is a fixed random projection matrix.
    """
    def __init__(self, in_dim: int = 2, n_features: int = 64, sigma: float = 1.0):
        super().__init__()
        self.n_features = n_features
        # Fixed (non-trainable) random projection
        B = torch.randn(n_features, in_dim) * sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self):
        return 2 * self.n_features

    def forward(self, x):
        # x shape: (batch, in_dim)
        proj = x @ self.B.T  # (batch, n_features)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def get_activation(name: str):
    """Return an activation module by name."""
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


# ===================================================================
# PINN Network
# ===================================================================

class AdvectionPINN(nn.Module):
    """
    Physics-Informed Neural Network for the 1D advection equation:
        u_t + beta * u_x = 0
        u(x, 0) = sin(x)         (initial condition)
        u(0, t) = u(2*pi, t)     (periodic BC)
    Exact solution: u(x, t) = sin(x - beta * t)

    Parameters
    ----------
    n_hidden : int
        Number of hidden layers.
    n_neurons : int
        Neurons per hidden layer.
    activation : str
        Activation function name ('tanh', 'sin', 'swish', 'gelu').
    fourier_features : bool
        If True, prepend a FourierFeaturesLayer before the MLP.
    fourier_sigma : float
        Scale of the random Fourier projection (only used if fourier_features=True).
    fourier_n_features : int
        Number of Fourier feature pairs (output dim = 2 * n_features).
    """

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

        # Optional Fourier feature embedding
        if fourier_features:
            self.ff_layer = FourierFeaturesLayer(
                in_dim=2,
                n_features=fourier_n_features,
                sigma=fourier_sigma,
            )
            in_dim = self.ff_layer.out_dim
        else:
            self.ff_layer = None
            in_dim = 2  # (x, t)

        # Input layer
        layers.append(nn.Linear(in_dim, n_neurons))
        layers.append(act_cls())

        # Hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(act_cls())

        # Output layer
        layers.append(nn.Linear(n_neurons, 1))

        self.net = nn.Sequential(*layers)

        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (N, 1)
        t : Tensor, shape (N, 1)

        Returns
        -------
        u : Tensor, shape (N, 1)
        """
        inp = torch.cat([x, t], dim=1)
        if self.ff_layer is not None:
            inp = self.ff_layer(inp)
        return self.net(inp)


# ===================================================================
# PDE residual via autograd
# ===================================================================

def advection_residual(model, x, t, beta):
    """
    Compute the PDE residual: r = u_t + beta * u_x.

    Parameters
    ----------
    model : AdvectionPINN
    x : Tensor (N, 1), requires_grad=True
    t : Tensor (N, 1), requires_grad=True
    beta : float

    Returns
    -------
    residual : Tensor (N, 1)
    """
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


# ===================================================================
# Exact solution
# ===================================================================

def exact_solution(x, t, beta):
    """u(x, t) = sin(x - beta * t)"""
    return np.sin(x - beta * t)


def exact_solution_torch(x, t, beta):
    """u(x, t) = sin(x - beta * t) — PyTorch version."""
    return torch.sin(x - beta * t)


# ===================================================================
# Sampling utilities
# ===================================================================

def sample_collocation(n_points, x_range=(0, 2 * np.pi), t_range=(0, 2),
                       method="lhs"):
    """
    Sample collocation points in the (x, t) domain.

    Parameters
    ----------
    n_points : int
    x_range : tuple
    t_range : tuple
    method : str — 'lhs' (Latin Hypercube) or 'random'

    Returns
    -------
    x, t : Tensors on DEVICE, shape (n_points, 1), requires_grad=True
    """
    if method == "lhs":
        # Simple stratified sampling (approximate LHS)
        n = n_points
        perms = np.random.permutation(n)
        x_vals = (perms + np.random.rand(n)) / n
        perms = np.random.permutation(n)
        t_vals = (perms + np.random.rand(n)) / n
        # Scale to domain
        x_vals = x_range[0] + x_vals * (x_range[1] - x_range[0])
        t_vals = t_range[0] + t_vals * (t_range[1] - t_range[0])
    else:
        x_vals = np.random.uniform(*x_range, n_points)
        t_vals = np.random.uniform(*t_range, n_points)

    x = torch.tensor(x_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    t = torch.tensor(t_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    return x, t


def sample_initial_condition(n_points, x_range=(0, 2 * np.pi)):
    """Sample IC points at t=0."""
    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    x = torch.tensor(x_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    t = torch.zeros(n_points, 1, dtype=DTYPE, device=DEVICE).requires_grad_(True)
    return x, t


def sample_boundary_condition(n_points, x_range=(0, 2 * np.pi), t_range=(0, 2)):
    """Sample periodic BC points: u(0, t) = u(2*pi, t)."""
    t_vals = np.linspace(t_range[0], t_range[1], n_points)

    t_left = torch.tensor(t_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    x_left = torch.full((n_points, 1), x_range[0], dtype=DTYPE, device=DEVICE).requires_grad_(True)

    t_right = torch.tensor(t_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
    x_right = torch.full((n_points, 1), x_range[1], dtype=DTYPE, device=DEVICE).requires_grad_(True)

    return x_left, t_left, x_right, t_right


# ===================================================================
# Training
# ===================================================================

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
    """
    Train a PINN on the advection equation.

    Parameters
    ----------
    model : AdvectionPINN
    beta : float — advection speed
    n_epochs : int — number of Adam steps
    lr : float — initial learning rate
    lr_min : float — minimum learning rate for cosine annealing
    n_collocation : int — interior collocation points
    n_ic : int — initial condition points
    n_bc : int — boundary condition points per side
    resample_every : int — resample collocation points every N epochs
    w_pde : float — PDE loss weight
    w_ic : float — IC loss weight
    w_bc : float — BC loss weight
    log_every : int — print loss every N epochs
    verbose : bool

    Returns
    -------
    dict with keys:
        'model': trained model
        'loss_history': list of total loss per epoch
        'pde_loss_history': list
        'ic_loss_history': list
        'bc_loss_history': list
        'training_time': float (seconds)
    """
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min
    )

    loss_history = []
    pde_loss_history = []
    ic_loss_history = []
    bc_loss_history = []

    # Initial sampling
    x_col, t_col = sample_collocation(n_collocation)
    x_ic, t_ic = sample_initial_condition(n_ic)
    x_bl, t_bl, x_br, t_br = sample_boundary_condition(n_bc)

    # IC target: u(x, 0) = sin(x)
    u_ic_target = torch.sin(x_ic).detach()

    start_time = time.time()

    for epoch in range(n_epochs):
        # Resample collocation points periodically
        if epoch > 0 and epoch % resample_every == 0:
            x_col, t_col = sample_collocation(n_collocation)

        optimizer.zero_grad()

        # PDE residual loss
        residual = advection_residual(model, x_col, t_col, beta)
        loss_pde = torch.mean(residual ** 2)

        # IC loss: u(x, 0) = sin(x)
        u_ic_pred = model(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)

        # Periodic BC loss: u(0, t) = u(2*pi, t)
        u_left = model(x_bl, t_bl)
        u_right = model(x_br, t_br)
        loss_bc = torch.mean((u_left - u_right) ** 2)

        # Total loss
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


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_on_grid(model, beta, nx=256, nt=100,
                     x_range=(0, 2 * np.pi), t_range=(0, 2)):
    """
    Evaluate the PINN and exact solution on a dense grid.

    Returns
    -------
    dict with keys:
        'x': 1D array (nx,)
        't': 1D array (nt,)
        'u_pred': 2D array (nx, nt)
        'u_exact': 2D array (nx, nt)
        'l2_error': float (relative L2 error)
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt)
    X, T = np.meshgrid(x, t, indexing="ij")  # (nx, nt)

    x_flat = X.flatten()
    t_flat = T.flatten()

    x_t = torch.tensor(x_flat, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    t_t = torch.tensor(t_flat, dtype=DTYPE, device=DEVICE).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_t, t_t).cpu().numpy().reshape(nx, nt)

    u_exact = exact_solution(X, T, beta)

    # Relative L2 error
    l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

    return {
        "x": x,
        "t": t,
        "u_pred": u_pred,
        "u_exact": u_exact,
        "l2_error": l2_error,
    }


# ===================================================================
# Fourier analysis utilities
# ===================================================================

def compute_spectrum(signal):
    """
    Compute the one-sided power spectrum of a 1D signal.

    Parameters
    ----------
    signal : 1D array of length N

    Returns
    -------
    freqs : array of frequency bins (non-negative)
    power : array of |FFT|^2 values (one-sided, normalized)
    """
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    power = (np.abs(fft_vals) ** 2) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / N)  # frequency in cycles per domain
    return freqs, power


def spectral_residual(pred, exact):
    """
    Compute the Fourier power spectrum of the prediction error.

    Parameters
    ----------
    pred : 1D array
    exact : 1D array

    Returns
    -------
    freqs, power : arrays
    """
    residual = pred - exact
    return compute_spectrum(residual)


def find_cutoff_frequency(power_pred, power_exact, threshold=0.01):
    """
    Find the frequency index where PINN power drops below
    `threshold` fraction of the exact solution power.

    Parameters
    ----------
    power_pred : 1D array — PINN power spectrum
    power_exact : 1D array — exact solution power spectrum
    threshold : float — ratio threshold

    Returns
    -------
    cutoff_idx : int — index into the frequency array
        Returns len(power_pred) - 1 if no cutoff found.
    """
    # Avoid division by zero
    max_exact = np.max(power_exact)
    if max_exact == 0:
        return len(power_pred) - 1

    for i in range(len(power_pred)):
        if power_exact[i] > threshold * max_exact:
            if power_pred[i] < threshold * power_exact[i]:
                return i
    return len(power_pred) - 1


def dominant_frequency(beta):
    """
    The dominant spatial frequency of u(x,t) = sin(x - beta*t).

    On domain [0, 2*pi], the spatial frequency is 1 cycle per domain,
    but the effective frequency content scales with beta due to the
    temporal oscillation. For a snapshot at fixed t, the spatial
    wavenumber is k=1, but the solution propagates at speed beta.

    Returns the spatial frequency in cycles per domain length.
    """
    # The spatial wavenumber is always k=1 for sin(x - beta*t).
    # However, when we look at the full (x,t) solution, the
    # characteristic frequency is beta/(2*pi).
    return beta / (2 * np.pi)


# ===================================================================
# Saving utilities
# ===================================================================

def save_results(results_dict, filepath):
    """Save results to JSON, converting numpy types."""

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
    """Save model state dict."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": model.config,
    }, filepath)
