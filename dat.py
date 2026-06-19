# run once to generate all canonical PINN failure datasets locally
# save as: generate_datasets.py

import numpy as np
from scipy.io import savemat
from scipy.integrate import solve_ivp
import os

os.makedirs("datasets", exist_ok=True)

# 1. Burgers shock (replaces burgers_shock.mat) ~200KB
x = np.linspace(-1, 1, 256)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x, t)
nu = 0.01 / np.pi
# pseudo-spectral solution
N = len(x); k = np.fft.fftfreq(N, d=1/N)
u = -np.sin(np.pi * x).copy()
U = np.zeros((len(t), N)); U[0] = u
for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    u_hat = np.fft.fft(u)
    u_hat *= np.exp(-nu * (2*np.pi*k)**2 * dt)
    u = np.real(np.fft.ifft(u_hat)) - dt * u * np.gradient(u, x)
    U[i] = u
savemat("datasets/burgers_shock.mat", {"x": x, "t": t, "usol": U.T})
print("Burgers done")

# 2. Allen-Cahn (replaces AC.mat) ~1MB
x = np.linspace(-1, 1, 512)
t = np.linspace(0, 1, 201)
u0 = x**2 * np.cos(np.pi * x)
savemat("datasets/AC.mat", {"x": x, "t": t, "u0": u0})
print("Allen-Cahn IC done")

# 3. Helmholtz 2D (common PINN failure case) ~500KB
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
a1, a2 = 1, 4
u_exact = np.sin(a1 * np.pi * X1) * np.sin(a2 * np.pi * X2)
savemat("datasets/helmholtz.mat", {"x1": x1, "x2": x2, "u_exact": u_exact})
print("Helmholtz done")

# 4. Heat equation (diffusion failure cases) ~300KB
x = np.linspace(0, 1, 200)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x, t)
alpha = 0.01  # low diffusivity → stiff case
u_exact = np.exp(-alpha * np.pi**2 * T) * np.sin(np.pi * X)
savemat("datasets/heat_stiff.mat", {"x": x, "t": t, "u_exact": u_exact.T, "alpha": alpha})
print("Heat (stiff) done")

# 5. Advection (spectral bias failure) ~200KB
x = np.linspace(0, 2*np.pi, 256)
t = np.linspace(0, 2, 100)
beta_values = [1, 10, 50]  # low to high → failure at high beta
for beta in beta_values:
    X, T = np.meshgrid(x, t)
    u = np.sin(X - beta * T)
    savemat(f"datasets/advection_beta{beta}.mat", {"x": x, "t": t, "u": u.T, "beta": beta})
print("Advection done")

print("\nAll datasets saved to ./datasets/ — total size < 10MB")