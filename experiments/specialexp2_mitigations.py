import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import json
import time
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from pinn_core import (
    DEVICE, DTYPE, AdvectionPINN, advection_residual,
    sample_collocation, sample_initial_condition,
    sample_boundary_condition, exact_solution,
    evaluate_on_grid, save_results,
)
from pinn_equations import (
    GenericPINN, burgers_residual, sample_burgers_domain,
    evaluate_burgers, load_burgers_reference, BURGERS_NU,
    helmholtz_residual, helmholtz_source, helmholtz_exact,
    sample_helmholtz_domain, evaluate_helmholtz,
    HELMHOLTZ_K_SQ, HELMHOLTZ_A1, HELMHOLTZ_A2,
)
from plot_utils import savefig, setup_style
setup_style()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "specialexp2"
SEED = 42
def load_checkpoint(path):
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}
def save_checkpoint(path, data):
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
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_default)
def pcgrad_project(grads_list):
    projected = [g.clone() for g in grads_list]
    n = len(projected)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = torch.dot(projected[i], grads_list[j])
            if dot < 0:
                projected[i] = projected[i] - (dot / (grads_list[j].norm()**2 + 1e-10)) * grads_list[j]
    return sum(projected)
def get_flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=False,
                                retain_graph=True, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p).flatten())
        else:
            flat.append(g.flatten())
    return torch.cat(flat)
def run_mitigation1():
    print(f"\n{'━' * 60}")
    print("MITIGATION 1: PCGrad for Gradient Pathology (Burgers)")
    print(f"{'━' * 60}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_epochs = 30000
    lr = 1e-3
    x_ref, t_ref, u_ref = load_burgers_reference()
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) =        sample_burgers_domain(10000, 200, 200)
    results = {"baseline": {}, "pcgrad": {}}
    for method in ["baseline", "pcgrad"]:
        print(f"\n  Training: {method}")
        torch.manual_seed(SEED)
        model = GenericPINN(in_dim=2, out_dim=1, n_hidden=4,
                            n_neurons=64, activation="tanh").to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-5)
        loss_hist = []
        cosine_hist = []
        params = list(model.parameters())
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            res = burgers_residual(model, x_int, t_int, BURGERS_NU)
            loss_pde = torch.mean(res ** 2)
            loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
            loss_bc  = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
            if epoch % 500 == 0:
                g_p = get_flat_grad(loss_pde, params).detach()
                g_b = get_flat_grad(loss_bc, params).detach()
                cos_sim = float(torch.dot(g_p, g_b) / (g_p.norm() * g_b.norm() + 1e-10))
                cosine_hist.append({"epoch": epoch, "cosine": cos_sim})
                optimizer.zero_grad()
            if method == "pcgrad":
                g_pde = get_flat_grad(loss_pde, params)
                g_ic  = get_flat_grad(10 * loss_ic, params)
                g_bc  = get_flat_grad(loss_bc, params)
                combined = pcgrad_project([g_pde, g_ic, g_bc])
                optimizer.zero_grad()
                idx = 0
                for p in params:
                    numel = p.numel()
                    p.grad = combined[idx:idx + numel].reshape(p.shape).clone()
                    idx += numel
            else:
                loss = loss_pde + 10 * loss_ic + loss_bc
                loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss = loss_pde.item() + 10 * loss_ic.item() + loss_bc.item()
            loss_hist.append(total_loss)
            if epoch % 10000 == 0:
                print(f"    [{epoch:6d}/{n_epochs}] Loss={total_loss:.4e}")
        _, l2 = evaluate_burgers(model, x_ref, t_ref, u_ref)
        print(f"  → {method}: L2={l2:.6f}")
        results[method] = {
            "l2_error": float(l2),
            "loss_history": loss_hist,
            "cosine_history": cosine_hist,
        }
        del model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for method, color, label in [("baseline", "#D32F2F", "Baseline"),
                                  ("pcgrad", "#2E7D32", "PCGrad")]:
        epochs = [c["epoch"] for c in results[method]["cosine_history"]]
        cosines = [c["cosine"] for c in results[method]["cosine_history"]]
        ax.plot(epochs, cosines, color=color, linewidth=1.5, label=label, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(-0.5, color="orange", linestyle=":", alpha=0.5,
               label="Conflict threshold (-0.5)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cosine Similarity (g_pde, g_bc)", fontsize=11)
    ax.set_title("Gradient Conflict Over Training", fontweight="bold")
    ax.legend(fontsize=9)
    ax = axes[1]
    step = 100
    for method, color, label in [("baseline", "#D32F2F", "Baseline"),
                                  ("pcgrad", "#2E7D32", "PCGrad")]:
        h = results[method]["loss_history"]
        ax.semilogy(range(0, len(h), step), h[::step],
                    color=color, linewidth=1.5, label=label, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.set_title("Loss Trajectories", fontweight="bold")
    ax.legend(fontsize=9)
    fig.suptitle(
        f"Mitigation 1: PCGrad for Gradient Pathology\n"
        f"Baseline L2={results['baseline']['l2_error']:.4f}  |  "
        f"PCGrad L2={results['pcgrad']['l2_error']:.4f}",
        fontweight="bold", fontsize=13)
    savefig(fig, OUTPUT_DIR / "mit1_pcgrad.png")
    improvement = (1 - results["pcgrad"]["l2_error"] / results["baseline"]["l2_error"]) * 100
    return {
        "failure_mode": "Gradient Pathology",
        "method": "PCGrad",
        "baseline_l2": results["baseline"]["l2_error"],
        "mitigated_l2": results["pcgrad"]["l2_error"],
        "improvement_pct": float(improvement),
        "verdict": "Success" if improvement > 50 else ("Partial" if improvement > 10 else "Fail"),
        "cosine_history_baseline": results["baseline"]["cosine_history"],
        "cosine_history_pcgrad": results["pcgrad"]["cosine_history"],
    }
def run_mitigation2():
    print(f"\n{'━' * 60}")
    print("MITIGATION 2: Adaptive Collocation for Spectral Bias (β=30)")
    print(f"{'━' * 60}")
    beta = 30
    n_epochs = 20000
    n_col = 1000
    adapt_every = 2000
    n_adapt_rounds = n_epochs // adapt_every
    lr = 1e-3
    results = {"static": {}, "adaptive": {}}
    for method in ["static", "adaptive"]:
        print(f"\n  Training: {method}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = AdvectionPINN(n_hidden=4, n_neurons=64, activation="tanh").to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-5)
        x_col, t_col = sample_collocation(n_col)
        x_ic, t_ic = sample_initial_condition(200)
        x_bl, t_bl, x_br, t_br = sample_boundary_condition(200)
        u_ic_target = torch.sin(x_ic).detach()
        loss_hist = []
        l2_per_round = []
        collocation_snapshots = {}
        for epoch in range(n_epochs):
            if method == "adaptive" and epoch > 0 and epoch % adapt_every == 0:
                round_num = epoch // adapt_every
                model.eval()
                x_dense, t_dense = sample_collocation(5000, method="random")
                with torch.enable_grad():
                    res_dense = advection_residual(model, x_dense, t_dense, beta)
                    res_mag = (res_dense ** 2).detach().cpu().numpy().flatten()
                n_top = int(0.20 * len(res_mag))
                top_idx = np.argsort(res_mag)[-n_top:]
                x_top = x_dense[top_idx].detach().cpu().numpy()
                t_top = t_dense[top_idx].detach().cpu().numpy()
                with torch.enable_grad():
                    res_current = advection_residual(model, x_col, t_col, beta)
                    res_cur_mag = (res_current ** 2).detach().cpu().numpy().flatten()
                n_replace = int(0.20 * n_col)
                bot_idx = np.argsort(res_cur_mag)[:n_replace]
                centers = np.column_stack([x_top.flatten(), t_top.flatten()])
                chosen = centers[np.random.choice(len(centers), n_replace, replace=True)]
                new_x = chosen[:, 0:1] + np.random.randn(n_replace, 1) * 0.05
                new_t = chosen[:, 1:2] + np.random.randn(n_replace, 1) * 0.05
                new_x = np.clip(new_x, 0, 2 * np.pi)
                new_t = np.clip(new_t, 0, 2)
                x_col_np = x_col.detach().cpu().numpy()
                t_col_np = t_col.detach().cpu().numpy()
                x_col_np[bot_idx] = new_x
                t_col_np[bot_idx] = new_t
                x_col = torch.tensor(x_col_np, dtype=DTYPE, device=DEVICE).requires_grad_(True)
                t_col = torch.tensor(t_col_np, dtype=DTYPE, device=DEVICE).requires_grad_(True)
                if round_num in [0, 3, 6, n_adapt_rounds - 1]:
                    collocation_snapshots[round_num] = {
                        "x": x_col.detach().cpu().numpy().flatten().tolist(),
                        "t": t_col.detach().cpu().numpy().flatten().tolist(),
                    }
                model.train()
                ev = evaluate_on_grid(model, beta)
                l2_per_round.append({"round": round_num, "l2": ev["l2_error"]})
                print(f"    Round {round_num}: L2={ev['l2_error']:.6f}")
            optimizer.zero_grad()
            res = advection_residual(model, x_col, t_col, beta)
            loss_pde = torch.mean(res ** 2)
            u_ic_pred = model(x_ic, t_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
            u_left = model(x_bl, t_bl)
            u_right = model(x_br, t_br)
            loss_bc = torch.mean((u_left - u_right) ** 2)
            loss = loss_pde + 10 * loss_ic + loss_bc
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_hist.append(loss.item())
            if epoch % 5000 == 0:
                print(f"    [{epoch:6d}/{n_epochs}] Loss={loss.item():.4e}")
        ev = evaluate_on_grid(model, beta)
        final_l2 = ev["l2_error"]
        print(f"  → {method}: L2={final_l2:.6f}")
        results[method] = {
            "l2_error": float(final_l2),
            "loss_history": loss_hist,
            "l2_per_round": l2_per_round,
            "collocation_snapshots": collocation_snapshots,
        }
        del model
    fig, ax = plt.subplots(figsize=(10, 6))
    if results["adaptive"]["l2_per_round"]:
        rounds = [r["round"] for r in results["adaptive"]["l2_per_round"]]
        l2s = [r["l2"] for r in results["adaptive"]["l2_per_round"]]
        ax.plot(rounds, l2s, "o-", color="#2E7D32", linewidth=2,
                markersize=8, label="Adaptive collocation")
    ax.axhline(results["static"]["l2_error"], color="#D32F2F",
               linestyle="--", linewidth=2,
               label=f"Static baseline (L2={results['static']['l2_error']:.4f})")
    ax.axhline(0.5, color="orange", linestyle=":", alpha=0.6,
               label="Success threshold (L2=0.5)")
    ax.set_xlabel("Adaptation Round", fontsize=12)
    ax.set_ylabel("L2 Relative Error", fontsize=12)
    ax.set_title(
        f"Adaptive Collocation vs Static (β={beta})\n"
        f"Static L2={results['static']['l2_error']:.4f}  |  "
        f"Adaptive final L2={results['adaptive']['l2_error']:.4f}",
        fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    savefig(fig, OUTPUT_DIR / "mit2_adaptive_collocation.png")
    improvement = (1 - results["adaptive"]["l2_error"] / results["static"]["l2_error"]) * 100
    return {
        "failure_mode": "Spectral Bias",
        "method": "Adaptive Collocation",
        "baseline_l2": results["static"]["l2_error"],
        "mitigated_l2": results["adaptive"]["l2_error"],
        "improvement_pct": float(improvement),
        "verdict": "Success" if results["adaptive"]["l2_error"] < 0.5 else "Partial" if improvement > 10 else "Fail",
    }
class HardBCHelmholtzPINN(nn.Module):
    def __init__(self, n_hidden=4, n_neurons=64, activation="tanh"):
        super().__init__()
        self.net = GenericPINN(in_dim=2, out_dim=1,
                               n_hidden=n_hidden, n_neurons=n_neurons,
                               activation=activation)
    def forward(self, x1, x2):
        D = (1 - x1**2) * (1 - x2**2)
        N = self.net(x1, x2)
        return D * N
def run_mitigation3():
    print(f"\n{'━' * 60}")
    print("MITIGATION 3: Hard BC Constraints (Helmholtz)")
    print(f"{'━' * 60}")
    n_epochs = 15000
    lr = 1e-3
    total_budget = 1000
    results = {}
    print("\n  Training: Soft constraint (99% BC — worst case from Exp 9)")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_bc_soft = int(total_budget * 0.99)
    n_int_soft = max(1, total_budget - n_bc_soft)
    model_soft = GenericPINN(in_dim=2, out_dim=1, n_hidden=4,
                              n_neurons=64, activation="tanh").to(DEVICE)
    optimizer = torch.optim.Adam(model_soft.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    (x1i, x2i), (x1b, x2b, ub) = sample_helmholtz_domain(n_int_soft, n_bc_soft)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        if n_int_soft > 0:
            res = helmholtz_residual(model_soft, x1i, x2i)
            loss_pde = torch.mean(res ** 2)
        else:
            loss_pde = torch.tensor(0.0, device=DEVICE)
        loss_bc = torch.mean((model_soft(x1b, x2b) - ub) ** 2)
        loss = loss_pde + 10 * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 5000 == 0:
            print(f"    [{epoch:6d}/{n_epochs}] Loss={loss.item():.4e}")
    ev_soft = evaluate_helmholtz(model_soft)
    print(f"  → Soft (99% BC): L2={ev_soft['l2_error']:.6f}")
    results["soft_worst"] = {
        "l2_error": float(ev_soft["l2_error"]),
        "pointwise_error": ev_soft["pointwise_error"].tolist(),
        "config": f"BC={n_bc_soft}, Int={n_int_soft}",
    }
    print("\n  Training: Soft constraint (15% BC — near optimal)")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_bc_opt = int(total_budget * 0.15)
    n_int_opt = total_budget - n_bc_opt
    model_opt = GenericPINN(in_dim=2, out_dim=1, n_hidden=4,
                             n_neurons=64, activation="tanh").to(DEVICE)
    optimizer = torch.optim.Adam(model_opt.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    (x1i, x2i), (x1b, x2b, ub) = sample_helmholtz_domain(n_int_opt, n_bc_opt)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        res = helmholtz_residual(model_opt, x1i, x2i)
        loss_pde = torch.mean(res ** 2)
        loss_bc = torch.mean((model_opt(x1b, x2b) - ub) ** 2)
        loss = loss_pde + 10 * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 5000 == 0:
            print(f"    [{epoch:6d}/{n_epochs}] Loss={loss.item():.4e}")
    ev_opt = evaluate_helmholtz(model_opt)
    print(f"  → Soft (15% BC): L2={ev_opt['l2_error']:.6f}")
    results["soft_optimal"] = {
        "l2_error": float(ev_opt["l2_error"]),
        "pointwise_error": ev_opt["pointwise_error"].tolist(),
        "config": f"BC={n_bc_opt}, Int={n_int_opt}",
    }
    print("\n  Training: Hard BC constraint (all 1000 points interior)")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_hard = HardBCHelmholtzPINN(n_hidden=4, n_neurons=64,
                                      activation="tanh").to(DEVICE)
    optimizer = torch.optim.Adam(model_hard.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    pts = np.random.rand(total_budget, 2) * 2 - 1
    x1_hard = torch.tensor(pts[:, 0:1], dtype=DTYPE, device=DEVICE).requires_grad_(True)
    x2_hard = torch.tensor(pts[:, 1:2], dtype=DTYPE, device=DEVICE).requires_grad_(True)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        u = model_hard(x1_hard, x2_hard)
        ones = torch.ones_like(u)
        u_x1 = torch.autograd.grad(u, x1_hard, ones, create_graph=True, retain_graph=True)[0]
        u_x1x1 = torch.autograd.grad(u_x1, x1_hard, torch.ones_like(u_x1),
                                       create_graph=True, retain_graph=True)[0]
        u_x2 = torch.autograd.grad(u, x2_hard, ones, create_graph=True, retain_graph=True)[0]
        u_x2x2 = torch.autograd.grad(u_x2, x2_hard, torch.ones_like(u_x2),
                                       create_graph=True, retain_graph=True)[0]
        q = helmholtz_source(x1_hard, x2_hard)
        res = u_x1x1 + u_x2x2 + HELMHOLTZ_K_SQ * u - q
        loss = torch.mean(res ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 5000 == 0:
            print(f"    [{epoch:6d}/{n_epochs}] Loss={loss.item():.4e}")
    model_hard.eval()
    nx, ny = 100, 100
    x1_eval = np.linspace(-1, 1, nx)
    x2_eval = np.linspace(-1, 1, ny)
    X1, X2 = np.meshgrid(x1_eval, x2_eval, indexing="ij")
    x1_t = torch.tensor(X1.flatten()[:, None], dtype=DTYPE, device=DEVICE)
    x2_t = torch.tensor(X2.flatten()[:, None], dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        u_pred_hard = model_hard(x1_t, x2_t).cpu().numpy().reshape(nx, ny)
    u_exact_h = helmholtz_exact(X1, X2)
    l2_hard = np.linalg.norm(u_pred_hard - u_exact_h) / (np.linalg.norm(u_exact_h) + 1e-30)
    pe_hard = np.abs(u_pred_hard - u_exact_h)
    print(f"  → Hard BC: L2={l2_hard:.6f}")
    results["hard_bc"] = {
        "l2_error": float(l2_hard),
        "pointwise_error": pe_hard.tolist(),
        "config": f"All {total_budget} pts interior, BCs exact by construction",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    configs = [
        ("soft_worst", "Soft 99% BC\n(worst case)"),
        ("soft_optimal", "Soft 15% BC\n(near optimal)"),
        ("hard_bc", "Hard BC\n(exact by construction)"),
    ]
    all_errs = [np.array(results[k]["pointwise_error"]) for k, _ in configs]
    vmax = float(np.percentile(np.concatenate([e.flatten() for e in all_errs]), 95))
    for ax, (key, title), err in zip(axes, configs, all_errs):
        im = ax.imshow(err.T, extent=[-1, 1, -1, 1], origin="lower",
                       cmap="hot", aspect="equal", vmin=0, vmax=vmax)
        ax.set_title(f"{title}\nL2={results[key]['l2_error']:.4f}",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    fig.colorbar(plt.cm.ScalarMappable(cmap="hot",
                 norm=plt.Normalize(0, vmax)), cax=cbar_ax,
                 label=f"|Error| (shared, 95th pct clip={vmax:.3f})")
    fig.suptitle("Mitigation 3: Hard BC Constraints vs Soft Constraints",
                 fontweight="bold", fontsize=13)
    savefig(fig, OUTPUT_DIR / "mit3_hard_bc_heatmaps.png")
    improvement = (1 - l2_hard / results["soft_worst"]["l2_error"]) * 100
    return {
        "failure_mode": "Boundary/Interior Ratio Sensitivity",
        "method": "Hard BC Constraints",
        "baseline_l2": results["soft_worst"]["l2_error"],
        "mitigated_l2": float(l2_hard),
        "improvement_pct": float(improvement),
        "optimal_soft_l2": results["soft_optimal"]["l2_error"],
        "verdict": "Success" if improvement > 50 else ("Partial" if improvement > 10 else "Fail"),
    }
def run_mitigation4():
    print(f"\n{'━' * 60}")
    print("MITIGATION 4: Causal Training (Advection β=50, t∈[0,5])")
    print(f"{'━' * 60}")
    beta = 50
    t_range = (0, 5)
    n_epochs = 20000
    n_col = 8000
    lr = 1e-3
    results = {"standard": {}, "causal": {}}
    for method in ["standard", "causal"]:
        print(f"\n  Training: {method}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = AdvectionPINN(n_hidden=4, n_neurons=64,
                              activation="tanh").to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-5)
        x_col, t_col = sample_collocation(n_col, t_range=t_range, method="random")
        x_ic, t_ic = sample_initial_condition(300)
        x_bl, t_bl, x_br, t_br = sample_boundary_condition(300, t_range=t_range)
        u_ic_target = torch.sin(x_ic).detach()
        loss_hist = []
        residual_snapshots = {}
        epsilon = 1.0
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            res = advection_residual(model, x_col, t_col, beta)
            res_sq = res ** 2
            if method == "causal":
                t_vals = t_col.detach()
                sort_idx = torch.argsort(t_vals.squeeze())
                res_sorted = res_sq[sort_idx].detach()
                dt = t_range[1] / n_col
                cum_res = torch.cumsum(res_sorted, dim=0) * dt
                weights = torch.exp(-epsilon * cum_res)
                inv_idx = torch.empty_like(sort_idx)
                inv_idx[sort_idx] = torch.arange(len(sort_idx), device=DEVICE)
                w = weights[inv_idx]
                loss_pde = torch.mean(w.detach() * res_sq)
                epsilon = 1.0 + (100.0 - 1.0) * epoch / n_epochs
            else:
                loss_pde = torch.mean(res_sq)
            u_ic_pred = model(x_ic, t_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
            u_left = model(x_bl, t_bl)
            u_right = model(x_br, t_br)
            loss_bc = torch.mean((u_left - u_right) ** 2)
            loss = loss_pde + 10 * loss_ic + loss_bc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_hist.append(loss.item())
            if epoch % 5000 == 0:
                print(f"    [{epoch:6d}/{n_epochs}] Loss={loss.item():.4e}")
            if epoch + 1 in [5000, 10000, 20000]:
                model.eval()
                nx, nt = 100, 200
                x_v = np.linspace(0, 2 * np.pi, nx)
                t_v = np.linspace(0, t_range[1], nt)
                XX, TT = np.meshgrid(x_v, t_v, indexing="ij")
                x_f = torch.tensor(XX.flatten(), dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
                t_f = torch.tensor(TT.flatten(), dtype=DTYPE, device=DEVICE).unsqueeze(1).requires_grad_(True)
                with torch.enable_grad():
                    r = advection_residual(model, x_f, t_f, beta)
                    r_map = (r ** 2).detach().cpu().numpy().reshape(nx, nt)
                residual_snapshots[epoch + 1] = r_map.tolist()
                model.train()
        ev = evaluate_on_grid(model, beta, t_range=t_range)
        final_l2 = ev["l2_error"]
        print(f"  → {method}: L2={final_l2:.6f}")
        x_v = np.linspace(0, 2 * np.pi, 200)
        t_v = np.linspace(0, t_range[1], 100)
        l2_vs_t = []
        model.eval()
        for ti in t_v:
            xi = torch.tensor(x_v, dtype=DTYPE, device=DEVICE).unsqueeze(1)
            ti_t = torch.full((len(x_v), 1), ti, dtype=DTYPE, device=DEVICE)
            with torch.no_grad():
                u_p = model(xi, ti_t).cpu().numpy().flatten()
            u_e = exact_solution(x_v, ti, beta)
            err = np.linalg.norm(u_p - u_e) / (np.linalg.norm(u_e) + 1e-10)
            l2_vs_t.append({"t": float(ti), "l2": float(err)})
        results[method] = {
            "l2_error": float(final_l2),
            "loss_history": loss_hist,
            "l2_vs_t": l2_vs_t,
        }
        del model
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, color, label in [("standard", "#D32F2F", "Standard"),
                                  ("causal", "#2E7D32", "Causal")]:
        ts = [r["t"] for r in results[method]["l2_vs_t"]]
        l2s = [r["l2"] for r in results[method]["l2_vs_t"]]
        ax.semilogy(ts, l2s, color=color, linewidth=2, label=label, alpha=0.8)
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("L2 Error at time t", fontsize=12)
    ax.set_title(
        f"Causal Training vs Standard (β={beta}, t∈[0,{t_range[1]}])\n"
        f"Standard L2={results['standard']['l2_error']:.4f}  |  "
        f"Causal L2={results['causal']['l2_error']:.4f}",
        fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    savefig(fig, OUTPUT_DIR / "mit4_causal_l2.png")
    improvement = (1 - results["causal"]["l2_error"] / results["standard"]["l2_error"]) * 100
    l2_at_end_std = results["standard"]["l2_vs_t"][-1]["l2"]
    l2_at_end_cau = results["causal"]["l2_vs_t"][-1]["l2"]
    improvement_end = (1 - l2_at_end_cau / l2_at_end_std) * 100
    return {
        "failure_mode": "Temporal Integration Failure",
        "method": "Causal Training (Wang et al. 2022)",
        "baseline_l2": results["standard"]["l2_error"],
        "mitigated_l2": results["causal"]["l2_error"],
        "improvement_pct": float(improvement),
        "l2_at_t5_baseline": float(l2_at_end_std),
        "l2_at_t5_causal": float(l2_at_end_cau),
        "improvement_at_t5_pct": float(improvement_end),
        "verdict": "Success" if improvement_end > 50 else ("Partial" if improvement_end > 10 else "Fail"),
    }
def run_experiment():
    t_start = time.time()
    print("=" * 70)
    print("SPECIAL EXP 2: Targeted Mitigation Experiments")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "specialexp2_checkpoint.json"
    ckpt = load_checkpoint(checkpoint_path)
    all_mitigations = {}
    if "mit1" not in ckpt:
        mit1 = run_mitigation1()
        ckpt["mit1"] = mit1
        save_checkpoint(checkpoint_path, ckpt)
    else:
        mit1 = ckpt["mit1"]
        print("\n  Mitigation 1: [loaded from checkpoint]")
    all_mitigations["Gradient Pathology"] = mit1
    if "mit2" not in ckpt:
        mit2 = run_mitigation2()
        ckpt["mit2"] = mit2
        save_checkpoint(checkpoint_path, ckpt)
    else:
        mit2 = ckpt["mit2"]
        print("\n  Mitigation 2: [loaded from checkpoint]")
    all_mitigations["Spectral Bias"] = mit2
    if "mit3" not in ckpt:
        mit3 = run_mitigation3()
        ckpt["mit3"] = mit3
        save_checkpoint(checkpoint_path, ckpt)
    else:
        mit3 = ckpt["mit3"]
        print("\n  Mitigation 3: [loaded from checkpoint]")
    all_mitigations["BC Ratio Sensitivity"] = mit3
    if "mit4" not in ckpt:
        mit4 = run_mitigation4()
        ckpt["mit4"] = mit4
        save_checkpoint(checkpoint_path, ckpt)
    else:
        mit4 = ckpt["mit4"]
        print("\n  Mitigation 4: [loaded from checkpoint]")
    all_mitigations["Temporal Failure"] = mit4
    print("\n── Mitigation Summary ──")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    col_labels = ["Failure Mode", "Method", "Baseline L2", "Mitigated L2",
                  "Improvement %", "Verdict"]
    table_data = []
    for name, mit in all_mitigations.items():
        table_data.append([
            mit["failure_mode"],
            mit["method"],
            f"{mit['baseline_l2']:.4f}",
            f"{mit['mitigated_l2']:.4f}",
            f"{mit['improvement_pct']:.1f}%",
            mit["verdict"],
        ])
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for i, row in enumerate(table_data):
        verdict = row[-1]
        cell = table[i + 1, len(col_labels) - 1]
        if verdict == "Success":
            cell.set_facecolor("#C8E6C9")
        elif verdict == "Partial":
            cell.set_facecolor("#FFF9C4")
        else:
            cell.set_facecolor("#FFCDD2")
    fig.suptitle("Mitigation Summary: One Remedy Per Failure Mode",
                 fontweight="bold", fontsize=14, y=0.95)
    plt.tight_layout()
    savefig(fig, OUTPUT_DIR / "mitigation_summary.png")
    for name, mit in all_mitigations.items():
        print(f"  {mit['failure_mode']:35s} | "
              f"Base={mit['baseline_l2']:.4f} → {mit['mitigated_l2']:.4f} "
              f"({mit['improvement_pct']:+.1f}%) | {mit['verdict']}")
    results = {
        "experiment": "Targeted Mitigation Experiments",
        "version": "specialexp2",
        "mitigations": {
            "pcgrad": mit1,
            "adaptive_collocation": mit2,
            "hard_bc": mit3,
            "causal_training": mit4,
        },
        "summary_note": (
            "Four targeted mitigations, one per confirmed failure mode. "
            "Each compares a mitigated model against the failure baseline. "
            "Success = >50% L2 improvement, Partial = 10-50%, Fail = <10%."
        ),
    }
    save_results(results, OUTPUT_DIR / "specialexp2_mitigation_results.json")
    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"SPECIAL EXP 2 — COMPLETE")
    print(f"  Total wall time: {total_elapsed / 60:.1f} min")
    print(f"  Results → {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    return results
if __name__ == "__main__":
    run_experiment()
