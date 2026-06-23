
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

from pinn_core import DEVICE, DTYPE, save_results
from pinn_equations import (
    GenericPINN, train_heat_pinn, heat_exact, evaluate_heat,
    heat_residual, sample_heat_domain,
    HEAT_X_RANGE,
)
from plot_utils import savefig, setup_style

setup_style()

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

ALPHA = 0.01
T_TOTAL = 10.0
N_WINDOWS = 10
WINDOW_SIZE = T_TOTAL / N_WINDOWS       

N_HIDDEN = 4
N_NEURONS = 64
EPOCHS_SINGLE = 30000
EPOCHS_PER_WINDOW = 5000

CATASTROPHIC_THRESHOLD = 0.5       

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp12"


def evaluate_at_times(model, alpha, times, nx=200, x_range=HEAT_X_RANGE):
    """Return list of L2 relative errors at each requested time slice."""
    x = np.linspace(x_range[0], x_range[1], nx)
    errors = []
    model.eval()
    for t_val in times:
        x_t = torch.tensor(x[:, None], dtype=DTYPE, device=DEVICE)
        t_t = torch.full((nx, 1), t_val, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
        u_exact = heat_exact(x, np.full_like(x, t_val), alpha)
        l2 = (np.linalg.norm(u_pred - u_exact)
              / (np.linalg.norm(u_exact) + 1e-30))
        errors.append(float(l2))
    return errors


def find_catastrophic_time(eval_times, errors, threshold):

    for t, e in zip(eval_times, errors):
        if e > threshold:
            return True, float(t)
    return False, None


def fit_error_growth(times, errors):

    times  = np.array(times)
    errors = np.array(errors)
    mask   = (errors > 1e-10) & np.isfinite(errors)
    t_fit  = times[mask]
    e_fit  = errors[mask]

    if len(t_fit) < 3:
        return "insufficient_data", {}

    fits = {}


    try:
        def linear(t, a, b): return a * t + b
        popt, _ = curve_fit(linear, t_fit, e_fit,
                            p0=[0.01, 0.01], maxfev=5000)
        pred = linear(t_fit, *popt)
        ss_res = np.sum((e_fit - pred) ** 2)
        ss_tot = np.sum((e_fit - np.mean(e_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fits["linear"] = {"params": popt.tolist(), "r2": float(r2)}
    except Exception:
        fits["linear"] = {"r2": -1.0}

    try:
        def poly(t, a, b, c): return a * t**2 + b * t + c
        popt, _ = curve_fit(poly, t_fit, e_fit,
                            p0=[0.001, 0.01, 0.01], maxfev=5000)
        pred = poly(t_fit, *popt)
        ss_res = np.sum((e_fit - pred) ** 2)
        ss_tot = np.sum((e_fit - np.mean(e_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fits["polynomial"] = {"params": popt.tolist(), "r2": float(r2)}
    except Exception:
        fits["polynomial"] = {"r2": -1.0}

    try:
        def exponential(t, a, b): return a * np.exp(b * t)
        popt, _ = curve_fit(exponential, t_fit, e_fit,
                            p0=[0.01, 0.1], maxfev=5000)
        pred = exponential(t_fit, *popt)
        ss_res = np.sum((e_fit - pred) ** 2)
        ss_tot = np.sum((e_fit - np.mean(e_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fits["exponential"] = {"params": popt.tolist(), "r2": float(r2)}
    except Exception:
        fits["exponential"] = {"r2": -1.0}

    best = max(fits, key=lambda k: fits[k]["r2"])
    return best, fits


def train_single_domain(ckpt=None, ckpt_path=None, model_path=None):
    print("\n  Single-domain training [0, 10]...")
    model = GenericPINN(in_dim=2, out_dim=1, n_hidden=N_HIDDEN,
                        n_neurons=N_NEURONS, activation="tanh")
    if ckpt is not None and ckpt.get("single_completed"):
        if model_path and model_path.exists():
            model.load_state_dict(torch.load(model_path))
            print("    [Loaded from checkpoint]")
            return model, ckpt.get("single_loss", [])

    train_out = train_heat_pinn(
        model, alpha=ALPHA, n_epochs=EPOCHS_SINGLE,
        t_range=(0, T_TOTAL), n_int=10000, n_ic=200, n_bc=200,
        log_every=5000,
    )
    
    if model_path:
        torch.save(model.state_dict(), model_path)
    if ckpt is not None and ckpt_path:
        ckpt["single_completed"] = True
        ckpt["single_loss"] = train_out["loss_history"]
        import json
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt, f)
            
    return model, train_out["loss_history"]


def train_windowed(ckpt=None, ckpt_path=None, model_path=None):
    print("\n  Windowed training with transfer learning...")
    model = GenericPINN(in_dim=2, out_dim=1, n_hidden=N_HIDDEN,
                        n_neurons=N_NEURONS, activation="tanh")
    all_loss = []
    start_w = 0

    if ckpt is not None and "windowed_completed" in ckpt:
        start_w = ckpt["windowed_completed"]
        all_loss = ckpt.get("windowed_loss", [])
        if model_path and model_path.exists():
            model.load_state_dict(torch.load(model_path))
            print(f"    [Resuming from window {start_w}]")

    if start_w == N_WINDOWS:
        return model, all_loss

    for w in range(start_w, N_WINDOWS):
        t_start = w * WINDOW_SIZE
        t_end   = (w + 1) * WINDOW_SIZE
        print(f"\n    Window [{t_start:.1f}, {t_end:.1f}]")

        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS_PER_WINDOW, eta_min=1e-5)

        (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = \
            sample_heat_domain(5000, 200, 200,
                               x_range=HEAT_X_RANGE,
                               t_range=(t_start, t_end))

        if w > 0:
  
            x_ic_np = np.linspace(HEAT_X_RANGE[0], HEAT_X_RANGE[1], 200)
            x_ic = torch.tensor(x_ic_np[:, None], dtype=DTYPE,
                                device=DEVICE).requires_grad_(True)
            t_ic = torch.full((200, 1), t_start, dtype=DTYPE,
                              device=DEVICE).requires_grad_(True)
            u_ic = torch.tensor(
                heat_exact(x_ic_np, np.full(200, t_start), ALPHA)[:, None],
                dtype=DTYPE, device=DEVICE,
            )

        window_loss = []
        for epoch in range(EPOCHS_PER_WINDOW):
            optimizer.zero_grad()
            res      = heat_residual(model, x_int, t_int, ALPHA)
            loss_pde = torch.mean(res ** 2)
            loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
            loss_bc  = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
            loss     = loss_pde + 10 * loss_ic + loss_bc
            loss.backward()
            optimizer.step()
            scheduler.step()
            window_loss.append(loss.item())

        all_loss.extend(window_loss)
        print(f"      Final loss: {window_loss[-1]:.4e}")

        if model_path:
            torch.save(model.state_dict(), model_path)
        if ckpt is not None and ckpt_path:
            ckpt["windowed_completed"] = w + 1
            ckpt["windowed_loss"] = all_loss
            import json
            with open(ckpt_path, 'w') as f:
                json.dump(ckpt, f)

    return model, all_loss



def run_experiment():
    print("=" * 70)
    print("EXP 12: Temporal Stiffness / Long-Time Integration  [v2 — fixed]")
    print(f"Device : {DEVICE}")
    print(f"α = {ALPHA},  T = [0, {T_TOTAL}]")
    print(f"Catastrophic threshold : L2 > {CATASTROPHIC_THRESHOLD}")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_times = np.linspace(0.5, T_TOTAL, 20)

    import json
    ckpt_path = OUTPUT_DIR / "exp12_checkpoint.json"
    model_single_path = OUTPUT_DIR / "model_single.pt"
    model_windowed_path = OUTPUT_DIR / "model_windowed.pt"

    ckpt = {}
    if ckpt_path.exists():
        try:
            with open(ckpt_path, 'r') as f:
                ckpt = json.load(f)
        except Exception:
            pass

    model_single,   loss_single   = train_single_domain(ckpt, ckpt_path, model_single_path)
    model_windowed, loss_windowed = train_windowed(ckpt, ckpt_path, model_windowed_path)

    errors_single   = evaluate_at_times(model_single,   ALPHA, eval_times)
    errors_windowed = evaluate_at_times(model_windowed, ALPHA, eval_times)


    failed_single,   t_fail_single   = find_catastrophic_time(
        eval_times, errors_single,   CATASTROPHIC_THRESHOLD)
    failed_windowed, t_fail_windowed = find_catastrophic_time(
        eval_times, errors_windowed, CATASTROPHIC_THRESHOLD)


    print("\n  ── Catastrophic failure summary ──")
    if failed_single:
        print(f"  Single-domain : FAILED  at t = {t_fail_single:.2f}  "
              f"(L2 > {CATASTROPHIC_THRESHOLD})")
    else:
        peak_s = max(errors_single)
        print(f"  Single-domain : NO FAILURE  "
              f"(peak L2 = {peak_s:.6f}, threshold = {CATASTROPHIC_THRESHOLD})")

    if failed_windowed:
        print(f"  Windowed      : FAILED  at t = {t_fail_windowed:.2f}  "
              f"(L2 > {CATASTROPHIC_THRESHOLD})")
    else:
        peak_w = max(errors_windowed)
        print(f"  Windowed      : NO FAILURE  "
              f"(peak L2 = {peak_w:.6f}, threshold = {CATASTROPHIC_THRESHOLD})")


    best_fit_single,   fits_single   = fit_error_growth(eval_times, errors_single)
    best_fit_windowed, fits_windowed = fit_error_growth(eval_times, errors_windowed)

    print(f"\n  Single-domain error growth : {best_fit_single}  "
          f"(R²={fits_single.get(best_fit_single, {}).get('r2', 0):.4f})")
    print(f"  Windowed error growth      : {best_fit_windowed}  "
          f"(R²={fits_windowed.get(best_fit_windowed, {}).get('r2', 0):.4f})")

    print("\n── Generating plots ──")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(eval_times, errors_single, "o-",
                color="#D32F2F", linewidth=2, markersize=7,
                label="Single Domain [0, 10]")
    ax.semilogy(eval_times, errors_windowed, "s-",
                color="#1565C0", linewidth=2, markersize=7,
                label=f"Windowed ({N_WINDOWS} × Δt={WINDOW_SIZE})")
    ax.axhline(CATASTROPHIC_THRESHOLD, color="orange", linestyle="--",
               linewidth=2, alpha=0.7,
               label=f"Catastrophic threshold ({CATASTROPHIC_THRESHOLD})")

    if failed_single:
        ax.axvline(t_fail_single, color="#D32F2F", linestyle=":",
                   alpha=0.6, label=f"Single fails at t={t_fail_single:.2f}")
    if failed_windowed:
        ax.axvline(t_fail_windowed, color="#1565C0", linestyle=":",
                   alpha=0.6, label=f"Windowed fails at t={t_fail_windowed:.2f}")

 
    if not failed_single:
        ax.annotate("No catastrophic\nfailure (single)",
                    xy=(eval_times[-1], errors_single[-1]),
                    xytext=(-80, 20), textcoords="offset points",
                    fontsize=8, color="#D32F2F",
                    arrowprops=dict(arrowstyle="->", color="#D32F2F"))
    if not failed_windowed:
        ax.annotate("No catastrophic\nfailure (windowed)",
                    xy=(eval_times[-1], errors_windowed[-1]),
                    xytext=(-80, -40), textcoords="offset points",
                    fontsize=8, color="#1565C0",
                    arrowprops=dict(arrowstyle="->", color="#1565C0"))

    ax.set_xlabel("Time t")
    ax.set_ylabel("L2 Relative Error")
    ax.set_title("Error Accumulation Over Time", fontweight="bold")
    ax.legend(fontsize=9)
    savefig(fig, OUTPUT_DIR / "error_over_time.png")

    snapshot_times = [0.5, 2.0, 5.0, 8.0, T_TOTAL]
    x = np.linspace(HEAT_X_RANGE[0], HEAT_X_RANGE[1], 200)
    fig, axes = plt.subplots(2, len(snapshot_times),
                             figsize=(4 * len(snapshot_times), 7))

    for j, t_val in enumerate(snapshot_times):
        x_t = torch.tensor(x[:, None], dtype=DTYPE, device=DEVICE)
        t_t = torch.full((len(x), 1), t_val, dtype=DTYPE, device=DEVICE)
        model_single.eval()
        model_windowed.eval()
        with torch.no_grad():
            u_s = model_single(x_t,   t_t).cpu().numpy().flatten()
            u_w = model_windowed(x_t, t_t).cpu().numpy().flatten()
        u_e = heat_exact(x, np.full_like(x, t_val), ALPHA)

        axes[0, j].plot(x, u_e, "k-",  linewidth=2,   label="Exact")
        axes[0, j].plot(x, u_s, "r--", linewidth=1.5, label="PINN")
        axes[0, j].set_title(f"Single | t={t_val}")
        axes[0, j].legend(fontsize=7)

        axes[1, j].plot(x, u_e, "k-",  linewidth=2,   label="Exact")
        axes[1, j].plot(x, u_w, "b--", linewidth=1.5, label="Windowed")
        axes[1, j].set_title(f"Windowed | t={t_val}")
        axes[1, j].legend(fontsize=7)

    axes[0, 0].set_ylabel("u(x, t) — Single")
    axes[1, 0].set_ylabel("u(x, t) — Windowed")
    for ax in axes[1, :]:
        ax.set_xlabel("x")
    fig.suptitle("Solution Snapshots: Single Domain vs Windowed",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    savefig(fig, OUTPUT_DIR / "solution_snapshots.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, errors, label, fits, best_fit, color in [
        (axes[0], errors_single,   "Single Domain", fits_single,
         best_fit_single,   "#D32F2F"),
        (axes[1], errors_windowed, "Windowed",      fits_windowed,
         best_fit_windowed, "#1565C0"),
    ]:
        ax.semilogy(eval_times, errors, "o", color=color,
                    markersize=8, label="Data", zorder=5)
        t_dense = np.linspace(eval_times[0], eval_times[-1], 100)

        if best_fit in fits and fits[best_fit]["r2"] > 0:
            p  = fits[best_fit].get("params", [])
            r2 = fits[best_fit]["r2"]
            if best_fit == "linear" and len(p) == 2:
                y = np.maximum(p[0] * t_dense + p[1], 1e-15)
                ax.semilogy(t_dense, y, "--", color="green",
                            label=f"Linear (R²={r2:.3f})")
            elif best_fit == "polynomial" and len(p) == 3:
                y = np.maximum(p[0]*t_dense**2 + p[1]*t_dense + p[2], 1e-15)
                ax.semilogy(t_dense, y, "--", color="orange",
                            label=f"Poly (R²={r2:.3f})")
            elif best_fit == "exponential" and len(p) == 2:
                y = p[0] * np.exp(p[1] * t_dense)
                ax.semilogy(t_dense, y, "--", color="purple",
                            label=f"Exp (R²={r2:.3f})")

        ax.set_xlabel("Time t")
        ax.set_ylabel("L2 Error")
        ax.set_title(f"{label}\nBest fit: {best_fit}", fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Error Growth Characterization", fontweight="bold", fontsize=14)
    savefig(fig, OUTPUT_DIR / "error_growth_fit.png")

    results = {
        "experiment":   "Temporal Stiffness / Long-Time Integration",
        "version":      "v2-fixed",
        "alpha":        ALPHA,
        "t_total":      T_TOTAL,
        "eval_times":   eval_times.tolist(),
        "errors_single":   errors_single,
        "errors_windowed": errors_windowed,

        "single_domain_failed":        failed_single,
        "catastrophic_time_single":    t_fail_single,   
        "windowed_failed":             failed_windowed,
        "catastrophic_time_windowed":  t_fail_windowed,  
    

        "error_growth_single":   best_fit_single,
        "error_growth_windowed": best_fit_windowed,
        "growth_fits_single": {
            k: {"r2": v["r2"]} for k, v in fits_single.items()
        },
        "growth_fits_windowed": {
            k: {"r2": v["r2"]} for k, v in fits_windowed.items()
        },
        "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
        "peak_error_single":   max(errors_single),
        "peak_error_windowed": max(errors_windowed),
    }
    save_results(results, OUTPUT_DIR / "exp12_results.json")

    print(f"\n{'=' * 70}")
    print("EXP 12 — COMPLETE  [v2]")
    if failed_single:
        print(f"  Single-domain : catastrophic failure at t={t_fail_single:.2f}")
    else:
        print(f"  Single-domain : NO failure  (peak L2={max(errors_single):.6f})")
    if failed_windowed:
        print(f"  Windowed      : catastrophic failure at t={t_fail_windowed:.2f}")
    else:
        print(f"  Windowed      : NO failure  (peak L2={max(errors_windowed):.6f})")
    print(f"  Results → {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    return results


if __name__ == "__main__":
    run_experiment()
