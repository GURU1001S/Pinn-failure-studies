import sys, os, json, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pinn_core import DEVICE, DTYPE, save_results
from pinn_equations import (
    GenericPINN, burgers_residual, burgers_ic,
    sample_burgers_domain, evaluate_burgers, load_burgers_reference,
    BURGERS_NU,
)
from plot_utils import savefig, setup_style

setup_style()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

N_HIDDEN = 4
N_NEURONS = 64
N_EPOCHS = 20000
N_SEEDS = 50
LR = 1e-3
LR_MIN = 1e-5
CONVERGENCE_THRESHOLD = 0.3
DIVERGENCE_THRESHOLD = 10.0
RANDOM_STDS = [0.001, 0.01, 0.1, 1.0, 10.0]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp13"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"

_STOP_REQUESTED = False

def _handle_sigint(sig, frame):
    global _STOP_REQUESTED
    if not _STOP_REQUESTED:
        print("\n\n  ⚠  Ctrl+C received — finishing current seed then saving "
              "checkpoint.\n     Press Ctrl+C again to force-quit immediately.\n")
        _STOP_REQUESTED = True
    else:
        print("\n  Force-quitting now.")
        sys.exit(1)

signal.signal(signal.SIGINT, _handle_sigint)



def load_checkpoint():

    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            ckpt = json.load(f)

        ckpt["completed"] = {
            strat: {int(k): v for k, v in seeds.items()}
            for strat, seeds in ckpt["completed"].items()
        }
        n_done = sum(len(v) for v in ckpt["completed"].values())
        total = len(ckpt.get("config", {}).get("strategies", [])) * N_SEEDS
        print(f"\n  ✔ Checkpoint found: {n_done} / {total} runs already done.")
        print(f"    Resuming from: {CHECKPOINT_PATH}\n")
        return ckpt
    return {"completed": {}, "config": {}}


def save_checkpoint(ckpt):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(ckpt, f)
    tmp_path.replace(CHECKPOINT_PATH)


def is_done(ckpt, strategy, seed):

    return seed in ckpt["completed"].get(strategy, {})


def record_run(ckpt, strategy, seed, l2_error, init_loss,
               final_loss, trajectory, status):

    if strategy not in ckpt["completed"]:
        ckpt["completed"][strategy] = {}
    ckpt["completed"][strategy][seed] = {
        "l2_error": l2_error,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "trajectory": trajectory,
        "status": status,
    }
    save_checkpoint(ckpt)


def reconstruct_results_from_checkpoint(ckpt, strategies, strategy_labels):

    all_results = {}
    for strat in strategies:
        label = strategy_labels[strat]
        seed_data = ckpt["completed"].get(strat, {})

        l2_errors, init_losses, final_losses, trajectories, statuses = \
            [], [], [], [], []

        for seed in range(N_SEEDS):
            if seed in seed_data:
                d = seed_data[seed]
                l2_errors.append(d["l2_error"])
                init_losses.append(d["init_loss"])
                final_losses.append(d["final_loss"])
                trajectories.append(d["trajectory"])
                statuses.append(d["status"])
            else:
    
                l2_errors.append(float("nan"))
                init_losses.append(float("nan"))
                final_losses.append(float("nan"))
                trajectories.append([float("nan")])
                statuses.append("pending")

        valid_errors = [e for e in l2_errors
                        if np.isfinite(e) and e == e]  
        n_converged  = statuses.count("converged")
        n_diverged   = statuses.count("diverged")
        n_stagnated  = statuses.count("stagnated")
        mean_l2   = np.mean(valid_errors)   if valid_errors else float("nan")
        std_l2    = np.std(valid_errors)    if valid_errors else float("nan")
        median_l2 = np.median(valid_errors) if valid_errors else float("nan")
        is_bimodal, sep_score = detect_bimodality(valid_errors)

        all_results[strat] = {
            "label": label,
            "l2_errors": l2_errors,
            "init_losses": init_losses,
            "final_losses": final_losses,
            "trajectories": trajectories,
            "statuses": statuses,
            "n_converged": n_converged,
            "n_diverged": n_diverged,
            "n_stagnated": n_stagnated,
            "mean_l2": mean_l2,
            "std_l2": std_l2,
            "median_l2": median_l2,
            "is_bimodal": is_bimodal,
            "bimodal_separation": sep_score,
        }
    return all_results



def apply_init(model, strategy):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if strategy == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif strategy == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="tanh")
            elif strategy == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif strategy.startswith("normal_"):
                std = float(strategy.split("_")[1])
                nn.init.normal_(m.weight, mean=0.0, std=std)
            else:
                raise ValueError(f"Unknown init strategy: {strategy}")
            nn.init.zeros_(m.bias)


def create_model_with_init(strategy, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GenericPINN(in_dim=2, out_dim=1, n_hidden=N_HIDDEN,
                        n_neurons=N_NEURONS, activation="tanh")
    apply_init(model, strategy)
    return model


def compute_initial_loss(model, x_int, t_int, x_ic, t_ic, u_ic,
                         x_bc, t_bc, u_bc, nu=BURGERS_NU):
    model.to(DEVICE)
    model.eval()
    with torch.enable_grad():
        res = burgers_residual(model, x_int, t_int, nu)
        loss_pde = torch.mean(res ** 2)
    with torch.no_grad():
        u_ic_pred = model(x_ic, t_ic)
        loss_ic   = torch.mean((u_ic_pred - u_ic) ** 2)
        u_bc_pred = model(x_bc, t_bc)
        loss_bc   = torch.mean((u_bc_pred - u_bc) ** 2)
    return (loss_pde.item() + 10 * loss_ic.item() + loss_bc.item())


def train_single_run(model, n_epochs=N_EPOCHS, lr=LR, lr_min=LR_MIN):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min)

    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = \
        sample_burgers_domain(10000, 200, 200)

    init_loss = compute_initial_loss(
        model, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc)

    loss_hist = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        res      = burgers_residual(model, x_int, t_int, BURGERS_NU)
        loss_pde = torch.mean(res ** 2)

        u_ic_pred = model(x_ic, t_ic)
        loss_ic   = torch.mean((u_ic_pred - u_ic) ** 2)

        u_bc_pred = model(x_bc, t_bc)
        loss_bc   = torch.mean((u_bc_pred - u_bc) ** 2)

        loss = loss_pde + 10 * loss_ic + loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_hist.append(loss_val)

        if not np.isfinite(loss_val) or loss_val > 1e10:
            break

    final_loss = loss_hist[-1] if loss_hist else float("nan")
    return init_loss, final_loss, loss_hist


def detect_bimodality(errors, threshold_ratio=3.0):
    errors = np.array(errors)
    errors = errors[np.isfinite(errors)]
    if len(errors) < 5:
        return False, 0.0

    log_errors = np.log10(errors + 1e-30)
    sorted_e   = np.sort(log_errors)
    gaps       = np.diff(sorted_e)
    max_gap_idx = np.argmax(gaps)
    max_gap     = gaps[max_gap_idx]
    std         = np.std(log_errors)

    bimodal = max_gap > threshold_ratio * std and std > 0.1

    cluster1 = sorted_e[:max_gap_idx + 1]
    cluster2 = sorted_e[max_gap_idx + 1:]
    if len(cluster1) > 0 and len(cluster2) > 0:
        separation = (np.mean(cluster2) - np.mean(cluster1)) / (std + 1e-10)
    else:
        separation = 0.0

    return bimodal, separation



def run_experiment():
    global _STOP_REQUESTED

    print("=" * 70)
    print("EXP 13: Initialization Failure Study  [PAUSE/RESUME ENABLED]")
    print(f"Device : {DEVICE}")
    print(f"Seeds  : {N_SEEDS}  |  Epochs: {N_EPOCHS}  |  LR: {LR}")
    print(f"Tip    : Ctrl+C to pause cleanly. Re-run to resume.")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  
    try:
        x_ref, t_ref, u_ref = load_burgers_reference()
    except FileNotFoundError:
        print("  ⚠  Reference not found. L2 errors will be NaN.")
        x_ref, t_ref, u_ref = None, None, None


    strategies = ["xavier", "he", "orthogonal"]
    for std in RANDOM_STDS:
        strategies.append(f"normal_{std}")

    strategy_labels = {
        "xavier":      "Xavier",
        "he":          "He (Kaiming)",
        "orthogonal":  "Orthogonal",
    }
    for std in RANDOM_STDS:
        strategy_labels[f"normal_{std}"] = f"Normal(σ={std})"


    ckpt = load_checkpoint()

    ckpt["config"] = {
        "strategies": strategies,
        "n_seeds":    N_SEEDS,
        "n_epochs":   N_EPOCHS,
        "lr":         LR,
    }
    save_checkpoint(ckpt)

    total_runs    = len(strategies) * N_SEEDS
    completed_now = sum(len(v) for v in ckpt["completed"].values())

    for strat in strategies:
        if _STOP_REQUESTED:
            break

        label = strategy_labels[strat]
        done_seeds = ckpt["completed"].get(strat, {})
        remaining  = [s for s in range(N_SEEDS) if s not in done_seeds]

        if not remaining:
            print(f"\n  ✔ {label}: all {N_SEEDS} seeds already done — skipping.")
            continue

        print(f"\n{'━' * 60}")
        print(f"Strategy : {label}  ({len(done_seeds)}/{N_SEEDS} seeds already done)")
        print(f"{'━' * 60}")

        for seed in remaining:
            if _STOP_REQUESTED:
                print(f"\n  ⏸  Pause requested — stopping after seed {seed-1}.")
                break

            model = create_model_with_init(strat, seed)
            init_loss, final_loss, loss_hist = train_single_run(model)

    
            l2 = float("nan")
            if x_ref is not None:
                try:
                    _, l2 = evaluate_burgers(model, x_ref, t_ref, u_ref)
                except Exception:
                    l2 = float("nan")

  
            if not np.isfinite(final_loss) or final_loss > DIVERGENCE_THRESHOLD:
                status = "diverged"
            elif np.isfinite(l2) and l2 < CONVERGENCE_THRESHOLD:
                status = "converged"
            else:
                status = "stagnated"


            step = max(1, len(loss_hist) // 500)
            traj_sub = [float(v) for v in loss_hist[::step]]


            record_run(ckpt, strat, seed,
                       l2_error   = float(l2),
                       init_loss  = float(init_loss),
                       final_loss = float(final_loss),
                       trajectory = traj_sub,
                       status     = status)

            completed_now += 1
            pct = 100 * completed_now / total_runs

            print(f"  Seed {seed+1:>3}/{N_SEEDS} | L2={l2:.6f} | "
                  f"{status:<10} | init_loss={init_loss:.4e} | "
                  f"final_loss={final_loss:.4e} | "
                  f"total progress: {completed_now}/{total_runs} ({pct:.1f}%)")


    total_done = sum(len(v) for v in ckpt["completed"].values())
    all_done   = (total_done == total_runs)

    if _STOP_REQUESTED and not all_done:
        print(f"\n{'=' * 70}")
        print(f"  ⏸  PAUSED — {total_done}/{total_runs} runs saved.")
        print(f"  Re-run the script to continue from where you stopped.")
        print(f"  Checkpoint: {CHECKPOINT_PATH}")
        print(f"{'=' * 70}")
        return None

    all_results = reconstruct_results_from_checkpoint(
        ckpt, strategies, strategy_labels)


    metastable_strategies = [
        s for s in strategies if all_results[s]["is_bimodal"]
    ]
    print(f"\n  ★ Metastable (bimodal) strategies: "
          f"{[strategy_labels[s] for s in metastable_strategies]}")

    if not all_done:
        print("\n  ⚠  Not all seeds done — skipping final plots.")
        print(f"     Re-run to complete and generate plots.")
        return None

    print("\n── Generating plots ──")

    n_strats = len(strategies)
    ncols = 4
    nrows = (n_strats + ncols - 1) // ncols


    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, strat in enumerate(strategies):
        ax   = axes[i]
        data = all_results[strat]
        valid = [e for e in data["l2_errors"] if np.isfinite(e)]
        if valid:
            ax.hist(valid, bins=20,
                    color="#D32F2F" if data["is_bimodal"] else "#1565C0",
                    alpha=0.75, edgecolor="white")
        ax.set_xlabel("L2 Relative Error")
        ax.set_ylabel("Count")
        tag = " ★BIMODAL" if data["is_bimodal"] else ""
        ax.set_title(f"{data['label']}{tag}", fontsize=10, fontweight="bold")
        ax.axvline(CONVERGENCE_THRESHOLD, color="green",
                   linestyle="--", alpha=0.6)
    for j in range(n_strats, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("L2 Error Distribution by Initialization Strategy",
                 fontweight="bold", fontsize=14)
    savefig(fig, OUTPUT_DIR / "error_histograms.png")

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, strat in enumerate(strategies):
        ax    = axes[i]
        data  = all_results[strat]
        trajs = data["trajectories"]
        max_len = max(len(t) for t in trajs)
        padded  = np.full((len(trajs), max_len), np.nan)
        for j, t in enumerate(trajs):
            padded[j, :len(t)] = t
        median_traj = np.nanmedian(padded, axis=0)
        q25 = np.nanpercentile(padded, 25, axis=0)
        q75 = np.nanpercentile(padded, 75, axis=0)
        epochs = np.arange(max_len)
        ax.semilogy(epochs, median_traj, color="#1565C0", linewidth=1.5)
        ax.fill_between(epochs, q25, q75, color="#1565C0", alpha=0.2)
        ax.set_xlabel("Epoch (subsampled)")
        ax.set_ylabel("Loss")
        ax.set_title(data["label"], fontsize=10, fontweight="bold")
    for j in range(n_strats, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Training Loss Trajectories (Median ± IQR)",
                 fontweight="bold", fontsize=14)
    savefig(fig, OUTPUT_DIR / "loss_trajectories.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = [all_results[s]["label"] for s in strategies]
    x_pos  = np.arange(len(strategies))

    ax = axes[0]
    conv = [all_results[s]["n_converged"]  for s in strategies]
    stag = [all_results[s]["n_stagnated"]  for s in strategies]
    div  = [all_results[s]["n_diverged"]   for s in strategies]
    ax.bar(x_pos, conv, color="#2E7D32", label="Converged")
    ax.bar(x_pos, stag, bottom=conv, color="#FF9800", label="Stagnated")
    ax.bar(x_pos, div,
           bottom=[c + s for c, s in zip(conv, stag)],
           color="#D32F2F", label="Diverged")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Training Outcomes", fontweight="bold")
    ax.legend()

    ax = axes[1]
    means  = [all_results[s]["mean_l2"] for s in strategies]
    stds   = [all_results[s]["std_l2"]  for s in strategies]
    colors_bar = ["#D32F2F" if all_results[s]["is_bimodal"]
                  else "#1565C0" for s in strategies]
    ax.bar(x_pos, means, yerr=stds, color=colors_bar, alpha=0.85,
           capsize=3, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean L2 Error")
    ax.set_yscale("log")
    ax.set_title("Mean L2 Error (red = bimodal)", fontweight="bold")

    ax = axes[2]
    init_means = [np.nanmean(all_results[s]["init_losses"]) for s in strategies]
    ax.bar(x_pos, init_means, color="#9C27B0", alpha=0.85, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Initial Loss")
    ax.set_yscale("log")
    ax.set_title("Initial Loss by Strategy", fontweight="bold")

    fig.suptitle("Initialization Strategy Comparison",
                 fontweight="bold", fontsize=14)
    savefig(fig, OUTPUT_DIR / "convergence_summary.png")

    fig, ax = plt.subplots(figsize=(14, 7))
    violin_data   = []
    violin_labels = []
    for strat in strategies:
        valid = [e for e in all_results[strat]["l2_errors"] if np.isfinite(e)]
        if valid:
            violin_data.append(valid)
            tag = " ★" if all_results[strat]["is_bimodal"] else ""
            violin_labels.append(all_results[strat]["label"] + tag)
    if violin_data:
        parts = ax.violinplot(violin_data, showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            strat = strategies[i]
            color = "#D32F2F" if all_results[strat]["is_bimodal"] else "#1565C0"
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for i, data in enumerate(violin_data):
            jitter = np.random.normal(0, 0.04, size=len(data))
            ax.scatter(np.full(len(data), i + 1) + jitter, data,
                       alpha=0.3, s=15, color="#333333", zorder=5)
        ax.set_xticks(range(1, len(violin_labels) + 1))
        ax.set_xticklabels(violin_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("L2 Relative Error")
    ax.set_yscale("log")
    ax.axhline(CONVERGENCE_THRESHOLD, color="green", linestyle="--",
               alpha=0.5, label="Convergence threshold")
    ax.set_title("Metastability Analysis — L2 Error Distributions\n"
                 "(★ = bimodal/metastable)", fontweight="bold", fontsize=13)
    ax.legend()
    savefig(fig, OUTPUT_DIR / "metastability_analysis.png")

    results = {
        "experiment":          "Initialization Failure Study",
        "n_seeds":             N_SEEDS,
        "n_epochs":            N_EPOCHS,
        "strategies":          strategies,
        "strategy_labels":     strategy_labels,
        "per_strategy": {
            s: {
                "label":              all_results[s]["label"],
                "l2_errors":          all_results[s]["l2_errors"],
                "init_losses":        all_results[s]["init_losses"],
                "final_losses":       all_results[s]["final_losses"],
                "n_converged":        all_results[s]["n_converged"],
                "n_diverged":         all_results[s]["n_diverged"],
                "n_stagnated":        all_results[s]["n_stagnated"],
                "mean_l2":            all_results[s]["mean_l2"],
                "std_l2":             all_results[s]["std_l2"],
                "median_l2":          all_results[s]["median_l2"],
                "is_bimodal":         all_results[s]["is_bimodal"],
                "bimodal_separation": all_results[s]["bimodal_separation"],
            }
            for s in strategies
        },
        "metastable_strategies": metastable_strategies,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
    }
    save_results(results, OUTPUT_DIR / "exp13_results.json")

    print(f"\n{'=' * 70}")
    print("EXP 13 — COMPLETE")
    print(f"  Metastable: {[strategy_labels[s] for s in metastable_strategies]}")
    print(f"  Results  → {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    return results


if __name__ == "__main__":
    run_experiment()
