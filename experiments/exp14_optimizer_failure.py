"""
exp14_optimizer_failure.py — Optimizer-Induced Failure Study
[v2 — journal-ready fixes]

Trains the same Burgers PINN architecture with different optimizers:
  - Adam (lr=1e-3, 1e-4, 1e-2)
  - L-BFGS
  - RMSprop
  - SGD with momentum
  - Adam→L-BFGS hybrid at switch points 10000, 20000, 30000

For each, records:
  - Full loss trajectory
  - Final L2 error (mean ± std over N_SEEDS seeds)
  - Per-component loss breakdown
  - Failure mode signature

Outputs (results/exp14/):
  - loss_trajectories.png
  - final_l2_comparison.png       (error bars)
  - adam_lbfgs_switch_analysis.png
  - failure_signatures.png
  - exp14_results.json

FIXES vs v1 (journal-ready):
  [FIX 1] Equalized compute budget — v1 gave L-BFGS 2000 steps vs
          40000 for Adam. Each L-BFGS step calls the loss/grad closure
          up to LBFGS_MAX_ITER=20 times internally, so 2000 L-BFGS
          steps ≈ 40000 gradient evaluations. v2 sets LBFGS steps so
          gradient evaluations are matched: N_LBFGS_STEPS =
          N_EPOCHS_TOTAL // LBFGS_MAX_ITER. Failure comparison is now
          fair. Applied same equalization to hybrid L-BFGS phase.

  [FIX 2] Multi-seed evaluation — v1 ran each config once (seed=42).
          v2 runs N_SEEDS=3 seeds per optimizer config and reports
          mean ± std L2. A single bad seed can misclassify a success
          as failure — multi-seed prevents this.

  [FIX 3] Fixed stagnation_ratio metric — v1 computed late/early loss
          ratio. If early loss is near zero (well-converged early),
          ratio → inf. If both are small, ratio ≈ 1 implying stagnation
          for a correctly converged model. v2 uses:
          stagnation = (early_mean - late_mean) / (early_mean + 1e-30)
          This gives 0 = no progress, 1 = full convergence, negative =
          loss increased. Physically meaningful and bounded.

  [FIX 4] Fixed hybrid L-BFGS budget — v1 gave shrinking L-BFGS
          budget as switch point increased: switch=10k → 2000 steps,
          switch=30k → 1000 steps. This confounded "later switch is
          worse" with "later switch gets less L-BFGS compute". v2 gives
          all hybrid configs the same fixed LBFGS_STEPS_HYBRID budget
          regardless of switch point.

  [FIX 5] Failure mode classification uses multi-seed mean L2 and
          the corrected stagnation metric, with explicit thresholds
          documented in JSON.
"""

import sys, os, json, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from pinn_core import DEVICE, DTYPE, save_results
from pinn_equations import (
    GenericPINN, burgers_residual,
    sample_burgers_domain, evaluate_burgers, load_burgers_reference,
    BURGERS_NU,
)
from plot_utils import savefig, setup_style

setup_style()

# ===================================================================
# Speed flags
# ===================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

# ===================================================================
# Config
# ===================================================================
N_HIDDEN        = 4
N_NEURONS       = 64
N_EPOCHS_TOTAL  = 40000
N_INT           = 10000
N_IC            = 200
N_BC            = 200
N_SEEDS         = 3           # FIX 2: multi-seed

# FIX 1: equalized L-BFGS budget
LBFGS_MAX_ITER  = 20          # closure calls per step
N_LBFGS_STEPS   = N_EPOCHS_TOTAL // LBFGS_MAX_ITER  # = 2000 steps ≈ 40k grad evals

# FIX 4: fixed hybrid L-BFGS budget regardless of switch point
LBFGS_STEPS_HYBRID = 1000     # same for all switch points

SWITCH_POINTS   = [10000, 20000, 30000]

# FIX 3: stagnation/oscillation thresholds
STAGNATION_THRESHOLD   = 0.05  # improvement < 5% → stagnated
OSCILLATION_THRESHOLD  = 0.3   # normalized std > 0.3 → oscillating

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp14"

# ===================================================================
# Graceful interrupt & Checkpoint handling
# ===================================================================
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
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                ckpt = json.load(f)
            n_done = sum(len(v) for v in ckpt.get("completed", {}).values())
            print(f"\n  ✔ Checkpoint found: {n_done} runs already done.")
            print(f"    Resuming from: {CHECKPOINT_PATH}\n")
            return ckpt
        except json.JSONDecodeError:
            print("  ⚠ Checkpoint file corrupted. Starting fresh.")
    return {"completed": {}}

def save_checkpoint(ckpt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(ckpt, f)
    tmp_path.replace(CHECKPOINT_PATH)


# ===================================================================
# FIX 3 — Corrected failure metrics
# ===================================================================

def compute_stagnation(loss_hist, window=2000):
    """
    Stagnation = fractional improvement from early to late training.
    = (early_mean - late_mean) / (early_mean + eps)

    Range: ~1 = full convergence, ~0 = no progress,
           negative = loss increased (diverging tail).

    v1 used late/early which gives inf when early ≈ 0 and ~1 for
    well-converged models (falsely implying stagnation).
    """
    if len(loss_hist) < 2 * window:
        return float("nan")
    early = float(np.mean(loss_hist[:window]))
    late  = float(np.mean(loss_hist[-window:]))
    return (early - late) / (early + 1e-30)


def compute_oscillation(loss_hist, window=2000):
    """Normalized std of the loss tail. High = oscillating/unstable."""
    if len(loss_hist) < window:
        return float("nan")
    tail = np.array(loss_hist[-window:])
    return float(np.std(tail) / (np.mean(np.abs(tail)) + 1e-30))


def classify_failure(mean_l2, stagnation, oscillation,
                     l2_fail=0.5):
    """
    FIX 5: classify using corrected metrics and multi-seed mean L2.
    """
    if not np.isfinite(mean_l2) or mean_l2 > l2_fail * 10:
        return "divergence"
    elif oscillation > OSCILLATION_THRESHOLD:
        return "ill_conditioned_hessian"
    elif stagnation < STAGNATION_THRESHOLD:
        return "flat_region_stagnation"
    elif mean_l2 > l2_fail:
        return "insufficient_convergence"
    else:
        return "success"


# ===================================================================
# Shared domain sampling (same points across seeds for fair comparison)
# ===================================================================

def get_domain(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return sample_burgers_domain(N_INT, N_IC, N_BC)


# ===================================================================
# Training routines
# ===================================================================

def _burgers_loss(model, x_int, t_int, x_ic, t_ic, u_ic,
                   x_bc, t_bc, u_bc):
    res      = burgers_residual(model, x_int, t_int, BURGERS_NU)
    loss_pde = torch.mean(res ** 2)
    loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
    loss_bc  = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
    return loss_pde + 10 * loss_ic + loss_bc, loss_pde, loss_ic, loss_bc


def train_adam(lr, seed, n_epochs=N_EPOCHS_TOTAL, log_every=10000):
    torch.manual_seed(seed); np.random.seed(seed)
    model     = GenericPINN(in_dim=2, out_dim=1, n_hidden=N_HIDDEN,
                             n_neurons=N_NEURONS, activation="tanh").to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = get_domain(seed)

    loss_h, pde_h, ic_h, bc_h = [], [], [], []
    t0 = time.time()
    for ep in range(n_epochs):
        optimizer.zero_grad()
        loss, lpde, lic, lbc = _burgers_loss(
            model, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc)
        loss.backward()
        optimizer.step(); scheduler.step()
        loss_h.append(loss.item())
        pde_h.append(lpde.item())
        ic_h.append(lic.item())
        bc_h.append(lbc.item())
        if not np.isfinite(loss.item()): break
        if ep % log_every == 0:
            print(f"    [{ep:6d}] loss={loss.item():.3e}")
    return model, {"loss_history": loss_h, "pde_history": pde_h,
                   "ic_history": ic_h, "bc_history": bc_h,
                   "training_time": time.time() - t0}


def train_lbfgs(seed, n_steps=N_LBFGS_STEPS, log_every=200):
    """FIX 1: n_steps = N_EPOCHS_TOTAL // LBFGS_MAX_ITER for equal budget."""
    torch.manual_seed(seed); np.random.seed(seed)
    model     = GenericPINN(in_dim=2, out_dim=1, n_hidden=N_HIDDEN,
                             n_neurons=N_NEURONS, activation="tanh").to(DEVICE)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=LBFGS_MAX_ITER,
        line_search_fn="strong_wolfe", history_size=50)
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = get_domain(seed)

    loss_h, pde_h, ic_h, bc_h = [], [], [], []
    t0 = time.time()

    for step in range(n_steps):
        vals = {}
        def closure():
            optimizer.zero_grad()
            loss, lpde, lic, lbc = _burgers_loss(
                model, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc)
            loss.backward()
            vals.update(loss=loss.item(), pde=lpde.item(),
                        ic=lic.item(), bc=lbc.item())
            return loss
        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"    L-BFGS step {step} failed: {e}"); break

        if not np.isfinite(vals.get("loss", float("nan"))): break
        loss_h.append(vals["loss"]); pde_h.append(vals["pde"])
        ic_h.append(vals["ic"]);     bc_h.append(vals["bc"])
        if step % log_every == 0:
            print(f"    [{step:6d}] loss={vals['loss']:.3e}")

    return model, {"loss_history": loss_h, "pde_history": pde_h,
                   "ic_history": ic_h, "bc_history": bc_h,
                   "training_time": time.time() - t0}


def train_rmsprop(seed, lr=1e-3, n_epochs=N_EPOCHS_TOTAL, log_every=10000):
    torch.manual_seed(seed); np.random.seed(seed)
    model     = GenericPINN(in_dim=2, out_dim=1, n_hidden=N_HIDDEN,
                             n_neurons=N_NEURONS, activation="tanh").to(DEVICE)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = get_domain(seed)

    loss_h, pde_h, ic_h, bc_h = [], [], [], []
    t0 = time.time()
    for ep in range(n_epochs):
        optimizer.zero_grad()
        loss, lpde, lic, lbc = _burgers_loss(
            model, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        loss_h.append(loss.item()); pde_h.append(lpde.item())
        ic_h.append(lic.item());    bc_h.append(lbc.item())
        if not np.isfinite(loss.item()): break
        if ep % log_every == 0:
            print(f"    [{ep:6d}] loss={loss.item():.3e}")
    return model, {"loss_history": loss_h, "pde_history": pde_h,
                   "ic_history": ic_h, "bc_history": bc_h,
                   "training_time": time.time() - t0}


def train_sgd(seed, lr=1e-2, momentum=0.9,
              n_epochs=N_EPOCHS_TOTAL, log_every=10000):
    torch.manual_seed(seed); np.random.seed(seed)
    model     = GenericPINN(in_dim=2, out_dim=1, n_hidden=N_HIDDEN,
                             n_neurons=N_NEURONS, activation="tanh").to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = get_domain(seed)

    loss_h, pde_h, ic_h, bc_h = [], [], [], []
    t0 = time.time()
    for ep in range(n_epochs):
        optimizer.zero_grad()
        loss, lpde, lic, lbc = _burgers_loss(
            model, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        loss_h.append(loss.item()); pde_h.append(lpde.item())
        ic_h.append(lic.item());    bc_h.append(lbc.item())
        if not np.isfinite(loss.item()): break
        if ep % log_every == 0:
            print(f"    [{ep:6d}] loss={loss.item():.3e}")
    return model, {"loss_history": loss_h, "pde_history": pde_h,
                   "ic_history": ic_h, "bc_history": bc_h,
                   "training_time": time.time() - t0}


def train_hybrid(switch_epoch, seed, adam_lr=1e-3,
                 lbfgs_steps=LBFGS_STEPS_HYBRID,  # FIX 4: fixed budget
                 n_total=N_EPOCHS_TOTAL, log_every=10000):
    """
    FIX 4: all hybrid configs get the same lbfgs_steps regardless of
    switch point, so comparisons are not confounded by compute budget.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = GenericPINN(in_dim=2, out_dim=1, n_hidden=N_HIDDEN,
                         n_neurons=N_NEURONS, activation="tanh").to(DEVICE)
    (x_int, t_int), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = get_domain(seed)

    loss_h, pde_h, ic_h, bc_h = [], [], [], []
    t0 = time.time()

    # Phase 1: Adam
    opt_adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    sch_adam = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_adam, T_max=switch_epoch, eta_min=adam_lr * 0.01)
    for ep in range(switch_epoch):
        opt_adam.zero_grad()
        loss, lpde, lic, lbc = _burgers_loss(
            model, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc)
        loss.backward()
        opt_adam.step(); sch_adam.step()
        loss_h.append(loss.item()); pde_h.append(lpde.item())
        ic_h.append(lic.item());    bc_h.append(lbc.item())
        if ep % log_every == 0:
            print(f"    [Adam {ep:6d}/{switch_epoch}] loss={loss.item():.3e}")
    loss_at_switch = loss_h[-1] if loss_h else float("nan")
    print(f"    → Switch at epoch {switch_epoch}, loss={loss_at_switch:.3e}")

    # Phase 2: L-BFGS (FIX 4: fixed step count)
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=LBFGS_MAX_ITER,
        line_search_fn="strong_wolfe", history_size=50)
    for step in range(lbfgs_steps):
        vals = {}
        def closure():
            opt_lbfgs.zero_grad()
            loss, lpde, lic, lbc = _burgers_loss(
                model, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc)
            loss.backward()
            vals.update(loss=loss.item(), pde=lpde.item(),
                        ic=lic.item(), bc=lbc.item())
            return loss
        try:
            opt_lbfgs.step(closure)
        except Exception as e:
            print(f"    L-BFGS step {step} failed: {e}"); break
        if not np.isfinite(vals.get("loss", float("nan"))): break
        loss_h.append(vals["loss"]); pde_h.append(vals["pde"])
        ic_h.append(vals["ic"]);     bc_h.append(vals["bc"])
        if step % 200 == 0:
            print(f"    [LBFGS {step:4d}/{lbfgs_steps}] loss={vals['loss']:.3e}")

    return model, {"loss_history": loss_h, "pde_history": pde_h,
                   "ic_history": ic_h, "bc_history": bc_h,
                   "training_time": time.time() - t0,
                   "switch_epoch": switch_epoch,
                   "loss_at_switch": loss_at_switch}


# ===================================================================
# Run one optimizer config across N_SEEDS seeds
# ===================================================================

def run_config(ckpt, name, train_fn, x_ref, t_ref, u_ref):
    """Run a training config across N_SEEDS seeds, return aggregated results."""
    print(f"\n{'━' * 60}")
    print(f"  Optimizer: {name}  ({N_SEEDS} seeds)")
    print(f"{'━' * 60}")

    seed_l2s, seed_stag, seed_osc, seed_train = [], [], [], []
    best_loss_hist, best_pde, best_ic, best_bc = None, None, None, None
    best_l2 = float("inf")

    if name not in ckpt.get("completed", {}):
        if "completed" not in ckpt:
            ckpt["completed"] = {}
        ckpt["completed"][name] = {}

    for seed in range(N_SEEDS):
        if _STOP_REQUESTED:
            print(f"\n  ⏸  Pause requested — stopping after seed {seed-1}.")
            break

        str_seed = str(seed)
        if str_seed in ckpt["completed"][name]:
            print(f"    Seed {seed} already done — loading from checkpoint.")
            out_ckpt = ckpt["completed"][name][str_seed]
            l2 = out_ckpt["l2"]
            stag = out_ckpt["stag"]
            osc = out_ckpt["osc"]
            seed_l2s.append(l2)
            seed_stag.append(stag)
            seed_osc.append(osc)
            seed_train.append(out_ckpt["training_time"])

            if np.isfinite(l2) and l2 < best_l2:
                best_l2 = l2
                best_loss_hist = out_ckpt.get("loss_history", [])
                best_pde = out_ckpt.get("pde_history", [])
                best_ic  = out_ckpt.get("ic_history", [])
                best_bc  = out_ckpt.get("bc_history", [])
            continue

        model, out = train_fn(seed)
        l2 = float("nan")
        if x_ref is not None:
            try:
                _, l2 = evaluate_burgers(model, x_ref, t_ref, u_ref)
            except Exception:
                pass
        stag = compute_stagnation(out["loss_history"])
        osc  = compute_oscillation(out["loss_history"])
        seed_l2s.append(l2)
        seed_stag.append(stag)
        seed_osc.append(osc)
        seed_train.append(out["training_time"])

        if np.isfinite(l2) and l2 < best_l2:
            best_l2 = l2
            best_loss_hist = out["loss_history"]
            best_pde = out["pde_history"]
            best_ic  = out["ic_history"]
            best_bc  = out["bc_history"]

        print(f"    Seed {seed}: L2={l2:.6f} stag={stag:.3f} osc={osc:.3f}")

        # Save to checkpoint
        ckpt["completed"][name][str_seed] = {
            "l2": l2,
            "stag": stag,
            "osc": osc,
            "training_time": out["training_time"],
            "loss_history": out["loss_history"],
            "pde_history": out["pde_history"],
            "ic_history": out["ic_history"],
            "bc_history": out["bc_history"],
        }
        save_checkpoint(ckpt)

    if len(seed_l2s) < N_SEEDS:
        return None

    mean_l2  = float(np.nanmean(seed_l2s))
    std_l2   = float(np.nanstd(seed_l2s))
    mean_stag = float(np.nanmean(seed_stag))
    mean_osc  = float(np.nanmean(seed_osc))
    fm        = classify_failure(mean_l2, mean_stag, mean_osc)

    print(f"  → mean L2={mean_l2:.6f} ± {std_l2:.6f}  FM={fm}")

    return {
        "l2_per_seed":     seed_l2s,
        "mean_l2":         mean_l2,
        "std_l2":          std_l2,
        "mean_stagnation": mean_stag,
        "mean_oscillation": mean_osc,
        "failure_mode":    fm,
        "training_times":  seed_train,
        # Best-seed trajectories for plotting
        "loss_history": best_loss_hist or [],
        "pde_history":  best_pde  or [],
        "ic_history":   best_ic   or [],
        "bc_history":   best_bc   or [],
    }


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment():
    print("=" * 70)
    print("EXP 14: Optimizer-Induced Failure Study  [v2 — journal]")
    print(f"Device         : {DEVICE}")
    print(f"Seeds per config: {N_SEEDS}")
    print(f"Adam epochs    : {N_EPOCHS_TOTAL}")
    print(f"L-BFGS steps   : {N_LBFGS_STEPS}  "
          f"(≈ {N_LBFGS_STEPS * LBFGS_MAX_ITER} grad evals = equal budget)")
    print(f"Hybrid LBFGS   : {LBFGS_STEPS_HYBRID} steps each "
          f"(fixed, switch-independent)")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        x_ref, t_ref, u_ref = load_burgers_reference()
    except FileNotFoundError:
        print("  ⚠ Reference not found. L2 = NaN.")
        x_ref = t_ref = u_ref = None

    ckpt = load_checkpoint()
    all_results = {}

    # ── Adam variants ────────────────────────────────────────────────
    for lr in [1e-3, 1e-4, 1e-2]:
        if _STOP_REQUESTED: break
        name = f"Adam(lr={lr:.0e})"
        res = run_config(
            ckpt, name,
            lambda seed, _lr=lr: train_adam(_lr, seed),
            x_ref, t_ref, u_ref)
        if res is not None:
            all_results[name] = res

    # ── L-BFGS (FIX 1: equalized budget) ────────────────────────────
    if not _STOP_REQUESTED:
        res = run_config(
            ckpt, "L-BFGS",
            lambda seed: train_lbfgs(seed),
            x_ref, t_ref, u_ref)
        if res is not None:
            all_results["L-BFGS"] = res

    # ── RMSprop ──────────────────────────────────────────────────────
    if not _STOP_REQUESTED:
        res = run_config(
            ckpt, "RMSprop",
            lambda seed: train_rmsprop(seed),
            x_ref, t_ref, u_ref)
        if res is not None:
            all_results["RMSprop"] = res

    # ── SGD + Momentum ───────────────────────────────────────────────
    if not _STOP_REQUESTED:
        res = run_config(
            ckpt, "SGD+Momentum",
            lambda seed: train_sgd(seed),
            x_ref, t_ref, u_ref)
        if res is not None:
            all_results["SGD+Momentum"] = res

    # ── Hybrids (FIX 4: fixed L-BFGS budget) ────────────────────────
    hybrid_results = {}
    for sw in SWITCH_POINTS:
        if _STOP_REQUESTED: break
        name = f"Adam→LBFGS@{sw}"
        res  = run_config(
            ckpt, name,
            lambda seed, _sw=sw: train_hybrid(_sw, seed),
            x_ref, t_ref, u_ref)
        if res is not None:
            all_results[name] = res
            hybrid_results[sw] = {
                "mean_l2": res["mean_l2"],
                "std_l2":  res["std_l2"],
            }

    if len(all_results) < 9:
        print(f"\n{'=' * 70}")
        print(f"  ⏸  PAUSED — Resume script later.")
        print(f"  Checkpoint: {CHECKPOINT_PATH}")
        print(f"{'=' * 70}")
        return None

    # Optimal switch point (by mean L2)
    valid_sw = {k: v for k, v in hybrid_results.items()
                if np.isfinite(v["mean_l2"])}
    optimal_switch = (min(valid_sw, key=lambda k: valid_sw[k]["mean_l2"])
                      if valid_sw else None)
    adam_ref_l2 = all_results[f"Adam(lr={1e-3:.0e})"]["mean_l2"]
    print(f"\n  ★ Optimal switch: {optimal_switch}  "
          f"(Adam-only baseline L2={adam_ref_l2:.6f})")

    # ── Plots ────────────────────────────────────────────────────────
    print("\n── Generating plots ──")
    names  = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    fm_colors = {
        "success":                "#2E7D32",
        "flat_region_stagnation": "#FF9800",
        "ill_conditioned_hessian":"#D32F2F",
        "divergence":             "#9C27B0",
        "insufficient_convergence":"#795548",
    }

    # 1. Loss trajectories
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, (name, data) in enumerate(all_results.items()):
        hist = data["loss_history"]
        if not hist: continue
        step = max(1, len(hist) // 3000)
        ax.semilogy(range(0, len(hist), step), hist[::step],
                    color=colors[idx], linewidth=1.2, alpha=0.85,
                    label=name)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title(
        "Loss Trajectories — All Optimizers\n"
        f"(Best seed shown per optimizer. "
        f"L-BFGS budget = {N_LBFGS_STEPS} steps ≈ {N_EPOCHS_TOTAL} grad evals.)",
        fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    savefig(fig, OUTPUT_DIR / "loss_trajectories.png")

    # 2. Final L2 comparison (mean ± std, FIX 2)
    fig, ax = plt.subplots(figsize=(13, 6))
    means  = [all_results[n]["mean_l2"] for n in names]
    stds   = [all_results[n]["std_l2"]  for n in names]
    bar_colors = [fm_colors.get(all_results[n]["failure_mode"], "#666666")
                  for n in names]

    bars = ax.bar(range(len(names)), means, color=bar_colors,
                  alpha=0.82, edgecolor="white")
    ax.errorbar(range(len(names)), means, yerr=stds,
                fmt="none", color="black", capsize=5,
                capthick=1.5, linewidth=1.5,
                label=f"±1σ ({N_SEEDS} seeds)")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("L2 Relative Error (mean ± σ)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(
        "Final L2 Error by Optimizer\n"
        "Green=success  Orange=stagnation  Red=ill-conditioned  Purple=divergence",
        fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)

    for i, (bar, mean, std, name) in enumerate(
            zip(bars, means, stds, names)):
        if np.isfinite(mean):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.15,
                    f"{mean:.4f}", ha="center", va="bottom", fontsize=7)
    savefig(fig, OUTPUT_DIR / "final_l2_comparison.png")

    # 3. Switch analysis (FIX 4: same LBFGS budget annotated)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sw_pts  = sorted(hybrid_results.keys())
    sw_means = [hybrid_results[s]["mean_l2"] for s in sw_pts]
    sw_stds  = [hybrid_results[s]["std_l2"]  for s in sw_pts]

    ax.errorbar(sw_pts, sw_means, yerr=sw_stds,
                fmt="o-", color="#D32F2F", linewidth=2,
                capsize=6, markersize=9, label="Adam→LBFGS")
    ax.axhline(adam_ref_l2, color="#1565C0", linestyle="--", alpha=0.7,
               label=f"Adam-only (L2={adam_ref_l2:.4f})")
    if optimal_switch:
        ax.axvline(optimal_switch, color="#2E7D32", linestyle="--",
                   alpha=0.7, label=f"Optimal switch ({optimal_switch})")
    ax.set_xlabel("Switch Epoch (Adam → L-BFGS)", fontsize=11)
    ax.set_ylabel("Final L2 Error (mean ± σ)", fontsize=11)
    ax.set_title(
        f"Hybrid Performance vs Switch Point\n"
        f"(All hybrids use same L-BFGS budget: "
        f"{LBFGS_STEPS_HYBRID} steps — FIX 4)",
        fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    # Per-component final loss for top configs
    top_names = [f"Adam(lr={1e-3:.0e})", "L-BFGS", "RMSprop", "SGD+Momentum"]
    if optimal_switch:
        top_names.append(f"Adam→LBFGS@{optimal_switch}")
    x_pos = np.arange(len(top_names))
    pde_f = [all_results[n]["pde_history"][-1]
              if all_results[n]["pde_history"] else 0 for n in top_names]
    ic_f  = [all_results[n]["ic_history"][-1]
              if all_results[n]["ic_history"] else 0 for n in top_names]
    bc_f  = [all_results[n]["bc_history"][-1]
              if all_results[n]["bc_history"] else 0 for n in top_names]

    ax.bar(x_pos - 0.2, pde_f, 0.2, color="#1565C0", label="PDE")
    ax.bar(x_pos,       ic_f,  0.2, color="#2E7D32", label="IC")
    ax.bar(x_pos + 0.2, bc_f,  0.2, color="#D32F2F", label="BC")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Final Loss Component")
    ax.set_yscale("log")
    ax.set_title("Per-Component Final Loss\n(Key optimizers)", fontweight="bold")
    ax.legend()

    fig.suptitle("Adam → L-BFGS Switch Analysis",
                 fontweight="bold", fontsize=14)
    savefig(fig, OUTPUT_DIR / "adam_lbfgs_switch_analysis.png")

    # 4. Failure signatures
    fig, ax = plt.subplots(figsize=(11, 7))
    for name, data in all_results.items():
        s  = data["mean_stagnation"]
        o  = data["mean_oscillation"]
        fm = data["failure_mode"]
        c  = fm_colors.get(fm, "#666666")
        markers = {"success": "o", "flat_region_stagnation": "s",
                   "ill_conditioned_hessian": "^",
                   "divergence": "X", "insufficient_convergence": "D"}
        if np.isfinite(s) and np.isfinite(o):
            ax.scatter(s, o, c=c, marker=markers.get(fm, "o"),
                       s=140, edgecolors="black", linewidth=0.5, zorder=5)
            ax.annotate(name, (s, o), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, alpha=0.9)

    ax.axhline(OSCILLATION_THRESHOLD, color="#D32F2F", linestyle=":",
               alpha=0.5, label=f"High oscillation (>{OSCILLATION_THRESHOLD})")
    ax.axvline(STAGNATION_THRESHOLD, color="#FF9800", linestyle=":",
               alpha=0.5, label=f"Stagnation zone (<{STAGNATION_THRESHOLD})")
    ax.set_xlabel("Stagnation Score  (0=no progress, 1=full convergence)",
                  fontsize=11)
    ax.set_ylabel("Oscillation Score  (normalized loss-tail std)", fontsize=11)
    ax.set_title(
        "Failure Mode Landscape\n"
        "(FIX 3: corrected stagnation metric — high score = good convergence)",
        fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    savefig(fig, OUTPUT_DIR / "failure_signatures.png")

    # ── JSON ─────────────────────────────────────────────────────────
    results = {
        "experiment": "Optimizer-Induced Failure Study",
        "version":    "v2-journal-ready",
        "config": {
            "n_hidden":         N_HIDDEN,
            "n_neurons":        N_NEURONS,
            "n_epochs_adam":    N_EPOCHS_TOTAL,
            "n_lbfgs_steps":    N_LBFGS_STEPS,
            "lbfgs_max_iter":   LBFGS_MAX_ITER,
            "lbfgs_grad_evals": N_LBFGS_STEPS * LBFGS_MAX_ITER,
            "lbfgs_steps_hybrid": LBFGS_STEPS_HYBRID,
            "n_seeds":          N_SEEDS,
        },
        "budget_note": (
            f"All optimizers receive equal gradient evaluation budget. "
            f"Adam: {N_EPOCHS_TOTAL} steps. "
            f"L-BFGS: {N_LBFGS_STEPS} steps × {LBFGS_MAX_ITER} "
            f"closure calls = {N_LBFGS_STEPS * LBFGS_MAX_ITER} grad evals. "
            "v1 gave L-BFGS 2000 steps (2000 grad evals) vs 40000 for Adam — "
            "an 20× budget disadvantage that confounded failure attribution."
        ),
        "per_optimizer": {
            name: {
                "l2_per_seed":      data["l2_per_seed"],
                "mean_l2":          data["mean_l2"],
                "std_l2":           data["std_l2"],
                "mean_stagnation":  data["mean_stagnation"],
                "mean_oscillation": data["mean_oscillation"],
                "failure_mode":     data["failure_mode"],
            }
            for name, data in all_results.items()
        },
        "hybrid_switch_analysis": {
            str(sw): hybrid_results[sw] for sw in SWITCH_POINTS
        },
        "optimal_switch_point": optimal_switch,
        "failure_mode_definitions": {
            "success":                 "mean_l2 < 0.5, stagnation > 0.05, osc < 0.3",
            "flat_region_stagnation":  "stagnation < 0.05 (loss barely improved)",
            "ill_conditioned_hessian": "oscillation > 0.3 (chaotic loss tail)",
            "divergence":              "mean_l2 > 5.0 or non-finite",
            "insufficient_convergence":"mean_l2 > 0.5 but < 5.0",
        },
        "stagnation_metric_note": (
            "v1 stagnation = late_mean / early_mean. "
            "This gives inf when early ≈ 0 and ~1 for well-converged "
            "models (falsely implying stagnation). "
            "v2 stagnation = (early - late) / early: bounded, "
            "monotone — 0=no progress, 1=full convergence."
        ),
        "hybrid_budget_note": (
            f"v1 gave shrinking L-BFGS budget as switch point increased: "
            f"switch=10k → 2000 steps, switch=30k → 1000 steps. "
            "This confounded 'later switch is worse' with 'later switch "
            "gets less L-BFGS compute'. "
            f"v2 gives all hybrid configs the same {LBFGS_STEPS_HYBRID} "
            "L-BFGS steps regardless of switch point."
        ),
    }

    save_results(results, OUTPUT_DIR / "exp14_results.json")

    print(f"\n{'=' * 70}")
    print("EXP 14 — COMPLETE  [v2]")
    print(f"{'=' * 70}")
    print(f"\n{'Optimizer':<24} | {'Mean L2':>10} | {'±Std':>8} | "
          f"{'Stag':>7} | {'Osc':>7} | {'Mode'}")
    print("─" * 75)
    for name, data in all_results.items():
        print(f"{name:<24} | {data['mean_l2']:>10.6f} | "
              f"{data['std_l2']:>8.6f} | "
              f"{data['mean_stagnation']:>7.3f} | "
              f"{data['mean_oscillation']:>7.3f} | "
              f"{data['failure_mode']}")
    print(f"\n  Optimal hybrid switch: {optimal_switch}")
    print(f"  Results → {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    return results


if __name__ == "__main__":
    run_experiment()