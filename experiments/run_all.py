"""
run_all.py — Sequential runner for ALL PINN failure experiment blocks.

Usage:
    python experiments/run_all.py                        # Run everything
    python experiments/run_all.py --block 1              # Block 1 only
    python experiments/run_all.py --block 2 3            # Blocks 2 and 3
    python experiments/run_all.py --prompt 2.1 2.3 4.1   # Specific prompts
    python experiments/run_all.py --help
"""

import sys
import os
import time
import argparse
import io

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ===================================================================
# Experiment registry
# ===================================================================

EXPERIMENTS = {
    # Block 1: Spectral Bias
    "1.1": ("Spectral Bias vs β", "exp1_spectral_bias"),
    "1.2": ("Activation Function Mitigation", "exp2_activation_study"),
    "1.3": ("Width × Depth NTK Analysis", "exp3_ntk_analysis"),
    # Block 2: Gradient Pathology
    "2.1": ("Gradient Pathology (Wang et al.)", "exp4_gradient_pathology"),
    "2.2": ("Loss Landscape Analysis", "exp5_loss_landscape"),
    "2.3": ("Gradient Flow & Conflict", "exp6_gradient_flow"),
    # Block 3: Collocation Point Failure
    "3.1": ("Collocation Sensitivity", "exp7_collocation_sensitivity"),
    "3.2": ("Collocation Starvation", "exp8_collocation_starvation"),
    "3.3": ("Boundary vs Interior Ratio", "exp9_boundary_ratio"),
    # Block 4: Stiff PDE & Multi-Scale
    "4.1": ("Stiffness Failure (Heat)", "exp10_stiffness_failure"),
    "4.2": ("Multi-Scale Failure (Allen-Cahn)", "exp11_multiscale_failure"),
    "4.3": ("Temporal Stiffness", "exp12_temporal_stiffness"),
    # Block 5: Initialization & Optimization Failure
    "5.1": ("Initialization Failure Study", "exp13_initialization_failure"),
    "5.2": ("Optimizer-Induced Failure", "exp14_optimizer_failure"),
    "5.3": ("LR Phase Diagram", "exp15_lr_phase_diagram"),
    # Block 6: Advanced & Compound Failures
    "6.1": ("Causality Failure", "exp16_causality_failure"),
    "6.2": ("Temporal Extrapolation", "exp17_temporal_extrapolation"),
    "6.3": ("Compound Failure", "exp18_compound_failure"),
    "6.4": ("Burgers Failure Map", "exp19_burgers_failure_map"),
    "6.5": ("Failure Fingerprinting", "exp20_failure_fingerprinting"),
}

BLOCKS = {
    1: ["1.1", "1.2", "1.3"],
    2: ["2.1", "2.2", "2.3"],
    3: ["3.1", "3.2", "3.3"],
    4: ["4.1", "4.2", "4.3"],
    5: ["5.1", "5.2", "5.3"],
    6: ["6.1", "6.2", "6.3", "6.4", "6.5"],
}


def run_prompt(prompt_id):
    """Dynamically import and run an experiment by prompt ID."""
    name, module_name = EXPERIMENTS[prompt_id]
    print(f"\n{'█' * 70}")
    print(f"█  PROMPT {prompt_id}: {name}")
    print(f"{'█' * 70}\n")

    module = __import__(module_name)
    return module.run_experiment()


def main():
    parser = argparse.ArgumentParser(
        description="Run PINN failure analysis experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiments:
  Block 1 -- Spectral Bias & Architecture
    1.1  Spectral Bias vs beta
    1.2  Activation Function Mitigation
    1.3  Width x Depth NTK Analysis

  Block 2 -- Gradient Pathology & Loss Imbalance
    2.1  Gradient Pathology (Wang et al. 2021)
    2.2  Loss Landscape Analysis (Li et al. 2018)
    2.3  Gradient Flow & Conflict Analysis

  Block 3 -- Collocation Point Failure
    3.1  Collocation Sensitivity (Sampling Strategies)
    3.2  Collocation Starvation & Adaptive Refinement
    3.3  Boundary vs Interior Ratio

  Block 4 -- Stiff PDE & Multi-Scale Failure
    4.1  Stiffness Failure (Heat Equation)
    4.2  Multi-Scale Failure (Allen-Cahn)
    4.3  Temporal Stiffness / Long-Time Integration

  Block 5 -- Initialization & Optimization Failure
    5.1  Initialization Failure Study
    5.2  Optimizer-Induced Failure
    5.3  Learning Rate Phase Diagram

  Block 6 -- Advanced & Compound Failures
    6.1  Causality Failure
    6.2  Temporal Extrapolation
    6.3  Compound Failure
    6.4  Burgers Failure Map
    6.5  Failure Fingerprinting

  Examples:
    python experiments/run_all.py                        # Run all
    python experiments/run_all.py --block 2              # Block 2 only
    python experiments/run_all.py --block 1 3            # Blocks 1 and 3
    python experiments/run_all.py --prompt 2.1 4.3       # Specific prompts
        """,
    )
    parser.add_argument(
        "--block", nargs="*", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Run all experiments in specified block(s).",
    )
    parser.add_argument(
        "--prompt", nargs="*", type=str,
        help="Run specific prompt(s) by ID (e.g., 2.1 3.3 4.1).",
    )
    args = parser.parse_args()

    # Determine which experiments to run
    if args.prompt:
        selected = args.prompt
        # Validate
        for p in selected:
            if p not in EXPERIMENTS:
                parser.error(f"Unknown prompt: {p}. "
                             f"Valid: {list(EXPERIMENTS.keys())}")
    elif args.block:
        selected = []
        for b in sorted(set(args.block)):
            selected.extend(BLOCKS[b])
    else:
        # Run everything
        selected = list(EXPERIMENTS.keys())

    print("=" * 70)
    print("PINN FAILURE ANALYSIS — EXPERIMENT SUITE")
    print(f"Running: {selected}")
    print(f"Total experiments: {len(selected)}")
    print("=" * 70)

    total_start = time.time()
    results = {}
    status = {}

    for prompt_id in selected:
        exp_start = time.time()
        try:
            results[prompt_id] = run_prompt(prompt_id)
            elapsed = time.time() - exp_start
            status[prompt_id] = f"SUCCESS ({elapsed:.1f}s)"
            print(f"\n✓ Prompt {prompt_id} completed in {elapsed:.1f}s "
                  f"({elapsed / 60:.1f} min)")
        except Exception as e:
            elapsed = time.time() - exp_start
            status[prompt_id] = f"FAILED ({elapsed:.1f}s): {e}"
            print(f"\n✗ Prompt {prompt_id} FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            results[prompt_id] = {"error": str(e)}

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETE")
    print(f"Total wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("=" * 70)

    for prompt_id in selected:
        name = EXPERIMENTS[prompt_id][0]
        print(f"  [{prompt_id}] {name}: {status[prompt_id]}")

    print(f"\nResults directories:")
    print(f"  Block 1: results/exp1/, results/exp2/, results/exp3/")
    print(f"  Block 2: results/exp4/, results/exp5/, results/exp6/")
    print(f"  Block 3: results/exp7/, results/exp8/, results/exp9/")
    print(f"  Block 4: results/exp10/, results/exp11/, results/exp12/")
    print(f"  Block 5: results/exp13/, results/exp14/, results/exp15/")
    print(f"  Block 6: results/exp16/, results/exp17/, results/exp18/, results/exp19/, results/exp20/")

    return results


if __name__ == "__main__":
    main()
