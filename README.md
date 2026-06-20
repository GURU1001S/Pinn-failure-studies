# An Atlas of PINN Failures: The Zugzwang Thesis - Official Code Repository

This repository contains the official code and reproducibility package for the paper *An Atlas of PINN Failures: The Zugzwang Thesis*. It systematically catalogs, reproduces, and analyzes the fundamental failure modes of Physics-Informed Neural Networks (PINNs) across canonical PDEs. By auditing both scalar errors and global conservation invariants, the codebase demonstrates how disparate network pathologies consistently shatter physical laws in distinct, unpredictable ways despite seemingly successful local convergence.

## Environment Setup

To reproduce the experiments in the manuscript, clone this repository and set up a Python virtual environment to install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/pinn-failure-atlas.git
cd pinn-failure-atlas

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/macOS:
# source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

## Reproducing the Atlas

All experiments are organized within the `experiments/` directory. To run the core scripts and reproduce the key findings of the manuscript, execute the following commands from the root directory:

```bash
# Reproduce the global phase space mapping (Experiment 25)
python experiments/exp25_phase_space.py

# Reproduce the comprehensive conservation law audit (Experiment 26)
python experiments/exp26_conservation_law_audit.py
```

## Random Seeds & Initialization

Stochasticity in neural network initialization is a key element evaluated in this manuscript. To ensure strict reproducibility and perfectly align with Section A.6 of the paper, the codebase adheres to the following seeding strategy:

*   **Baseline Runs:** All baseline models strictly default to `SEED = 42`.
*   **Robustness Checks:** For experiments assessing geometric sensitivity and stability (such as Experiment 5 and Experiment 8), we utilize explicit seed loops to aggregate statistical outcomes.
*   **Collocation Starvation (FM5):** In Experiment 26, the `FM5` configuration utilizes an explicitly configured seed offset (`SEED + 42`) to guarantee the network explores its distinct starvation trajectory without collapsing into identical basins occupied by completely different failure modes.

## Results Structure

The repository is designed to be fully self-contained. When executing any experiment, all artifacts are automatically routed and saved to the corresponding folder inside the `results/` directory (e.g., `results/exp26/`). This includes:
*   **Raw Logs:** Extracted metrics and analytical logs.
*   **Checkpoints:** PyTorch network states (`.pt` files) saved at the final evaluation epoch.
*   **Generated Plots:** All figures, heatmaps, and diagnostic visualizer outputs utilized throughout the manuscript.
