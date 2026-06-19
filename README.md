# PINN Zugzwang: Mapping the Failure Landscape of Physics-Informed Neural Networks

This repository contains the complete experimental codebase and reproducibility package for our systematic evaluation of Physics-Informed Neural Network (PINN) failures under stiff partial differential equations (PDEs). 

## Overview

While PINNs demonstrate theoretical promise, they frequently fail to converge on stiff PDEs. This repository provides the tools and scripts used to isolate and quantify these failure mechanisms through spectral profiling, variance decomposition, and optimization diagnostics.

**Key Empirical Findings (as reported in the manuscript):**
1. **Loss Landscape Pathologies**: Failed models converge to exceptionally sharp minima ($\lambda_{\max} \approx 9.4 \times 10^7$) compared to successful architectures ($\lambda_{\max} \approx 3.4 \times 10^3$).
2. **Structural Dominance**: Structural factors (architecture depth/width, activation choices) dominate hyperparameter tuning (learning rate, boundary weighting) in determining outcome variance ($\eta^2 = 0.673$ vs $0.0023$ for Burgers).
3. **Metric Illusions**: Standard pointwise $L^2$ errors obscure underlying physical violations. Distinct models with near-identical $L^2$ errors exhibit up to a $57\times$ divergence in mass conservation violations.
4. **The Zugzwang Condition**: PINNs encounter multiple, mathematically independent failure mechanisms simultaneously under stiffness. No single available algorithmic intervention provides a structurally complete remedy.

## Repository Structure

The repository is stripped of temporary outputs and is focused strictly on reproducible scientific artifacts:

```text
.
├── environment.yml             # Conda environment specifications
├── requirements.txt            # Python pip dependencies
├── experiments/                # Core reproducibility codebase
│   ├── pinn_core.py            # Neural network architecture definitions
│   ├── pinn_equations.py       # PDE definitions (Advection, Burgers, etc.)
│   ├── plot_utils.py           # Evaluation and visualization utilities
│   ├── exp1_*.py to exp27_*.py # 27 primary experimental sweeps
│   └── specialexp1_*.py        # Special mitigation and compound failure tests
└── results/                    # (Generated locally) JSON logs and plots
```

## Setup and Installation

All experiments were validated using `PyTorch`. We provide exact environment snapshots to guarantee computational reproducibility.

**Using Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate pinn_atlas_env
```

**Using Pip:**
```bash
pip install -r requirements.txt
```

## Running the Experiments

Each experiment is standalone and fully reproducible. To run a specific failure analysis sweep (e.g., Experiment 8: Collocation Starvation), simply execute the corresponding Python script:

```bash
python experiments/exp8_collocation_starvation.py
```

- **Metrics:** Scripts output raw measurements to `results/exp{N}/exp{N}_results.json`.
- **Figures:** All manuscript plots are generated natively inside the `results/` subdirectories after completion.
- **Checkpoints:** Large network weights (`.pt` / `checkpoint.json`) are disabled from version control to maintain repository health, but are saved locally during execution.

## Hardware Requirements
While simple advection sweeps can run on a standard CPU, full convergence mapping (e.g., Experiment 27 Variance Decomposition) is highly computationally intensive and requires a CUDA-capable GPU. The framework defaults to `cuda` if available.

## Citation
If you utilize this codebase or our failure metrics in your work, please cite our manuscript. *(Citation details will be updated upon publication).*
