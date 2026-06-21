"""
plot_utils.py — Publication-quality plotting utilities for PINN experiments.

Provides:
  - Consistent figure styling (fonts, colors, DPI)
  - Spectral comparison plots
  - Heatmap helpers
  - Loss curve overlays
  - Multi-panel figure layouts
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from pathlib import Path


# ===================================================================
# Global style configuration
# ===================================================================

def setup_style():
    """Apply publication-quality defaults."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "figure.figsize": (10, 6),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.framealpha": 0.8,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.constrained_layout.use": True,
        "font.family": "serif",
    })
    sns.set_palette("husl")

setup_style()


# Color palettes
COLORS = {
    "tanh": "#2196F3",
    "sin": "#FF5722",
    "swish": "#4CAF50",
    "gelu": "#9C27B0",
    "fourier_1": "#FF9800",
    "fourier_10": "#E91E63",
    "fourier_100": "#00BCD4",
    "exact": "#333333",
    "pinn": "#2196F3",
    "residual": "#F44336",
}

ACTIVATION_LABELS = {
    "tanh": "tanh",
    "sin": "sin (SIREN)",
    "swish": "Swish (SiLU)",
    "gelu": "GELU",
    "fourier_1": "FF(σ=1)+tanh",
    "fourier_10": "FF(σ=10)+tanh",
    "fourier_100": "FF(σ=100)+tanh",
}


def get_activation_color(act_name):
    """Get the color for a given activation."""
    return COLORS.get(act_name, "#666666")


def get_activation_label(act_name):
    """Get the display label for a given activation."""
    return ACTIVATION_LABELS.get(act_name, act_name)


def savefig(fig, filepath, **kwargs):
    """Save figure and close."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save original format
    fig.savefig(filepath, bbox_inches="tight", **kwargs)
    
    # Save alternate format
    if filepath.suffix == '.pdf':
        png_path = filepath.with_suffix('.png')
        fig.savefig(png_path, bbox_inches="tight", **kwargs)
        print(f"  Figure saved: {filepath} & {png_path}")
    elif filepath.suffix == '.png':
        pdf_path = filepath.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches="tight", **kwargs)
        print(f"  Figure saved: {filepath} & {pdf_path}")
    else:
        print(f"  Figure saved: {filepath}")
        
    plt.close(fig)


# ===================================================================
# Spectral plots
# ===================================================================

def plot_spectral_comparison(freqs_list, power_pred_list, power_exact_list,
                             beta_list, cutoff_indices=None, filepath=None):
    """
    Multi-panel spectral comparison: PINN vs exact power spectrum for each beta.

    Parameters
    ----------
    freqs_list : list of arrays — frequency bins for each beta
    power_pred_list : list of arrays — PINN power spectra
    power_exact_list : list of arrays — exact solution power spectra
    beta_list : list of floats
    cutoff_indices : list of ints or None — cutoff freq index per beta
    filepath : str or None — save path
    """
    n = len(beta_list)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (freqs, pp, pe, beta) in enumerate(
        zip(freqs_list, power_pred_list, power_exact_list, beta_list)
    ):
        ax = axes[i]
        ax.semilogy(freqs, pe + 1e-30, color=COLORS["exact"],
                     label="Exact", linewidth=2)
        ax.semilogy(freqs, pp + 1e-30, color=COLORS["pinn"],
                     label="PINN", linewidth=1.5, alpha=0.8)

        if cutoff_indices is not None and cutoff_indices[i] < len(freqs):
            ax.axvline(freqs[cutoff_indices[i]], color=COLORS["residual"],
                       linestyle="--", alpha=0.7,
                       label=f"Cutoff f={freqs[cutoff_indices[i]]:.1f}")

        ax.set_title(f"β = {beta}")
        ax.set_xlabel("Frequency (cycles/domain)")
        ax.set_ylabel("|FFT|²")
        ax.legend(loc="upper right")
        ax.set_xlim(left=0)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Spectral Energy Distribution: PINN vs Exact Solution",
                 fontsize=14, fontweight="bold")

    if filepath:
        savefig(fig, filepath)
    return fig


def plot_l2_vs_beta(beta_list, l2_errors, dominant_freqs, cutoff_beta=None,
                    filepath=None):
    """
    Dual-axis plot: L2 error vs beta + dominant frequency overlay.

    Parameters
    ----------
    beta_list : list of floats
    l2_errors : list of floats
    dominant_freqs : list of floats — beta / (2*pi)
    cutoff_beta : float or None — beta where failure begins
    filepath : str or None
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "#D32F2F"
    color2 = "#1565C0"

    # L2 error (left axis)
    ax1.semilogy(beta_list, l2_errors, "o-", color=color1, linewidth=2,
                 markersize=8, label="L2 Relative Error", zorder=5)
    ax1.set_xlabel("β (advection speed)", fontsize=13)
    ax1.set_ylabel("L2 Relative Error", color=color1, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(0.1, color=color1, linestyle=":", alpha=0.5, label="10% threshold")

    # Dominant frequency (right axis)
    ax2 = ax1.twinx()
    ax2.plot(beta_list, dominant_freqs, "s--", color=color2, linewidth=2,
             markersize=7, label="Dominant freq β/(2π)")
    ax2.set_ylabel("Dominant Frequency (cycles/domain)", color=color2, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Cutoff vertical line
    if cutoff_beta is not None:
        ax1.axvline(cutoff_beta, color="#FF6F00", linewidth=2, linestyle="--",
                    alpha=0.8, label=f"Empirical cutoff β={cutoff_beta}")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=10, framealpha=0.9)

    ax1.set_title("PINN Failure vs. Advection Speed",
                  fontsize=14, fontweight="bold")

    if filepath:
        savefig(fig, filepath)
    return fig


def plot_solution_snapshots(eval_results_list, beta_list, t_snapshot_idx=50,
                            filepath=None):
    """
    Side-by-side PINN vs exact solution at a fixed time for each beta.

    Parameters
    ----------
    eval_results_list : list of dicts from evaluate_on_grid()
    beta_list : list of floats
    t_snapshot_idx : int — time index for snapshot
    filepath : str or None
    """
    n = len(beta_list)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (res, beta) in enumerate(zip(eval_results_list, beta_list)):
        ax = axes[i]
        x = res["x"]
        t_val = res["t"][t_snapshot_idx]
        u_pred = res["u_pred"][:, t_snapshot_idx]
        u_exact = res["u_exact"][:, t_snapshot_idx]

        ax.plot(x, u_exact, color=COLORS["exact"], linewidth=2, label="Exact")
        ax.plot(x, u_pred, color=COLORS["pinn"], linewidth=1.5,
                linestyle="--", label="PINN")
        ax.set_title(f"β={beta} | t={t_val:.2f} | L2={res['l2_error']:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.legend(loc="upper right", fontsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Solution Snapshots: PINN vs Exact",
                 fontsize=14, fontweight="bold")

    if filepath:
        savefig(fig, filepath)
    return fig


# ===================================================================
# Training loss curves
# ===================================================================

def plot_training_curves(loss_histories, labels, colors=None, filepath=None,
                         title="Training Loss Curves"):
    """
    Overlay multiple training loss curves.

    Parameters
    ----------
    loss_histories : list of lists
    labels : list of str
    colors : list of str or None
    filepath : str or None
    title : str
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (hist, label) in enumerate(zip(loss_histories, labels)):
        c = colors[i] if colors else None
        ax.semilogy(hist, label=label, color=c, alpha=0.8, linewidth=1.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    if filepath:
        savefig(fig, filepath)
    return fig


# ===================================================================
# Heatmaps
# ===================================================================

def plot_heatmap(data, row_labels, col_labels, title, xlabel, ylabel,
                 cmap="RdYlBu_r", fmt=".3f", vmin=None, vmax=None,
                 log_scale=False, annotate=True, filepath=None):
    """
    Generic heatmap.

    Parameters
    ----------
    data : 2D array (n_rows, n_cols)
    row_labels : list of str
    col_labels : list of str
    title : str
    xlabel, ylabel : str
    cmap : colormap name
    fmt : annotation format string
    log_scale : if True, plot log10(data)
    filepath : str or None
    """
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.5),
                                    max(5, len(row_labels) * 0.8)))

    plot_data = np.log10(data + 1e-30) if log_scale else data
    plot_fmt = ".1f" if log_scale else fmt

    sns.heatmap(
        plot_data,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=annotate,
        fmt=plot_fmt,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    title_suffix = " (log₁₀)" if log_scale else ""
    ax.set_title(f"{title}{title_suffix}", fontsize=14, fontweight="bold")

    if filepath:
        savefig(fig, filepath)
    return fig


# ===================================================================
# Spectral residual overlay
# ===================================================================

def plot_spectral_residuals_overlay(
    results_by_beta,
    filepath=None,
):
    """
    All spectral residuals on one figure — one subplot per beta,
    with all activation functions overlaid.

    Parameters
    ----------
    results_by_beta : dict
        {beta: {act_name: {'freqs': ..., 'power': ...}, ...}, ...}
    filepath : str or None
    """
    betas = sorted(results_by_beta.keys())
    n = len(betas)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, beta in enumerate(betas):
        ax = axes[i]
        for act_name, spec_data in results_by_beta[beta].items():
            freqs = spec_data["freqs"]
            power = spec_data["power"]
            color = get_activation_color(act_name)
            label = get_activation_label(act_name)
            ax.semilogy(freqs, power + 1e-30, color=color,
                        label=label, linewidth=1.3, alpha=0.85)

        ax.set_title(f"β = {beta}", fontsize=12)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("|FFT(residual)|²")
        ax.legend(loc="upper right", fontsize=7, ncol=1)
        ax.set_xlim(left=0)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Spectral Residuals by Activation Function",
                 fontsize=14, fontweight="bold")

    if filepath:
        savefig(fig, filepath)
    return fig


# ===================================================================
# Stability bar chart
# ===================================================================

def plot_stability_bars(act_names, variances, filepath=None):
    """
    Bar chart of training loss variance (last 1000 steps) per activation.

    Parameters
    ----------
    act_names : list of str
    variances : list of float
    filepath : str or None
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [get_activation_color(a) for a in act_names]
    labels = [get_activation_label(a) for a in act_names]

    bars = ax.bar(labels, variances, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=1.2)

    ax.set_ylabel("Loss Variance (last 1000 steps)")
    ax.set_title("Training Stability by Activation Function",
                 fontsize=14, fontweight="bold")
    ax.set_yscale("log")

    # Annotate bars
    for bar, val in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2e}", ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=30, ha="right")

    if filepath:
        savefig(fig, filepath)
    return fig


# ===================================================================
# NTK eigenvalue spectra
# ===================================================================

def plot_eigenvalue_spectra(eigen_data_dict, filepath=None):
    """
    Plot eigenvalue decay curves for selected architectures.

    Parameters
    ----------
    eigen_data_dict : dict
        {(width, depth): eigenvalues_array, ...}
    filepath : str or None
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    cmap = plt.cm.viridis
    n = len(eigen_data_dict)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for idx, ((w, d), eigvals) in enumerate(eigen_data_dict.items()):
        k = np.arange(1, len(eigvals) + 1)
        ax.loglog(k, eigvals, color=colors[idx], linewidth=1.5,
                  label=f"W={w}, D={d}", alpha=0.85)

    ax.set_xlabel("Eigenvalue Index k")
    ax.set_ylabel("λ_k (NTK Eigenvalue)")
    ax.set_title("NTK Eigenvalue Spectra by Architecture",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    if filepath:
        savefig(fig, filepath)
    return fig
