import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to import setup_style and savefig from experiments.plot_utils
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from experiments.plot_utils import savefig, setup_style
    setup_style()
except ImportError:
    # Fallback to standard matplotlib settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "Roboto", "DejaVu Sans"]
    def savefig(fig, filepath):
        fig.savefig(filepath, dpi=300, bbox_inches="tight")

def plot_sharpness_with_ci(sharp_fail_mean, sharp_fail_std,
                            sharp_success_mean, sharp_success_std,
                            samples_fail, samples_success,
                            significant, filepath):
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Failed Model\n(Normal σ=1.0)", "Success Model\n(Xavier)"]
    means      = [sharp_fail_mean, sharp_success_mean]
    stds       = [sharp_fail_std,  sharp_success_std]
    colors_bar = ["#D32F2F", "#2E7D32"]

    bars = ax.bar(categories, means, color=colors_bar, alpha=0.80,
                  edgecolor="white", width=0.5)

    # Error bars: ±2σ (95% CI assuming normal)
    ax.errorbar(categories, means,
                yerr=[2 * s for s in stds],
                fmt="none", color="black",
                capsize=8, capthick=2, linewidth=2,
                label="±2σ (5 runs)")

    # Individual run scatter
    jitter = 0.06
    for xi, (samples, c) in enumerate(
            zip([samples_fail, samples_success], colors_bar)):
        x_jitter = np.random.normal(xi, jitter, size=len(samples))
        ax.scatter(x_jitter, samples, color=c, s=50, zorder=5,
                   alpha=0.8, edgecolors="white", linewidths=0.5)

    # Value annotations
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.15,
                f"{mean:.1f} ± {std:.1f}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Hessian Max Eigenvalue λ_max", fontsize=12)
    ax.set_title(
        f"Loss Sharpness Comparison (mean ± 2σ, n=5 runs)\n"
        f"Counter-intuitive result: failed model has FLATTER landscape",
        fontweight="bold", fontsize=11)

    ax.text(0.5, 0.04,
            f"Both models trained 20k epochs. Init method is the only difference.\n"
            f"Flat-minima inversion: failed model converged to a flatter wrong solution.",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#FFF9C4",
                      edgecolor="#F57F17", alpha=0.9))

    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig, filepath)
    print(f"  Sharpness report saved: {filepath}")

def main():
    exp5_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(exp5_dir, "exp5_results.json")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    sharpness = data.get("sharpness", {})
    failed_data = sharpness.get("failed", {})
    success_data = sharpness.get("success", {})
    
    sharp_fail_mean = failed_data.get("mean")
    sharp_fail_std = failed_data.get("std")
    samples_fail = failed_data.get("samples")
    
    sharp_success_mean = success_data.get("mean")
    sharp_success_std = success_data.get("std")
    samples_success = success_data.get("samples")
    
    significant = sharpness.get("significant_2sigma", False)
    
    output_img_path = os.path.join(exp5_dir, "sharpness_report.png")
    
    print("Regenerating sharpness report...")
    plot_sharpness_with_ci(
        sharp_fail_mean, sharp_fail_std,
        sharp_success_mean, sharp_success_std,
        samples_fail, samples_success,
        significant, output_img_path
    )
    print("Done!")

if __name__ == "__main__":
    main()
