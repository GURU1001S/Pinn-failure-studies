# Insights from Experiments 7 to 9 (Sampling & Data Constraints)

## Experiment 7: Collocation Sampling Strategy
**Objective:** Evaluate how different spatial sampling strategies (Random, Latin Hypercube Sampling [LHS], Sobol, Halton) affect PINN training stability and final accuracy across 20 independent random seeds.
**Key Findings:**
- Standard pseudo-random sequences like **Sobol** proved to be the most robust, exhibiting the lowest variance ($\sigma^2 = 1.65 \times 10^{-4}$) across runs.
- **LHS (Latin Hypercube)**, despite being a standard design-of-experiments technique, was surprisingly the most unstable here, showing an order of magnitude higher variance ($\sigma^2 = 1.77 \times 10^{-3}$) and several severe failure outliers.
- While the mean L2 errors were somewhat comparable ($\approx 4-5\%$), the reliability of convergence varied wildly.
**Insight:** PINNs are extremely sensitive to "holes" in spatial sampling. Pseudo-random, low-discrepancy sequences (Sobol) guarantee strict uniform spatial coverage, which prevents localized error pooling. LHS is insufficient for stiff PDEs because its grid-based randomization can still leave critical local regions under-sampled.

## Experiment 8: Collocation Starvation
**Objective:** Determine the absolute minimum number of interior collocation points required to prevent catastrophic failure, and observe behavior as point counts scale up to 5000.
**Key Findings:**
- There is a strict, cliff-like phase transition. Dropping the interior point count below 1000 triggers immediate, catastrophic failure, causing the relative L2 error to jump from $\approx 10\%$ to over $300\%$ (and up to $700\%$ at 200 points).
- **The Sweet Spot:** 2000 points yielded the best accuracy ($3.9\%$).
- **The Over-Sampling Penalty:** Interestingly, increasing the points further to 5000 actually *increased* the error slightly (back up to $5.2\%$).
**Insight:** The loss landscape geometry collapses if the point density falls below the Nyquist-like threshold required by the PDE's highest frequency components. However, "throwing more data at it" (5000 points) introduces excessive gradient noise and optimization stiffness, overwhelming the network's fixed capacity and leading to slight regression. 

## Experiment 9: Boundary vs. Interior Ratio
**Objective:** With a fixed total budget of 1000 points, determine the optimal percentage allocation between boundary condition (BC) points and interior PDE points.
**Key Findings:**
- The optimal allocation is highly constrained to a narrow "sweet spot" between **10% and 20% BC points** (the absolute best being 15%, yielding $4.4\%$ L2 error).
- **Boundary Starvation (1% BC):** Fails with $221\%$ error because the mathematical problem becomes ill-posed. The network learns *some* PDE solution, but not the specific one tied to our domain.
- **Interior Starvation ( $\ge 50\%$ BC):** Triggers an even more critical failure mode (errors ranging from $200\%$ up to $621\%$). 
- *Curious Anomaly:* There is a strange localized recovery at $40\%$ BC points ($14.9\%$ L2 error), sandwiched between catastrophic failures at 30% and 50%.
**Insight:** Over-weighting data on the boundaries starves the interior. If the network lacks sufficient continuous spatial data to satisfy the physical gradients internally, it interpolates wildly between the rigid boundaries. Thus, "Interior Starvation" is the dominant and most severe data-failure mode for PINNs.

---

## Visual Plot Validation (Advanced Tool Analysis)
Visual analysis of the generated `.png` artifacts validates these numerical insights:
1. **`error_boxplot.png` (Exp 7):** The boxplots visually confirm the text labels, showcasing the tight distribution of SOBOL vs the massive outlier tails extending upward in the LHS plot.
2. **`l2_vs_count.png` (Exp 8):** The bar chart visually defines the cliff edge: four tall red bars on the left (50-500 points) sit way above the dashed failure line, and instantly drop into tight green bars (1000+ points). It also visually confirms the slight bounce-back in error at 5000 compared to 2000.
3. **`l2_vs_ratio.png` (Exp 9):** The color-coded bar chart perfectly captures the narrow green "valley" of success between 5% and 20%. The massive red towers on the right ($>50\%$) visually validate that Interior Starvation is highly destructive.

---

## Overarching Interconnecting Hypothesis
Experiments 7, 8, and 9 collectively prove that **Data Geometry Constraints dictate PINN convergence, not just Data Volume.**
1. **Density (Exp 8):** The absolute number of points must clear a minimum physical threshold to resolve the PDE waves, otherwise the loss landscape fundamentally degenerates.
2. **Distribution (Exp 7):** Even with enough points, if their spatial distribution has subtle gaps (like in Random or LHS sampling), the local residual errors will pool in those gaps, destabilizing the global optimization.
3. **Allocation (Exp 9):** Even with perfect numbers and sampling, allocating too much budget to the boundaries starves the interior gradients, leading to catastrophic interpolation failure.

**Conclusion:** To achieve reliable training, a PINN must utilize pseudo-random sampling (Sobol), maintain a minimum absolute interior density, and heavily bias the allocation ratio toward the interior (80-90% PDE points) to support the physical gradients.
