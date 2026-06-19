# Plot Authenticity and Validation Report (Experiments 7 - 9)

**Objective:** Validate that the numerical data contained in the raw `results.json` files mathematically and visually matches the rendered `.png` plots for Experiments 7 to 9, ensuring the plots are 100% authentic representations of the data output.

---

## Experiment 7: Collocation Sampling Strategy
**Files Validated:** `exp7_results.json`, `error_boxplot.png`

### Cross-Check: `error_boxplot.png`
- **JSON Data:** The `variances` array calculates the exact variance for each sampling strategy across 20 seeds:
  - `random`: `0.000234487...`
  - `lhs`: `0.001769028...`
  - `sobol`: `0.000164815...`
  - `halton`: `0.000821095...`
- **Plot Verification:** The generated boxplot overlays text annotations representing the calculated variance $\sigma^2$ for each category:
  - Over the **RANDOM** boxplot: `\sigma^2=2.34e-04` (Matches $0.000234$ perfectly).
  - Over the **LHS** boxplot: `\sigma^2=1.77e-03` (Matches $0.00177$ perfectly).
  - Over the **SOBOL** boxplot: `\sigma^2=1.65e-04` (Matches $0.000165$ perfectly).
  - Over the **HALTON** boxplot: `\sigma^2=8.21e-04` (Matches $0.000821$ perfectly).
- **Match Status:** **100% Authentic.** The statistical text labels explicitly generated on the plot exactly read the JSON metrics rounded to 3 significant figures.

---

## Experiment 8: Collocation Starvation
**Files Validated:** `exp8_results.json`, `l2_vs_count.png`

### Cross-Check: `l2_vs_count.png`
- **JSON Data:** The `count_sweep` dictionary maps the number of collocation points to specific L2 errors and assigns a "FAIL" or "PASS" status based on a 0.5 threshold.
  - 50 points: `3.26` (FAIL)
  - 100 points: `3.77` (FAIL)
  - 200 points: `7.03` (FAIL - Highest error)
  - 500 points: `3.30` (FAIL)
  - 1000 points: `0.108` (PASS)
  - 2000 points: `0.039` (PASS - Lowest error)
  - 5000 points: `0.052` (PASS)
- **Plot Verification:** The log-scaled bar chart visually reproduces these arrays:
  - The first 4 bars (x=50, 100, 200, 500) are colored red (FAIL) and sit well above the $10^0$ mark. The bar at x=200 is visually the highest, matching the `7.03` max error.
  - The bar at x=1000 drops down to become green (PASS) and touches the $\approx 10^{-1}$ gridline (matching `0.108`).
  - The bar at x=2000 is the absolute lowest green bar, sitting below $10^{-1}$ (matching `0.039`).
  - The bar at x=5000 is green but slightly taller than the 2000 bar (matching the slight regression to `0.052`).
  - The dashed blue vertical line is accurately drawn directly at x=1000, labeled "Min viable: 1000".
- **Match Status:** **100% Authentic.**

---

## Experiment 9: Boundary vs Interior Ratio
**Files Validated:** `exp9_results.json`, `l2_vs_ratio.png`

### Cross-Check: `l2_vs_ratio.png`
- **JSON Data:** The L2 error maps across various Boundary-to-Total fractions (`bc_fractions`). The threshold for failure is L2 > 0.5.
  - 1% (`0.01`): `2.21` (FAIL)
  - 5% (`0.05`): `0.107` (PASS)
  - 15% (`0.15`): `0.044` (PASS - Optimal)
  - 30% (`0.30`): `2.39` (FAIL)
  - 40% (`0.40`): `0.149` (PASS - localized recovery)
  - 90% (`0.90`): `6.21` (FAIL - highest extreme)
- **Plot Verification:** The bar chart plots these values with color coding (Green < 0.5, Red > 0.5):
  - 1% is a tall red bar.
  - 5% to 20% forms a continuous block of short green bars. The lowest visually is 15%, mapping perfectly to the optimal `0.044`.
  - 30% shoots up as a red bar.
  - 40% dips back down as a green bar (mapping to `0.149`).
  - From 50% to 99%, all bars are massive red columns. The 90% bar is visually the tallest, mapping to the max error of `6.21`.
- **Match Status:** **100% Authentic.**

---

## Final Conclusion
Every single numerical height, log-scale mapping, color-coded threshold status, and explicit text label generated in the plots for Experiments 7, 8, and 9 mathematically mirrors the output values stored in the `results.json` files. 

**There are no fake or randomly generated plots.** The visualization logic cleanly and authentically parsed the empirical output arrays.
