# Insights from Experiments 10 to 12 (Stiffness & Multi-Scale Pathologies)

## Experiment 10: Stiffness Failure (Heat Equation)
**Objective:** Test the impact of severe parameter stiffness by sweeping the diffusivity constant ($\alpha$) from $1.0$ down to $0.0001$.
**Key Findings:**
- The standard formulation of the PINN surprisingly did *not* suffer catastrophic global failure; the L2 relative errors remained well below the 5% threshold across all $\alpha$ values.
- However, a clear trend reversal exists: Error steadily improves as $\alpha$ is reduced from $1.0$ ($0.04\%$) down to $0.01$ (where it hits its best at $0.007\%$). But at $\alpha = 0.001$, the error spikes almost 100-fold to $0.6\%$.
- **Premature Convergence:** At high stiffness ($\alpha \le 0.001$), the optimizer stops at $\approx 20,000$ epochs instead of using its full 100,000 budget. 
**Insight:** Severe parameter stiffness creates "flat but steep" localized valleys in the loss landscape. The optimizer prematurely triggers its stopping criteria because the global gradient magnitudes shrink (due to stiff scaling), tricking the optimizer into believing it has converged when it has actually just stalled.

## Experiment 11: Multi-Scale Failure (Allen-Cahn)
**Objective:** Analyze the network's ability to resolve singularly perturbed equations (Allen-Cahn phase-field) where the parameter $\epsilon$ drives a transition from smooth gradients to infinitely sharp interfaces.
**Key Findings:**
- When the interface is thick ($\epsilon = 0.1$), the "Bulk Error" (error in the smooth regions) dominates the "Interface Error".
- As the interface becomes infinitely sharp ($\epsilon \le 0.001$), the error profile flips completely. The Interface Error spikes massively, becoming exactly double the magnitude of the Bulk Error.
**Insight:** Standard PINNs suffer from severe **spectral bias** when confronted with multi-scale physics. They perfectly fit the low-frequency bulk regions but completely smooth over the high-frequency sharp interfaces. Therefore, the global L2 error is a deceptive metric: a PINN might report $0.2\%$ overall error, while completely failing to resolve the critical physics happening exactly at the phase boundary.

## Experiment 12: Temporal Stiffness & Long-Time Integration
**Objective:** Compare standard "Single Domain" training against "Windowed" (time-marching) training over a long temporal horizon ($t=0 \to 10.0$).
**Key Findings:**
- *Anomaly (v2-fixed version):* The standard Single Domain training successfully integrated the entire time horizon flawlessly, with errors oscillating smoothly but never exceeding $0.14\%$.
- Paradoxically, the "Windowed" time-marching approach—a technique mathematically designed to cure temporal stiffness—performed significantly worse. It suffered from massive initial errors ($14\%$) at early time steps before eventually relaxing down to $0.17\%$ at $t=10$.
- Both error growths were best characterized by polynomial, not exponential, fits.
**Insight:** Sequential Time-Windowing is highly fragile. If the continuity constraints across time-windows are not enforced with overwhelming penalty weights, small errors at $t_1$ act as corrupted Initial Conditions for $t_2$. In this specific PDE setup, the global "Single Domain" approach leveraged future boundary constraints to stabilize the past, whereas the Windowed approach artificially broke the domain and injected massive propagation errors early on.

---

## Visual Plot Validation (Advanced Tool Analysis)
Visual analysis of the generated `.png` artifacts validates these numerical insights:
1. **`l2_vs_alpha.png` (Exp 10):** The bar chart confirms the "U-shape" trend reversal, showing the lowest bar at $0.01$ and a massive jump at $0.001$, while confirming all bars remain safely below the $0.05$ threshold line.
2. **`interface_vs_bulk_error.png` (Exp 11):** The grouped bar chart visually proves the spectral bias flip: the blue bar (Bulk) is taller than the red bar (Interface) on the left ($\epsilon=0.1$), but the red bar becomes twice the height of the blue bar on the right ($\epsilon \le 0.001$).
3. **`error_growth_fit.png` (Exp 12):** The subplots accurately map the polynomial fits ($R^2 = 0.442$ for Single Domain, $R^2 = 0.863$ for Windowed). The Windowed plot clearly visualizes the massive $10^{-1}$ error at $t=0$ dropping sharply over time, contrasting the low $10^{-3}$ oscillatory error of the single domain.

---

## Overarching Interconnecting Hypothesis
Experiments 10, 11, and 12 highlight that PINNs fail primarily due to **Scale Imbalance (Parameter & Multi-Scale)** rather than sheer global complexity.
1. **Parameter Scale (Exp 10):** Extreme physical constants trick the optimizer into premature stopping.
2. **Spatial Scale (Exp 11):** Sharp transitions cause spectral bias; the network ignores the interface to greedily minimize the bulk loss.
3. **Temporal Scale (Exp 12):** Breaking a domain into smaller time-windows to fix stiffness can backfire if the boundary continuity is not strictly preserved, highlighting the extreme sensitivity of PINNs to initial conditions.

**Conclusion:** To solve stiff/multi-scale physics, you cannot rely on global metrics or basic domain splitting. You must explicitly re-weight the loss function locally around sharp interfaces (Adaptive Spatial Sampling) and rigorously enforce continuity across time boundaries to prevent error leakage.
