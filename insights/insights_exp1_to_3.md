# Insights from Experiments 1 to 3 (PINN Failure Studies)

## Experiment 1: Spectral Bias vs. Beta
**Objective:** Evaluate how the dominant frequency of the target advection equation (controlled by $\beta$) affects the PINN's ability to learn the solution.
**Key Findings:**
- The network successfully learned the solution for $\beta \in \{1, 5, 10\}$, maintaining a high cutoff frequency of $128.0$ and an $L_2$ error $< 0.05$.
- A critical failure threshold is observed at $\beta \ge 30$. At this point, the $L_2$ error spikes to $\approx 0.90 - 0.99$.
- More importantly, the model's empirical cutoff frequency crashes abruptly to $1.0$ for all failing $\beta$ values, completely failing to capture the true dominant frequencies (e.g., $f_{dom} \approx 4.77$ for $\beta=30$).
**Insight:** Standard PINNs exhibit a severe "spectral bias," effectively acting as a low-pass filter. Once the physical advection speed demands higher frequency representation, the network fundamentally fails to resolve the high-frequency components of the PDE.

## Experiment 2: Activation Function Mitigation
**Objective:** Determine if changing the activation function (e.g., to `sin`, `swish`, `gelu`, or Fourier features) can mitigate the spectral bias at failure speeds ($\beta \in \{30, 50, 100\}$).
**Key Findings:**
- Across all tested activation functions, the $L_2$ error remained consistently high ($\ge 0.88$). 
- Frequency-aware activations like SIREN (`sin`) and Random Fourier Features (`fourier_10`, `fourier_100`) did **not** resolve the failure; their cutoff frequencies identically flatlined at $1.0$.
- High-frequency Fourier features induced significantly higher loss variance but no structural improvement in the PDE solution.
**Insight:** Spectral failure in advection equations is not merely an activation bottleneck. Frequency-embedding techniques (which typically help in pure coordinate-MLPs like NeRFs) are insufficient to cure the pathology when the physics-informed residual (PDE loss) is involved.

## Experiment 3: Width x Depth NTK Analysis
**Objective:** Analyze the conditioning of the Neural Tangent Kernel (NTK) across varying network widths (16 to 256) and depths (2 to 8) at a fixed failing speed ($\beta=50$).
**Key Findings:**
- **Total Collapse:** Every single depth-width configuration failed to learn the solution, with $L_2$ errors hovering around $\approx 0.98$ (with smaller networks like width 16/depth 2 performing even worse at $\approx 2.1$).
- **Ill-Conditioning:** The NTK condition numbers are astronomically high across the board (ranging from $4 \times 10^9$ to $1.2 \times 10^{11}$). 
- **Eigenvalue Decay:** The NTK eigenvalue decay exponents are steep (around 2.8 to 3.3), corroborating that the gradient flow is overwhelmingly dominated by only a few low-frequency eigenvectors.
**Insight:** Scaling the network capacity does absolutely nothing to alleviate the failure. The extreme condition numbers prove that the optimization landscape is pathological. 

---

## Visual Plot Validation (Advanced Tool Analysis)
Using advanced visual analysis tools on the generated `.png` plots, the following phenomenological details were confirmed:
1. **`l2_vs_beta.png` & `spectral_comparison.png` (Exp 1):** The L2 vs. Beta plot shows a massive, discontinuous jump in error crossing the 10% threshold exactly at $\beta=30$. In the spectral comparison plots, for $\beta \ge 30$, the PINN's spectral energy (blue line) exhibits a "dead drop" right after frequency 1.0, utterly failing to track the broad spectral peaks of the exact solution (black line). This visually confirms the hard low-pass filtering effect.
2. **`l2_heatmap.png` (Exp 2):** The heatmap visually emphasizes that all activation functions fail. Notably, high-frequency Fourier Features ($FF(\sigma=10)+\tanh$) actually show a darker red heat signature (1.1715 L2 error at $\beta=30$), indicating that naive frequency injection destabilizes the loss rather than fixing the bias.
3. **`eigenvalue_spectra.png` & `ntk_decay_heatmap.png` (Exp 3):** The log-log spectra plot shows a catastrophic cliff for all architectures: the eigenvalues plummet from $10^4$ down to $10^{-4}$ within the first 10 indices. The decay heatmap shows that while increasing network width (e.g., to 256) slightly softens the decay exponent from 3.35 to 2.89, it remains far too steep to overcome the conditioning threshold, visually proving that capacity scaling is ineffective.

---

## Overarching Interconnecting Hypothesis
The catastrophic failure of PINNs at higher advection speeds ($\beta \ge 30$) is fundamentally a **gradient pathology driven by NTK ill-conditioning**, not a capacity or representational limit. 
1. **The Core Mechanism:** As the PDE wave speed ($\beta$) increases, the physics loss induces a highly anisotropic NTK where high-frequency eigenfunctions are suppressed by extreme condition numbers ($>10^9$).
2. **The Symptom:** This NTK collapse manifests as the observed **spectral bias** (Exp 1), locking the network's cutoff frequency at $1.0$.
3. **The Futility of Standard Interventions:** Because the optimization landscape itself is numerically stiff and degenerate, passive structural interventions—such as swapping activation functions (Exp 2) or increasing network width/depth (Exp 3)—fail to alter the underlying ill-conditioning. 

**Conclusion:** To solve this failure mode, interventions must actively reshape the loss landscape or the gradient flow (e.g., adaptive loss weighting, modified PDE formulations, or specialized curriculum learning), rather than relying on standard architectural tweaks.
