# An Atlas of PINN Failures: Experiments 1-6 Analysis

This document presents a rigorous mathematical breakdown of PINN failure modes across Experiments 1 through 6, rejecting standard deep learning tropes in favor of the theoretical frameworks of Physics-Informed Neural Networks.

---

## Experiment 1: Spectral Bias vs. Beta

**1. The Primary Pathology:** Dynamic Bandwidth Compression resulting in Spectral Failure.

**2. The Mechanism:** PINNs possess a fundamental spectral bias toward dissipative (parabolic) systems. As the advection speed ($\beta$) increases, the network's high-frequency cutoff drops drastically. At $\beta=30$, the empirical cutoff frequency plunges to $4.77$ Hz, which falls strictly below the PDE's dominant required frequency of $15.91$ Hz (for $\beta=100$). The network physically cannot resolve the wave, leading to catastrophic failure ($L^2 \to 0.99$).

**3. The Rebuttal:** A naive deep learning reviewer would examine the training curves and point to the periodic loss spikes, claiming optimization instability or an inadequate learning rate. We rebut this by defining these dense, regular spikes as **Pattern A**—a harmless collocation point resampling artifact with tail variance $< 10^{-6}$. The noise-floor-gated ratio test proves this is an inherent topological bandwidth limit, not a hyperparameter tuning issue.

---

## Experiment 2: Activation Function Mitigation

**1. The Primary Pathology:** Trivial Collapse and Genuine Gradient Instability.

**2. The Mechanism:** Employing Fourier Features with a high variance ($\sigma=100$) creates a zero-function attractor. The near-orthogonal random projections flatten the PDE gradient landscape near zero, trapping the optimizer into predicting $u \equiv 0$ across all $\beta$ values ($L^2 \geq 0.995$). Furthermore, these configurations exhibit **Pattern B** spikes—large, isolated gradient instabilities with slow recovery (tail variance $> 10^{-4}$) caused by near-resonant interactions with the PDE gradient.

**3. The Rebuttal:** The "Silver Bullet" fallacy leads reviewers to suggest that simply swapping to SIREN or adding Fourier Features solves high-frequency advection. The data decisively disproves this: no activation achieved $L^2 < 0.1$. Fourier Features merely trade spectral failure for landscape shattering and trivial collapse.

---

## Experiment 3: Capacity Scaling (Width $\times$ Depth)

**1. The Primary Pathology:** Asymptotic Limits of Capacity Scaling via Ill-Conditioning.

**2. The Mechanism:** NTK Condition Number ($\kappa$) and Spectral Decay ($\alpha$) evaluated *at initialization* reveal that scaling network capacity is futile. Across all 25 width-depth configurations, $\kappa > 4 \times 10^9$. The landscape is a pathologically ill-conditioned ravine before a single gradient step is taken. Scaling width from 16 to 256 yields only a marginal ~14% improvement in the decay exponent ($\alpha$ drops from 3.35 to 2.89), which is mathematically insufficient to overcome the steep spectral cliff.

**3. The Rebuttal:** A standard DL practitioner will claim the network is simply under-parameterized for a complex conservative system. The NTK initialization metrics rebut this by proving a hard geometric limit: the failure is baked into the initial projection landscape, and adding more parameters does not fundamentally alter the catastrophic decay rate of the eigenvalues.

---

## Experiment 4: Gradient Pathology ($\lambda$ Sweeps)

**1. The Primary Pathology:** The Late-Stage Re-emergence of Pathology.

**2. The Mechanism:** Static boundary weighting is an unwinnable game of "Whack-a-Mole." Standard training follows a strict 3-phase lifecycle: Initial BC collapse $\to$ PDE dominance (norms 10–133x larger) $\to$ Convergence. Inflating $\lambda_{BC}$ to 100 successfully prevents early PDE dominance, but the optimizer eventually over-optimizes the boundary at the expense of the interior. This causes the gradient norm ratio to spike back up to ~22x between epochs 15,000–19,000. 

**3. The Rebuttal:** A reviewer might argue that static $\lambda$ grid search is sufficient to balance gradients. We rebut this by showing that static weights cannot adapt to the evolving gradient geometry. An elevated $\lambda$ only delays the pathology, manifesting as late-stage re-emergence rather than true resolution.

---

## Experiment 5: Loss Landscape (Hessian)

**1. The Primary Pathology:** Landscape Shattering.

**2. The Mechanism:** A saturated initialization (Normal, $\sigma=1.0$) causes $\tanh$ neurons to saturate immediately ($\approx \pm1$), destroying local curvature. This failure is defined by an astronomical variance in the Hessian's maximum eigenvalue ($\lambda_{max}$ mean $\approx 9.39 \times 10^7$, std $\approx 9.79 \times 10^7$). It is not a smooth basin, but a chaotic, discontinuous landscape.

**3. The Rebuttal:** A naive reviewer will rely on standard deep learning theory, which states that poor generalization is caused by sharp minima, and advocate for flattening the landscape. We rebut this by showing that in PINNs, a saturated initialization doesn't just create a simple "sharp minimum"—it induces Landscape Shattering. The failed model's $\lambda_{max}$ exhibits astronomical magnitude and variance ($9.39 \times 10^7 \pm 9.79 \times 10^7$) compared to the stable curvature of the successful physics minimum ($3362 \pm 810$). The failure is caused by chaotic, discontinuous curvature that completely breaks gradient-based navigation, not just "sharpness."

---

## Experiment 6: Gradient Conflict (Cosine Similarity)

**1. The Primary Pathology:** U-Shaped Non-Monotonic Gradient Conflict.

**2. The Mechanism:** Tracking *unweighted* gradients reveals the pure vector geometry of the loss components. The PDE-IC conflict is not linear. It experiences an "Early Shock" ($\cos \approx -0.99$), enters a "False Hope" phase mid-training where the conflict appears resolved ($\cos > 0$), and then undergoes a severe "Late-Stage Re-emergence" where cosine similarity plummets back below $-0.60$ as components reach their asymptotic floors.

**3. The Rebuttal:** A naive reviewer will only analyze the first 10,000 epochs or use weighted gradients, concluding that the conflict gracefully resolves over time. We rebut this by extending the unweighted vector tracking to the full 50,000 epochs, proving that early resolution is a mathematical illusion. The optimizer inevitably loses vector alignment as loss components stagnate.
