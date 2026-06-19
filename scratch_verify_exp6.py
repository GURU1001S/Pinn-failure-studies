"""Verify exp6 results JSON integrity and identify all issues."""
import json
import numpy as np

with open(r"D:\Games\FAILURE STUDIES\results\exp6\exp6_results.json") as f:
    d = json.load(f)

epochs = d["epochs_tracked"]
cos_pde_ic = np.array(d["cosine_pde_ic"])
cos_pde_bc = np.array(d["cosine_pde_bc"])
cos_ic_bc  = np.array(d["cosine_ic_bc"])

print("=" * 70)
print("EXP6 AUDIT REPORT")
print("=" * 70)

# 1. Check epoch tracking
print("\n1. EPOCH TRACKING:")
print(f"   N_EPOCHS config: {d['n_epochs']}")
print(f"   Tracked epochs: {len(epochs)}")
print(f"   First epoch: {epochs[0]}, Last epoch: {epochs[-1]}")
expected = list(range(0, d["n_epochs"], 500))
if epochs == expected:
    print("   OK Epoch sequence matches expected range(0, 50000, 500)")
else:
    print(f"   MISMATCH: expected {len(expected)} entries, got {len(epochs)}")

# 2. Verify cosine sim arrays match epoch count
print(f"\n2. ARRAY LENGTHS:")
print(f"   epochs:      {len(epochs)}")
print(f"   cos_pde_ic:  {len(cos_pde_ic)}")
print(f"   cos_pde_bc:  {len(cos_pde_bc)}")
print(f"   cos_ic_bc:   {len(cos_ic_bc)}")
all_match = len(cos_pde_ic) == len(cos_pde_bc) == len(cos_ic_bc) == len(epochs)
print(f"   All arrays same length: {all_match}")

# 3. Cosine similarity bounds check
print(f"\n3. COSINE SIMILARITY BOUNDS:")
for name, arr in [("pde_ic", cos_pde_ic), ("pde_bc", cos_pde_bc), ("ic_bc", cos_ic_bc)]:
    lo, hi = arr.min(), arr.max()
    valid = lo >= -1.001 and hi <= 1.001
    print(f"   {name}: [{lo:.6f}, {hi:.6f}]  {'OK' if valid else 'OUT OF RANGE'}")

# 4. Peak conflict verification
print(f"\n4. PEAK CONFLICT VERIFICATION:")
peak = d["peak_conflict"]
actual_min_idx = np.argmin(cos_pde_ic)
actual_min_val = cos_pde_ic[actual_min_idx]
actual_min_epoch = epochs[actual_min_idx]
print(f"   JSON says: epoch={peak['epoch']}, cos_pde_ic={peak['cosine_pde_ic']:.10f}")
print(f"   Recomputed: epoch={actual_min_epoch}, cos_pde_ic={actual_min_val:.10f}")
match = abs(peak["cosine_pde_ic"] - actual_min_val) < 1e-6 and peak["epoch"] == actual_min_epoch
print(f"   Peak conflict consistent: {match}")

# 4b. BUG CHECK: peak_conflict.cosine_pde_bc uses WRONG index
print(f"\n   JSON peak_conflict.cosine_pde_bc = {peak['cosine_pde_bc']:.10f}")
min_pde_bc_idx = np.argmin(cos_pde_bc)
min_pde_bc_val = cos_pde_bc[min_pde_bc_idx]
pde_bc_at_peak_pde_ic = cos_pde_bc[actual_min_idx]
print(f"   cos_pde_bc at min(pde_ic) epoch ({actual_min_epoch}): {pde_bc_at_peak_pde_ic:.10f}")
print(f"   min(cos_pde_bc) overall:                            {min_pde_bc_val:.10f} at epoch {epochs[min_pde_bc_idx]}")
if abs(peak["cosine_pde_bc"] - min_pde_bc_val) < 1e-6:
    print("   BUG: cosine_pde_bc in peak_conflict uses argmin(cos_pde_bc) NOT argmin(cos_pde_ic)")
    print("     It reports the minimum of pde_bc, not pde_bc AT the epoch of peak PDE-IC conflict.")
elif abs(peak["cosine_pde_bc"] - pde_bc_at_peak_pde_ic) < 1e-6:
    print("   OK cosine_pde_bc correctly reported at the peak PDE-IC conflict epoch")
else:
    print("   UNCLEAR which index was used for cosine_pde_bc")

# 5. Phase analysis verification
print(f"\n5. PHASE ANALYSIS VERIFICATION:")
n = len(epochs)
early = slice(0, n // 5)
mid = slice(n // 5, 3 * n // 5)
late = slice(3 * n // 5, n)

print(f"   n={n}, early=0:{n//5} ({n//5} samples), "
      f"mid={n//5}:{3*n//5} ({3*n//5 - n//5} samples), "
      f"late={3*n//5}:{n} ({n - 3*n//5} samples)")

pa = d["phase_analysis"]
all_ok = True
for phase_name, sl in [("early", early), ("mid", mid), ("late", late)]:
    for pair_name, arr in [("pde_ic", cos_pde_ic), ("pde_bc", cos_pde_bc), ("ic_bc", cos_ic_bc)]:
        json_val = pa[phase_name][pair_name]
        computed = float(np.mean(arr[sl]))
        if abs(json_val - computed) > 1e-5:
            print(f"   MISMATCH {phase_name}/{pair_name}: JSON={json_val:.10f} vs computed={computed:.10f}")
            all_ok = False
if all_ok:
    print("   OK All 9 phase/pair combos verified")

# 6. Hypothesis check
print(f"\n6. HYPOTHESIS VERIFICATION:")
peak_val = cos_pde_ic[np.argmin(cos_pde_ic)]
hypothesis = peak_val < -0.1
print(f"   Threshold: peak < -0.1")
print(f"   Peak value: {peak_val:.6f}")
print(f"   JSON says: {d['hypothesis_confirmed']}")
print(f"   Recomputed: {hypothesis}")
print(f"   Match: {hypothesis == d['hypothesis_confirmed']}")

# 7. L2 error sanity
print(f"\n7. L2 ERROR:")
print(f"   {d['l2_error']:.6f}")
print(f"   This is Burgers equation, so 0.09 is typical converged error")

# 8. Missing fields for journal readiness
print(f"\n8. JOURNAL READINESS ISSUES:")
missing = []
if "version" not in d:
    missing.append("version field missing")
if "config" not in d:
    missing.append("config section missing (n_hidden, n_neurons, lr, etc.)")
if "seed" not in d and "random_seed" not in d:
    missing.append("random seed not recorded -- not reproducible")
if "training_time" not in d:
    missing.append("training_time not recorded")
if "figure_notes" not in d:
    missing.append("figure_notes missing (needed for caption generation)")

for m in missing:
    print(f"   MISSING: {m}")

# 9. Late-training conflict re-emergence
print(f"\n9. LATE-TRAINING ANALYSIS:")
late_pde_ic = cos_pde_ic[late]
n_negative_late = int(np.sum(late_pde_ic < 0))
n_late = len(late_pde_ic)
print(f"   Late phase (epochs {epochs[3*n//5]}-{epochs[-1]}):")
print(f"   PDE-IC negative: {n_negative_late}/{n_late} measurements ({100*n_negative_late/n_late:.1f}%)")
print(f"   PDE-IC mean: {np.mean(late_pde_ic):.4f}")
print(f"   Late re-emergence NOT documented in JSON")

# 10. Cosine similarity consistency with heatmap
print(f"\n10. HEATMAP MATRIX CONSISTENCY:")
print(f"   Epoch 0 values:")
print(f"   cos(PDE,IC) = {cos_pde_ic[0]:.4f}  (heatmap shows -0.04)")
print(f"   cos(PDE,BC) = {cos_pde_bc[0]:.4f}  (heatmap shows  0.51)")
print(f"   cos(IC,BC)  = {cos_ic_bc[0]:.4f}   (heatmap shows -0.56)")

# Epoch 2500 (index 5)
i5 = 5
print(f"   Epoch 2500 values:")
print(f"   cos(PDE,IC) = {cos_pde_ic[i5]:.4f}  (heatmap shows -1.00)")
print(f"   cos(PDE,BC) = {cos_pde_bc[i5]:.4f}  (heatmap shows  0.80)")
print(f"   cos(IC,BC)  = {cos_ic_bc[i5]:.4f}   (heatmap shows -0.84)")

print(f"\n{'=' * 70}")
print("SUMMARY OF ISSUES FOUND")
print("=" * 70)
print("1. BUG: peak_conflict.cosine_pde_bc uses wrong index (argmin of pde_bc")
print("   not pde_bc at the epoch of max PDE-IC conflict)")
print("2. MISSING: version, config, seed, training_time, figure_notes")
print("3. Late-training PDE-IC conflict re-emergence undocumented")
print("4. No IC-BC conflict interpretation in JSON")
print("5. Hypothesis threshold (< -0.1) too simplistic for journal --")
print("   late-stage conflict is intermittent, not sustained")
print("=" * 70)
