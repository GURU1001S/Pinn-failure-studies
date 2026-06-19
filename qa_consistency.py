import json
import math
import sys
import os

def check_close(val1, val2, tol=1e-2):
    return abs(val1 - val2) < tol

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"[FAIL] Missing file: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def run_qa():
    print("=" * 60)
    print("Numeric Consistency QA Check")
    print("=" * 60)
    
    passed = True

    # 1. Exp 27: Eta^2 and Ratios
    print("\n--- Checking Exp 27 (Variance Decomposition) ---")
    exp27 = load_json(r"d:\Games\FAILURE STUDIES\results\exp27\exp27_results.json")
    if exp27:
        # Advection
        s_adv = exp27["eta2_advection"]["S_group"]
        h_adv = exp27["eta2_advection"]["H_group"]
        ratio_adv = s_adv / h_adv
        print(f"Advection S/H Ratio: {ratio_adv:.2f} (expected ~1.7)")
        if not check_close(ratio_adv, 1.7, 0.1):
            print(f"  [FAIL] Advection S/H ratio is {ratio_adv:.2f}, not 1.7")
            passed = False
            
        # Burgers
        s_burg = exp27["eta2_burgers"]["S_group"]
        h_burg = exp27["eta2_burgers"]["H_group"]
        ratio_burg = s_burg / h_burg
        print(f"Burgers S/H Ratio: {ratio_burg:.2f} (expected ~287-292)")
        if not check_close(ratio_burg, 287.5, 5.0):
            print(f"  [FAIL] Burgers S/H ratio is {ratio_burg:.2f}, not ~287.5-292")
            passed = False
            
    # 2. Exp 5: Sharpness Ratio
    print("\n--- Checking Exp 5 (Loss Landscape Sharpness) ---")
    exp5 = load_json(r"d:\Games\FAILURE STUDIES\results\exp5\exp5_results.json")
    if exp5:
        failed_s = exp5["sharpness_failed"]
        success_s = exp5["sharpness_success"]
        sharp_ratio = failed_s / success_s
        print(f"Sharpness Ratio (Failed/Success): {sharp_ratio:,.2f} (expected ~27,947)")
        if not check_close(sharp_ratio, 27947, 50):
            print(f"  [FAIL] Sharpness ratio is {sharp_ratio:.2f}, not ~27947")
            passed = False
            
    # 3. Exp 22.2: Recovery Rates
    print("\n--- Checking Exp 22.2 (Recovery Rates) ---")
    exp22 = load_json(r"d:\Games\FAILURE STUDIES\results\exp22_2\exp22_2_results.json")
    if exp22:
        rates = exp22["intervention_recovery_rates"]
        print(f"Recovery Rates: {rates}")
        if rates.get("I1: 10\u00d7 Collocation", 0) != 0.45:
            print("  [FAIL] I1 rate mismatch")
            passed = False
        if rates.get("I2: Fourier Features", 0) != 0.55:
            print("  [FAIL] I2 rate mismatch")
            passed = False
        if rates.get("I3: L-BFGS", 0) != 0.15:
            print("  [FAIL] I3 rate mismatch")
            passed = False

    print("\n" + "=" * 60)
    if passed:
        print("OVERALL STATUS: ALL CHECKS PASSED")
    else:
        print("OVERALL STATUS: FAILED")
    print("=" * 60)

if __name__ == "__main__":
    run_qa()
