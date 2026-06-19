import json
import numpy as np

with open(r"d:\Games\FAILURE STUDIES\results\exp26\exp26_results.json", "r") as f:
    data = json.load(f)

for fm_key, fm_data in data["per_failure_mode"].items():
    print(f"\n--- {fm_key} ---")
    
    # Values we are looking for
    if fm_key == "FM1":
        targets = [0.125, 3.45]
    elif fm_key == "FM3":
        targets = [4.25, 8.90]
    elif fm_key == "FM4":
        targets = [0.0]
    elif fm_key == "FM6":
        targets = [689.67]
    else:
        continue
        
    for target in targets:
        print(f"Looking for {target} in {fm_key}")
        for key, val in fm_data.items():
            if isinstance(val, float):
                if np.isclose(val, target, atol=0.01):
                    print(f"  MATCH in {key}: {val}")
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, float) and np.isclose(v2, target, atol=0.01):
                        print(f"  MATCH in {key}.{k2}: {v2}")
            elif isinstance(val, list):
                for i, v2 in enumerate(val):
                    if isinstance(v2, float) and np.isclose(v2, target, atol=0.01):
                        print(f"  MATCH in {key}[{i}]: {v2}")
