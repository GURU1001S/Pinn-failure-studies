import json

with open('results/exp26/exp26_results.json') as f:
    d = json.load(f)

for fm in d['per_failure_mode']:
    if 'C4_entropy' in d['per_failure_mode'][fm]:
        c4 = d['per_failure_mode'][fm]['C4_entropy']
        val = abs(c4[-1] - c4[0]) / (abs(c4[0]) + 1e-8)
        print(f"{fm}: {val}")
