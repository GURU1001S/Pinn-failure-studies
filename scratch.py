import json
d = json.load(open(r"d:\Games\FAILURE STUDIES\results\exp22_2\exp22_2_results.json"))
m = d["recovery_matrix"]
labels = d["config"]["intervention_labels"]
print("Intervention | RECOVERED | PARTIAL | FAILED | Rate")
for j, lbl in enumerate(labels):
    rec = sum(1 for row in m if row[j]["outcome"] == "RECOVERED")
    part = sum(1 for row in m if row[j]["outcome"] == "PARTIAL")
    fail = sum(1 for row in m if row[j]["outcome"] == "FAILED")
    rate = rec / 20.0
    print(f"{lbl} | {rec} | {part} | {fail} | {rate*100:.0f}%")
