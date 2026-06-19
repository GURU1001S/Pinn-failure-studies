import json

d = json.load(open(r"d:\Games\FAILURE STUDIES\results\exp22_2\exp22_2_results.json"))
m = d["recovery_matrix"]
labels = d["config"]["intervention_labels"]

print(r"\begin{table}[htpb]")
print(r"    \centering")
print(r"    \caption{Recovery rates by post-hoc intervention across 20 distinct failure configurations.}")
print(r"    \label{tab:exp22_recovery}")
print(r"    \begin{tabular}{@{}lrrrr@{}}")
print(r"        \toprule")
print(r"        Intervention & $N_{recovered}$ & $N_{partial}$ & $N_{failed}$ & Recovery Rate \\")
print(r"        \midrule")

for j, lbl in enumerate(labels):
    rec = sum(1 for row in m if row[j]["outcome"] == "RECOVERED")
    part = sum(1 for row in m if row[j]["outcome"] == "PARTIAL")
    fail = sum(1 for row in m if row[j]["outcome"] == "FAILED")
    rate = rec / 20.0
    # Clean label for latex
    lbl_tex = lbl.replace("×", r"\times")
    print(f"        {lbl_tex} & ${rec}$ & ${part}$ & ${fail}$ & ${rate*100:.0f}\\%$ \\\\")

print(r"        \bottomrule")
print(r"    \end{tabular}")
print(r"\end{table}")
