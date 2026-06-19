import os
import glob
import json
import ast

def extract_docstring(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)
            docstring = ast.get_docstring(tree)
            return docstring if docstring else "No docstring"
    except Exception as e:
        return f"Error: {e}"

def extract_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return f"Error: {e}"

exp_dir = r"d:\Games\FAILURE STUDIES\experiments"
results_dir = os.path.join(exp_dir, "results")

output_lines = []

output_lines.append("=== EXPERIMENTS OVERVIEW ===")
for i in range(1, 23):
    py_files = glob.glob(os.path.join(exp_dir, f"exp{i}_*.py"))
    if not py_files:
        output_lines.append(f"exp{i}: NOT FOUND")
        continue
    
    py_file = py_files[0]
    doc = extract_docstring(py_file)
    output_lines.append(f"--- {os.path.basename(py_file)} ---")
    lines = doc.split('\n')
    output_lines.append("\n".join(lines[:10]))  # Just get the top part
    
    json_files = glob.glob(os.path.join(results_dir, f"exp{i}", "*.json"))
    if json_files:
        json_data = extract_json(json_files[0])
        output_lines.append(f"--- JSON Data for exp{i} ---")
        if isinstance(json_data, dict):
            for k in list(json_data.keys()):
                if "note" in k or k in ["optimal_bc_fraction", "optimal_l2", "failure_rate_pct", "rankings", "summary_statistics", "final_mean_error"]:
                    output_lines.append(f"Key: {k}")
                    val = json_data[k]
                    if isinstance(val, dict):
                        for k2, v2 in val.items():
                            if isinstance(v2, (int, float, str)):
                                output_lines.append(f"  {k2}: {v2}")
                    else:
                        output_lines.append(f"  {val}")
            
            # Additional extraction for specific keys
            if 'per_ratio' in json_data:
                output_lines.append("Has 'per_ratio' data.")
            if 'metrics' in json_data:
                output_lines.append("Has 'metrics' data.")
        else:
            output_lines.append("JSON Data not dict.")
    else:
         output_lines.append(f"No JSON results found for exp{i}")
         
    output_lines.append("")

out_file = r"d:\Games\FAILURE STUDIES\aggregated_data.txt"
with open(out_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))
print(f"Data saved to {out_file}")
