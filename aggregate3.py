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

exp_dir = r"D:\Games\FAILURE STUDIES\experiments"
results_dir_1 = r"D:\Games\FAILURE STUDIES\results"
results_dir_2 = os.path.join(exp_dir, "results")

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
    if doc:
        lines = doc.split('\n')
        output_lines.append("\n".join(lines[:10]))
    
    # Check both potential results directories
    json_files = []
    for rdir in [results_dir_1, results_dir_2]:
        json_files.extend(glob.glob(os.path.join(rdir, f"exp{i}", "*.json")))
        json_files.extend(glob.glob(os.path.join(rdir, f"exp{i}_*.json")))
        json_files.extend(glob.glob(os.path.join(rdir, f"*exp{i}*.json")))
    
    # Remove duplicates
    json_files = list(set(json_files))
        
    if json_files:
        json_data = extract_json(json_files[0])
        output_lines.append(f"--- JSON Data for exp{i} ({os.path.basename(json_files[0])}) ---")
        if isinstance(json_data, dict):
            def print_dict(d, indent=2):
                for k, v in d.items():
                    if isinstance(v, dict):
                        output_lines.append(" " * indent + str(k) + ":")
                        print_dict(v, indent + 2)
                    elif isinstance(v, list) and len(v) > 5:
                        output_lines.append(" " * indent + f"{k}: List of {len(v)} items")
                    elif "note" in str(k) or "finding" in str(k) or k in ["optimal_bc_fraction", "lowest_variance", "failure_threshold", "improvement_pct", "rankings", "summary_statistics"]:
                        output_lines.append(" " * indent + f"{k}: {v}")
            print_dict(json_data)
        else:
            output_lines.append("JSON Data not dict.")
    else:
         output_lines.append(f"No JSON results found for exp{i}")
         
    output_lines.append("")

out_file = r"D:\Games\FAILURE STUDIES\aggregated_data3.txt"
with open(out_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))
print(f"Data saved to {out_file}")
