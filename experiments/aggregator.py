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
results_dir = os.path.join(exp_dir, "results")
results_dir_old = r"D:\Games\FAILURE STUDIES\results"

output_lines = []

experiments_to_check = [
    "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp9", 
    "exp10", "exp11", "exp12", "exp13", "exp14", "exp15", "exp16", "exp17", 
    "exp18", "exp19", "exp20", "exp21", "exp22_2", "exp23", "exp24",
    "specialexp2", "specialexp3", "specialexp4", "specialexp5", "specialexp6"
]

output_lines.append("=== EXPERIMENTS OVERVIEW ===")
for exp in experiments_to_check:
    py_files = glob.glob(os.path.join(exp_dir, f"{exp}_*.py"))
    if not py_files and exp == "exp22_2":
        py_files = glob.glob(os.path.join(exp_dir, f"exp22_*.py"))
    if not py_files and exp == "exp23":
        py_files = glob.glob(os.path.join(exp_dir, f"specialexp5_*.py"))
    if not py_files and exp == "exp24":
        py_files = glob.glob(os.path.join(exp_dir, f"specialexp6_*.py"))
    if not py_files and "specialexp" in exp:
        py_files = glob.glob(os.path.join(exp_dir, f"{exp}_*.py"))
    if not py_files:
        py_files = glob.glob(os.path.join(exp_dir, f"{exp}.py"))

    output_lines.append(f"\n{'='*40}\n{exp.upper()}\n{'='*40}")
    
    if py_files:
        py_file = py_files[0]
        doc = extract_docstring(py_file)
        output_lines.append(f"--- Docstring: {os.path.basename(py_file)} ---")
        if doc:
            lines = doc.split('\n')
            output_lines.append("\n".join(lines[:15]))
    else:
        output_lines.append("Python file NOT FOUND")
    
    # JSONs
    json_files = glob.glob(os.path.join(results_dir, exp, "*.json"))
    if not json_files:
        json_files = glob.glob(os.path.join(results_dir, f"{exp}_*.json"))
    if not json_files:
        json_files = glob.glob(os.path.join(results_dir_old, exp, "*.json"))
    if not json_files:
        json_files = glob.glob(os.path.join(results_dir_old, f"{exp}_*.json"))
        
    if json_files:
        json_data = extract_json(json_files[0])
        output_lines.append(f"--- JSON Data ({os.path.basename(json_files[0])}) ---")
        if isinstance(json_data, dict):
            def print_dict(d, indent=2):
                for k, v in d.items():
                    if isinstance(v, dict):
                        output_lines.append(" " * indent + str(k) + ":")
                        print_dict(v, indent + 2)
                    elif isinstance(v, list) and len(v) > 5:
                        output_lines.append(" " * indent + f"{k}: List of {len(v)} items")
                    else:
                        output_lines.append(" " * indent + f"{k}: {str(v)[:200]}") 
            print_dict(json_data)
        else:
            output_lines.append("JSON Data not dict.")
    else:
         output_lines.append("No JSON results found.")

out_file = os.path.join(exp_dir, "aggregator_output.txt")
with open(out_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))
print(f"Data saved to {out_file}")
