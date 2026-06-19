import os, re

def check_latex_refs():
    tex_files = []
    for root, _, files in os.walk(r'd:\Games\FAILURE STUDIES'):
        for f in files:
            if f.endswith('.tex'):
                tex_files.append(os.path.join(root, f))
    
    labels = {}
    refs = {}
    
    for f in tex_files:
        content = open(f, encoding='utf-8').read()
        
        for m in re.findall(r'\\label\{([^}]+)\}', content):
            labels[m] = f
            
        for m in re.findall(r'\\(?:ref|pageref)\{([^}]+)\}', content):
            refs.setdefault(m, []).append(f)
            
    missing = set(refs.keys()) - set(labels.keys())
    unused = set(labels.keys()) - set(refs.keys())
    
    print("=== MISSING LABELS ===")
    for m in sorted(missing):
        print(f"✗ \\ref{{{m}}} -> NOT FOUND anywhere (used in {refs[m][0]})")
        
    print("\n=== UNUSED LABELS ===")
    for u in sorted(unused):
        print(f"[UNUSED LABEL] {u} in {labels[u]} - consider removing or check if \\ref was accidentally deleted")
        
    print("\n=== VALID LABELS ===")
    for m in sorted(set(refs.keys()) & set(labels.keys())):
        print(f"✓ \\ref{{{m}}} -> found in {labels[m]}")

if __name__ == '__main__':
    check_latex_refs()
