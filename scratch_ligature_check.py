import os
import re

root_dir = r"D:\Games\FAILURE STUDIES"
sections_dir = os.path.join(root_dir, "sections")

patterns = {
    "ne-scale": "fine-scale",
    "dierent": "different",
    "diculty": "difficulty",
    "oating": "floating",
    "eective": "effective",
    "oset": "offset",
    "pro le": "profile",
    "coef cient": "coefficient"
}

# Boundary/context-based patterns:
# "rst" at word boundaries or start/end of word: " rst ", etc.
# "xed" at word start/boundaries: " xed", etc.
# "nite" -> "finite" (usually \bnite\b or similar)
# "uid" -> "fluid" (usually \buid\b or similar)
# "cient" -> "efficient" or "coefficient" (context-dependent)
# "ow" -> "flow" (context-dependent, e.g. "gradient ow", "fluid ow", "velocity ow", "uniform ow")

for f in os.listdir(sections_dir):
    if f.endswith(".tex"):
        filepath = os.path.join(sections_dir, f)
        content = open(filepath, encoding="utf-8").read()
        lines = content.splitlines()
        for idx, line in enumerate(lines, 1):
            # 1. Simple replacements
            for pat, repl in patterns.items():
                if pat in line:
                    print(f"[SIMPLE] {f}:{idx} | {pat} | {line.strip()}")
            
            # 2. "rst" -> "first"
            if re.search(r"\brst\b", line):
                print(f"[RST] {f}:{idx} | {line.strip()}")
            
            # 3. "xed" -> "fixed" (at start of word, e.g., \bxed)
            if re.search(r"\bxed\w*", line):
                print(f"[XED] {f}:{idx} | {line.strip()}")
                
            # 4. "nite" -> "finite" (\bnite\b or similar)
            if re.search(r"\bnite\b", line) or "nite-volume" in line or "nite-difference" in line:
                print(f"[NITE] {f}:{idx} | {line.strip()}")
                
            # 5. "uid" -> "fluid"
            if re.search(r"\buid\b", line) or "uid-dynamics" in line or "uid-dominated" in line:
                print(f"[UID] {f}:{idx} | {line.strip()}")
                
            # 6. "cient" -> "efficient" or "coefficient"
            if "cient" in line:
                print(f"[CIENT] {f}:{idx} | {line.strip()}")
                
            # 7. "ow" -> "flow"
            if re.search(r"\bow\b", line) or "gradient ow" in line or "uid ow" in line or "flow" not in line and " ow" in line:
                # print candidates
                if any(x in line for x in [" ow ", " ow.", " ow,", " ow}", " ow)"]):
                    print(f"[OW] {f}:{idx} | {line.strip()}")
