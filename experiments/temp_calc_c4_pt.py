import torch
import numpy as np
from pathlib import Path

fm_ids = ["FM1", "FM2", "FM3", "FM4", "FM5", "FM6"]
print("C4 Changes:")
for fm in fm_ids:
    p = Path(f'results/exp26/{fm}_checkpoint.pt')
    if p.exists():
        ckpt = torch.load(str(p), weights_only=False)
        snaps = ckpt.get("snaps", {})
        c4 = snaps.get("C4_entropy")
        if c4 is not None:
            c4 = np.array(c4)
            val = abs(c4[-1] - c4[0]) / (abs(c4[0]) + 1e-8)
            print(f"{fm}: {val:.6f}")
        else:
            print(f"{fm}: C4_entropy not found in snaps")
    else:
        print(f"{fm}: file not found")
