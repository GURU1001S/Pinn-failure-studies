import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_num_threads(1)
import numpy as np
from pathlib import Path

fm_ids = ["FM1", "FM2", "FM3", "FM4", "FM5", "FM6"]
print("C4 Changes:")
for fm in fm_ids:
    p = Path(f'results/exp26/{fm}_checkpoint.pt')
    if p.exists():
        # Map location cpu prevents CUDA init hangs
        ckpt = torch.load(str(p), map_location='cpu', weights_only=False)
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
