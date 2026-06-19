import sys, os
import traceback
import numpy as np

with open('grid_test_output.txt', 'w') as f:
    try:
        f.write('Starting test...\n')
        import torch
        sys.path.insert(0, r'd:\Games\FAILURE STUDIES\experiments')
        from exp26_conservation_law_audit import PINN, load_checkpoint, FAILURE_MODES, DEVICE, DTYPE
        
        fm2_cfg = next(c for c in FAILURE_MODES if c['id'] == 'FM2')
        ckpt = load_checkpoint('FM2')
        if ckpt is None:
            f.write('FM2 checkpoint not found.\n')
            sys.exit(1)
            
        model = PINN(n_hidden=fm2_cfg['n_hidden'], n_neurons=fm2_cfg['n_neurons']).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        resolutions = [128, 256, 512, 1024, 2048]
        t_val = 2.0  # Final time for advection

        f.write('Grid Convergence for FM2 (Success Control) at t=2.0\n')
        f.write('Nx      | C1 (Mass)\n')
        f.write('-------------------\n')
        for nx in resolutions:
            x_vals = np.linspace(0.0, 2*np.pi, nx)
            x_t = torch.tensor(x_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1)
            t_t = torch.full((nx, 1), t_val, dtype=DTYPE, device=DEVICE)
            with torch.no_grad():
                u = model(x_t, t_t).cpu().numpy().flatten()
            
            mass = float(np.trapezoid(u, x_vals))
            f.write(f'{nx:<7} | {mass:.8f}\n')
        f.write('Done.\n')
    except Exception as e:
        f.write(f'Error: {e}\n')
        f.write(traceback.format_exc())
