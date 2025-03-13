from sweep_ray_tune import sweep
import numpy as np

grid = {
    'init_scale_percent': np.linspace(.1, 4.5, 400).tolist(),
    'rerun_idx': np.arange(1).tolist(), # This is a little trick. Add an extra field to each run with the rerun_idx. Allows us to do multiple runs per trial with different inits.
}
sweep('scale_sweeps_1_20', grid, 2, 1/4.)
