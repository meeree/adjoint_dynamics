from sweep_ray_tune import sweep
import numpy as np

grid = {
    'init_scale_percent': np.linspace(1.0, 3.0, 12).tolist(),
    'rerun_idx': np.arange(1).tolist(), # This is a little trick. Add an extra field to each run with the rerun_idx. Allows us to do multiple runs per trial with different inits.
}
sweep('scale_sweeps_same_init', grid, 2, 1/2.)
