# An example of how to use sweeping code to run a sweep of different initializaiton scales with multiple reruns using ray-tune to distribute the over GPUs/CPUs.

import sys
sys.path.append('../')
from sweep_ray_tune import sweep
import numpy as np

grid = {
    'init_scale_percent': np.linspace(.1, 4.5, 400).tolist(),
    'rerun_idx': np.arange(5).tolist(), # This is a little trick. Add an extra field to each run with the rerun_idx. Allows us to do multiple runs per trial with different inits.
}
cpus_per_run, gpus_per_run = 2, 1/4.
sweep('scale_sweeps_1_20', grid, cpus_per_run, gpus_per_run)
