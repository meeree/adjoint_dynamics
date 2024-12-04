# Perform a sweep where we train model on memory pro and category pro tasks
# with a variety of different weight scales. Also, look at transfer learning
# between the two tasks. 
from train import train, ping_dir, parse_args
import numpy as np
import copy
import json
from tqdm import tqdm

root = 'sweep_correlated_x/'
ping_dir(root)

args = parse_args() # Init args. Some will be set by the sweeping code itself, irrespective of what user inputted.

# Grid of hyperparams.
scales = np.linspace(.2, 3.0, 40)
#n_hidden = [100, 500]
tasks = ['d1fp2_correlated_x']

grid = []
grid_idx = 0
for scale in scales:
    for task in tasks:
        run_args = copy.deepcopy(args)
        run_args.task = task
        run_args.init_scale_percent = scale
        run_args.prefix = root + f'grid_{grid_idx}/'
        grid.append(run_args)
        grid_idx += 1 

# Save the grid of args to a "manifest" json file that can help to navigate it.
# This will be a list of dictionaries. Each dict contains arguments for a run.
mfst = root + 'grid_manifest.json'
print(f"Saving manifest of grid of {len(grid)} runs to {mfst}")
with open(mfst, 'w') as fout:
    json.dump([vars(run_args) for run_args in grid], fout)

print("Starting Sweep ... ")
for run_args in tqdm(grid):
    train(run_args)
