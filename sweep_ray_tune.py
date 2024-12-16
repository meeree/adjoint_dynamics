# Attempt to use ray-tune to make sweeping faster.
from train import train, ping_dir, parse_args
import json, copy
import numpy as np
from tqdm import tqdm

# Ray tune imports and wandb integration
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from ray.air.integrations.wandb import WandbLoggerCallback

args = parse_args() # Init args. Some will be set by the sweeping code itself, irrespective of what user inputted.
args = vars(args)

#root = 'sweep_scale_many/'
root = 'single/'
ping_dir(root)

# Grid of hyperparams.
#scales = np.linspace(1., 3.5, 16)
scales = [3.16666]
reruns = 1
#n_hidden = [100, 500]
#tasks = ['memory_pro', 'category_pro', 'd1fp2']
tasks = ['memory_pro']

grid = []
grid_idx = 0
for rerun_idx in range(reruns):
    for scale in scales:
        for task in tasks:
            run_args = copy.deepcopy(args)
            run_args['task'] = task
            run_args['init_scale_percent'] = scale
            run_args['prefix'] = root + f'grid_{grid_idx}/'
            ping_dir(run_args['prefix'])
            run_args['rerun_idx'] = rerun_idx
            grid.append(run_args)
            grid_idx += 1 

# Save the grid of args to a "manifest" json file that can help to navigate it.
# This will be a list of dictionaries. Each dict contains arguments for a run.
mfst = root + 'grid_manifest.json'
print(f"Saving manifest of grid of {len(grid)} runs to {mfst}")
with open(mfst, 'w') as fout:
    json.dump(grid, fout)


# Hyperparameter grid
tune_grid = copy.deepcopy(args)
tune_grid['task'] = tune.grid_search(tasks)
tune_grid['init_scale_percent'] = tune.grid_search(scales)
tune_grid['rerun_idx'] = tune.grid_search(np.arange(reruns))
tune_grid['wandb'] = '' # DISABLE WANDB INTEGRATION DIRECTLY. TRY USE TUNE.

tune_grid['device'] = 'cuda'

def tune_with_callback():
    # FIFOScheduler will run until max iterations without any culling. Good for analysis.
    scheduler = FIFOScheduler(
        #metric="loss",       # The metric to optimize
        #mode="min",          # Minimize the metric
        #max_t=args['niters'] # Maximum iterations (or epochs)
    )

    # Use WandbLoggerCallback to log to wandb while training :D
    if args['wandb'] != '':
        print("Logging to weights-and-biases project : " + args['wandb'])

    tuner = tune.Tuner(
#        tune.with_resources(train, {'cpu': 1, 'gpu': 1./6.}), # Fractional GPU usage.
#        tune.with_resources(train, {'cpu': 2, 'gpu': 1./3.}), # Fractional GPU usage.
        tune.with_resources(train, {'cpu': 8, 'gpu': 1}), # Fractional GPU usage.
#        tune.with_resources(train, {'cpu': 1, 'gpu': 0}), # No GPU usage.
        tune_config=tune.TuneConfig(
            scheduler = scheduler
        ),
        run_config=ray.train.RunConfig(
            callbacks=[WandbLoggerCallback(project=args['wandb'])]
        ),
        param_space = tune_grid,
    )
    tuner.fit()

print("Starting Ray Tune Sweep ... ")
tune_with_callback()
