# Attempt to use ray-tune to make sweeping faster.
from train import train, ping_dir, parse_args
import json
from copy import deepcopy
from itertools import product
import os

# Ray tune imports and wandb integration
import ray
from ray import tune
from ray.tune.schedulers import FIFOScheduler
from ray.air.integrations.wandb import WandbLoggerCallback


def sweep(sweep_name, sweep_grid, cpus_per_run, gpus_per_run):
    ###############################################################################################################
    # Sweep a parameter space. gpus_per_run can be fractional, e.g. 1/10 for a tenth of a gpu for each run.
    ###############################################################################################################

    # Init args as for a single run. These set the default arguments. Sweep_grid sets the parameters that will be swept.
    # For example, if we want to sweep 'n_hidden,' sweep_grid could be {'n_hidden': tune.grid_search(values)} where values are what we want to try in the sweep.
    args = parse_args() 
    args = vars(args)

    # Set a couple arguments myself based on what they MUST be. 
    if gpus_per_run > 0:
        args['device'] = 'cuda'
    else:
        args['device'] = 'cpu'
    args['use_ray'] = True # Definitely true :)
    args['verbose'] = False # Ray logs a ton already.

    # All runs in sweep will be under this root directory.
    root = sweep_name + '/'
    ping_dir(root)
    root = root + args['task'] + '/'
    ping_dir(root)

    # Generate cartesian product over all sweep parameters.
    base_cfg = deepcopy(args)
    keys, values = sweep_grid.keys(), sweep_grid.values()
    dynamic = [dict(zip(keys, combination)) for combination in product(*values)]

    all_cfgs = [deepcopy(base_cfg) for i in range(len(dynamic))]
    for i in range(len(dynamic)):
        all_cfgs[i].update(dynamic[i])
        all_cfgs[i]['prefix'] = os.getcwd() + '/' + root + 'grid_' + str(i)
    tune_grid = tune.grid_search(all_cfgs)

    # Save the swept parameters cartesian product dictionary to a "grid_manifest." 
    # This will allow user to clearly see which parameters are swept and which runs to consider.
    mfst = root + 'grid_manifest.json'
    print(f"Saving manifest of grid of {len(dynamic)} runs to {mfst}")
    with open(mfst, 'w') as fout:
        json.dump(dynamic, fout)

    print("All runs should be located in " + root)

    # FIFOScheduler will run until max iterations without any culling. Good for analysis.
    scheduler = FIFOScheduler()

    if args['wandb'] != '':
        print("Logging to weights-and-biases project : " + args['wandb'])

    tuner = tune.Tuner(
        tune.with_resources(train, {
                'cpu': cpus_per_run, 
                'gpu': gpus_per_run 
        }), 
        tune_config=tune.TuneConfig(
            scheduler = scheduler
        ),
        run_config=ray.train.RunConfig(
            callbacks=[WandbLoggerCallback(project=args['wandb'])] # Logging is indirectly done through ray report.
        ),
        param_space = tune_grid,
    )
    print("-------------------------- Starting Ray Tune Sweep --------------------------")
    tuner.fit()
