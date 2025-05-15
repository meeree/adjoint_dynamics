#######################################################################################################
# Train a model with a variety of arguments.                                                          |
# Creates a directory and saves checkpoints of model throughout training.                             | 
# Also, integrates with weights-and-biases, so user can log to the web (recommended).                 |
# Also, integrates with ray-tune, so user can run over many distribued jobs (see sweep_ray_tune.py).  |
#######################################################################################################

import torch
from torch import nn
import numpy as np
import copy, argparse
import json # For saving config file.

from architecture import SequentialModel # My model. In future, might want to make this customizable.

def parse_args(args = None):
    parser = argparse.ArgumentParser('RNN Training')

    # Selecting task:
    parser.add_argument('--task_suite', type=str, default='custom', choices = ['custom', 'neurogym'])
    parser.add_argument('--task', type=str, default='MotorTiming-v0')

    # Training related:
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--method', type=str, default='sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--noise_std', type=float, default=0.)
    parser.add_argument('--niters', type=int, default=50000)
    parser.add_argument('--stopping_acc', type=float, default=0.99)
    parser.add_argument('--running-window', type=int, default=50)
    parser.add_argument('--terminal_loss', type=bool, default=False)
    parser.add_argument('--use_ray', type=bool, default=False)

    # Initialization related:
    parser.add_argument('--random_seed', type=int, default=-1)
    parser.add_argument('--n_hidden', type=int, default=100)
    parser.add_argument('--init_scale_percent', type=float, default=1.)

    # For starting from a checkpoint:
    parser.add_argument('--checkpoint_file', type=str, default='')

    # Saving results related:
    parser.add_argument('--prefix', type=str, default='.')
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--wandb', type=str, default='')
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args(args)

def get_default_args():
    return vars(parse_args([]))

def ping_dir(directory, clear = False):
    # Check if directory exists and make if not. If clear flag is True, clear any contents of the directory if it exists.
    import os
    if len(directory) == 0:
        return 

    if os.path.exists(directory):
        if clear:
            import shutil
            shutil.rmtree(directory)
            os.mkdir(directory)
    else:
        os.mkdir(directory)

def compute_loss(args, loss_fn, out, target):
    if args.terminal_loss:
        return loss_fn(out[:, -1:, :], target[:, -1:, :])
    return loss_fn(out, target)

def retrieve_task_from_args(args):
    # Get the task from a suite of them.
    from tasks import get_task_wrapper
    task_wrapper = get_task_wrapper(args.task_suite)
    task = task_wrapper(**vars(args)) # Pass kwargs.
    return task
    
def train(args, task = None):
    if isinstance(args, dict):
        defaults = get_default_args() # For any missing arguments.
        defaults.update(args)
        args = argparse.Namespace(**defaults) # Dict to arguments.

    # The user can pass a task (e.g. like in tasks.py) directly. 
    # Alternatively, they can specify it through args, i.e. keeping task = None:
    # Sending a task directly without args is more flexible but requires more work.
    if task is None:
        task = retrieve_task_from_args(args)

    root_dir = args.prefix + '/'
    ping_dir(root_dir)
    ping_dir(root_dir + 'checkpoints', clear = True)
    if args.verbose:
        from tqdm import tqdm # progress bar.
        print(f'Beginning Training in root directory {root_dir}...')
        if args.wandb:
            print(f'Logging to Weights-and-Biases. Project ' + args.wandb)
        if args.use_ray:
            print(f'Using ray-tune to distribute compute!')

    # Save the config for this run to the root_dir. It'll also be available on wandb.
    with open(root_dir + 'config.json', 'w') as cfg_out:
        json.dump(vars(args), cfg_out)

    # Load the data and model.
    if args.random_seed != -1:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    if args.use_ray:
        import ray
        if args.wandb:
            args.wandb = '' # Disable direct logging with wandb. Will be done indirectly, as in sweep_ray_tune.py.
    elif args.wandb: # No ray but still use weights-and-biases:
        import wandb
        wandb.init(project = args.wandb, config = vars(args))

    # Setup model. Use a sample input to set the shapes.
    example_batch, example_target = task()
    model = torch.jit.script(SequentialModel(args.n_hidden, example_batch.shape[-1], example_target.shape[-1]))
    model.noise_std = args.noise_std

    # Scale weights by a percent to play with different initializations.
    with torch.no_grad():
        for param in model.parameters():
            param.data *= args.init_scale_percent

    if args.checkpoint_file != '':
        ld = torch.load(args.checkpoint_file)['model']
        model.load_state_dict(ld)
    
    model = model.to(args.device)
    model_prev = model

    if args.method == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.method == 'adagrad':
        optim = torch.optim.Adagrad(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.method == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    acc_fn = lambda out, target: {'acc': -1} # Undefined.
    if hasattr(task, 'accuracy') and callable(getattr(task, 'accuracy')): # Check if accuracy function exists in task.
        acc_fn = task.accuracy

    example_acc_dict = acc_fn(example_target, example_target)
    accs = {key: [] for key in example_acc_dict.keys()}

    losses = []
    loss_fn = nn.MSELoss()

    my_range = range(args.niters) if not args.verbose else tqdm(range(args.niters))
    for itr in my_range:
        # Training Step.
        optim.zero_grad()
        batch, target = task()
        batch, target = batch.to(args.device), target.to(args.device)
        out, hidden = model(batch)
        loss = compute_loss(args, loss_fn, out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        # Checkpoint Saving.
        losses.append(loss.item())
        
        # Accuracy metrics.
        acc_dict = acc_fn(out, target) # Dictionary of accuracy metrics.
        for key in acc_dict.keys():
            accs[key].append(acc_dict[key])

        checkpoint = {
            'losses': losses,
            'accuracies': accs,
            'iterations': len(losses),
            'optim': optim.state_dict(),
            'model': model.state_dict(),
            'model_prev': model_prev.state_dict()
        }

        # Logging to ray-tune, wandb, saving to file, and checking accuracy threshold for early stopping.
        log_entry = {"loss": losses[-1]}
        log_entry.update(acc_dict)
        if args.use_ray:
            ray.train.report(log_entry)
        if args.wandb != '':
            wandb.log(log_entry)

        if len(losses) > args.running_window:
            # Check ALL accuracies are above stopping_acc.
            stop = True 
            for key, vals in accs.items():
                smooth_acc = sum(vals[-args.running_window:]) / args.running_window
                if smooth_acc < args.stopping_acc:
                    stop = False
                    break

            if stop:
                if args.verbose:
                    print("Hit stopping accuracy. Terminating early.")
                break

        if itr == args.niters - 1 or itr % args.save_freq == 0:
            torch.save(checkpoint, root_dir + f'checkpoints/checkpoint_{itr}.pt')
            print("Saving to, " + root_dir + f'checkpoints/checkpoint_{itr}.pt')

        model_prev = copy.deepcopy(model) # Save previous model for analysis of single GD step changes.

    if args.wandb != '':
        wandb.finish()

    return sum(losses[-args.running_window:]) / args.running_window

if __name__ == "__main__":
    # For user to run. Can also run training code in other python code, e.g. sweep_ray_tune.py.
    args = parse_args()
    train(args) # Task is specified through arguments (task_suite and task).
