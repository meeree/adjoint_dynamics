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

from architecture import ModelRNNv3 # My model. In future, might want to make this customizable.

def parse_args():
    parser = argparse.ArgumentParser('RNN Training')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--n_hidden', type=int, default=100)
    parser.add_argument('--noise_std', type=float, default=0.)
    parser.add_argument('--niters', type=int, default=50000)
    parser.add_argument('--task', type=str, default='memory_pro')
    parser.add_argument('--terminal_loss', type=bool, default=False)
    parser.add_argument('--checkpoint_file', type=str, default='')
    parser.add_argument('--prefix', type=str, default='.')
    parser.add_argument('--method', type=str, default='adam')
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--init_scale_percent', type=float, default=1.)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--stopping_acc', type=float, default=0.99)
    parser.add_argument('--wandb', type=str, default='')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', type=bool, default=False)
    return parser.parse_args()

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

def train(args, task):
    if isinstance(args, dict):
        args = argparse.Namespace(**args) # Dict to arguments.

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

    # Load the data and model.
    if args.random_seed != -1:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    if use_ray:
        import ray
        if args.wandb:
            from ray.air.integrations.wandb import setup_wandb # Ray-tune has wandb support, for logging many distirbuted jobs to the cloud (wow!)
            wandb = setup_wandb(vars(args), project=args.wandb)
    elif args.wandb: # No ray but still use weights-and-biases:
        import wandb
        wandb.init(project = args.wandb, config = vars(args))

    # Generate the data.
    inps, targets = task.generate()
    dset = torch.utils.data.TensorDataset(inps, targets)
    dloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
    model = torch.jit.script(ModelRNNv3(args.n_hidden, inps.shape[-1], targets.shape[-1]))
    model.noise_std = args.noise_std

    # Scale weights by a percent to play with different initializations.
    with torch.no_grad():
        model.W.weight.data *= args.init_scale_percent
        model.W_in.weight.data *= args.init_scale_percent
        model.W_out.weight.data *= args.init_scale_percent

    if args.checkpoint_file != '':
        ld = torch.load(args.checkpoint_file)['model']
        model.load_state_dict(ld)

    if args.method == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.method == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    losses, accs = [], []
    max_acc = 0.
    loss_fn = nn.MSELoss()
    glog = []
    
    model = model.to(args.device)
    model_prev = model
    my_range = range(args.niters) if not args.verbose else tqdm(range(args.niters))
    for itr in my_range:
        # Training Step.
        optim.zero_grad()
        batch, target = next(iter(dloader))
        batch, target = batch.to(args.device), target.to(args.device)
        out, hidden = model.evaluate(batch)
        loss = compute_loss(args, loss_fn, out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        # Checkpoint Saving.
        losses.append(loss.item())
        accs.append(task.accuracy(out, target).item())
        checkpoint = {
            'losses': losses,
            'accuracies': accs,
            'iterations': len(losses),
            'optim': optim.state_dict(),
            'model': model.state_dict(),
            'model_prev': model_prev.state_dict()
        }

        # Logging to ray-tune, wandb, saving to file, and checking accuracy threshold for early stopping.
        if use_ray:
            ray.train.report({"accuracy": accs[-1], "loss": losses[-1]})
        if args.wandb != '':
            wandb.log({"accuracy": accs[-1], "loss": losses[-1]})

        if len(accs) > 50:
            smooth_acc = sum(accs[-50:]) / 50.
            if smooth_acc > max_acc:
                max_acc = smooth_acc
                torch.save(checkpoint, root_dir + 'best.pt')

            if smooth_acc >= args.stopping_acc:
                if args.verbose:
                    print("Hit stopping accuracy. Terminating early.")
                break

        if itr == args.niters - 1 or itr % args.save_freq == 0:
            torch.save(checkpoint, root_dir + f'checkpoints/checkpoint_{itr}.pt')
            print("Saving to, " + root_dir + f'checkpoints/checkpoint_{itr}.pt')

        model_prev = copy.deepcopy(model) # Save previous model for analysis of single GD step changes.

    return sum(losses[-50:]) / 50.

if __name__ == "__main__":
    args = parse_args()
    task = task_importer(args.task) # Get the task from a consortium of them.
    train(args, task)
