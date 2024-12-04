import argparse
from tqdm import tqdm
import numpy as np
import scipy as sp
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import time
import pickle
import glob
import re, os
import copy
from architecture import ModelRNNv2
import ray

def parse_args():
    parser = argparse.ArgumentParser('Adjoint Paper Training')
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
    return parser.parse_args()

def ping_dir(directory, clear = False):
    # Check if directory exists and make if not.
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

def train(args, plot = False, verbose = False):
    if isinstance(args, dict):
        args = argparse.Namespace(**args) # Dict to arguments.
    root_dir = args.prefix + '/'
    ping_dir(root_dir)
    ping_dir(root_dir + 'checkpoints', clear = True)
    if verbose:
        print(f'Root Directory {root_dir}')

    device = 'cpu'
    if args.task == 'memory_pro':
        import memory_pro as task
    elif args.task == 'category_pro':
        import category_pro as task
    elif args.task == 'mix_mem_cat':
        import memory_category_mix as task
    elif args.task == 'single_fp':
        import single_fp_task as task
    elif args.task == 'flip_flop':
        import flip_flop as task
    elif args.task == 'd1fp2':
        import d1fp2 as task
    elif args.task == 'd1fp2_correlated_x':
        import d1fp2_correlated_x as task

    # Load the data and model.
    if args.random_seed != -1:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    if args.wandb:
        # Import weights-and-biases and its integration with ray-tune hyperparameter sweeper.
        from ray.air.integrations.wandb import setup_wandb
        wandb = setup_wandb(vars(args), project=args.wandb)

    inps, targets = task.generate()
#    inps, targets = inps.to(args.device), targets.to(args.device) # Put everything on GPU. For small datasets, this is fine. For big, we need to do on batch basis. 
    dset = torch.utils.data.TensorDataset(inps, targets)
    dloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
    model = torch.jit.script(ModelRNNv2(args.n_hidden, inps.shape[-1], targets.shape[-1]))
    model.noise_std = args.noise_std

    # Scale weights by a percent to play with different initializations.
    with torch.no_grad():
        model.W.weight.data *= args.init_scale_percent
        model.W_in.weight.data *= args.init_scale_percent
        model.W_out.weight.data *= args.init_scale_percent

    if args.checkpoint_file != '':
        ld = torch.load(args.checkpoint_file)['model']
        model.load_state_dict(ld)

    # Training with Adam.
    losses, accs = [], []
    if args.method == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.method == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    loss_fn = nn.MSELoss()
    max_acc = 0.
    model = model.to(args.device)
    glog = []
    
    target_shape = next(iter(dloader))[1].shape
    model_prev = model
    my_range = range(args.niters) if not verbose else tqdm(range(args.niters))
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

        ray.train.report({"accuracy": accs[-1], "loss": losses[-1]})
        if args.wandb != '':
            wandb.log({"accuracy": accs[-1], "loss": losses[-1]})

        if len(accs) > 50:
            smooth_acc = sum(accs[-50:]) / 50.
            if smooth_acc > max_acc:
                max_acc = smooth_acc
                torch.save(checkpoint, root_dir + 'best.pt')

            if smooth_acc >= args.stopping_acc:
                if verbose:
                    print("Hit stopping accuracy. Terminating early.")
                break

        if itr == args.niters - 1 or itr % args.save_freq == 0:
            torch.save(checkpoint, root_dir + f'checkpoints/checkpoint_{itr}.pt')

#        if plot and (itr % 60 == 0 or itr == args.niters -1):
#            params = list(model.parameters())
#            glog.append(torch.sqrt(torch.sum(params[2].grad**2)).item())

        model_prev = copy.deepcopy(model)

    if plot:
        plt.figure()
        plt.subplot(1,2,1)
        losses_avg = np.convolve(losses, np.ones(20)/20., mode='valid')
        plt.plot(losses)
        plt.plot(losses_avg)
        plt.title('Training Loss') 
        plt.ylabel('Train Loss (mse)')
        plt.xlabel('Iteration, s')

        plt.subplot(1,2,2)
        accs_avg = np.convolve(accs, np.ones(20)/20., mode='valid')
        plt.plot(accs)
        plt.plot(accs_avg)
        plt.title('Training Acccuracy') 
        plt.ylabel('Train Accuracy [0-1]')
        plt.xlabel('Iteration, s')

        plt.figure()
        plt.plot(glog)
        plt.title('Grad Norm')
        plt.xlabel('Iteration, s')
        plt.ylabel('$\|\theta(s)\|$')

        plt.show()

    return sum(losses[-50:]) / 50.

if __name__ == "__main__":
    args = parse_args()
    train(args, plot = True)
