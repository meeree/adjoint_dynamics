import argparse
from adjoint_calculation import adjoint_calculate, adjoint_calculate_RNN
from torchdiffeq import odeint, odeint_adjoint
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import time
import pickle
import glob
import re, os
import memory_pro, category_pro

parser = argparse.ArgumentParser('Single Task Adjoints RNN Example')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--batch_time', type=int, default=400)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--grad_clip', type=float, default=1.)
parser.add_argument('--n_hidden', type=int, default=100)
parser.add_argument('--niters', type=int, default=50000)
parser.add_argument('--retrain', type=bool, default=False)
parser.add_argument('--repca', type=bool, default=False)
parser.add_argument('--analyze_best', type=bool, default=False)
args = parser.parse_args()
device = 'cpu'

class Model(nn.Module):
    ''' The model in this case is just an RNN (hand built). '''
    ''' I need this hand built RNN for now since it fits into the framework of my adjoint code. '''
    def __init__(self, Win, Wout, bout, W):
        super().__init__()
        self.Win = nn.Parameter(torch.from_numpy(Win).float())
        self.Wout = nn.Parameter(torch.from_numpy(Wout).float())
        self.bout = nn.Parameter(torch.from_numpy(bout).float())
        self.W = nn.Parameter(torch.from_numpy(W).float())
        self.X = None
        
    def set_x(self, X): 
        ''' Set the input data, X, to sample from over time. '''
        ''' Times should be discrete integer indices. '''
        self.X = X.float()

    def eval_output(self, s):
        out = torch.matmul(s, self.Wout) + self.bout
        return out

    def forward(self, t, s):
        ''' Times should be discrete integer indices. '''
        ''' t is timestep, s is state at this time. '''
        if len(s.shape) == 1:
            s = s[None, :] # Not batched. Just add a 1.
        x = self.X[:, int(torch.round(t))-1, :]
        s_new = torch.tanh(torch.mm(s.clone(), self.W) + torch.mm(x, self.Win))
        return s_new

    def evaluate(self, X, grad_record = False):
        # X is shape [batch size, timesteps, input dim].
        self.set_x(X)
        N = self.W.shape[0]
        B = X.shape[0]
        T = X.shape[1]
        t = torch.arange(T).float()
        s = torch.zeros((B, T, N))
        for tv in t[1:]:
            i = int(torch.round(tv))
            s[:, i] = self(tv, s[:, i-1]) # Forward step.
        s.requires_grad_(grad_record)
        out = self.eval_output(s)
        return s, out

    def evaluate_ode_mode(self, X, method, tol):
        self.set_x(X)
        N = self.W.shape[0]
        B = X.shape[0]
        T = X.shape[1]

        s0 = torch.zeros((B, N)).to(self.Win.device)
        t = torch.arange(T).float().to(self.Win.device)

        s = odeint(self, s0, t, method = method, rtol = tol, atol = tol) # Simulate ODE with forward function.
        s = s.transpose(0,1)
        out = self.eval_output(s)
        return s, out

class ModelRNN(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_out, self.n_in = 3, 3 
        self.n_hidden = n_hidden

        self.Wout = nn.Linear(n_hidden, self.n_out)
        self.rnn = nn.RNN(self.n_in, self.n_hidden, batch_first = True, bias = False)

    def evaluate(self, X):
        # Pass through RNN over time.
        hidden, s_end = self.rnn(X.float())

        # Pass through output layer.
        return self.Wout(hidden), hidden

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

def train(root_dir, plot = True):
    # Load the data and model.

    # Uncomment for deterministic training.
#    torch.manual_seed(2)
#    np.random.seed(2)

    inps, targets = category_pro.generate()
    dset = torch.utils.data.TensorDataset(inps, targets)
    dloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
    model = ModelRNN(args.n_hidden)

    # Training with Adam.
    losses, accs = [], []
    optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    loss_fn = nn.MSELoss()
    min_loss = 1e10
    model = model.cuda()
    
    target_shape = next(iter(dloader))[1].shape
    for itr in tqdm(range(args.niters)):
        # Training Step.
        optim.zero_grad()
        batch, target = next(iter(dloader))
        batch, target = batch.cuda(), target.cuda()
        out, hidden = model.evaluate(batch)
        loss = loss_fn(out[:, :, :], target[:, :, :])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        # Checkpoint Saving.
        losses.append(loss.item())
        accs.append(category_pro.accuracy(out, target).item())
        checkpoint = {
            'losses': losses,
            'accuracies': accs,
            'iterations': len(losses),
            'optim': optim.state_dict(),
            'model': model.state_dict()
        }
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(checkpoint, root_dir + 'best.pt')

        if itr == 0 or (itr+1) % (args.niters // 25) == 0:
            torch.save(checkpoint, root_dir + f'checkpoints/checkpoint_{itr}.pt')

    if plot:
        plt.figure()
        plt.subplot(1,2,1)
        losses_avg = np.convolve(losses, np.ones(20)/20., mode='same')
        plt.plot(losses)
        plt.plot(losses_avg)
        plt.title('Training Loss') 
        plt.ylabel('Train Loss (mse)')
        plt.xlabel('Iteration')

        plt.subplot(1,2,2)
        accs_avg = np.convolve(accs, np.ones(20)/20., mode='same')
        plt.plot(accs)
        plt.plot(accs_avg)
        plt.title('Training Acccuracy') 
        plt.ylabel('Train Accuracy [0-1]')
        plt.xlabel('Iteration')

def plot_example_single_trial(checkpoint):
    X, Y = category_pro.generate()
    t = torch.arange(X.shape[1]).float()

    model = torch.load(checkpoint, map_location=torch.device('cpu'))['model']
    tWout = model['Wout.weight']
    tbout = model['Wout.bias']
    tWin = model['rnn.weight_ih_l0']
    tW = model['rnn.weight_hh_l0']
    with torch.no_grad():
        # Transfer pytorch trained model. 
        Win = tWin.numpy().T 
        Wout = tWout.numpy().T
        W = tW.numpy().T
        bout = tbout.numpy()

        ode = Model(Win, Wout, bout, W)
        ode.set_x(X)
        with torch.enable_grad():
            true_s, out = ode.evaluate(X, grad_record=True)
            loss_fn = nn.MSELoss()
            loss = loss_fn(out, Y)
            grad_s = torch.autograd.grad(loss, true_s)[0]

        # Measure adjoints and state over time and record everything.
        data = adjoint_calculate_RNN(t, true_s, ode, grad_s)

    # Plot an example.
    b = 0
    plt.figure()
    for idx, name in enumerate(['Fixation', 'Stimulus Cos', 'Stimulus Sin']):
        plt.subplot(3, 1, 1 + idx)
        plt.plot(X[b, :, idx].detach().cpu(), alpha = .5, linewidth = 3, label = 'Input')
        plt.plot(out[b, :, idx].detach().cpu(), label = 'Output')
        plt.plot(Y[b, :, idx].detach().cpu(), label = 'Target')
        plt.title(name, fontsize = 15)
        plt.ylim(-1.2, 1.2)

    ax = plt.gca()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 15)
    plt.tight_layout()

def compute_pca(S, n_components = 10):
    n_components = min(n_components, S.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(S)
    s = pca.transform(S)
    return pca, s

def analyze_over_training(checkpoints, iteration, save_raw = False, debug = False):
    ''' Runs through training checkpoints and analyzes, mainly through PCA. '''
    ''' save_raw toggles saving of the full network output. This can be large. '''
    X, Y = category_pro.generate()
    t = torch.arange(X.shape[1]).float()
    pbar = tqdm(list(zip(checkpoints, iteration)))
    records_all = []
    for ch, itr in pbar:
        pbar.set_description(ch)
        record = {'iteration': itr} # All the data is stored in a dict per iteration.

        model = torch.load(ch, map_location=torch.device('cpu'))['model']
        tWout = model['Wout.weight']
        tbout = model['Wout.bias']
        tWin = model['rnn.weight_ih_l0']
        tW = model['rnn.weight_hh_l0']
        with torch.no_grad():
            # Transfer pytorch trained model. 
            Win = tWin.numpy().T 
            Wout = tWout.numpy().T
            W = tW.numpy().T
            bout = tbout.numpy()

            ode = Model(Win, Wout, bout, W)
            ode.set_x(X)
            with torch.enable_grad():
                true_s, out = ode.evaluate(X, grad_record=True)
                loss_fn = nn.MSELoss()
                loss = loss_fn(out, Y)
                grad_s = torch.autograd.grad(loss, true_s, retain_graph=True)[0]

            # Measure adjoints and state over time and record everything.
            data = adjoint_calculate_RNN(t, true_s, ode, grad_s)

            # Outer product approach.
            z = np.array([v[1].detach().numpy() for v in data])[::-1]
            a = np.array([v[2].detach().numpy() for v in data])[::-1]
            adjoint_sigma_prime = (1 - z[1:]**2) * a[1:] # ASSUMES TANH ACTIVATION (I.E sig' = (1-sig**)). 

            if debug: # Validate the computed gradient is correct for the W parameter. Could validate others, but any should be good.
                loss.backward() # Get the gradients with autograd.
                w_grad_true = ode.W.grad.numpy()
                w_grad_mine = data[-1][6].detach().numpy()

                g_W = np.zeros((z.shape[1], z.shape[2], z.shape[2])) # [B, N, N].
                g_W_record = np.zeros((len(t), *g_W.shape))
                for i in range(z.shape[0]-2, -1, -1):
                    # Batched outer product.
                    dot_g = -np.einsum('ij,ik->ijk', z[i], adjoint_sigma_prime[i]) 
                    g_W = g_W - dot_g # Timestep of length 1.
                    g_W_record[i] = np.copy(g_W)
                w_grad_outer = np.sum(g_W, 0)

#                plt.figure(figsize=(9,3))
#                vmin, vmax = min(np.min(w_grad_true), np.min(w_grad_mine)), max(np.max(w_grad_true), np.max(w_grad_mine))
#                plt.subplot(1,3,1)
#                plt.imshow(w_grad_true, vmin = vmin, vmax = vmax)
#                plt.title('Pytorch based $\\nabla_W Loss$')
#                plt.subplot(1,3,2)
#                plt.imshow(w_grad_mine, vmin = vmin, vmax = vmax)
#                plt.title('Adjoint based $\\nabla_W Loss = g_W(0)$')
#                plt.subplot(1,3,3)
#                plt.imshow(w_grad_true - w_grad_mine, vmin = vmin, vmax = vmax)
#                plt.title('Error, True - Mine')
#                plt.suptitle('Validating Runninng Gradient. Same Color Scale for all.')
#                plt.savefig('Validated_Grad.pdf')
#                plt.show()
                plt.figure(figsize=(12,3))
                vmin, vmax = min(np.min(w_grad_true), np.min(w_grad_mine)), max(np.max(w_grad_true), np.max(w_grad_mine))
                plt.subplot(1,4,1)
                plt.imshow(w_grad_true, vmin = vmin, vmax = vmax)
                plt.colorbar()
                plt.title('Pytorch Based $\\nabla_W Loss$')
                plt.subplot(1,4,2)
                plt.imshow(w_grad_mine)
                plt.colorbar()
                plt.title('Adjoint Based $\\nabla_W Loss = g_W(0)$')
                vmin, vmax = min(np.min(w_grad_true), np.min(w_grad_mine)), max(np.max(w_grad_true), np.max(w_grad_mine))
                plt.subplot(1,4,3)
                plt.imshow(w_grad_outer, vmin = vmin, vmax = vmax)
                plt.colorbar()
                plt.title('Adjoint Outer Product Based')
                plt.subplot(1,4,4)
                plt.imshow(w_grad_true - w_grad_mine, vmin = vmin, vmax = vmax)
                plt.title('Error, True - Mine')
                plt.suptitle('Validating Runninng Gradient. Same Color Scale for all.')
                plt.savefig('Validated_Grad.pdf')
                plt.show()

            # TODO: MAKE THIS CLEANER. ESSENTIALLY, to get grad for parameters, we need to pass in each individual data point and concat results ;(.
            for param_idx in [3, 6]:
                for idx in range(len(data)):
                    data[idx][param_idx] = torch.empty((0, *data[idx][param_idx].shape))

                count = 50 if param_idx == 3 else 25 
                for b in range(count): # Just do a couple data samples since things can get slow and big quick...
                    X_sub, Y_sub = X[b:b+1], Y[b:b+1]
                    ode.set_x(X_sub)
                    with torch.enable_grad():
                        true_s_sub, out_sub = ode.evaluate(X_sub, grad_record=True)
                        loss_sub = loss_fn(out_sub, Y_sub)
                        grad_s_sub = torch.autograd.grad(loss_sub, true_s_sub)[0]

                    # Measure adjoints and set_zlimstate over time and record everything.
                    data_sub = adjoint_calculate_RNN(t, true_s_sub, ode, grad_s_sub)
                    for idx in range(len(data)):
                        data[idx][param_idx] = torch.concatenate([data[idx][param_idx], data_sub[idx][param_idx].unsqueeze(0)])

            # Calculate the scale of the two terms contributing to the gradient.
            g1 = 2 / float(out.nelement()) * torch.matmul(out, ode.Wout.transpose(0,1))
            g2 = 2 / float(out.nelement()) * torch.matmul(-Y, ode.Wout.transpose(0,1))
            record['g1'] = torch.mean(g1.norm(dim=1)).item() # Mean of norm over time.
            record['g2'] = torch.mean(g2.norm(dim=1)).item()
            record['loss'] = loss.item()

            vel = z[1:] - z[:-1] # f(x) - x.
            plt.plot(t[:-1], (vel * a[:-1]).sum(-1), label = f'h_{itr}', color = [itr / iteration[-1], 0., 0.])

            inds = [1, 2, 3, 6, 1]
            names = ['state', 'adjoint', 'win_grad', 'w_grad', 'adjoint_sigma_prime']
            for ind, name in zip(inds, names):
                states = np.array([a[ind].detach().numpy() for a in data])[::-1] # reverse time.
                if name == 'adjoint_sigma_prime':
                    states = adjoint_sigma_prime

                if save_raw:
                    record[f'{name}_raw'] = states

                # NOTE: Different PCA per iteration! See function below to do same PCA for all.
                pca, s = compute_pca(states.reshape(states.shape[0] * states.shape[1], -1))
                s = s.reshape((states.shape[0], states.shape[1], -1)) # shape (tsteps, batch, projection components).
                record[f'{name}_pca'] = s
                record[f'{name}_pca_struct'] = pca

            records_all.append(record)

#    plt.title('Hamiltonian')
#    plt.xlabel('Time')
#    plt.legend()
#    plt.show()

    return records_all

def analyze_over_training_SAME_PCA(checkpoints, iteration):
    ''' Uses a single consistent PCA over all iterations per data column stored. '''
    record = analyze_over_training(checkpoints, iteration, save_raw = True)
    names = ['state', 'adjoint', 'win_grad', 'w_grad']
    for name in names:
        all_iters = [entry[f'{name}_raw'] for entry in record]
        entry_shape = all_iters[0].shape
        all_iters = [entry.reshape(entry.shape[0] * entry.shape[1], -1) for entry in all_iters]
        all_iters = np.concatenate(all_iters)
        pca, s = compute_pca(all_iters)
        s = s.reshape((len(record), entry_shape[0], entry_shape[1], -1))
        # Update the PCA entries in the record.
        for entry, s_itr in zip(record, s):
            entry[f'{name}_pca'] = s_itr
            entry[f'{name}_pca_struct'] = pca
    return record 

def make_grid_fig(records):
    M = int(len(records) ** .5)
    N = int(np.ceil(len(records) / float(M)))
    return plt.figure(figsize=(N * 4, M * 3)), M, N

def plot_raw_over_training(records, quantity_name, display_name, cfg, trial_idx = 0):
    fig, M, N = make_grid_fig(records)
    for plot_idx, record in enumerate(records):
        ax = fig.add_subplot(M, N, plot_idx + 1)
        s = record[f'{quantity_name}_raw']
        for color, off1, off2 in zip(['red', 'green', 'blue'], [0, cfg['T_stim'], cfg['T_stim']+cfg['T_memory']], [cfg['T_stim'], cfg['T_stim']+cfg['T_memory'], s.shape[0]]):
            end = min(off2+1, s.shape[0])
            sub = s[off1:end, trial_idx] # Only plot one trial.
            sub = sub.reshape((sub.shape[0], -1))
            ax.plot(np.arange(off1, end), sub, c = color, alpha = 0.5)

        ax.set_xlabel('Time Step')
        plt.title(f'Iteration {record["iteration"]}')
    plt.suptitle(f'{display_name} Over Training. Single Trial.')
    plt.tight_layout()

def plot_raw_over_training_multitrial(records, quantity_name, display_name, cfg, trial_inds = list(range(10))):
    fig, M, N = make_grid_fig(records)
    for plot_idx, record in enumerate(records):
        ax = fig.add_subplot(M, N, plot_idx + 1)
        s = record[f'{quantity_name}_raw']
        for trial_idx in trial_inds:
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][trial_idx % 10] # Default matplotlib colors.
            for style, off1, off2 in zip(['-', 'dashed', 'dotted'], [0, cfg['T_stim'], cfg['T_stim']+cfg['T_memory']], [cfg['T_stim'], cfg['T_stim']+cfg['T_memory'], s.shape[0]]):
                end = min(off2+1, s.shape[0])
                sub = s[off1:end, trial_idx] # Only plot one trial.
                ax.plot(np.arange(off1, end), sub, alpha = 0.5, linestyle = style, c = color)

        ax.set_xlabel('Time Step')
        plt.title(f'Iteration {record["iteration"]}')
    plt.suptitle(f'{display_name} Over Training. {len(trial_inds)} Trials.')
    plt.tight_layout()

def plot_phase_space_over_training(records, cfg, trial_idx = 0, sample_idx = 0):
    ''' Plot phase space comparing adjoint and forward state over training iterations. '''
    fig, M, N = make_grid_fig(records)
    for plot_idx, record in enumerate(records):
        ax = fig.add_subplot(M, N, plot_idx + 1)
        s, adj = record[f'state_raw'], record[f'adjoint_raw']
        for color, off1, off2 in zip(['red', 'green', 'blue'], [0, cfg['T_stim'], cfg['T_stim']+cfg['T_memory']], [cfg['T_stim'], cfg['T_stim']+cfg['T_memory'], s.shape[0]]):
            sub_s, sub_adj = s[off1:off2+1, trial_idx], adj[off1:off2+1, trial_idx] # Only plot one trial.
            ax.plot(sub_s[:, sample_idx], sub_adj[:, sample_idx], c = color, alpha = 0.5) 
        ax.set_xlabel(f'State[{sample_idx}]')
        ax.set_ylabel(f'Adjoint[{sample_idx}]')
        plt.title(f'Iteration {record["iteration"]}')

    plt.suptitle(f'Phase Space Over Training. Single Trial.')
    plt.tight_layout()

def plot_pca_over_training(records, quantity_name, display_name, cfg):
    fig, M, N = make_grid_fig(records)
    for plot_idx, record in enumerate(records):
        ax = fig.add_subplot(M, N, plot_idx + 1, projection='3d')
        ax.view_init(azim=0, elev=90)
        s = record[f'{quantity_name}_pca']
        for color, off1, off2 in zip(['red', 'green', 'blue'], [0, cfg['T_stim'], cfg['T_stim']+cfg['T_memory']], [cfg['T_stim'], cfg['T_stim']+cfg['T_memory'], s.shape[0]]):
            sub = s[off1:off2+1]
            for b in range(0, sub.shape[1], 10):
                ax.plot3D(sub[:, b, 0], sub[:, b, 1], sub[:, b, 2], c = color, alpha = 0.5)

        # Consistent scaling of all axes. If a PC has a very small scale, we don't really care about it! 
#        min_dim, max_dim = np.min(s), np.max(s)
#        ax.set_xlim(min_dim, max_dim)
#        ax.set_ylim(min_dim, max_dim)
#        ax.set_zlim(min_dim, max_dim)

        accuracy = sum(record[f'{quantity_name}_pca_struct'].explained_variance_ratio_[:3]) * 100
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title(f'Iteration {record["iteration"]}; 3D acc = {accuracy:.1f}%')
    plt.suptitle(f'{display_name} PCA Projection Over Time')
    plt.tight_layout()

def plot_pca_fixed_points_over_training(records, quantity_name, display_name, cfg):
    fig, M, N = make_grid_fig(records)
    for plot_idx, record in enumerate(records):
        ax = fig.add_subplot(M, N, plot_idx + 1, projection='3d')
        s = record[f'{quantity_name}_pca']
        for color, off in zip(['black', 'red', 'green', 'blue'], [0, cfg['T_stim']-1, cfg['T_stim']+cfg['T_memory']-1, s.shape[0]-1]):
            ax.scatter(s[off, :, 0], s[off, :, 1], s[off, :, 2], c = color, alpha = 0.7, marker='^', s = .4)

        # Consistent scaling of all axes. If a PC has a very small scale, we don't really care about it! 
#        min_dim, max_dim = np.min(s), np.max(s)
#        ax.set_xlim(min_dim, max_dim)
#        ax.set_ylim(min_dim, max_dim)
#        ax.set_zlim(min_dim, max_dim)

        accuracy = sum(record[f'{quantity_name}_pca_struct'].explained_variance_ratio_[:3]) * 100
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title(f'Iteration {record["iteration"]}; 3D acc = {accuracy:.1f}%')
    plt.suptitle(f'{display_name} PCA Projection Final Points of Each Phase')
    plt.tight_layout()

if __name__ == "__main__":
    root = './CategoryPro/'
    ping_dir(root)
    if args.retrain:
        ping_dir(root + 'checkpoints', clear = True)
        train(root, plot = True)
        plt.show()

    files = glob.glob(root + 'checkpoints/*.pt')
    files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    iteration = [int(re.findall(r'\d+', file)[0]) for file in files]
    files = [root + 'checkpoints/' + os.path.basename(p) for p in files]

    if args.analyze_best:
        analyze_model_old(files[-1])

    if args.repca:
        training_run = analyze_over_training_SAME_PCA(files, iteration)
        with open(root + 'training_run_SAME_PCA_.pt', "wb") as fp:
            pickle.dump(training_run, fp)

    with open(root + 'training_run_SAME_PCA_.pt', "rb") as fp:
        training_run = pickle.load(fp)

    save_dir = root + 'May_7/' # CUSTOMIZE.
    ping_dir(save_dir)

    # Plot loss on task.
    plt.figure()
    plt.plot(iteration, [record['loss'] for record in training_run])
    plt.title('Train Loss')
    plt.savefig(save_dir + 'Loss.png')

    # Terms contributing to adjoint.
    plt.figure()
    plt.plot(iteration, [record['g1'] for record in training_run], color = "#ffab40", linewidth=4)
    plt.plot(iteration, [record['g2'] for record in training_run], color = "#c27ba0", linewidth=4)
    plt.xlabel('Training Iteration')
    plt.legend(['Term 1 Norm', 'Term 2 Norm'])
    plt.savefig(save_dir + 'Terms.png')

    # Just to illustrate the task on a single trial.
    plot_example_single_trial(files[-1])
    plt.savefig(save_dir + 'Figure_Single_Trial.pdf')

#    training_run = training_run[:18:2] # Subset for plots.
    training_run = training_run[:9] # Subset for plots.

    # Analyze PCA over training.
    ping_dir(save_dir + 'pca_plots/')
    plot_pca_over_training(training_run, 'state', 'Hidden States', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_1.png')
    plot_pca_over_training(training_run, 'adjoint', 'Adjoint', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_2.png')
    plt.show()
    plot_pca_over_training(training_run, 'win_grad', 'Running $W_{in}$ Gradient', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_3.png')
    plot_pca_over_training(training_run, 'w_grad', 'Running $W$ Gradient', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_4.png')

    # Analyze PCA fixed points over training.
    plot_pca_fixed_points_over_training(training_run, 'state', 'Hidden States', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_10.png')
    plot_pca_fixed_points_over_training(training_run, 'adjoint', 'Adjoint', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_11.png')
    plot_pca_fixed_points_over_training(training_run, 'win_grad', 'Running $W_{in}$ Gradient', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_12.png')
    plot_pca_fixed_points_over_training(training_run, 'w_grad', 'Running $W$ Gradient', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_13.png')

    # Analyze raw hidden state and adjoint (no PCA) over training.
    ping_dir(save_dir + 'raw_plots/')
    plot_raw_over_training(training_run, 'state', 'Hidden States', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'raw_plots/Figure_13.png')
    plot_raw_over_training(training_run, 'adjoint', 'Adjoint', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'raw_plots/Figure_14.png')
    plot_raw_over_training(training_run, 'win_grad', 'Running $W_{in}$ Gradient', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'raw_plots/Figure_18.png')

    plot_raw_over_training_multitrial(training_run, 'state', 'Hidden States', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'raw_plots/Figure_15.png')
    plot_raw_over_training_multitrial(training_run, 'adjoint', 'Adjoint', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'raw_plots/Figure_16.png')

    plot_phase_space_over_training(training_run, cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'raw_plots/Figure_17.png')

    plot_pca_over_training(training_run, 'adjoint_sigma_prime', 'Adjoint Sigma Prime', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_20.png')
    plot_pca_fixed_points_over_training(training_run, 'adjoint_sigma_prime', 'Adjoint Sigma Prime', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_21.png')
    plot_raw_over_training(training_run, 'adjoint_sigma_prime', 'Adjoint Sigma Prime', cfg = category_pro.DEFAULT_CFG)
    plt.savefig(save_dir + 'pca_plots/Figure_22.png')

    plt.show()
