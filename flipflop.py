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

parser = argparse.ArgumentParser('3 Flip Flop Adjoint Demo')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--batch_time', type=int, default=400)
parser.add_argument('--niters', type=int, default=450)
parser.add_argument('--continuous', type=bool, default=False)
parser.add_argument('--retrain', type=bool, default=False)
parser.add_argument('--repca', type=bool, default=False)
parser.add_argument('--analyze_best', type=bool, default=False)
args = parser.parse_args()
device = 'cpu'

class Model(nn.Module):
    ''' The model in this case is just an RNN (hand built). '''
    def __init__(self, Win, Wout, bout, W):
        super().__init__()
        self.Win = nn.Parameter(torch.from_numpy(Win).float())
        self.Wout = nn.Parameter(torch.from_numpy(Wout).float())
        self.bout = nn.Parameter(torch.from_numpy(bout).float())
        self.W = nn.Parameter(torch.from_numpy(W).float())
        self.X = None
        self.t = None
        self.s0 = None
        
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
#        s_new = torch.where(s_new < 1e-3, 0., s_new) # For stability.
        return s_new.squeeze()

    def evaluate(self, X, grad_record = False):
        # X is shape [batch size, timesteps, 3].
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
        self.Wout.requires_grad = False # Don't train this layer.
        self.rnn = nn.RNN(self.n_in, self.n_hidden, batch_first = True, bias = False)

    def evaluate(self, X):
        self.X = X
#        s_init = torch.rand((1, self.n_hidden)) # Random initial conditions

        # Pass through RNN over time.
        hidden, s_end = self.rnn(X.float())

        # Pass through output layer.
        return self.Wout(hidden), hidden

def import_data(preload_model_hidden = True, debug = False):
    # import data
    X = np.loadtxt('Data/x_test_array.gz') # input
    Y = np.loadtxt('Data/y_test_array.gz') # target output
    Wout = np.loadtxt('Data/wout.gz') # output weights
    Win = np.loadtxt('Data/win.gz') # input weigths
    if preload_model_hidden:
        W = np.loadtxt('Data/w.gz') # trained connectivity matrix
    else:
        W = np.random.normal(size=[100,100]) # random connectivity matrix

    if args.continuous:
        # Adapt the task inputs and outputs so that instead of [-1, 0, 1] as only states
        # the pulses are anywhere in the range [-1, 1] and we should match them.
        rng = np.random.default_rng(seed=0)
        new_X = rng.random(X.shape)
        new_X = np.where(new_X < .1, 0., new_X) # Make all values > 0.1.
        new_X = new_X * X # Only input where original X is nonzero.
        new_Y = np.zeros_like(new_X)
        prev = np.zeros(3)
        for t in range(new_Y.shape[0]):
            for d in range(3):
                if np.abs(new_X[t, d]) > .05:
                    prev[d] = new_X[t, d]
            new_Y[t] = prev
        X, Y = new_X, new_Y

    if debug:
        plt.plot(X[:1000])
        plt.plot(Y[:1000], linestyle = 'dotted')
        plt.show()

    return Win, Wout, W, X, Y

def visualize(y, ax, nplot=1000, style = '-'):
    ax.plot(y[:nplot, 0], 'b', linestyle = style)
    ax.plot(y[:nplot, 1], 'r', linestyle = style)
    ax.plot(y[:nplot, 2], 'g', linestyle = style)

class FlipFlopDataset():
    ''' Splits up single long input into multiple chunks for a dataset.'''
    def __init__(self, X, Y, chunk_size):        
        # Make X and Y divisible by chunk_size by truncating end.
        self.X = torch.from_numpy(X)[:X.shape[0] - X.shape[0] % chunk_size].float()
        self.Y = torch.from_numpy(Y)[:X.shape[0] - X.shape[0] % chunk_size].float()
        self.chunk_size = chunk_size

    def __len__(self):
        return self.X.shape[0] // self.chunk_size

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        return self.X[start:end], self.Y[start:end]


def train_flipflop(root_dir, debug = False):
    # Load the data and model.
    torch.manual_seed(2)
    np.random.seed(2)
    _, _, W, X, Y = import_data(preload_model_hidden = False)
    model = ModelRNN(W.shape[0])
    dset = FlipFlopDataset(X, Y, args.batch_time)
    dloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
    print(f'Data split into {len(dset)} samples, each of length {dset.chunk_size}')

#    model = model.state_dict()
#    tWout = model['Wout.weight']
#    tbout = model['Wout.bias']
#    tWin = model['rnn.weight_ih_l0']
#    tW = model['rnn.weight_hh_l0']
#    Win = tWin.numpy().T 
#    Wout = tWout.numpy().T
#    W = tW.numpy().T
#    bout = tbout.numpy()
#    model = Model(Win, Wout, bout, W)

    # Plot an example input.
    if debug:
        x, y = dset[0] 
        ax1 = plt.subplot(211)
        visualize(x, ax1, style='--')
        ax2 = plt.subplot(212)
        visualize(y, ax2)
        plt.show()

    # Training with Adam.
    losses = []
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.MSELoss()
    min_loss = 1e10
    model = model.cuda()
    for itr in tqdm(range(args.niters)):
        optim.zero_grad()
        batch, target = next(iter(dloader))
        batch, target = batch.cuda(), target.cuda()
        out, hidden = model.evaluate(batch)
        loss = loss_fn(out, target)
        loss.backward()
        optim.step()

        losses.append(loss.item())
        checkpoint = {
            'losses': losses,
            'iterations': len(losses),
            'optim': optim.state_dict(),
            'model': model.state_dict()
        }
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(checkpoint, root_dir + 'flip_flop_best.pt')

        if itr == 0 or (itr+1) % (args.niters // 25) == 0:
            torch.save(checkpoint, root_dir + f'checkpoints/flip_flop_checkpoint_{itr}.pt')

        if (itr+1) % args.niters == 0:
            plt.figure()
            plt.plot(losses)
            plt.title('Training on 3-Flip-Flop Task')
            plt.ylabel('Loss (mse)')
            plt.xlabel('Iteration')
            plt.show()

def compute_pca(S, n_components = 10):
    n_components = min(n_components, S.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(S)
    s = pca.transform(S)
    return pca, s

def compute_and_plot_pca(S, name, ax, plot_explain_variances = False):
    # activation projections
    pca, s = compute_pca(S)
    plot_3d = (ax.name == '3d')
    T = s.shape[0]
    inc = T // 10
    for i in range(0, T, inc):
        c = [i / float(T - 1), 0.0, 0.0]
        if plot_3d:
            ax.plot(s[i:i+inc,0], s[i:i+inc,1], s[i:i+inc,2], '-', linewidth=1, color=c)
        else:
            ax.plot(s[i:i+inc,0], s[i:i+inc,1], '-', linewidth=1, color=c)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if plot_3d:
        ax.set_zlabel('PC3')
        ax.view_init(0,360) 
    plt.title(f'{name} PCA Projection Over Time (black->red)')
    plt.tight_layout()

    # explained variance plot
    if plot_explain_variances:
        plt.figure()
        plt.plot(pca.explained_variance_ratio_.cumsum(),'k.')
        plt.plot([0,pca.n_components],[1,1],'k--')
        plt.xlabel('# of included components')
        plt.ylabel('variance explained ratio')
        plt.title(f'{name}, cumulative variance explained')

    return pca

def analyze_model_old(checkpoint_file):
    # Load the data and model.
    _, _, W, X, Y = import_data(preload_model_hidden = False)

    model = torch.load(checkpoint_file)['model']
#    model = ModelRNN(W.shape[0])
#    model = model.state_dict()
    tWout = model['Wout.weight'].cpu()
    tbout = model['Wout.bias'].cpu()
    tWin = model['rnn.weight_ih_l0'].cpu()
    tW = model['rnn.weight_hh_l0'].cpu()

    with torch.no_grad():
        # Transfer pytorch trained model. 
        Win = tWin.numpy().T 
        Wout = tWout.numpy().T
        W = tW.numpy().T
        bout = tbout.numpy()

        X = torch.from_numpy(X[None, :10000, :])
        target = torch.from_numpy(Y[None, :10000, :]).float()
        ode = Model(Win, Wout, bout, W)
        ode.set_x(X)
        ode = ode.cpu()

        N = W.shape[0]
#        true_s, out = ode.evaluate_ode_mode(X, 'rk4', 1e-5)

        with torch.enable_grad():
            true_s, out = ode.evaluate(X, grad_record=True)
            loss_fn = nn.MSELoss()
            loss = loss_fn(out, target)
            grad_s = torch.autograd.grad(loss, true_s)[0]
            guess = 2 / float(out.nelement()) * (out - target)
            guess = torch.matmul(guess, ode.Wout.transpose(0,1))
            print(torch.max(torch.abs(guess - grad_s)), guess.max(), grad_s.max())

        true_s, out, grad_s = true_s[0], out[0], grad_s[0]

        nplot = 3000
#        fig = plt.figure()
#        ax1 = fig.add_subplot(311)
#        visualize(X, ax1)
#        plt.title('inputs')
#
#        ax2 = fig.add_subplot(312)
#        visualize(out, ax2)
#        plt.title('outputs')
#
#        ax3 = fig.add_subplot(313)
#        visualize(true_s, ax3)
#        plt.title('activations')
#        plt.xlabel('time')

        t = torch.arange(X.shape[1]).float()
        record = adjoint_calculate_RNN(t, true_s, ode, grad_s)

        # Record is a list of augmented states. 
        # Each augmented state is a list [<UNUSED>, state, adjoint, running grad].
        states = np.array([a[1].detach().numpy() for a in record])
        adjoints = np.array([a[2].detach().numpy() for a in record])

        # Look at grads for Win and W. Wout and bout do not have running gradients.
        for idx, (name, _) in enumerate(ode.named_parameters()):
            print(name, 3+idx, np.array([a[3+idx].detach().numpy() for a in record]).shape)

        grads_Win = np.array([a[3].detach().numpy() for a in record])
        grads_W = np.array([a[6].detach().numpy() for a in record])
        grads_Win = grads_Win.reshape(len(t), -1) # Flatten last two dimensions. [T, 3, H] -> [T, 3*H].
        grads_W = grads_W.reshape(len(t), -1) # Flatten last two dimensions. [T, H, H] -> [T, H*H].

        # Reverse temporally since these were simulated backwards in time.
        states = states[::-1]
        adjoints = adjoints[::-1]
        grads_Win = grads_Win[::-1]
        grads_W = grads_W[::-1]

        plt.figure(figsize=(8,6))
        ax = plt.subplot(511)
        visualize(X[0], ax, nplot = nplot)
        plt.title('Inputs')
        ax = plt.subplot(512)
        visualize(out, ax, nplot = nplot)
        plt.title('RNN Output')
        ax = plt.subplot(513)
        plt.plot(t[:nplot], states[:nplot, :10])
        plt.title('Hidden States')
        ax = plt.subplot(514)
#        visualize(adjoints[:, :10], ax, nplot = nplot)
        plt.plot(t[:nplot], adjoints[:nplot, :10])
#        plt.imshow(adjoints[:nplot].T, aspect='auto', cmap = 'seismic')
        plt.title('Adjoint Flow')
        ax = plt.subplot(515)
#        plt.imshow(grads[:nplot].T, aspect='auto', cmap = 'seismic')
        plt.plot(t[:nplot], grads_W[:nplot, :10])
        plt.title('Running Gradient')
        plt.tight_layout()

        plt.figure()
        for i in range(5):
            plt.subplot(5,1,1+i)
            plt.plot(true_s[:, i], adjoints[:, i])
        plt.suptitle('Phase Space Plots')

        plt.figure()
        for i in range(1):
            plt.subplot(1,1,1+i)
            plt.plot(t[:nplot*10], true_s[:nplot*10, i])
            plt.gca().twinx()
            plt.plot(t[:nplot*10], adjoints[:nplot*10, i], c = 'red')
        plt.suptitle('Comparitive Plots')

        plt.figure()
        for i in range(1):
            plt.subplot(1,1,1+i)
            v1, v2 = true_s[:, i], adjoints[:, i]
            v1 = (v1 - v1.min()) / (v1.max() - v1.min())
            v2 = (v2 - v2.min()) / (v2.max() - v2.min())
            plt.imshow(np.stack([v1, v2], 0), aspect='auto', cmap = 'binary', interpolation='none')
            plt.colorbar()
        plt.suptitle('Comparitive Plots')

        # dimensionality reduction and activity projection
        # principal component analysis
        accumulated = np.cumsum(true_s, 0)
        quantities = [
            {'val': true_s, 'name': 'Hidden States'},
            {'val': grads_W, 'name': 'Running $W$ Gradient'},
            {'val': grads_Win, 'name': 'Running $W_{in}$ Gradient'},
            {'val': adjoints, 'name': 'Adjoint'},
            {'val': accumulated, 'name': 'Accumulated'}
        ]
        for quant in quantities:
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            compute_and_plot_pca(quant['val'], quant['name'], ax, True)

        plt.show()

def analyze_over_training(checkpoints, iteration, T = 10000, save_raw = False):
    records_all = []
    _, _, _, X, Y = import_data(preload_model_hidden = False)
    X = torch.from_numpy(X[None, :T, :]).float()
    target = torch.from_numpy(Y[None, :T, :]).float()
    t = torch.arange(X.shape[1]).float()
    pbar = tqdm(list(zip(checkpoints, iteration)))
    for ch, itr in pbar:
        pbar.set_description(ch)
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

            record = {'iteration': itr}
            ode = Model(Win, Wout, bout, W)
            ode.set_x(X)
            with torch.enable_grad():
                true_s, out = ode.evaluate(X, grad_record=True)
                loss_fn = nn.MSELoss()
                loss = loss_fn(out, target)
                grad_s = torch.autograd.grad(loss, true_s)[0]

                # Calculate the scale of the two terms contributing to the gradient.
                g1 = 2 / float(out.nelement()) * torch.matmul(out, ode.Wout.transpose(0,1))
                g2 = 2 / float(out.nelement()) * torch.matmul(-target, ode.Wout.transpose(0,1))
                record['g1'] = torch.mean(g1.norm(dim=-1)).item() # Mean of norm over time.
                record['g2'] = torch.mean(g2.norm(dim=-1)).item()
                record['loss'] = loss.item()

                true_s, out, grad_s = true_s[0], out[0], grad_s[0]

            # Measure adjoints and state over time and record everything.
            data = adjoint_calculate_RNN(t, true_s, ode, grad_s)

            x = np.array([a[1].detach().numpy() for a in data])[::-1]
            p = np.array([a[2].detach().numpy() for a in data])[::-1]
            vel = x[1:] - x[:-1] # f(x) - x.

            plt.plot(t[:-1], (vel * p[:-1]).sum(-1), label = f'h_{itr}', color = [itr / iteration[-1], 0., 0.])
            inds = [1, 2, 3, 6, 7]
            names = ['state', 'adjoint', 'win_grad', 'w_grad']
            for ind, name in zip(inds, names):
                states = np.array([a[ind].detach().numpy() for a in data])
                states = states.reshape(len(t), -1)[::-1] # Flatten and reverse.
#                if name == 'win_grad' or name == 'w_grad':
#                    states = states[1:] - states[:-1] # Analyze increments.

                if save_raw:
                    record[f'{name}_raw'] = states
                pca, s = compute_pca(states)
                record[f'{name}_pca'] = s
                record[f'{name}_pca_struct'] = pca
            records_all.append(record)

    plt.title('Hamiltonian')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

    return records_all

def plot_pca_over_training(records, quantity_name, display_name):
    M = int(len(records) ** .5)
    N = int(np.ceil(len(records) / float(M)))
    fig = plt.figure(figsize=(N * 4, M * 3))
    for plot_idx, record in enumerate(records):
        ax = fig.add_subplot(M, N, plot_idx + 1)
        s = record[f'{quantity_name}_pca']

        T = s.shape[0]
        inc = T // 10
        for i in range(0, T, inc):
            c = [i / float(T - 1), 0.0, 0.0]
            ax.plot(s[i:i+inc,0], s[i:i+inc,1], '-', linewidth=.7, color=c)

        accuracy = sum(record[f'{quantity_name}_pca_struct'].explained_variance_ratio_[:3]) * 100

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.title(f'Iteration {record["iteration"]}; 3D acc = {accuracy:.1f}%')
    plt.suptitle(f'{display_name} PCA Projection Over Time (black->red)')
    plt.tight_layout()

def wiener_process_analyze(records, quantity_name, display_name):
    M = int(len(records) ** .5)
    N = int(np.ceil(len(records) / float(M)))
    fig = plt.figure(figsize=(N * 4, M * 3))
    for plot_idx, record in enumerate(records):
        ax = fig.add_subplot(M, N, plot_idx + 1)
        s = record[f'{quantity_name}_pca']
        incs = s[1:] - s[:-1]

        H, xedges, yedges = np.histogram2d(incs[:,0], incs[:,1], bins=60)
        H = H + 1 # +1 just so to not have log(0) below.
        ax.imshow(H.T, cmap='seismic', interpolation='bilinear', norm=mpl.colors.LogNorm(), extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]], origin = 'lower')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.title(f'Iteration {record["iteration"]}')
    plt.suptitle(f'{display_name} PCA Increment Histograms')
    plt.tight_layout()

def plot_pca_matrix_over_training(records, quantity_name, display_name):
    M = int(len(records) ** .5)
    N = int(np.ceil(len(records) / float(M)))
    fig = plt.figure(figsize=(N * 4, M * 3))
    for plot_idx, record in enumerate(records):
        ax = fig.add_subplot(M, N, plot_idx + 1)
        pca = record[f'{quantity_name}_pca_struct']
        plt.imshow(pca.components_[:3, :], aspect = 'auto', interpolation = 'none', cmap = 'gist_rainbow')
        plt.title(f'Iteration {record["iteration"]}')
    plt.suptitle(f'{display_name} PCA Matrices over training')
    plt.tight_layout()

if __name__ == "__main__":
    root = '' if not args.continuous else 'continuous_task/'

    if args.retrain:
        train_flipflop(root)

    if args.analyze_best:
        analyze_model_old(root + 'checkpoints/flip_flop_checkpoint_449.pt')


    files = glob.glob(root + 'checkpoints/*.pt')
    files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    iteration = [int(re.findall(r'\d+', file)[0]) for file in files]
    files = [root + 'checkpoints/' + os.path.basename(p) for p in files]
    if args.repca:
        training_run = analyze_over_training(files, iteration)
        with open(root + 'training_run.pt', "wb") as fp:
            pickle.dump(training_run, fp)

    with open(root + 'training_run.pt', "rb") as fp:
        training_run = pickle.load(fp)

    plt.figure()
    plt.plot([record['loss'] for record in training_run])
    plt.title('Loss on Full Timeframe')

    plt.figure()
    plt.plot(iteration, [record['g1'] for record in training_run], color = "#ffab40", linewidth=4)
    plt.plot(iteration, [record['g2'] for record in training_run], color = "#c27ba0", linewidth=4)
    plt.xlabel('Training Iteration')
    plt.legend(['Term 1 Norm', 'Term 2 Norm'])
    plt.savefig(root + 'Terms.png')
    plt.show()

    plot_pca_over_training(training_run, 'state', 'Hidden States')
    plt.savefig(root + 'pca_plots/Figure_1.png')
    plot_pca_over_training(training_run, 'adjoint', 'Adjoint')
    plt.savefig(root + 'pca_plots/Figure_2.png')
    plot_pca_over_training(training_run, 'win_grad', 'Running $W_{in}$ Gradient')
    plt.savefig(root + 'pca_plots/Figure_3.png')
    plot_pca_over_training(training_run, 'w_grad', 'Running $W$ Gradient')
    plt.savefig(root + 'pca_plots/Figure_4.png')

    plot_pca_matrix_over_training(training_run, 'win_grad', 'Running $W_{in}$ Gradient')
    plt.savefig(root + 'pca_plots/Figure_5.png')
    plot_pca_matrix_over_training(training_run, 'w_grad', 'Running $W$ Gradient')
    plt.savefig(root + 'pca_plots/Figure_6.png')

    wiener_process_analyze(training_run, 'win_grad', 'Running $W_{in}$ Gradient')
    plt.savefig(root + 'pca_plots/Figure_7.png')
    wiener_process_analyze(training_run, 'w_grad', 'Running $W$ Gradient')
    plt.savefig(root + 'pca_plots/Figure_8.png')

    plot_pca_over_training(training_run, 'joint', 'Joint States')
    plt.savefig(root + 'pca_plots/Figure_9.png')

    plt.show()
    exit()
