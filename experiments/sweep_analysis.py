# Same as sweep_analysis.ipynb but for many tasks at once.
import sys
sys.path.append('../')
from train import ping_dir
from analysis_utils import rerun_trials, load_checkpoints, batched_cov_and_pcs
import numpy as np
import torch
from torch import nn
import neurogym as ngym
import matplotlib.pyplot as plt
import matplotlib as mpl

def produce_plots(neurogym_root, output_root, tasks):
    ping_dir(output_root)

    # plot_spec tells us how to interpet data and what to plot.
    fn1 = lambda zs_all, adjs_all: zs_all[-2:-1]
    fn2 = lambda zs_all, adjs_all: zs_all[:, :, -1:].swapaxes(0,2)
    fn3 = lambda zs_all, adjs_all: adjs_all[-2:-1]
    fn4 = lambda zs_all, adjs_all: adjs_all[:, :, -1:].swapaxes(0,2)
    plot_spec = [
            {'traj': fn1, 'traj2': fn3, 'title': f'Cross-Covariance, $Cov(a,z)$, Post Training Dynamics', 'xlabel': 'Time, t', 'smoothing': 1},
            {'traj': fn2, 'traj2': fn4, 'title': f'Cross-Covariance, $Cov(a,z)$,  Final Timestep GD flow', 'xlabel': 'GD Iteration, s', 'smoothing': 1},
    ]

    plt.rcParams['font.family'] = 'Times New Roman'
    for spec_idx, spec in enumerate(plot_spec):
        plt.figure(figsize = (11, 9))
        traj_fn, title, xlabel, smoothing = spec['traj'], spec['title'], spec['xlabel'], spec['smoothing']
        for task_idx, task in enumerate(tasks):
            plt.subplot(3, 3, task_idx + 1)
            root = neurogym_root + task + '/'
            print(f'Data root file: {root}')

            checkpoints, GD_iteration = load_checkpoints(root)
            print(f'{len(checkpoints)} Checkpoints, over {GD_iteration[-1]} GD iterations')

            kwargs = {'dt': 100}
            seq_len = 100
            ntrials = 256
            dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=ntrials, seq_len=seq_len)
            inputs, labels = dataset()
            inputs, labels = inputs.swapaxes(0,1), labels.swapaxes(0,1)
            targets = nn.functional.one_hot(torch.from_numpy(labels)).numpy()

            print(f'Input size [B, T, n_in]: {inputs.shape}, Target size [B, T, n_out]: {targets.shape}')

            zs_all, adjs_all, losses_all = rerun_trials(inputs, targets, checkpoints, compute_adj = True)
            zs_all, adjs_all = zs_all / np.mean(np.abs(zs_all)), adjs_all / np.mean(np.abs(adjs_all)) # Normalize scales for PCA. Adj can have a very small scale otherwise.
            print(f'zs shape is [checkpoints, trials, timesteps, n_hidden]: {zs_all.shape}')

            def smooth(y, box_pts, mode = 'same'):
                box = np.ones(box_pts)/box_pts
                y_smooth = np.convolve(y, box, mode=mode)
                return y_smooth

            traj = traj_fn(zs_all, adjs_all)
            traj2 = None
            if 'traj2' in spec:
                traj2 = spec['traj2'](zs_all, adjs_all)
            covs, evals, pcs, variance_ratios, dims = batched_cov_and_pcs(traj, traj2)

#            plt.subplot(1,2,1)
#            for ev_idx in range(4):
#                plt.plot(evals[0, :, ev_idx], label  = f'$\lambda_{ev_idx+1}$')
#            plt.ylabel('Top Estimated Variances', fontsize = 15)
#            plt.yscale('log')
#            plt.xlabel(xlabel, fontsize = 15)
#            plt.legend(frameon=False, fontsize = 15)

#            plt.subplot(1,2,2)
            dim_smooth = smooth(dims[0], smoothing, mode = 'valid')
            plt.plot(dim_smooth)
            plt.xlabel(xlabel, fontsize = 15)
            plt.ylabel('Estimated Dimension', fontsize = 15)
            plt.title(task, fontsize = 15)
        plt.suptitle(title, fontsize = 15, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(output_root + '/dimension_plot_' + str(spec_idx+4) + '.pdf')

tasks = 'PerceptualDecisionMaking-v0 OneTwoThreeGo-v0 MultiSensoryIntegration-v0 MotorTiming-v0 IntervalDiscrimination-v0 GoNogo-v0 DelayPairedAssociation-v0 DelayComparison-v0 ContextDecisionMaking-v0'
tasks = tasks.split() # str -> list of str.
produce_plots('/home/ws3/Desktop/james/neurogym/examples/', 'outputs/', tasks) 
