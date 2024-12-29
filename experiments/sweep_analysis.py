# Same as sweep_analysis.ipynb but for many tasks at once.
import sys
sys.path.append('../')
from train import ping_dir
from analysis_utils import rerun_trials, load_checkpoints, batched_cov_and_pcs
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import neurogym as ngym
import matplotlib.pyplot as plt
import matplotlib as mpl

def produce_plots(neurogym_root, output_root, tasks, naming_lambda = lambda task: task):
    ping_dir(output_root)

    # plot_spec tells us how to interpet data and what to plot.
    fn1 = lambda zs_all, adjs_all: zs_all[-2:-1]
    fn2 = lambda zs_all, adjs_all: zs_all[:, :, -1:].swapaxes(0,2)
    fn3 = lambda zs_all, adjs_all: adjs_all[-2:-1]
    fn4 = lambda zs_all, adjs_all: adjs_all[:, :, -1:].swapaxes(0,2)
    plot_spec = [
            {'traj': fn1, 'title': f'Hidden, Post Training Dynamics', 'xlabel': 'Time, t', 'smoothing': 1},
            {'traj': fn2, 'title': f'Hidden,  Final Timestep GD flow', 'xlabel': 'GD Iteration, s', 'smoothing': 1},
            {'traj': fn3, 'title': f'Adjoint, Post Training Dynamics', 'xlabel': 'Time, t', 'smoothing': 1},
            {'traj': fn4, 'title': f'Adjoint,  Final Timestep GD flow', 'xlabel': 'GD Iteration, s', 'smoothing': 1},
            {'traj': fn1, 'traj2': fn3, 'title': f'Cross-Covariance, $Cov(a,z)$, Post Training Dynamics', 'xlabel': 'Time, t', 'smoothing': 1},
            {'traj': fn2, 'traj2': fn4, 'title': f'Cross-Covariance, $Cov(a,z)$,  Final Timestep GD flow', 'xlabel': 'GD Iteration, s', 'smoothing': 1},
    ]

    plt.rcParams['font.family'] = 'Times New Roman'
    for spec_idx, spec in enumerate(plot_spec):
        print("Making plot ", spec['title'])
        plt.figure(figsize = (11, 9))
        traj_fn, title, xlabel, smoothing = spec['traj'], spec['title'], spec['xlabel'], spec['smoothing']
        for task_idx, task in enumerate(tqdm(tasks)):
            plt.subplot(3, 3, task_idx + 1)
            for rerun in range(5): # Rerun a couple times.
                root = neurogym_root + naming_lambda(task)  + '/'
                print(f'Data root file: {root}')

                checkpoints, GD_iteration = load_checkpoints(root)
#                checkpoints, GD_iteration = checkpoints[::10], GD_iteration[::10] # SUBSET.
                print(f'{len(checkpoints)} Checkpoints, over {GD_iteration[-1]} GD iterations')

                kwargs = {'dt': 100}
                seq_len = 100
                ntrials = 256 
                dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=ntrials, seq_len=seq_len)
                inputs, labels = dataset()
                inputs, labels = inputs.swapaxes(0,1), labels.swapaxes(0,1)
                targets = nn.functional.one_hot(torch.from_numpy(labels)).numpy()

                print(f'Input size [B, T, n_in]: {inputs.shape}, Target size [B, T, n_out]: {targets.shape}')

                zs_all, adjs_all, outs_all, losses_all = rerun_trials(inputs, targets, checkpoints, compute_adj = True, verbose = False, device = 'cpu')
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

def produce_plots_simple(neurogym_root, output_root, tasks, naming_lambda = lambda task: task):
    ping_dir(output_root)

    dims_all = []
    plt.figure(figsize = (11, 9))
    smoothing = 1
    for task_idx, task in enumerate(tqdm(tasks)):
        root = neurogym_root + naming_lambda(task)  + '/'
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

        zs_all, adjs_all, outs_all, losses_all = rerun_trials(inputs, targets, checkpoints, compute_adj = True, verbose = False, device = 'cpu')
        zs_all, adjs_all = zs_all / np.mean(np.abs(zs_all)), adjs_all / np.mean(np.abs(adjs_all)) # Normalize scales for PCA. Adj can have a very small scale otherwise.
        print(f'zs shape is [checkpoints, trials, timesteps, n_hidden]: {zs_all.shape}')

        dims = batched_cov_and_pcs(zs_all[:, :, -1:].swapaxes(0,2))[-1][0] # [gd iters]
        dims_all.append(dims)

#    dims_all = np.stack(dims_all)
#    np.savetxt('dims_all.csv', dims_all)
#    print(dims_all.shape)

    plt.figure(figsize = (10, 6))
    def condense_name(name):
        name = name[:-3] # Cut out -v0
        lower = lambda s: ''.join([s[0]] + [c for c in s[1:] if c.islower()])
        name = ''.join(lower(name[i:i+3]) for i, c in enumerate(name) if c.isupper())
        return name

    abbr_task_names = [condense_name(name) for name in tasks]
    for idx, (dims, task_name) in enumerate(zip(dims_all, abbr_task_names)):
#        marker = '*' if idx >= 20 else ('x' if idx >= 10 else '^') # 10 distinct colors, so use different marker for every 10
        plt.plot(np.arange(len(dims)), dims, label=task_name)

    plt.legend(frameon=False, ncol=2, fontsize = 12)
#    plt.plot(dims_all.mean(0))
    plt.xlabel('GD Iteration, $s$', fontsize = 15)
    plt.ylabel('Estimated Dimension', fontsize = 15)
    plt.title('Dimension of Final Hidden State Over GD')
    plt.tight_layout()
    plt.savefig(output_root  + 'Dimension_all_tasks.pdf')
    plt.show()

#tasks = 'PerceptualDecisionMaking-v0 OneTwoThreeGo-v0 MultiSensoryIntegration-v0 MotorTiming-v0 IntervalDiscrimination-v0 GoNogo-v0 DelayPairedAssociation-v0 DelayComparison-v0 ContextDecisionMaking-v0'
tasks = 'MotorTiming-v0 AntiReach-v0 DelayComparison-v0 ProbabilisticReasoning-v0 MultiSensoryIntegration-v0 PerceptualDecisionMaking-v0 PerceptualDecisionMakingDelayResponse-v0'
tasks = tasks.split() # str -> list of str.
produce_plots('/home/ws3/Desktop/james/neurogym/examples/', 'outputs_run2/', tasks, naming_lambda = lambda task: 'run2_' + task) 
#produce_plots_simple('/home/ws3/Desktop/james/neurogym/examples/', 'outputs_run2/', tasks, naming_lambda = lambda task: 'run2_' + task)
