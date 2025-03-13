# Same as sweep_analysis.ipynb but for many tasks at once.
import sys, glob, re
sys.path.append('../')
from train import ping_dir
from analysis_utils import rerun_trials, load_checkpoints, batched_cov_and_pcs, import_checkpoint, load_sweep_checkpoints
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import neurogym as ngym
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

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
                checkpoints, GD_iteration = checkpoints[::4], GD_iteration[::4] # SUBSET.
                print(f'{len(checkpoints)} Checkpoints, over {GD_iteration[-1]} GD iterations')

                kwargs = {'dt': 100}
                seq_len = 100
                ntrials = 256 
                dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=ntrials, seq_len=seq_len)
                inputs, labels = dataset()
                inputs, labels = inputs.swapaxes(0,1), labels.swapaxes(0,1)
                targets = nn.functional.one_hot(torch.from_numpy(labels)).numpy()

                print(f'Input size [B, T, n_in]: {inputs.shape}, Target size [B, T, n_out]: {targets.shape}')

                zs_all, adjs_all, outs_all, losses_all, _ = rerun_trials(inputs, targets, checkpoints, compute_adj = True, verbose = False, device = 'cpu')
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
        checkpoints, GD_iteration = checkpoints[::4], GD_iteration[::4] # SUBSET.
        print(f'{len(checkpoints)} Checkpoints, over {GD_iteration[-1]} GD iterations')

        kwargs = {'dt': 100}
        seq_len = 100
        ntrials = 500
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=ntrials, seq_len=seq_len)
        inputs, labels = dataset()
        inputs, labels = inputs.swapaxes(0,1), labels.swapaxes(0,1)
        targets = nn.functional.one_hot(torch.from_numpy(labels)).numpy()

        print(f'Input size [B, T, n_in]: {inputs.shape}, Target size [B, T, n_out]: {targets.shape}')

        zs_all, adjs_all, outs_all, losses_all, _ = rerun_trials(inputs, targets, checkpoints, compute_adj = True, verbose = False, device = 'cpu')
        print(f'zs shape is [checkpoints, trials, timesteps, n_hidden]: {zs_all.shape}')

        fn = lambda data: data.reshape((data.shape[0], -1, 1, data.shape[-1])).swapaxes(0,2)
        dims = batched_cov_and_pcs(fn(zs_all))[-1][0] # [gd iters]
        dims_all.append(dims)
#        inputs_stacked = np.repeat(fn(inputs[None]), fn(adjs_all).shape[2], axis=2)
#        print(inputs_stacked.shape, fn(adjs_all).shape)
        dims = batched_cov_and_pcs(fn(adjs_all))[-1][0] # [gd iters]
        dims_all.append(dims)

#    dims_all = np.stack(dims_all)
#    np.savetxt('dims_all.csv', dims_all)
#    print(dims_all.shape)

    plt.figure(figsize = (12, 6))
    def condense_name(name):
        name = name[:-3] # Cut out -v0
        lower = lambda s: ''.join([s[0]] + [c for c in s[1:] if c.islower()])
        name = ''.join(lower(name[i:i+3]) for i, c in enumerate(name) if c.isupper())
        return name

    abbr_task_names = [condense_name(name) for name in tasks]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, dims in enumerate(dims_all):
        plt.subplot(2,3,idx//2+1)
        task_name = abbr_task_names[idx // 2]
#        marker = '*' if idx >= 20 else ('x' if idx >= 10 else '^') # 10 distinct colors, so use different marker for every 10
        plt.title(tasks[idx//2], fontsize = 15)
        plt.plot(np.arange(len(dims)), dims, color = colors[(idx // 2) % len(colors)], linestyle = 'solid' if idx % 2 == 0 else 'dashed')
        plt.legend(['State', 'Adjoint'], frameon=False, fontsize = 14)

#    plt.legend(frameon=False, ncol=2, fontsize = 12)
#    plt.plot(dims_all.mean(0))
    plt.xlabel('GD Iteration, $s$', fontsize = 15)
    plt.ylabel('Estimated Dimension', fontsize = 15)
    plt.tight_layout()
    plt.savefig(output_root  + 'Dimension_all_tasks.pdf')
    plt.show()

def plot_dim_over_g(neurogym_root, output_root, task):
    ping_dir(output_root)

    dims_all = []
    plt.figure(figsize = (11, 9))
    smoothing = 1

    manifest, all_checkpoints, all_iters = load_sweep_checkpoints(neurogym_root + '/' + task)
    gs = [dc['init_scale_percent'] for dc in manifest]
    cmap = plt.get_cmap('viridis')
    accs_all = []
    for g_idx, group in enumerate(zip(tqdm(all_checkpoints), gs, all_iters)):
        checkpoints, g, GD_iteration = group
        checkpoints, GD_iteration = checkpoints[::1], GD_iteration[::1] # SUBSET.
        if len(dims_all) > 0 and len(checkpoints) != len(dims_all[-1]):
            continue # Incomplete run. They should all have same length.
        accs = np.array([sum(vals[-50:])/50. for vals in import_checkpoint(checkpoints[-1])['accuracies'].values()])
        accs_all.append(accs)

        success = np.all(accs > 0.95)
        print(success, accs)

        print(f'{len(checkpoints)} Checkpoints, over {GD_iteration[-1]} GD iterations')

        kwargs = {'dt': 100}
        seq_len = 100
        ntrials = 64
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=ntrials, seq_len=seq_len)
        inputs, labels = dataset()
        inputs, labels = inputs.swapaxes(0,1), labels.swapaxes(0,1)
        targets = nn.functional.one_hot(torch.from_numpy(labels)).numpy()

        print(f'Input size [B, T, n_in]: {inputs.shape}, Target size [B, T, n_out]: {targets.shape}')

        zs_all, adjs_all, outs_all, losses_all, _ = rerun_trials(inputs, targets, checkpoints, compute_adj = True, verbose = False, device = 'cpu')
        has_nan = np.isnan(zs_all).any()
        if has_nan: # Invalid.
            continue
        print(f'zs shape is [checkpoints, trials, timesteps, n_hidden]: {zs_all.shape}')

        fn = lambda data: data.reshape((data.shape[0], -1, 1, data.shape[-1])).swapaxes(0,2)
        dims = batched_cov_and_pcs(fn(zs_all))[-1][0] # [gd iters]
        dims_all.append(dims)
        plt.plot(dims, color = cmap(g_idx / (len(gs) - 1)), linestyle = 'dashed' if not success else 'solid')

    accs_all = np.stack(accs_all)
    np.savetxt('accs_all_' + task + '.csv', accs_all)

    dims_all = np.stack(dims_all)
    np.savetxt('dims_all_' + task + '.csv', dims_all)
    plt.suptitle(f'{task} Dimension Over Training, Many Init Scales')
    plt.xlabel('GD Iteration, s')
    plt.ylabel('Estimated Dimension')
    plt.show()

    for idx, dims in enumerate(dims_all):
        task_name = abbr_task_names[idx // 2]
#        marker = '*' if idx >= 20 else ('x' if idx >= 10 else '^') # 10 distinct colors, so use different marker for every 10
        plt.title(tasks[idx//2])
        plt.plot(np.arange(len(dims)), dims, color = colors[(idx // 2) % len(colors)], linestyle = 'solid' if idx % 2 == 0 else 'dashed')
        plt.legend(['Cov(a,sig(z))'], frameon=False)

#    plt.legend(frameon=False, ncol=2, fontsize = 12)
#    plt.plot(dims_all.mean(0))
    plt.xlabel('GD Iteration, $s$', fontsize = 15)
    plt.ylabel('Estimated Dimension', fontsize = 15)
    plt.tight_layout()
    plt.savefig(output_root  + 'Dimension_all_tasks.pdf')
    plt.show()

def sig_prime_means(neurogym_root, output_root, tasks, naming_lambda = lambda task: task):
    ping_dir(output_root)

    mean_sig_prime = []
    for task_idx, task in enumerate(tqdm(tasks)):
        root = neurogym_root + naming_lambda(task)  + '/'
        print(f'Data root file: {root}')
        checkpoints, GD_iteration = load_checkpoints(root)
        checkpoints, GD_iteration = checkpoints[::4], GD_iteration[::4] # SUBSET.
        print(f'{len(checkpoints)} Checkpoints, over {GD_iteration[-1]} GD iterations')

        kwargs = {'dt': 100}
        seq_len = 100
        ntrials = 64
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=ntrials, seq_len=seq_len)
        inputs, labels = dataset()
        inputs, labels = inputs.swapaxes(0,1), labels.swapaxes(0,1)
        targets = nn.functional.one_hot(torch.from_numpy(labels)).numpy()

        print(f'Input size [B, T, n_in]: {inputs.shape}, Target size [B, T, n_out]: {targets.shape}')

        zs_all, adjs_all, outs_all, losses_all, _ = rerun_trials(inputs, targets, checkpoints, compute_adj = True, verbose = False, device = 'cpu')
        print(f'zs shape is [checkpoints, trials, timesteps, n_hidden]: {zs_all.shape}')

        mean_sig_prime.append(np.mean(zs_all, 1))

    mean_sig_prime = np.stack(mean_sig_prime)
    for i in range(mean_sig_prime.shape[0]):
        plt.subplot(3,3,i+1)
        plt.plot(mean_sig_prime[i, 1:, :, 0].T)
    plt.show()

def subspace_angles(neurogym_root, output_root, tasks, naming_lambda = lambda task: task):
    ping_dir(output_root)

    dims_all = []
    plt.figure(figsize = (11, 9))
    smoothing = 1
    angles_all = []
    for task_idx, task in enumerate(tqdm(tasks)):
        root = neurogym_root + naming_lambda(task)  + '/'
        print(f'Data root file: {root}')
        checkpoints, GD_iteration = load_checkpoints(root)
        checkpoints, GD_iteration = checkpoints[::4], GD_iteration[::4] # SUBSET.
        print(f'{len(checkpoints)} Checkpoints, over {GD_iteration[-1]} GD iterations')

        kwargs = {'dt': 100}
        seq_len = 100
        ntrials = 500
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=ntrials, seq_len=seq_len)
        inputs, labels = dataset()
        inputs, labels = inputs.swapaxes(0,1), labels.swapaxes(0,1)
        targets = nn.functional.one_hot(torch.from_numpy(labels)).numpy()

        print(f'Input size [B, T, n_in]: {inputs.shape}, Target size [B, T, n_out]: {targets.shape}')

        zs_all, adjs_all, outs_all, losses_all, _ = rerun_trials(inputs, targets, checkpoints, compute_adj = True, verbose = False, device = 'cpu')
        print(f'zs shape is [checkpoints, trials, timesteps, n_hidden]: {zs_all.shape}')

        fn = lambda data: data.reshape((data.shape[0], -1, data.shape[-1])).swapaxes(1,2)
        zs_all, adjs_all = fn(zs_all), fn(adjs_all)

        angles = []
        from scipy.linalg import subspace_angles
        for s in tqdm(range(zs_all.shape[0])):
            U_z = PCA(3).fit_transform(zs_all[s])
            U_a = PCA(3).fit_transform(adjs_all[s])
            angles.append(subspace_angles(U_z, U_a))
        angles_all.append(angles)


    plt.figure(figsize = (12, 6))
    def condense_name(name):
        name = name[:-3] # Cut out -v0
        lower = lambda s: ''.join([s[0]] + [c for c in s[1:] if c.islower()])
        name = ''.join(lower(name[i:i+3]) for i, c in enumerate(name) if c.isupper())
        return name

    abbr_task_names = [condense_name(name) for name in tasks]
    angles_all = np.array(angles_all)
    for i in range(angles_all.shape[0]):
        plt.plot(angles_all[i], label = abbr_task_names[i])
    plt.legend()
    plt.show()

#    dims_all = np.stack(dims_all)
#    np.savetxt('dims_all.csv', dims_all)
#    print(dims_all.shape)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, dims in enumerate(dims_all):
        plt.subplot(2,3,idx//2+1)
        task_name = abbr_task_names[idx // 2]
#        marker = '*' if idx >= 20 else ('x' if idx >= 10 else '^') # 10 distinct colors, so use different marker for every 10
        plt.title(tasks[idx//2], fontsize = 15)
        plt.plot(np.arange(len(dims)), dims, color = colors[(idx // 2) % len(colors)], linestyle = 'solid' if idx % 2 == 0 else 'dashed')
        plt.legend(['State', 'Adjoint'], frameon=False, fontsize = 14)

#    plt.legend(frameon=False, ncol=2, fontsize = 12)
#    plt.plot(dims_all.mean(0))
    plt.xlabel('GD Iteration, $s$', fontsize = 15)
    plt.ylabel('Estimated Dimension', fontsize = 15)
    plt.tight_layout()
    plt.savefig(output_root  + 'Dimension_all_tasks.pdf')
    plt.show()

SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 15
BIGGEST_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGEST_SIZE)  # fontsize of the figure title

#tasks = 'PerceptualDecisionMaking-v0 OneTwoThreeGo-v0 MultiSensoryIntegration-v0 MotorTiming-v0 IntervalDiscrimination-v0 GoNogo-v0 DelayPairedAssociation-v0 DelayComparison-v0 ContextDecisionMaking-v0'
tasks = 'MotorTiming-v0 AntiReach-v0 DelayComparison-v0 ProbabilisticReasoning-v0 MultiSensoryIntegration-v0 PerceptualDecisionMaking-v0 PerceptualDecisionMakingDelayResponse-v0'
tasks = tasks.split() # str -> list of str.
tasks.pop(-2) # Remove one to make it 6
subspace_angles('/home/ws3/Desktop/james/neurogym/examples/', 'outputs_clean/', tasks, naming_lambda = lambda task: 'default_init_no_stopping_' + task)
produce_plots_simple('/home/ws3/Desktop/james/neurogym/examples/', 'outputs_clean/', tasks, naming_lambda = lambda task: 'default_init_no_stopping_' + task)
produce_plots('/home/ws3/Desktop/james/neurogym/examples/', 'outputs_clean/', tasks, naming_lambda = lambda task: 'default_init_no_stopping_' + task) 

#tasks = tasks.split() # str -> list of str.
#plot_dim_over_g('../scale_sweeps_1_20/', 'outputs_run4/', 'MultiSensoryIntegration-v0')
##plot_dim_over_g('../scale_sweeps_1_10/', 'outputs_run4/', 'MotorTiming-v0')
#sig_prime_means('/home/ws3/Desktop/james/neurogym/examples/', 'outputs_run3/', tasks, naming_lambda = lambda task: 'default_init_no_stopping_' + task)
