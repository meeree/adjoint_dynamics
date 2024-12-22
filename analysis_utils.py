# Utils for analysis of checkpoints during training. 
import torch
import numpy as np
from tqdm import tqdm
from architecture import ModelRNNv3

def load_checkpoints(root):
    # Given a root directoy, return a list of filenames corresponding to all checkpoints in that root directory. 
    # Also return iteration count for each file. Each file should be of the form root + checkpoints/checkpoint_<number>.pt.
    import glob, re, os
    if len(root) > 0 and root[-1] != '/' and root[-1] != '\\':
        root = root + '/'

    checkpoints = glob.glob(root + 'checkpoints/*.pt')
    checkpoints = sorted(checkpoints, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    iteration = [int(re.findall(r'\d+', file)[-1]) for file in checkpoints]
    checkpoints = [root + 'checkpoints/' + os.path.basename(p) for p in checkpoints]
    return checkpoints, iteration

def rerun_trials(X, Y, checkpoints, compute_adj = False, device = 'cuda', verbose = True):
    # #####################################################################################################
    # Given a list of checkpoints, rerun on the same consistent data and possibly compute adjoints, etc.  |
    # Checkpoints can either be a list of file names or a list of pytorch state_dicts.                    |
    # X Should be shape [trials, timesteps, n_in], Y shape [trials, timesteps, n_out].                    |
    # #####################################################################################################
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)

    inp = X.to(device).float()
    if compute_adj:
        targ = Y.to(device).float()

    n_in, n_out = X.shape[-1], Y.shape[-1]

    zs_all, adjs_all, out_all, losses_all = [], [], [], []
    to_np = lambda x: x.detach().cpu().numpy()

    if verbose:
        print("Re-evaluating on the Same Data.")
    pbar = tqdm(list(checkpoints)) if verbose else list(checkpoints)
    for ch in pbar:
        if isinstance(ch, str):
            if verbose:
                pbar.set_description(ch)
            state_dict = torch.load(ch, map_location = device, weights_only=True)['model']
        else:
            state_dict = ch # Assume ch IS just a state dict, not a str indicating where it should be.
        n_hidden = state_dict['W.weight'].shape[0]

        with torch.set_grad_enabled(compute_adj):
            model = torch.jit.script(ModelRNNv3(n_hidden = n_hidden, n_in = n_in, n_out = n_out))
            model.load_state_dict(state_dict)
            model = model.to(device)

            if not compute_adj:
                zs_all.append(to_np(model(inp)[1]))
                continue # Done in this case.

            zs, adj, out, loss_unreduced, loss = model.analysis_mode(inp, targ)
            zs_all.append(to_np(zs)) # [B, T, H]
            adjs_all.append(to_np(adj)) # [B, T, H]
            out_all.append(to_np(out)) # [B, T, n_out]
            losses_all.append(to_np(loss_unreduced.mean(-1))) # [B, T].

    if not compute_adj:
        return np.stack(zs_all)

    zs_all, adjs_all, out_all, losses_all = np.stack(zs_all), np.stack(adjs_all), np.stack(out_all), np.stack(losses_all)
    return np.stack(zs_all), np.stack(adjs_all), np.stack(out_all), np.stack(losses_all)

def batched_cov_and_pcs(traj, traj2 = None, dim_thresh = 0.95, abs_thresh = 1e-6):
    # Get the covariance and principle components for data over checkpoints (D), batches (B), time (T), with hidden dimension (H). 
    # traj is shape [D, B, T, H]. D is trials, B is baches (what we mean over), T is time, H is hidden index.
    # If traj2 is not None, we take cross covariances, assuming traj2 has the same shape.
    centered = traj - traj.mean(1)[:, None]
    if traj2 is not None:
        centered2 = traj2 - traj2.mean(1)[:, None]

    covs = np.zeros((*traj.shape[:-3], traj.shape[-2], traj.shape[-1], traj.shape[-1])) # [D, T, H, H]
    for idx, zs in enumerate(traj):
        for tidx, z_t in enumerate(zs.transpose(1,2,0)): # Shape [H, B]
            if traj2 is None:
                cov = np.cov(z_t)
            else:
                cov = np.mean(centered[idx, :, tidx, :, None] * centered2[idx, :, tidx, None, :], 0) # [H, H]
            covs[idx, tidx] = cov


    evals, pcs = np.linalg.eigh(covs) # Use symmery. Get principle components.
    evals, pcs = evals[:, :, ::-1], pcs[:, :, :, ::-1] # Make descending order. Shapes [D, T, H], [D, T, H, H]
    total_variances = np.sum(evals, axis = -1) # [D, T]
    variance_ratios = np.cumsum(evals / total_variances[..., None], axis = -1)
    dimensions = np.zeros_like(total_variances)
    for i1 in range(variance_ratios.shape[0]):
        for i2 in range(variance_ratios.shape[1]):
            if total_variances[i1,i2] < abs_thresh:
                continue
            dim_idx = np.argwhere(variance_ratios[i1, i2] > dim_thresh)[0,0]
            v0 = 0. if dim_idx == 0 else variance_ratios[i1, i2, dim_idx-1]
            v1 = variance_ratios[i1, i2, dim_idx]
            dimensions[i1, i2] = dim_idx + (dim_thresh - v0) / (v1 - v0) + 1

    dimensions[total_variances < abs_thresh] = 0 # If the total variance is super low, the covariance is a point, so it doesn't make sense to use variance ratios.
    return covs, evals, pcs, variance_ratios, dimensions
