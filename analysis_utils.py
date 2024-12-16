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

def rerun_trials(X, Y, checkpoints, compute_adj = False, device = 'cuda'):
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

    zs_all, adjs_all, losses_all = [], [], []
    to_np = lambda x: x.detach().cpu().numpy()

    print("Re-evaluating on the Same Data.")
    pbar = tqdm(list(checkpoints))
    for ch in pbar:
        if isinstance(ch, str):
            pbar.set_description(ch)
            state_dict = torch.load(ch, map_location = device, weights_only=True)['model']
            n_hidden = state_dict['W.weight'].shape[0]
        else:
            state_dict = ch # Assume ch IS just a state dict, not a str indicating where it should be.

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
            losses_all.append(to_np(loss_unreduced.mean(-1))) # [B, T].

    if not compute_adj:
        return np.stack(zs_all)
    return np.stack(zs_all), np.stack(adjs_all), np.stack(losses_all)
