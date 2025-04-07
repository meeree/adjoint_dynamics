# Parameter operator from main paper. 
import torch
import numpy as np

# Utility function
torch_to_np = lambda x: x.detach().cpu().numpy()

# Compute the entire operator. If there are T discrete timesteps and B batch trials, then the discretized operator is a (B*T, B*T) matrix (a flattened (B, T, B, T) tensor). 
# It has no H dimension since it acts as a scalar over the neuron index (see Theorem in our paper).
#
# Features is a list of relevant features for the kernel.
# Explictly, K(t, b, t0, b0) = sum_{f=1}^F innerproduct(features[f, b, t], features[f, b0, t0]) / B.
# For an RNN, f(z,x) = -z + W sigma(z) + W_{in} x(t), the features are [x(t), sigma(z(t))].
def compute_discretized_operator(features, to_np = True, dev = 'cuda'):
    if len(features) == 0:
        return 0.

    B, T = features[0].shape[:2] 
    K = torch.zeros((B, T, B, T)).to(dev) # Compute in cuda for more speed.
    with torch.no_grad():
        for feat in features:
            feat_dev = torch.from_numpy(feat).to(dev) # [B, T, H].
            re_ordered = feat_dev.movedim(-1, 0) # [H, B, T].
            for b in range(B): # Iterate over b since trying to do one giant tensor dot often gives memory issues.
                K[b] += torch.tensordot(feat_dev[b], re_ordered, dims=1) / B

    K = K.reshape((B * T, B * T))
    return torch_to_np(K) if to_np else K

# Compute spectral decomposition by first computing the full discretized operator then using eigsh.
def spectral_decomposition(inputs, zs, components, return_matrix = False):
    from scipy.sparse.linalg import eigsh
    K = compute_discretized_operator(inputs, zs, to_np = True)
    evals, efuns = eigsh(K, components)
    evals, efuns = evals[::-1], efuns[:, ::-1] # Biggest -> smallest ordering.
    if return_matrix:
        return K, evals, efuns
    return evals, efuns

# Compute what the operator K does to a quantity q of shape (..., B, T, H) = (..., batches, timesteps, hidden neuron index).
# This can be done by computing the full discretized matrix and doing matrix vector multiplication. 
# However, this is a very memory heavy approach and we can instead directly compute the action.
def compute_action(features, q, to_np = True, dev = 'cuda'):
    # q is shape (..., B, T, H) where ... can be any dimension.
    if len(features) == 0:
        return 0. * q

    q_dev = torch.from_numpy(q).to(dev)
    out = torch.zeros_like(q_dev)

    B, T = features[0].shape[:2] 
    K = torch.zeros((B, T, B, T)).to(dev) # Compute in cuda for more speed.
    with torch.no_grad():
        for feat in features:
            feat_dev = torch.from_numpy(feat).to(dev) # [B, T, H2].
            f_times_q = feat_dev[..., None, :] * q_dev[..., :, None]  # [..., B, T, H, H2]
            f_times_q = f_times_q.sum((-4,-3)) / B # [..., H, H2].
            out += (f_times_q[..., None, None, :, :] * feat_dev[..., :, :, None, :]).sum(-1) # [..., B, T, H, H2] -> [..., B, T, H] with sum.

    return torch_to_np(out) if to_np else out
