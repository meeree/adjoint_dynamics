# One dimensional task where there are two fixed point attractors in ideal solution.
import torch
from torch import nn
import numpy as np

DEFAULT_CFG = {
    'T_stim': 30, 
    'T_memory': 30, 
    'T_response': 30, 
    'n_samples': 1000
}

def generate(cfg = DEFAULT_CFG, debug = False, noise = True):
    # Inputs are 2 dimensional of format,
    # [fixation, stim].
    T = cfg["T_stim"] + cfg["T_memory"] + cfg["T_response"]
    D = 2

    memory_start = cfg["T_stim"] 
    response_start = memory_start + cfg["T_memory"]
    
    # Generate data.
    inp = torch.zeros((cfg["n_samples"], T, D))
    target = torch.zeros_like(inp)

    # Fixate until response time. 
    inp[:, :response_start, 0] = 1.
    target[:, :, 0] = inp[:, :, 0] # Same target for fixation.

    discr = torch.rand(cfg["n_samples"]) * 2 - 1. # Inputs in range [-1, 1]
    decay_over_t = torch.linspace(1., 0., cfg['T_stim'])
    inp[:, :memory_start, 1] += discr[:, None] * decay_over_t[None, :]
    decay_over_t = torch.linspace(1., 0., cfg['T_response'])
    target[:, response_start:, 1] += discr[:, None] * decay_over_t[None, :]

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        for chan, name in enumerate(['Fixation', 'Stim']):
            for qidx, quant in enumerate([inp, target]):
                plt.subplot(2, 2, 1 + 2*chan + qidx)
                plt.plot(quant[0, :, chan], linewidth = 4)
                plt.title(name)
                plt.axvline(memory_start, c = 'black', alpha = .5, linewidth = 3, linestyle = 'dashed')
                plt.axvline(response_start, c = 'black', alpha = .5, linewidth = 3, linestyle = 'dashed')
                plt.ylim(-1.2, 1.2)
            plt.suptitle('Input Left, Target Right')

    # Add noise to inputs.
    if noise:
        with torch.no_grad():
            inp += torch.normal(torch.zeros_like(inp), 1.) * .1 * (2 ** .5)
    return inp, target

def accuracy(X, Y):
    cnd1 = torch.sum(torch.abs(X[:, -1, 0] - Y[:, -1, 0]) < .01) # Fixation.
    cnd2 = (torch.abs(X[:, -1, 1] - Y[:, -1, 1]) < .01) # Stim
    return torch.mean(torch.logical_and(cnd1, cnd2).float())
