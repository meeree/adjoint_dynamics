# Take two tasks (e.g., memory pro) and mix them together by adding a context duration and rule input.
import torch
from torch import nn
import numpy as np

MIX_DEFAULT_CFG = {
    'T_context': 30,
}

def generate(inp1, target1, inp2, target2, cfg = MIX_DEFAULT_CFG, noise = True, debug = False):
    # Inputs are 4 dimensional: 
    # [fixation, stim1 cos, stim1 sin, rule].
    # Rule specifies what task to focus on: 0 for task1, 1 for task2. 
    # 4 durations in order: context, stim, memory, response.
    T = inp1.shape[1] + cfg["T_context"]
    n_samples = inp1.shape[0] + inp2.shape[0]

    context_end = cfg["T_context"]

    # Generate data.
    inp = torch.zeros((n_samples, T, 4))
    target = torch.zeros((n_samples, T, 3)) # No rule channel in output.

    # Concatenate. We shuffle samples further down.
    inp[:inp1.shape[0], context_end:, :3] = inp1.clone()
    inp[inp1.shape[0]:, context_end:, :3] = inp2.clone()
    inp[inp1.shape[0]:, :, 3] = 1. # Rule input: 0 for task1, 1 for task2.
    target[:inp1.shape[0], context_end:] = target1.clone()
    target[inp1.shape[0]:, context_end:] = target2.clone()

    # Shuffling.
    inds = torch.randperm(n_samples)
    inp, target = inp[inds], target[inds]

    # Add noise to missing parts.
    if noise:
        with torch.no_grad():
            inp[:, :context_end, :3] += torch.normal(torch.zeros_like(inp[:, :context_end, :3]), 1.) * .1 * (2 ** .5)
            inp[:, :, 3] += torch.normal(torch.zeros_like(inp[:, :, 3]), 1.) * .1 * (2 ** .5)

    if debug:
        import matplotlib.pyplot as plt

        for b in range(4):
            plt.figure()
            for chan, name in enumerate(['Fixation', 'Stim1 Cos', 'Stim1 Sin']):
                for qidx, quant in enumerate([inp, target]):
                    plt.subplot(4, 2, 1 + 2*chan + qidx)
                    plt.plot(quant[b, :, chan], linewidth = 4)
                    plt.title(name)
                    plt.ylim(-1.2, 1.2)

            plt.subplot(4, 2, 7)
            plt.plot(inp[b, :, 3], linewidth = 4)
            plt.title('Rule')
            plt.ylim(-1.2, 1.2)

            plt.suptitle('Input Left, Target Right')

    return inp, target
