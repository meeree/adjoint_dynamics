import torch
from torch import nn
import numpy as np
from . import memory_pro

DEFAULT_CFG = {
    'T_context': 30,
}
DEFAULT_CFG.update(memory_pro.DEFAULT_CFG)

def generate(cfg = DEFAULT_CFG, noise = True, debug = False):
    # Inputs are 5 dimensional: 
    # [fixation, stim1 cos, stim1 sin, anti, delay].
    # anti specifies if task is pro or anti (0 or 1), delay specifies if memory or delay task (0 or 1)
    # 4 durations in order: context, stim, memory, response.
    ctx_end = cfg["T_context"]
    
    # mem-pro, mem-anti, delay-pro, delay-anti.
    flags = {'anti': [False, True, False, True], 'delay': [False, False, True, True]}
    
    inps, targets = [], []
    anti_chans, delay_chans = [], []
    cfg['n_samples'] = cfg['n_samples'] // 4 # Divide up samples between four tasks.
    for i in range(4):
        cfg['anti'] = flags['anti'][i]
        cfg['delay'] = flags['delay'][i]
        inp_i, target_i = memory_pro.generate(cfg, debug=debug, noise=noise)
        ctx_period = torch.zeros((inp_i.shape[0], ctx_end, inp_i.shape[2]))
        ctx_period[:, :, 0] = 1. # Fixation
        inp_i = torch.cat((ctx_period, inp_i), 1)
        target_i = torch.cat((ctx_period, target_i), 1)

        anti_chan_i = torch.zeros_like(inp_i[:, :, 0:1])
        delay_chan_i = torch.zeros_like(inp_i[:, :, 0:1])
        anti_chan_i[:, :ctx_end] = int(cfg['anti']) # Anti input.
        delay_chan_i[:, :ctx_end] = int(cfg['delay']) # Delay input.

        inp_i = torch.cat((inp_i, anti_chan_i, delay_chan_i), -1) # [B, T, 3] -> [B, T, 5]

        inps.append(inp_i)
        targets.append(target_i)

    inp = torch.cat(inps, 0)
    target = torch.cat(targets, 0)

#    # Shuffling.
#    inds = torch.randperm(n_samples)
#    inp, target = inp[inds], target[inds]

    # Add noise to missing parts.
    if noise:
        with torch.no_grad():
            inp[:, :ctx_end, :3] += torch.normal(torch.zeros_like(inp[:, :ctx_end, :3]), 1.) * .1 * (2 ** .5)
            inp[:, :, 3:] += torch.normal(torch.zeros_like(inp[:, :, 3:]), 1.) * .1 * (2 ** .5)

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

def accuracy(X, Y):
    return memory_pro.accuracy(X, Y)
