# Simplified CategoryPro task based on Flexible multitask computation in recurrent networks utilizes shared dynamical motifs
import torch
from torch import nn
import numpy as np
import memory_pro # MOST of the definition is the same. Just the target changes.

DEFAULT_CFG = memory_pro.DEFAULT_CFG

def generate(cfg = DEFAULT_CFG, noise = True, debug = False):
    # See memory_pro.py for most of the details. 
    inp, target = memory_pro.generate(cfg, noise = noise)
    # Check theta < pi or > pi, i.e. sin > 0 or < a.
    # Goal: if theta < pi, converge to fixed point at pi/2. If theta > pi, at 3*pi/2.
    target[:, :, 1] = 0.
    target[:, :, 2] = torch.where(target[:, :, 2] > 0., 1., -1.)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        for chan, name in enumerate(['Fixation', 'Stim1 Cos', 'Stim1 Sin']):
            for qidx, quant in enumerate([inp, target]):
                plt.subplot(3, 2, 1 + 2*chan + qidx)
                plt.plot(quant[0, :, chan], linewidth = 4)
                plt.title(name)
                plt.axvline(memory_start, c = 'black', alpha = .5, linewidth = 3, linestyle = 'dashed')
                plt.axvline(response_start, c = 'black', alpha = .5, linewidth = 3, linestyle = 'dashed')
                plt.ylim(-1.2, 1.2)
            plt.suptitle('Input Left, Target Right')

    return inp, target

def accuracy(X, Y):
    return memory_pro.accuracy(X,Y)
