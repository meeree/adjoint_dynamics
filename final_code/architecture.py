''' RNN architecture with internal noise and a version that is compatible with my adjoint code. '''
import torch
from torch import nn

class ModelRNNv2(nn.Module):
    def __init__(self, n_hidden = 100, n_in = 3, n_out = 3):
        super().__init__()
        self.n_out, self.n_in = n_out, n_in
        self.n_hidden = n_hidden

        self.W_out = nn.Linear(n_hidden, self.n_out, bias = False)
        self.W = nn.Linear(n_hidden, n_hidden, bias = False)
        self.W_in = nn.Linear(n_in, n_hidden, bias = False)
        self.noise_std = 0.
        self.X = torch.zeros(()) # For adjoint code.

    @torch.jit.export
    def forward(self, t, h):
        ''' Compute flow from time t-1 to time t, NOT time t to t+1 !!! '''
        ''' Operates like a non-autonomous ODE where t indexes into a pre-stored self.X array. '''
        ''' Times should be discrete integer indices. '''
        ''' t is timestep, h is state at this time. '''
        if len(h.shape) == 1:
            h = h[None, :] # Not batched. Just add a 1.
        return self.W(torch.tanh(h)) + self.W_in(self.X[:, torch.round(t).int(), :])

    @torch.jit.export
    def evaluate(self, X, h0 = torch.Tensor()):
        # Pass through RNN over time.
        self.X = X # To let user know what current X value is.
        h0 = torch.zeros(X.shape[0], self.n_hidden).to(X.device) if not h0.numel() else h0.to(X.device)
        h = torch.normal(h0, self.noise_std) # Noisy init.
        hidden = []
        for t in range(X.shape[1]):
            h = self.W(torch.tanh(h)) + self.W_in(X[:, t, :])
            hidden.append(h)

        hidden = torch.stack(hidden, 1)
        hidden += torch.normal(torch.zeros_like(hidden), self.noise_std) # Hidden noise.

        # Pass through output layer.
        return self.W_out(hidden), hidden

    @torch.jit.export
    def eval_autonomous_simple(self, h):
        # h shape [..., H].
        hout = self.W(torch.tanh(h)) 
        return self.W_out(hout), hout

# Turns the forward function of ModelRNNv2 into eval_autonomous_simple, which makes it very fast for FP finding, etc.
# This is just used to make it compatible with things that use a forward function directly.
class ModelWrapper(nn.Module):
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn
        self.batch_first = True

    def forward(self, h0):
        return self.rnn.eval_autonomous_simple(h0)
