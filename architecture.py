''' RNN architecture with internal noise and a version that is compatible with my adjoint code. '''
import torch
from torch import nn

class ModelRNNv3(nn.Module):
    def __init__(self, n_hidden = 100, n_in = 3, n_out = 3):
        super().__init__()
        self.n_out, self.n_in = n_out, n_in
        self.n_hidden = n_hidden

        self.W_out = nn.Linear(n_hidden, self.n_out, bias = False)
        self.W = nn.Linear(n_hidden, n_hidden, bias = False)
        self.W_in = nn.Linear(n_in, n_hidden, bias = False)
        self.noise_std = 0.

        self.loss_fn_no_reduce = nn.MSELoss(reduction='none') # User can change.

    @torch.jit.export
    def eval_hidden_with_noise(self, X, h0):
        h = torch.zeros(X.shape[0], self.n_hidden).to(X.device) if not h0.numel() else h0.to(X.device)
        if self.noise_std > 0:
            h = torch.normal(h, self.noise_std) # Noisy init.

        hidden = []
        for t in range(X.shape[1]):
            h_next = self.W(torch.tanh(h)) + self.W_in(X[:, t, :])
            hidden.append(h_next)
            h = h_next.clone()
            if self.noise_std > 0:
                h += torch.normal(torch.zeros_like(h), self.noise_std) # Hidden noise.
        return hidden

    def forward(self, X, h0 = torch.Tensor()):
        # Default function for training:
        # X is shape [batch size, timesteps, n_in]. 
        # Will return a pair: [batch size, timesteps, n_out], [batch size, timesteps, n_hidden].
        hidden = torch.stack(self.eval_hidden_with_noise(X, h0), 1)
        return self.W_out(hidden), hidden

    @torch.jit.export
    def analysis_mode(self, X, target, h0 = torch.Tensor()):
        # Intended for deep analysis of the GD flow:
        # Run RNN and compute losses, 
        # returning hidden, adjoints, outputs, unreduced losses, reduced loss.
        # Shapes: [B, T, H], [B, T, H], [B, T, O], [B, T, O], scalar.
        hidden = self.eval_hidden_with_noise(X, h0) # A list.
        for h in hidden:
            h.retain_grad() # For adjoints.
        output = self.W_out(torch.stack(hidden, 1))
        loss_unreduced = self.loss_fn_no_reduce(output, target)
        loss = loss_unreduced.mean()
        loss.backward() # Perform BPTT.

        adjoint = torch.stack([h.grad for h in hidden], 1) # dL/dz defn of adjoint.
        hidden = torch.stack(hidden, 1)
        return hidden, adjoint, output, loss_unreduced, loss

    @torch.jit.export
    def eval_autonomous_simple(self, h):
        # h shape [..., H].
        hout = self.W(torch.tanh(h)) 
        return self.W_out(hout), hout

# Turns the forward function of ModelRNNv3 into eval_autonomous_simple, which makes it very fast for FP finding, etc.
# This is just used to make it compatible with things that use a forward function directly.
class ModelWrapper(nn.Module):
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn
        self.batch_first = True

    def forward(self, h0):
        return self.rnn.eval_autonomous_simple(h0)
