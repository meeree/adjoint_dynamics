# Propagation operator from main paper. 
import torch
import numpy as np
from .op_common import Operator

# Utility function
np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

# Compute what the operator P does to a quantity q of shape (..., B, T, H) = (..., batches, timesteps, hidden neuron index).
# This is done by forwards(f + q) - forwards(f) where f is the orginal model right hand side and forwards simulates 
# the ode z'(t | x) = f(z(t|x), x(t), t) given the function f.
# Model_f is the time stepper for the model. For example, for an RNN it is RNNCell and for a GRU it is GRUCell.
class PropagatorOperator(Operator):
    def __init__(self, model_f, x, h0 = None, dev = 'cpu'):
        self.dev = dev
        self.model_f = model_f.to(dev)
        self.x = np_to_torch(x).to(dev) # [B, T, Nin]
        self.h0 = None if h0 is None else np_to_torch(h0).to(dev)

    def __call__(self, q):
        # q is shape (..., B, T, H) where ... can be any dimension.
        # Evaluate forwards(model_f) and forwards(model_f + q).
        with torch.no_grad():
            q_dev = np_to_torch(q).to(self.dev) # [B, T, H].

            eval1, eval2 = torch.zeros_like(q_dev), torch.zeros_like(q_dev)
            h1, h2 = torch.zeros_like(q_dev[:, 0]), torch.zeros_like(q_dev[:, 0])
            if self.h0 is not None:
                h1, h2 = self.h0.clone(), self.h0.clone()

            for t in range(q_dev.shape[1]):
                h1 = self.model_f(self.x[:, t], h1)
                h2 = self.model_f(self.x[:, t], h2 + q_dev[:, t])
                eval1[:, t], eval2[:, t] = h1.clone(), h2.clone()

        return torch_to_np(eval2 - eval1)
