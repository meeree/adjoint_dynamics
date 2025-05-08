import torch
import numpy as np
import matplotlib.pyplot as plt
import utils

def test_linearized_propagation(plot = False):
    # First, test on a linear ODE model, z_{n+1} = W * z_n + x(t). 
    # This has dynamics z_n = Phi(n, 0) z_0 + sum_{i=1}^n Phi(n, i) x_i,
    # where Phi(n, n0) = W^{n - n0}.
    from analysis.propagation_op_general import PropagationOperator_LinearForm as POLF

    B, T, H = 5, 20, 10 # Batch size, Timesteps, Hidden count.
    x = torch.randn(B, T, H) * 1e-3 # Inputs.
    W = torch.randn(10, 10) / (H**0.5)
    model_f = lambda x, z: z @ W.T + x 

    # Simulate an example.
    hidden = [torch.zeros((B, H))]
    for t in range(T):
        hidden.append(model_f(x[:, t], hidden[-1]).clone())
    hidden = torch.stack(hidden[1:], 1) # [B, T, H]

    if plot:
        plt.plot(hidden[0, :, :])
        plt.show()

    polf = POLF(model_f, 0.*x, hidden)

    print(" ----- ")
    print("Jacobian Matrices Relative Error:")
    true_jac = torch.zeros((B, T, H, H))
    true_jac[:, :] = W
    print(utils.relative_error(true_jac, polf.jacs))
    print(" ----- ")

    print("Fundamental Matrices Relative Error:")
    true_U = torch.zeros((B, T+1, H, H))
    for t in range(T+1):
        true_U[:, t] = torch.linalg.matrix_power(W, t)
    print(utils.relative_error(true_U, polf.Us)) 
    print(" ----- ")

    print("Trajectory Reconstruction Relative Error:")
    guess_z = polf(x) # Feeding x into the state-transition form should give perfect reconstruction since it's a linear ODE.
    print(utils.relative_error(guess_z, hidden)) 
    print(" ----- ")


if __name__ == '__main__':
    test_linearized_propagation()
