import torch
from torchdiffeq import odeint
def adjoint_calculate(t, y, func, method, tol, user_adjoint = None):
    ''' Adapted from torchdiffeq so that I can plot and save values of adjoint. 
        Input is timesteps for eval, state over time (outputted from solver) and solver parameters. '''
    with torch.no_grad():
        adjoint_rtol, adjoint_atol = tol, tol 
        adjoint_method = method
        adjoint_params = tuple(list(func.parameters()))
        grad_y = torch.zeros_like(y)
        grad_y.fill_(1 / y.nelement()) # FOR MEAN TASK, WHERE GRAD IS JUST 1/TIMESTEPS.
        grad_y = torch.ones_like(grad_y)

        ##################################
        #      Set up initial state      #
        ##################################

        # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
        aug_state = [torch.zeros((), dtype=y.dtype, device=y.device), y[:, -1], grad_y[:, -1]]  # vjp_t, y, vjp_y
        aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params

        ##################################
        #    Set up backward ODE func    #
        ##################################

        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y = y_aug[1]
            adj_y = y_aug[2]
            # ignore gradients wrt time and parameters

            if user_adjoint is not None:
                adj_y = user_adjoint(t)

            with torch.enable_grad():
                t_ = t.detach()
                t = t_.requires_grad_(True)
                y = y.detach().requires_grad_(True)

                # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
                # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
                # wrt t here means we won't compute that if we don't need it.
                func_eval = func(t, y)

                vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                    func_eval, (t, y) + adjoint_params, -adj_y,
                    allow_unused=True, retain_graph=True
                )

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
            vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                          for param, vjp_param in zip(adjoint_params, vjp_params)]

            # J_t, f, J_y, J_theta. See eqns (49-50) in Neural ODEs paper.
            return (vjp_t, func_eval, vjp_y, *vjp_params)

        ##################################
        #       Solve adjoint ODE        #
        ##################################

        # RECORD AUGMENTED SYSTEM OVER ODE.
        record = []
        time_vjps = None
        record.append(aug_state)
        for i in range(len(t) - 1, 0, -1):
            # Run the augmented system backwards in time.
            aug_state = odeint(
                augmented_dynamics, tuple(aug_state),
                t[i - 1:i + 1].flip(0),
                rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method
            )
            aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
            aug_state[1] = y[:, i - 1]  # update to use our forward-pass estimate of the state
            aug_state[2] += grad_y[:, i - 1]  # update any gradients wrt state at this time point
            record.append(aug_state)

    return record

def adjoint_calculate_RNN(t, y, func, grad_y, user_adjoint = None):
    ''' Adapted to RNN adjoint calculation. '''
    with torch.no_grad():
        adjoint_params = tuple(list(func.parameters()))

        ##################################
        #      Set up initial state      #
        ##################################

        # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
        aug_state = [torch.zeros((), dtype=y.dtype, device=y.device), y[:, -1], grad_y[:, -1]]  # vjp_t, y, vjp_y
        aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params

        ##################################
        #    Set up backward ODE func    #
        ##################################

        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y = y_aug[1]
            adj_y = y_aug[2]
            # ignore gradients wrt time and parameters

            if user_adjoint is not None:
                adj_y = user_adjoint(t)

            with torch.enable_grad():
                t_ = t.detach()
                t = t_.requires_grad_(True)
                y = y.detach().requires_grad_(True)

                # Evaluate f(z).
                func_eval = func(t, y)

                # a(t) = Df_z^T * a(t+1). Note no minus sign for RNN update!
                # Nicely, autograd.grad computes the Jacobians then products them in one step (third parameter adj_y).
                vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                    func_eval, (t, y) + adjoint_params, adj_y,
                    allow_unused=True, retain_graph=True
                )

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
            vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                          for param, vjp_param in zip(adjoint_params, vjp_params)]

            # RNN SPECIAL CASE: WE NEED TO ACCUMULATE HERE.
            vjp_t += y_aug[0]
            vjp_params = [cur + vjp_param for cur, vjp_param in zip(y_aug[3:], vjp_params)]

            # J_t, f, J_y, J_theta. See eqns (49-50) in Neural ODEs paper.
            return (vjp_t, func_eval, vjp_y, *vjp_params)

        ##################################
        #       Solve adjoint ODE        #
        ##################################

        # RECORD AUGMENTED SYSTEM OVER ODE.
        record = []
        time_vjps = None
        record.append(aug_state)
        for i in range(len(t) - 1, 0, -1):
            # Run the augmented system backwards in time.
            # In RNN CASE, no odeint involved!
            aug_state = augmented_dynamics(t[i], tuple(aug_state))
            aug_state = list(aug_state)
            aug_state[1] = y[:, i - 1]  # update to use our forward-pass estimate of the state
            aug_state[2] += grad_y[:, i - 1]  # For accumulation task where loss involves every timestep, we need to add dL/dz(t) contribution from the straight loss to the term. For mean task, this is just 1/timesteps.
            record.append(aug_state)

    return record
