# Defining what an operator is in general. 
from abc import ABC, abstractmethod

class Operator(ABC):
    @abstractmethod
    def __init__(self):
        pass 

    # The input q is shape [..., B, T, H] = [..., batch count, timesteps, hidden count]. Compute action, which has same shape as q.
    @abstractmethod
    def __call__(self, q):
        pass

    def rayleigh_coef(self, q):
        Kq = self(q) # [..., B, T, H]
        return (Kq * q).sum((-3, -2, -1)) / (q * q).sum((-3, -2, -1))

    def to_scipy(self, shape):
        # Convert to a scipy LinearOperator. Shape should be the shape of a typical input to __call__.
        from scipy import LinearOperator
        from math import prod
        shape_flat = prod(shape)
        flat_action = lambda q_flat: self(q_flat.reshape(shape)).flatten()
        return LinearOperator((shape_flat, shape_flat), flat_action, dtype = float)

class ComposedOperator(Operator):
    def __init__(self, op1, op2):
        # Define operator op1 * op2.
        self.op1 = op1
        self.op2 = op2

    def __call__(self, q):
        return self.op1(self.op2(q))
