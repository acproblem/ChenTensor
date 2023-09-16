from ._optim import *
from .._tensor import *


class RMSprop:
    """
    Momentum gradient descent optimizer

    Attributes:

    Methods:
        __init__(self, paras, lr=0.01, alpha=0.99, eps=1e-8) : Constructor.
        step(self) : Update learnable parameters.
        zero_grad(self) : Clear the parameter gradient to zero.
    """
    def __init__(self, paras, lr=0.01, alpha=0.99, eps=1e-8):
        """
        Constructor.

        Parameters:
            paras : list of tensor
                List of learnable parameters.
            lr : float or int
                learning rate.
            alpha : float or int
            eps : float
                A value for making the denominator not zero.
        """
        if not isinstance(paras, list):
            raise TypeError("The parameter `paras` must be a list of tensor.")

        paras_float32, paras_float64, paras_int32, paras_int64 = [], [], [], []
        for i, para in enumerate(paras):
            if isinstance(para, TensorFloat32):
                paras_float32.append(para)
            elif isinstance(para, TensorFloat64):
                paras_float64.append(para)
            elif isinstance(para, TensorInt32):
                paras_int32.append(para)
            elif isinstance(para, TensorInt64):
                paras_int64.append(para)
            else:
                raise TypeError(f"The {i}-th variable of `paras` is not a tensor.")

        self.opts = []
        if len(paras_float32) > 0:
            self.opts.append(RMSpropFloat32(paras_float32, lr, alpha, eps))
        if len(paras_float64) > 0:
            self.opts.append(RMSpropFloat64(paras_float32, lr, alpha, eps))
        if len(paras_int32) > 0:
            self.opts.append(RMSpropInt32(paras_float32, lr, alpha, eps))
        if len(paras_int64) > 0:
            self.opts.append(RMSpropInt64(paras_float32, lr, alpha, eps))

    def step(self):
        """Update learnable parameters."""
        for opt in self.opts:
            opt.step()

    def zero_grad(self):
        """Clear the parameter gradient to zero."""
        for opt in self.opts:
            opt.zero_grad()
