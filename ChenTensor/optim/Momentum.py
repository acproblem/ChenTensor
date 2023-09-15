from ._optim import *
from .._tensor import *


class Momentum:
    def __init__(self, paras, lr=0.01, momentum=0.9):
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
            self.opts.append(MomentumFloat32(paras_float32, lr, momentum))
        if len(paras_float64) > 0:
            self.opts.append(MomentumFloat64(paras_float64, lr, momentum))
        if len(paras_int32) > 0:
            self.opts.append(MomentumInt32(paras_int32, lr, momentum))
        if len(paras_int64) > 0:
            self.opts.append(MomentumInt64(paras_int64, lr, momentum))

    def step(self):
        for opt in self.opts:
            opt.step()

    def zero_grad(self):
        for opt in self.opts:
            opt.zero_grad()
