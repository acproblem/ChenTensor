from .network import *


class LeakyReLU(Network):
    def __init__(self, alpha=0.01, dtype=Dtype.float32):
        super().__init__()
        if dtype == Dtype.float32:
            self._net = LeakyReLUFloat32(alpha)
        elif dtype == Dtype.float64:
            self._net = LeakyReLUFloat64(alpha)
        elif dtype == Dtype.int32:
            self._net = LeakyReLUInt32(alpha)
        elif dtype == Dtype.int64:
            self._net = LeakyReLUInt64(alpha)
        else:
            raise RuntimeError("Please pass correct dtype.")

    def forward(self, inputs):
        return self._net.forward(inputs)

    def parameters(self):
        return self._net.parameters()

    def type(self):
        return NetType.LeakyReLU

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    @property
    def alpha(self):
        return self._net.alpha

    def __str__(self):
        return f"LeakyReLU(alpha={self.alpha})"
