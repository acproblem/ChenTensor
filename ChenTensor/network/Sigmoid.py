from .network import *

class Sigmoid(Network):
    def __init__(self, dtype=Dtype.float32):
        super().__init__()
        if dtype == Dtype.float32:
            self._net = SigmoidFloat32()
        elif dtype == Dtype.float64:
            self._net = SigmoidFloat64()
        elif dtype == Dtype.int32:
            self._net = SigmoidInt32()
        elif dtype == Dtype.int64:
            self._net = SigmoidInt64()
        else:
            raise RuntimeError("Please pass correct dtype.")

    def forward(self, inputs):
        return self._net.forward(inputs)

    def parameters(self):
        return self._net.parameters()

    def type(self):
        return NetType.Sigmoid

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    def __str__(self):
        return "Sigmoid()"
