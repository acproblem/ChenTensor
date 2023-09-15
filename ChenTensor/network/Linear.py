from .network import *


class Linear(Network):
    def __init__(self, in_features, out_features, bias=True, dtype=Dtype.float32):
        super().__init__()
        if dtype == Dtype.float32:
            self._net = LinearFloat32(in_features, out_features, bias)
        elif dtype == Dtype.float64:
            self._net = LinearFloat64(in_features, out_features, bias)
        elif dtype == Dtype.int32:
            self._net = LinearInt32(in_features, out_features, bias)
        elif dtype == Dtype.int64:
            self._net = LinearInt64(in_features, out_features, bias)
        else:
            raise RuntimeError("Please pass correct dtype.")

    def forward(self, inputs):
        return self._net.forward(inputs)

    def parameters(self):
        return self._net.parameters()

    def type(self):
        return NetType.Linear

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    @property
    def requires_bias(self):
        return self._net.requires_bias

    @property
    def weight(self):
        return self._net.weight

    @property
    def bias(self):
        return self._net.bias

    @property
    def in_features(self):
        return self._net.in_features

    @property
    def out_features(self):
        return self._net.out_features

    def __str__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.requires_bias})"
