from .network import *

class ReLU(Network):
    """
    This is ReLU Layer class that inherit Network.
        f(x) = x if x >= 0 else 0

    Attributes:

    Methods:
        __init__(self, dtype=Dtype.float32) : Constructor.
    """
    def __init__(self, dtype=Dtype.float32):
        """
        Constructor.

        Parameters:
            dtype : ChenTensor.Dtype
                Data type.
        """
        super().__init__()
        if dtype == Dtype.float32:
            self._net = ReLUFloat32()
        elif dtype == Dtype.float64:
            self._net = ReLUFloat64()
        elif dtype == Dtype.int32:
            self._net = ReLUInt32()
        elif dtype == Dtype.int64:
            self._net = ReLUInt64()
        else:
            raise RuntimeError("Please pass correct dtype.")

    def forward(self, inputs):
        return self._net.forward(inputs)

    def parameters(self):
        return self._net.parameters()

    def type(self):
        return NetType.ReLU

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    def __str__(self):
        return f"ReLU()"
