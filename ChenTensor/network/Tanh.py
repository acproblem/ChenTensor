from .network import *

class Tanh(Network):
    """
    This is Tanh Layer class that inherit Network.
        f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

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
            self._net = TanhFloat32()
        elif dtype == Dtype.float64:
            self._net = TanhFloat64()
        elif dtype == Dtype.int32:
            self._net = TanhInt32()
        elif dtype == Dtype.int64:
            self._net = TanhInt64()
        else:
            raise RuntimeError("Please pass correct dtype.")

    def forward(self, inputs):
        return self._net.forward(inputs)

    def parameters(self):
        return self._net.parameters()

    def type(self):
        return NetType.Tanh

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    def __str__(self):
        return "LeakyReLU()"
