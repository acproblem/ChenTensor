from .network import *


class Dropout(Network):
    """
    This is Dropout Layer class that inherit Network.

    Attributes:
        probability : float
            The dropout probability.

    Methods:
        __init__(self, p=0.5, dtype=Dtype.float32) : Constructor.
    """
    def __init__(self, p=0.5, dtype=Dtype.float32):
        """
        Constructor.

        Parameters:
            p : int or float
                The dropout probability. It must in [0, 1].
            dtype : ChenTensor.Dtype
                Data type.
        """
        super().__init__()
        if dtype == Dtype.float32:
            self._net = DropoutFloat32(p)
        elif dtype == Dtype.float64:
            self._net = DropoutFloat64(p)
        elif dtype == Dtype.int32:
            self._net = DropoutInt32(p)
        elif dtype == Dtype.int64:
            self._net = DropoutInt64(p)
        else:
            raise RuntimeError("Please pass correct dtype.")

    def forward(self, inputs):
        return self._net.forward(inputs) if self._mode == 'train' else inputs

    def parameters(self):
        return self._net.parameters()

    def type(self):
        return NetType.Dropout

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    @property
    def probability(self):
        return self._net.probability

    def __str__(self):
        return f"Dropout(p={self.probability})"
