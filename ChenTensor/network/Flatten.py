from .network import *


class Flatten(Network):
    """
    This is Flatten Layer class that inherit Network.
    It flattens the data with dimensions within [start_dim, end_dim].

    Attributes:
        start_dim : int
            The start dimension.
        end_dim : int
            The end dimension.

    Methods:
        __init__(self, start_dim=1, end_dim=-1, dtype=Dtype.float32) : Constructor.
    """
    def __init__(self, start_dim=1, end_dim=-1, dtype=Dtype.float32):
        """
        Constructor.

        Parameters:
            start_dim : int
                The start dimension.
            end_dim : int
                The end dimension.
            dtype : ChenTensor.Dtype
                Data type.
        """
        super().__init__()
        if dtype == Dtype.float32:
            self._net = FlattenFloat32(start_dim, end_dim)
        elif dtype == Dtype.float64:
            self._net = FlattenFloat64(start_dim, end_dim)
        elif dtype == Dtype.int32:
            self._net = FlattenInt32(start_dim, end_dim)
        elif dtype == Dtype.int64:
            self._net = FlattenInt64(start_dim, end_dim)
        else:
            raise RuntimeError("Please pass correct dtype.")

    def forward(self, inputs):
        """
        Forward propagation. Return calculation result.

        Parameters:
            inputs : Tensor

        Returns:
            Tensor
                shape : [batch_size, out_features]
        """
        return self._net.forward(inputs)

    def parameters(self):
        return self._net.parameters()

    def type(self):
        return NetType.Flatten

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    @property
    def start_dim(self):
        return self._net.start_dim

    @property
    def end_dim(self):
        return self._net.end_dim

    def __str__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"