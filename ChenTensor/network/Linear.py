from .network import *


class Linear(Network):
    """
    This is Linear Layer class that inherit Network.

    Attributes:
        in_features : int
            The number of input's features.
        out_features : int
            The number of output's features.
        requires_bias : bool
            Whether offset item is required.
        weight : tensor (TensorFloat32 or TensorFloat64 or TensorInt32 or TensorInt64)
            Weight term.
        bias : tensor (TensorFloat32 or TensorFloat64 or TensorInt32 or TensorInt64)
            Bias term.

    Methods:
        __init__(self, in_features, out_features, bias=True, dtype=Dtype.float32) : Constructor.
    """
    def __init__(self, in_features, out_features, bias=True, dtype=Dtype.float32):
        """
        Constructor.

        Parameters:
            in_features : int
                The number of input's features.
            out_features : int
                The number of output's features.
            bias : bool
                Whether offset item is required.
            dtype : ChenTensor.Dtype
                Data type.
        """
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
        """
        Forward propagation. Return calculation result.

        Parameters:
            inputs : tensor (TensorFloat32 or TensorFloat64 or TensorInt32 or TensorInt64)
                shape : [batch_size, num_features]

        Returns:
            tensor (TensorFloat32 or TensorFloat64 or TensorInt32 or TensorInt64)
                shape : [batch_size, num_features]
        """
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
