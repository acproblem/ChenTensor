from .network import *


class BatchNorm1D(Network):
    """
    This is Batch Normalization 1-D Layer class that inherit Network.

    Attributes:
        num_features : int
            The number of features.
        momentum : float
            It is used to compute mean and variance of running time.
        eps : float
            A value for making the denominator not zero.

    Methods:
        __init__(self, num_features, eps=1e-5, momentum=0.9, dtype=Dtype.float32) : Constructor.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.9, dtype=Dtype.float32):
        """
        Constructor.

        Parameters:
            num_features : int
                The number of features.
            momentum : float
                It is used to compute mean and variance of running time.
            eps : float
                A value for making the denominator not zero.
            dtype : ChenTensor.Dtype
                Data type.
        """
        super().__init__()

        if dtype == Dtype.float32:
            self._net = BatchNorm1DFloat32(num_features, eps, momentum)
        elif dtype == Dtype.float64:
            self._net = BatchNorm1DFloat64(num_features, eps, momentum)
        elif dtype == Dtype.int32:
            self._net = BatchNorm1DInt32(num_features, eps, momentum)
        elif dtype == Dtype.int64:
            self._net = BatchNorm1DInt64(num_features, eps, momentum)
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
        return self._net.forward(inputs) if self.mode == "train" else self._net.eval_forward(inputs)

    def parameters(self):
        return self._net.parameters()

    def type(self):
        return NetType.BatchNorm1D

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    @property
    def num_features(self):
        return self._net.num_features

    @property
    def momentum(self):
        return self._net.momentum

    @property
    def eps(self):
        return self._net.eps

    def __str__(self):
        return f"BatchNorm1D(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"
