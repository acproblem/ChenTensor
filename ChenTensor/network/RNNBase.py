from .network import *
from .._tensor import Dtype, tensor
from .. import functional as f
from .ReLU import *
from .Tanh import *
from .Sigmoid import *
from .Dropout import *
import numpy as np


class RNNBase(Network):
    """
    This is a RNN network abstract class. All custom RNN network classes must inherit this class.
    Subclasses need to implement the `forward` method.

    Attributes:
        input_size : int
            The number of input's features.
        hidden_size : int
            The number of hidden data's features.
        requires_bias : bool
            Whether offset item is required.

    Methods:
        __init__(self, input_size, hidden_size, bias=True, dtype=Dtype.float32) : Constructor.
    """
    def __init__(self, input_size, hidden_size, bias=True, dtype=Dtype.float32):
        """
        Constructor.

        Parameters:
            input_size : int
                The number of input's features.
            hidden_size : int
                The number of hidden data's features.
            bias : bool
                Whether offset item is required.
            dtype : ChenTensor.Dtype
                Data type.
        """
        super().__init__()
        if type(input_size) is not int or type(hidden_size) is not int:
            raise RuntimeError("The parameters `input_size` and `hidden_size` must be `int`.")
        if type(bias) is not bool:
            raise RuntimeError("The parameter `bias` must be `bool`.")

        # 网络信息
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._requires_bias = bias
        self._dtype = dtype

    def forward(self, inputs, hidden):
        """
        Forward propagation. Return calculation result.

        Parameters:
            inputs : Tensor
                Input of time t. shape : [batch_size, input_size]
            hidden : Tensor
                Hidden input of time t. shape : [batch_size, hidden_size]

        Returns:
            Tensor
                Hidden output of time t. shape : [batch_size, hidden_size]
        """
        pass

    def __call__(self, inputs, hidden):
        return self.forward(inputs, hidden)

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    @property
    def input_size(self):
        return self._input_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def requires_bias(self):
        return self._requires_bias

    def __str__(self):
        return f"RNNBase(input_size={self.input_size}, hidden_size={self.hidden_size}, " + \
            f"bias={self.requires_bias}, dtype={self._dtype})"
