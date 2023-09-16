from .network import *
from .RNNBase import *
from .._tensor import Dtype, tensor
from .. import functional as f
from .ReLU import *
from .Tanh import *
from .Sigmoid import *
from .Dropout import *
import numpy as np


class RNN(RNNBase):
    """
    This is RNN cell class that inherit RNNBase.

    Attributes:
        activation : str
            "tanh" or "sigmoid" or "relu".

    Methods:
        __init__(self, input_size, hidden_size, activation='tanh', bias=True, dtype=Dtype.float32) : Constructor.
    """
    def __init__(self, input_size, hidden_size, activation='tanh', bias=True, dtype=Dtype.float32):
        """
        Constructor.

        Parameters:
            input_size : int
                The number of input's features.
            hidden_size : int
                The number of hidden data's features.
            activation : str
                "tanh" or "sigmoid" or "relu".
            bias : bool
                Whether offset item is required.
            dtype : ChenTensor.Dtype
                Data type.
        """
        super().__init__(input_size, hidden_size, bias, dtype)

        if activation not in ['tanh', 'relu', 'sigmoid']:
            raise RuntimeError("The parameter `activation` must be one of ['tanh'. 'relu', 'sigmoid'].")

        # 网络信息
        self._activation = Tanh() if activation == 'tanh' else ReLU() if activation == 'relu' else Sigmoid()

        # 可学习参数
        def _init_para(shape):  # 初始化参数
            return tensor(np.random.uniform(-np.sqrt(1 / hidden_size), np.sqrt(1 / hidden_size), shape), dtype, True)

        self._Whh = _init_para([hidden_size, hidden_size])
        self._Wih = _init_para([input_size, hidden_size])
        if bias:
            self._bh = _init_para(hidden_size)

    def forward(self, inputs, hidden):
        hidden = f.mm(hidden, self._Whh) + f.mm(inputs, self._Wih)
        if self._requires_bias:
            hidden = hidden + self._bh
        hidden = self._activation(hidden)

        return hidden

    def parameters(self):
        return [self._Whh, self._Wih, self._bh]

    def type(self):
        return NetType.RNN

    @property
    def activation(self):
        return 'tanh' if self._activation.type() == NetType.Tanh else \
            'relu' if self._activation.type() == NetType.ReLU else 'sigmoid'

    def __str__(self):
        return f"RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, " + \
            f"activation='{self.activation}', bias={self.requires_bias}, dtype={self._dtype})"

# import ChenTensor as ct
# from ChenTensor import network
# import numpy as np
#
# a = ct.tensor(np.arange(20).reshape([4, 5]), dtype=ct.float32)
# h = ct.tensor(np.arange(24).reshape([4, 6]), dtype=ct.float32)
# net = network.RNN(5, 6)
#
# net(a, h)
