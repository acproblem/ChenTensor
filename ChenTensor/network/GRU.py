from .network import *
from .RNNBase import *
from .._tensor import Dtype, tensor
from .. import functional as f
from .ReLU import *
from .Tanh import *
from .Sigmoid import *
from .Dropout import *
import numpy as np


class GRU(RNNBase):
    """
    This is GRU cell class that inherit RNNBase.

    Attributes:

    Methods:
        __init__(self, input_size, hidden_size, activation='tanh', bias=True, dtype=Dtype.float32) : Constructor.
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
        super().__init__(input_size, hidden_size, bias, dtype)

        # 激活函数
        self._sigmoid = Sigmoid()
        self._tanh = Tanh()

        # 可学习参数
        def _init_para(shape):  # 初始化参数
            return tensor(np.random.uniform(-np.sqrt(1 / hidden_size), np.sqrt(1 / hidden_size), shape), dtype, True)

        self._Whr = _init_para([hidden_size, hidden_size])
        self._Wir = _init_para([input_size, hidden_size])
        self._Whz = _init_para([hidden_size, hidden_size])
        self._Wiz = _init_para([input_size, hidden_size])
        self._Whn = _init_para([hidden_size, hidden_size])
        self._Win = _init_para([input_size, hidden_size])
        if bias:
            self._br = _init_para(hidden_size)
            self._bz = _init_para(hidden_size)
            self._bhn = _init_para(hidden_size)
            self._bin = _init_para(hidden_size)

    def forward(self, inputs, hidden):
        r = f.mm(hidden, self._Whr) + f.mm(inputs, self._Wir)
        z = f.mm(hidden, self._Whz) + f.mm(inputs, self._Wiz)
        if self._requires_bias:
            r = r + self._br
            z = z + self._bz

        r = self._sigmoid(r)
        z = self._sigmoid(z)

        n = f.mm(hidden, self._Whn)
        if self._requires_bias:
            n = n + self._bhn
        n = r * n + f.mm(inputs, self._Win)
        if self._requires_bias:
            n = n + self._bin
        n = self._tanh(n)

        hidden = (tensor(1) - z) * n + z * hidden

        return hidden

    def parameters(self):
        return [self._Whr, self._Wir, self._Whz, self._Wiz, self._Whn, self._Win,
                self._br, self._bz, self._bhn, self._bin]

    def type(self):
        return NetType.GRU

    def __str__(self):
        return f"GRU(input_size={self.input_size}, hidden_size={self.hidden_size}, " + \
            f"bias={self.requires_bias}, dtype={self._dtype})"

# import ChenTensor as ct
# from ChenTensor import network
# import numpy as np
#
# a = ct.tensor(np.arange(20).reshape([4, 5]), dtype=ct.float32)
# h = ct.tensor(np.arange(24).reshape([4, 6]), dtype=ct.float32)
# net = network.GRU(5, 6, 3, dropout=0.5)
#
# net(a, h)
