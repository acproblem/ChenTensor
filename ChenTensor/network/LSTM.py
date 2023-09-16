from .network import *
from .RNNBase import *
from .._tensor import Dtype, tensor
from .. import functional as f
from .ReLU import *
from .Tanh import *
from .Sigmoid import *
from .Dropout import *
import numpy as np


class LSTM(RNNBase):
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

        self._Whi = _init_para([hidden_size, hidden_size])
        self._Wii = _init_para([input_size, hidden_size])
        self._Whf = _init_para([hidden_size, hidden_size])
        self._Wif = _init_para([input_size, hidden_size])
        self._Whg = _init_para([hidden_size, hidden_size])
        self._Wig = _init_para([input_size, hidden_size])
        self._Who = _init_para([hidden_size, hidden_size])
        self._Wio = _init_para([input_size, hidden_size])
        if bias:
            self._bi = _init_para(hidden_size)
            self._bf = _init_para(hidden_size)
            self._bg = _init_para(hidden_size)
            self._bo = _init_para(hidden_size)

    def forward(self, inputs, hidden, c):
        i = f.mm(hidden, self._Whi) + f.mm(inputs, self._Wii)
        f_ = f.mm(hidden, self._Whf) + f.mm(inputs, self._Wif)
        g = f.mm(hidden, self._Whg) + f.mm(inputs, self._Wig)
        o = f.mm(hidden, self._Who) + f.mm(inputs, self._Wio)

        if self._requires_bias:
            i = i + self._bi
            f_ = f_ + self._bf
            g = g + self._bg
            o = o + self._bo

        i = self._sigmoid(i)
        f_ = self._sigmoid(f_)
        g = self._tanh(g)
        o = self._sigmoid(o)

        c = f_ * c + i * g
        hidden = o * self._tanh(c)

        return hidden, c

    def __call__(self, inputs, hidden, c):
        return self.forward(inputs, hidden, c)

    def parameters(self):
        return [self._Whr, self._Wir, self._Whz, self._Wiz, self._Whn, self._Win, self._Who,
                self._br, self._bz, self._bhn, self._bin, self._bo]

    def type(self):
        return NetType.GRU

    def __str__(self):
        return f"RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, " + \
            f"output_size={self.output_size}, bias={self.requires_bias}, " + \
            f"dropout={self.dropout}, dtype={self._dtype})"

# import ChenTensor as ct
# from ChenTensor import network
# import numpy as np
#
# a = ct.tensor(np.arange(20).reshape([4, 5]), dtype=ct.float32)
# h = ct.tensor(np.arange(24).reshape([4, 6]), dtype=ct.float32)
# c = ct.tensor(np.arange(24).reshape([4, 6]), dtype=ct.float32)
# net = network.LSTM(5, 6)
#
# net(a, h, c)
