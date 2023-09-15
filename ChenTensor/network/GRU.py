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
    def __init__(self, input_size, hidden_size, output_size, bias=True, dropout=0, dtype=Dtype.float32):
        super().__init__(input_size, hidden_size, output_size, bias, dropout, dtype)

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
        self._Who = _init_para([hidden_size, output_size])
        if bias:
            self._br = _init_para(hidden_size)
            self._bz = _init_para(hidden_size)
            self._bhn = _init_para(hidden_size)
            self._bin = _init_para(hidden_size)
            self._bo = _init_para(output_size)

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

        if self._dropout:
            hidden = self._dropout(hidden)

        output = f.mm(hidden, self._Who)
        if self._requires_bias:
            output = output + self._bo
        output = self._tanh(output)

        return output, hidden

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
# net = network.GRU(5, 6, 3, dropout=0.5)
#
# net(a, h)
