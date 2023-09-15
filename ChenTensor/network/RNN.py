from .network import *
from .._tensor import Dtype, tensor
from .. import functional as f
from .ReLU import *
from .Tanh import *
from .Sigmoid import *
from .Dropout import *
import numpy as np


class RNN(Network):
    def __init__(self, input_size, hidden_size, output_size, activation='tanh', bias=True, dropout=0,
                 dtype=Dtype.float32):
        super().__init__()
        if type(input_size) is not int or type(hidden_size) is not int or type(output_size) is not int:
            raise RuntimeError("The parameters `input_size`, `hidden_size` and `output_size` must be `int`.")
        if activation not in ['tanh', 'relu', 'sigmoid']:
            raise RuntimeError("The parameter `activation` must be one of ['tanh'. 'relu', 'sigmoid'].")
        if type(bias) is not bool:
            raise RuntimeError("The parameter `bias` must be `bool`.")
        if not isinstance(dropout, (int, float)):
            raise RuntimeError("The parameter `dropout` must be `int` or `float`.")
        if not 0 <= dropout <= 1:
            raise RuntimeError("The parameter `dropout` must be in [0, 1].")

        # 网络信息
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._activation = Tanh() if activation == 'tanh' else ReLU() if activation == 'relu' else Sigmoid()
        self._requires_bias = bias
        self._dropout = Dropout(dropout) if dropout > 0 else None
        self._dtype = dtype

        # 可学习参数
        def _init_para(shape):  # 初始化参数
            return tensor(np.random.uniform(-np.sqrt(1 / hidden_size), np.sqrt(1 / hidden_size), shape), dtype, True)

        self._Whh = _init_para([hidden_size, hidden_size])
        self._Wih = _init_para([input_size, hidden_size])
        self._Who = _init_para([hidden_size, output_size])
        if bias:
            self._bh = _init_para(hidden_size)
            self._bo = _init_para(output_size)

    def forward(self, inputs, hidden):
        """
        RNN forward propagation.
        :param inputs: shape[batch_size, input_size]
        :param hidden: shape[batch_size, hidden_size]
        :return: (output, hidden)
            output: shape[batch_size, output_size]
            hidden: shape[batch_size, hidden_size]
        """
        hidden = f.mm(hidden, self._Whh) + f.mm(inputs, self._Wih)
        if self._requires_bias:
            hidden = hidden + self._bh
        if self._dropout:
            hidden = self._dropout(hidden)
        hidden = self._activation(hidden)

        output = f.mm(hidden, self._Who)
        if self._requires_bias:
            output = output + self._bo
        output = self._activation(output)

        return output, hidden

    def __call__(self, inputs, hidden):
        return self.forward(inputs, hidden)

    def parameters(self):
        return [self._Whh, self._Wih, self._Who, self._bh, self._bo]

    def type(self):
        return NetType.RNN

    def train(self):
        self._dropout.train()
        self._mode = 'train'

    def eval(self):
        self._dropout.eval()
        self._mode = 'eval'

    @property
    def input_size(self):
        return self._input_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def activation(self):
        return 'tanh' if self._activation.type() == NetType.Tanh else \
            'relu' if self._activation.type() == NetType.ReLU else 'sigmoid'

    @property
    def requires_bias(self):
        return self._requires_bias

    @property
    def dropout(self):
        return self._dropout.probability if self._dropout is not None else 0

    def __str__(self):
        return f"RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, " + \
            f"output_size={self.output_size}, activation='{self.activation}', bias={self.requires_bias}, " + \
            f"dropout={self.dropout}, dtype={self._dtype})"

# import ChenTensor as ct
# from ChenTensor import network
# import numpy as np
#
# a = ct.tensor(np.arange(20).reshape([4, 5]), dtype=ct.float32)
# h = ct.tensor(np.arange(24).reshape([4, 6]), dtype=ct.float32)
# net = network.RNN(5, 6, 3, dropout=0.5)
#
# net(a, h)
