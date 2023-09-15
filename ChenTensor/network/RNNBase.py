from .network import *
from .._tensor import Dtype, tensor
from .. import functional as f
from .ReLU import *
from .Tanh import *
from .Sigmoid import *
from .Dropout import *
import numpy as np


class RNNBase(Network):
    def __init__(self, input_size, hidden_size, output_size, bias=True, dropout=0, dtype=Dtype.float32):
        super().__init__()
        if type(input_size) is not int or type(hidden_size) is not int or type(output_size) is not int:
            raise RuntimeError("The parameters `input_size`, `hidden_size` and `output_size` must be `int`.")
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
        self._requires_bias = bias
        self._dropout = Dropout(dropout) if dropout > 0 else None
        self._dtype = dtype

    def forward(self, inputs, hidden):
        """
        RNNBase forward propagation.
        :param inputs: shape[batch_size, input_size]
        :param hidden: shape[batch_size, hidden_size]
        :return: (output, hidden)
            output: shape[batch_size, output_size]
            hidden: shape[batch_size, hidden_size]
        """
        pass

    def __call__(self, inputs, hidden):
        return self.forward(inputs, hidden)

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
    def requires_bias(self):
        return self._requires_bias

    @property
    def dropout(self):
        return self._dropout.probability if self._dropout is not None else 0

    def __str__(self):
        return f"RNNBase(input_size={self.input_size}, hidden_size={self.hidden_size}, " + \
            f"output_size={self.output_size}, bias={self.requires_bias}, " + \
            f"dropout={self.dropout}, dtype={self._dtype})"
