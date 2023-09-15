from ._network import *
from .._tensor import Dtype

class Network:
    def __init__(self):
        self._mode = 'train'  # 训练模式，两种模式：'train', 'eval'

    def forward(self, inputs):
        pass

    def parameters(self):
        paras = []
        for key, value in self.__dict__.items():
            if isinstance(value, (Network, NetworkFloat32, NetworkFloat64, NetworkInt32, NetworkInt64)):
                paras.extend(value.parameters())
        return paras

    def __call__(self, inputs):
        return self.forward(inputs)

    def type(self, inputs):
        return NetType.Undefined

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def __repr__(self):
        return self.__str__()

    @property
    def mode(self):
        return self._mode

    def train(self):
        for key, value in self.__dict__.items():
            if isinstance(value, (Network, NetworkFloat32, NetworkFloat64, NetworkInt32, NetworkInt64)):
                value.train()
        self._mode = 'train'

    def eval(self):
        for key, value in self.__dict__.items():
            if isinstance(value, (Network, NetworkFloat32, NetworkFloat64, NetworkInt32, NetworkInt64)):
                value.eval()
        self._mode = 'eval'