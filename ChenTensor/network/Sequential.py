from .network import *

class Sequential(Network):
    def __init__(self, net_lst):
        super().__init__()
        if not isinstance(net_lst, (list, tuple)):
            raise TypeError("The parameter `net_lst` must be a list (or tuple) of Networks.")

        for i, net in enumerate(net_lst):
            if not isinstance(net, (Network, NetworkFloat32, NetworkFloat64, NetworkInt32, NetworkInt64)):
                raise TypeError(f"The {i}-th parameter is not a Network.")

        self._net_lst = net_lst

    def forward(self, inputs):
        for net in self._net_lst:
            inputs = net(inputs)
        return inputs

    def parameters(self):
        paras = []
        for net in self._net_lst:
            paras.extend(net.parameters())
        return paras

    def type(self):
        return NetType.Sequential

    def get(self, idx):
        return self._net_lst[idx]

    def size(self):
        return len(self._net_lst)

    def train(self):
        for net in self._net_lst:
            net.train()
        self._mode = 'train'

    def eval(self):
        for net in self._net_lst:
            net.train()
        self._mode = 'eval'

    def __str__(self):
        return f"Sequential(size={self.size()}) {{\n" +\
            "".join(["\t" + str(net) + "\n" for net in self._net_lst]) + "}"
