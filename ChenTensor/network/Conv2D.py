from .network import *


class Conv2D(Network):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, padding_value=0,
                 bias=True, dtype=Dtype.float32):
        super().__init__()

        # 参数转化
        if type(kernel_size) is int:
            kernel_size = [kernel_size, kernel_size]
        if type(stride) is int:
            stride = [stride, stride]
        if type(padding) is int:
            padding = [padding, padding]
        if type(dilation) is int:
            dilation = [dilation, dilation]

        if dtype == Dtype.float32:
            self._net = Conv2DFloat32(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_value,
                                      bias)
        elif dtype == Dtype.float64:
            self._net = Conv2DFloat64(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_value,
                                      bias)
        elif dtype == Dtype.int32:
            self._net = Conv2DInt32(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_value,
                                    bias)
        elif dtype == Dtype.int64:
            self._net = Conv2DInt64(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_value,
                                    bias)
        else:
            raise RuntimeError("Please pass correct dtype.")

    def forward(self, inputs):
        return self._net.forward(inputs)

    def parameters(self):
        return self._net.parameters()

    def type(self):
        return NetType.Conv2D

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    @property
    def requires_bias(self):
        return self._net.requires_bias

    @property
    def weight(self):
        return self._net.weight

    @property
    def bias(self):
        return self._net.bias

    @property
    def in_channels(self):
        return self._net.in_channels

    @property
    def out_channels(self):
        return self._net.out_channels

    @property
    def kernel_size(self):
        return self._net.kernel_size

    @property
    def stride(self):
        return self._net.stride

    @property
    def padding(self):
        return self._net.padding

    @property
    def dilation(self):
        return self._net.dilation

    @property
    def padding_value(self):
        return self._net.padding_value

    def __str__(self):
        return f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, " + \
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, " + \
            f"dilation={self.dilation}, padding_value={self.padding_value}, bias={self.requires_bias})"

