from .network import *


class MaxPool2D(Network):
    """
    This is Maximum Pooling 2-D Layer class that inherit Network.

    Attributes:
        kernel_size : tuple or list of int (length = 2)
            Pooling kernel size.
        stride : tuple or list of int (length = 2)
            Pooling kernel step size.
        padding : tuple or list of int (length = 2)
            Padding size.
        dilation : tuple or list of int (length = 2)
            Dilation size.

    Methods:
        __init__(self, kernel_size, stride=1, padding=0, dilation=1, dtype=Dtype.float32) : Constructor.
    """
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, dtype=Dtype.float32):
        """
        Constructor.

        Parameters:
            kernel_size : tuple or list of int (length = 2)
                Pooling kernel size.
            stride : tuple or list of int (length = 2)
                Pooling kernel step size.
            padding : tuple or list of int (length = 2)
                Padding size.
            dilation : tuple or list of int (length = 2)
                Dilation size.
            dtype : ChenTensor.Dtype
                Data type.
        """
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
            self._net = MaxPool2DFloat32(kernel_size, stride, padding, dilation)
        elif dtype == Dtype.float64:
            self._net = MaxPool2DFloat64(kernel_size, stride, padding, dilation)
        elif dtype == Dtype.int32:
            self._net = MaxPool2DInt32(kernel_size, stride, padding, dilation)
        elif dtype == Dtype.int64:
            self._net = MaxPool2DInt64(kernel_size, stride, padding, dilation)
        else:
            raise RuntimeError("Please pass correct dtype.")

    def forward(self, inputs):
        """
        Forward propagation. Return calculation result.

        Parameters:
            inputs : Tensor
                shape : [batch_size, in_channels, input_height, input_width]

        Returns:
            Tensor
                shape : [batch_size, out_channels, output_height, output_width]
        """
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

    def __str__(self):
        return f"MaxPool2D(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, " + \
            f"dilation={self.dilation})"
