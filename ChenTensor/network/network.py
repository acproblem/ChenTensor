from ._network import *
from .._tensor import Dtype

class Network:
    """
    This is a network abstract class. All custom network classes must inherit this class.
    Subclasses need to implement the `forward` method.

    Attributes:
        mode : str
            "train" or "eval".

    Methods:
        __init__(self) : Constructor.
        forward(self, inputs) : Forward propagation. Return calculation result.
        __call__(self, inputs) : Equivalent to the `forward` method.
        parameters(self) : Get learnable parameters. Subclasses can override it.
        type(self) : Get type of network.
        __str__(self) : Return a description for object.
        __repr__(self) : Equivalent to the `__str__` method.
        train(self) : Set the network to training mode.
        eval(self) : Set the network to evaluation mode.
    """
    def __init__(self):
        """
        Constructor.

        Parameters:

        """
        self._mode = 'train'  # 训练模式，两种模式：'train', 'eval'

    def forward(self, inputs):
        """
        Forward propagation. Return calculation result.

        Parameters:
            inputs : Tensor

        Returns:
            Tensor
        """
        pass

    def parameters(self):
        """
        Get learnable parameters. Subclasses can override it.

        Parameters:

        Returns:
            list of tensors : Learnable parameters.
        """
        paras = []
        for key, value in self.__dict__.items():
            if isinstance(value, (Network, NetworkFloat32, NetworkFloat64, NetworkInt32, NetworkInt64)):
                paras.extend(value.parameters())
        return paras

    def __call__(self, inputs):
        """Equivalent to the `forward` method."""
        return self.forward(inputs)

    def type(self):
        """Get type of network."""
        return NetType.Undefined

    def __str__(self):
        """Return a description for object."""
        return f"{self.__class__.__name__}()"

    def __repr__(self):
        """Equivalent to the `__str__` method."""
        return self.__str__()

    @property
    def mode(self):
        return self._mode

    def train(self):
        """Set the network to training mode."""
        for key, value in self.__dict__.items():
            if isinstance(value, (Network, NetworkFloat32, NetworkFloat64, NetworkInt32, NetworkInt64)):
                value.train()
        self._mode = 'train'

    def eval(self):
        """Set the network to evaluation mode."""
        for key, value in self.__dict__.items():
            if isinstance(value, (Network, NetworkFloat32, NetworkFloat64, NetworkInt32, NetworkInt64)):
                value.eval()
        self._mode = 'eval'