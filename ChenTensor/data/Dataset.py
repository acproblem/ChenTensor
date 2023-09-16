

class Dataset:
    """
    This is a abstract class. All custom dataset classes must inherit this class.
    Subclasses need to implement the `__getitem__` method and the `__len__` method.

    Attributes:

    Methods:
        __getitem__(self, item) : Get tensor.
        __len__(self) : Get the number of data.
    """
    def __init__(self):
        pass

    def __getitem__(self, item):
        """
        Get tensor.

        Parameters:
            item : int

        Returns:
            tensor (TensorFloat32 or TensorFloat64 or TensorInt32 or TensorInt64)
        """
        pass

    def __len__(self):
        """
        Get the number of data.

        Parameters:

        Returns:
            int : The number of data.
        """
        pass