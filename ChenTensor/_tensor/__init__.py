from ._tensor import *


# Create a tensor
def tensor(arr=0.0, dtype=float32, requires_grad=False):
    """
    Create a tensor.

    Parameters:
        arr : numpy.ndarray, list, int, float
        dtype : ChenTensor.Dtype, default = ChenTensor.float32
        requires_grad : bool, default = False
            Create a tensor with gradient if requires_grad is True.

    Returns:
        TensorFloat32 if dtype is ChenTensor.float32.
        TensorFloat64 if dtype is ChenTensor.float64.
        TensorInt32 if dtype is ChenTensor.int32.
        TensorInt64 if dtype is ChenTensor.int64.

    Example:
        >>> import ChenTensor as ct
        >>> import numpy as np
        >>> ct.tensor([0, 1, 2])
        { 0.,  1.,  2.}
        >>> ct.tensor(1)
         1.
        >>> ct.tensor(np.array([1, 2, 3]), dtype=ct.int32, requires_grad=True)
        {1, 2, 3}
    """

    if dtype == float32:
        return TensorFloat32(arr, requires_grad)
    elif dtype == float64:
        return TensorFloat64(arr, requires_grad)
    elif dtype == int32:
        return TensorInt32(arr, requires_grad)
    elif dtype == int64:
        return TensorInt64(arr, requires_grad)
    else:
        raise RuntimeError("Please pass correct dtype.")

