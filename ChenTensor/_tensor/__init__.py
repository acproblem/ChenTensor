from ._tensor import *

# Create a tensor
def tensor(arr=0.0, dtype=float32, requires_grad=False):
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

# __all__ = ["tensor", "Dtype", ""]
