from .Dataset import Dataset
from .._tensor import tensor


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        super().__init__()
        if len(tensors) == 0:
            raise RuntimeError("Pass at least one parameter while instantiating class `TensorDataset`.")
        if any(tensors[0].shape()[0] != t.shape()[0] for t in tensors):
            raise RuntimeError("Size of tensors mismatch.")
        self.tensors = tensors

    def __getitem__(self, item):
        return tuple(t[item] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape()[0]
