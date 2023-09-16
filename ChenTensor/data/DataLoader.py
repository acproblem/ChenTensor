from .Dataset import Dataset
import random
from ..functional import _functional as f


class DataLoader:
    """
    This is a class for loading data in small batches.

    Attributes:
        dataset : Dataset
        batch_size : int

    Methods:
        __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) : Constructor.
        __iter__(self) : Return a iterable object.
        __next__(self) : Get the next data.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        if batch_size <= 0:
            raise RuntimeError("The parameter `batch_size` must be greater than zero.")

        self.dataset = dataset
        self.batch_size = batch_size
        self._pos = 0  # 当前迭代的位置
        self.idxs = [i for i in range(len(dataset))]  # 数据索引，记录了迭代次序
        if shuffle:
            random.shuffle(self.idxs)

    def __iter__(self):
        """Return a iterable object."""
        return self

    def __next__(self):
        """Get the next data."""
        if self._pos < len(self.dataset):
            batch = [self.dataset[j] for j in self.idxs[self._pos: min(self._pos + self.batch_size, len(self.dataset))]]
            res = tuple(f.union_tensor([batch[row][col] for row in range(len(batch))]) for col in range(len(batch[0])))
            self._pos += self.batch_size
            return res
        else:
            self._pos = 0
            raise StopIteration
