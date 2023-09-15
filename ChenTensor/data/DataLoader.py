from .Dataset import Dataset
import random
from ..functional import _functional as f


class DataLoader:
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
        return self

    def __next__(self):
        if self._pos < len(self.dataset):
            batch = [self.dataset[j] for j in self.idxs[self._pos: min(self._pos + self.batch_size, len(self.dataset))]]
            res = tuple(f.union_tensor([batch[row][col] for row in range(len(batch))]) for col in range(len(batch[0])))
            self._pos += self.batch_size
            return res
        else:
            self._pos = 0
            raise StopIteration
