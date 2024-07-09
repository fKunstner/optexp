import torch
from attr import frozen
from torch.utils.data import DataLoader

from optexp.datasets.dataset import Dataset, HasClassCounts, TrVa


@frozen
class ImageNet(Dataset, HasClassCounts):
    def input_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    def get_num_samples(self, tr_va: TrVa) -> int:
        raise NotImplementedError()

    def output_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    def get_dataloader(self, b: int, tr_va: TrVa, num_workers: int) -> DataLoader:
        raise NotImplementedError()

    def class_counts(self, tr_va: TrVa) -> torch.Tensor:
        raise NotImplementedError()
