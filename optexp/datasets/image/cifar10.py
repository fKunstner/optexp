from functools import lru_cache
from typing import Optional

import torch
import torchvision
from attrs import frozen
from torch.types import Device
from torch.utils.data import TensorDataset

from optexp.config import Config
from optexp.datasets.dataset import (
    Dataset,
    Downloadable,
    HasClassCounts,
    InMemory,
    Split,
)
from optexp.datasets.utils import make_dataloader

MEAN, STD = torch.tensor([0.491, 0.482, 0.446]), torch.tensor([0.247, 0.243, 0.261])


@frozen
class CIFAR10(Dataset, HasClassCounts, Downloadable, InMemory):
    """The `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset, provided through
    `TorchVision <https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html>`_.
    """

    def get_num_samples(self, split: Split) -> int:
        if split == "tr":
            return 50_000
        if split == "va":
            return 10_000
        raise ValueError(f"Invalid tr_va: {split}")

    @lru_cache()
    def class_counts(self, split: Split) -> torch.Tensor:
        return torch.bincount(torch.tensor(self._get_dataset(split).targets))

    def data_input_shape(self, batch_size) -> torch.Size:
        return torch.Size([batch_size, 3, 32, 32])

    def model_output_shape(self, batch_size) -> torch.Size:
        return torch.Size([batch_size, 10])

    def has_test_set(self) -> bool:
        return False

    def is_downloaded(self):
        return all(
            (Config.get_dataset_directory() / "cifar-10-batches-py" / file).exists()
            for file in [
                "batches.meta",
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
                "readme.html",
                "test_batch",
            ]
        )

    def download(self):
        path = str(Config.get_dataset_directory())
        for train in [True, False]:
            torchvision.datasets.CIFAR10(path, download=True, train=train)

    @staticmethod
    def _get_dataset(split: Split):
        return torchvision.datasets.CIFAR10(
            root=str(Config.get_dataset_directory()),
            train=split == "tr",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=MEAN, std=STD),
                ]
            ),
        )

    def _get_tensor_dataset(self, split: Split, to_device: Optional[Device] = None):
        dataset = self._get_dataset(split)
        data, targets = torch.tensor(dataset.data), torch.tensor(dataset.targets)
        data = data.permute(0, 3, 1, 2).to(torch.float32)
        data = torchvision.transforms.Normalize(mean=MEAN, std=STD)(data / 255.0)
        if to_device is not None:
            data, targets = data.to(to_device), targets.to(to_device)
        return TensorDataset(data, targets)

    def get_dataloader(
        self, b: int, split: Split, num_workers: int
    ) -> torch.utils.data.DataLoader:
        return make_dataloader(self._get_dataset(split), b, num_workers)

    def get_in_memory_dataloader(
        self,
        b: int,
        split: Split,
        num_workers: int,
        to_device: Optional[Device] = None,
    ) -> torch.utils.data.DataLoader:
        return make_dataloader(
            self._get_tensor_dataset(split, to_device), b, num_workers
        )
