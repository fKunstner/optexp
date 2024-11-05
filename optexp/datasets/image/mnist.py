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

MEAN, STD = 0.1307, 0.3081


@frozen
class MNIST(Dataset, HasClassCounts, Downloadable, InMemory):
    """The `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset, provided through
    `TorchVision <https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html>`_.
    """

    def get_num_samples(self, split: Split) -> int:
        if split == "tr":
            return 60_000
        if split == "va":
            return 10_000
        raise ValueError(f"Invalid tr_va: {split}")

    @lru_cache()
    def class_counts(self, split: Split) -> torch.Tensor:
        return torch.bincount(self._get_dataset(split).targets)

    def data_input_shape(self, batch_size) -> torch.Size:
        return torch.Size([batch_size, 1, 28, 28])

    def model_output_shape(self, batch_size) -> torch.Size:
        return torch.Size([batch_size, 10])

    def has_test_set(self) -> bool:
        return False

    def is_downloaded(self):
        return all(
            (Config.get_dataset_directory() / "MNISTDataset" / "raw" / file).exists()
            for file in [
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte",
            ]
        )

    def download(self):
        path = str(Config.get_dataset_directory())
        for train in [True, False]:
            torchvision.datasets.MNIST(path, download=True, train=train)

    @staticmethod
    def _get_dataset(split: Split):
        return torchvision.datasets.MNIST(
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
        data, targets = dataset.data, dataset.targets
        data = torchvision.transforms.Normalize(mean=MEAN, std=STD)(data / 255.0)
        data = data.unsqueeze(1)
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
