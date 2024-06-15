from dataclasses import dataclass
from typing import Optional

import torch
import torchvision
from torch.types import Device
from torch.utils.data import TensorDataset

from optexp import config
from optexp.datasets.dataset import (
    AvailableAsTensor,
    Dataset,
    Downloadble,
    HasClassCounts,
    TrVa,
)

MEAN, STD = 0.1307, 0.3081


@dataclass(frozen=True)
class MNIST(Dataset, HasClassCounts, Downloadble, AvailableAsTensor):

    def class_counts(self, tr_va: TrVa) -> torch.Tensor:
        return torch.bincount(self._get_dataset(tr_va).targets)

    def input_shape(self, batch_size) -> torch.Size:
        return torch.Size([batch_size, 1, 28, 28])

    def output_shape(self, batch_size) -> torch.Size:
        return torch.Size([10])

    def is_downloaded(self):
        return all(
            (config.get_dataset_directory() / "MNISTDataset" / "raw" / file).exists()
            for file in [
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte",
            ]
        )

    def download(self):
        path = str(config.get_dataset_directory())
        for train in [True, False]:
            torchvision.datasets.MNIST(path, download=True, train=train)

    def _get_dataset(self, tr_va: TrVa):
        return torchvision.datasets.MNIST(
            root=str(config.get_dataset_directory()),
            train=tr_va == "tr",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=MEAN, std=STD),
                ]
            ),
        )

    def _get_tensor_dataset(self, tr_va: TrVa, to_device: Optional[Device] = None):
        dataset = self._get_dataset(tr_va)
        X, y = dataset.data, dataset.targets
        X = torchvision.transforms.Normalize(mean=MEAN, std=STD)(X / 255.0)
        if to_device is not None:
            X, y = X.to(to_device), y.to(to_device)
        return TensorDataset(X, y)

    def get_dataloader(
        self,
        b: int,
        tr_va: TrVa,
        num_workers: int = 4,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self._get_dataset(tr_va),
            batch_size=b,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_tensor_dataloader(
        self,
        b: int,
        tr_va: TrVa,
        to_device: Optional[Device] = None,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self._get_tensor_dataset(tr_va, to_device),
            batch_size=b,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
