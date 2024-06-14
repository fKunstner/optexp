from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST as TorchMNIST

from optexp import config
from optexp.config import get_logger
from optexp.datasets.dataset import Dataset
from optexp.datasets.dataset_getter import get_image_dataset


@dataclass(frozen=True)
class ImageDataset(Dataset):
    """An Image Dataset

    Attributes:
        flatten: 2D images will be flattened into a 1D vector if true
    """

    flatten: bool = False

    def load(
        self,
    ) -> Tuple[DataLoader, DataLoader, torch.Size, torch.Size, torch.Tensor]:
        get_logger().info("Loading dataset: " + self.name)

        loaders, input_shape, output_shape, class_freqs = get_image_dataset(
            dataset_name=self.name,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            normalize=True,
            flatten=self.flatten,
        )

        return (
            loaders["train_loader"],
            loaders["val_loader"],
            input_shape,
            output_shape,
            class_freqs,
        )


@dataclass(frozen=True)
class MNIST(ImageDataset):
    batch_size: int
    name: str = field(default="MNIST", init=False)
    flatten: bool = False

    def load(
        self,
    ) -> Tuple[DataLoader, DataLoader, torch.Size, torch.Size, torch.Tensor]:
        train_set = TorchMNIST(
            str(config.get_dataset_directory()),
            download=False,
            train=True,
            transform=torchvision.transforms.ToTensor(),
        )
        val_set = TorchMNIST(
            str(config.get_dataset_directory()),
            download=False,
            train=False,
            transform=torchvision.transforms.ToTensor(),
        )
        tr_ld = DataLoader(
            train_set,
            batch_size=self.batch_size,
        )
        va_ld = DataLoader(val_set, batch_size=self.batch_size)

        return (
            tr_ld,
            va_ld,
            torch.Size([1]),
            torch.Size([10]),
            torch.bincount(train_set.targets),
        )

    def download(self):
        raise NotImplementedError


@dataclass(frozen=True)
class ImageNet(ImageDataset):
    batch_size: int
    name: str = field(default="ImageNet", init=False)
    flatten: bool = False
