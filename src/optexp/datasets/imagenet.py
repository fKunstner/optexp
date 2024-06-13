import os
from pathlib import Path

import numpy as np
import torch

from optexp import Dataset, get_logger, config
from dataclasses import dataclass, field
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ImageNet(Dataset):
    batch_size: int
    name: str = field(default="ImageNet", init=False)
    normalize: bool = True
    flatten: bool = False
    num_workers: int = 8
    shuffle: bool = True

    def load(self):
        get_logger().info("Loading dataset: " + self.name)

        dataset_dir = config.get_dataset_directory() / Path("ImageNet")
        traindir = os.path.join(dataset_dir, "train")
        valdir = os.path.join(dataset_dir, "val")

        if self.normalize:
            normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            normalize_transform = transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]
            )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize_transform,
                ]
            ),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize_transform,
                ]
            ),
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        targets = torch.tensor(train_dataset.targets)
        output_shape = np.array([targets.max().item() + 1])
        features, _ = next(iter(train_loader))
        input_shape = np.array(list(features[0].shape))
        return (
            train_loader,
            val_loader,
            input_shape,
            output_shape,
            torch.bincount(targets),
        )

    def download(self):
        pass
