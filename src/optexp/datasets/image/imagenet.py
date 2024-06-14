import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from optexp import config
from optexp.config import get_logger
from optexp.datasets import Dataset
from optexp.datasets.dataset import TrVa


@dataclass(frozen=True)
class ImageNet(Dataset):
    batch_size: int
    name: str = field(default="ImageNet", init=False)
    normalize: bool = True
    flatten: bool = False
    num_workers: int = 8
    shuffle: bool = True

    def load(
        self,
        b: int,
        tr_va: TrVa,
        on_gpu: bool = False,
    ) -> DataLoader:
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
        # targets = torch.tensor(train_dataset.targets)
        # output_shape = np.array([targets.max().item() + 1])
        # features, _ = next(iter(train_loader))
        # input_shape = np.array(list(features[0].shape))
        # return (
        #     train_loader,
        #     val_loader,
        #     input_shape,
        #     output_shape,
        #     torch.bincount(targets),
        # )

        if tr_va == "tr":
            return train_loader
        return val_loader

    def download(self):
        pass


def load_imagenet(save_path, batch_size, shuffle, num_workers, normalize):

    dataset_dir = str(save_path) + "/ImageNet"

    local_dataset = extract(dataset_dir)

    traindir = os.path.join(local_dataset, "train")
    valdir = os.path.join(local_dataset, "val")

    if normalize:
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
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
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
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    targets = torch.tensor(train_dataset.targets)
    output_shape = np.array([targets.max().item() + 1])
    loaders = {"train_loader": train_loader, "val_loader": val_loader}
    features, _ = next(iter(train_loader))
    input_shape = np.array(list(features[0].shape))
    return loaders, input_shape, output_shape, torch.bincount(targets)


def extract(dataset_dir):
    local_disk = os.getenv("SLURM_TMPDIR")
    if local_disk is None:
        raise ValueError(
            "Cannot locate node scratch, it is only visible on the main node process."
        )
    if not os.path.isfile(local_disk + "/imagenet" + "/EXTRACTED"):
        get_logger().info("Extracting ImageNet Dataset")
        os.system(f"bash {dataset_dir}/extract_ILSVRC.sh {dataset_dir} {local_disk}")
    else:
        get_logger().info("Dataset Already Extracted")
    return local_disk + "/imagenet"


if __name__ == "__main__":
    extract("/home/anon/optexp/datasets/ImageNet")
