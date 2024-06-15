import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from optexp import config
from optexp.config import get_logger
from optexp.datasets.dataset import DatasetNotDownloadableError, HasClassCounts, TrVa


class ImagenetNotDownloadableError(DatasetNotDownloadableError):
    def __init__(self):
        message = textwrap.dedent(
            f"""
            ImageNet cannot be downloaded automatically.
            
            Please download the dataset manually and place it in the workspace, at 
                {config.get_dataset_directory() / "ImageNet"}
            """
        )
        super().__init__(message)


@dataclass(frozen=True)
class ImageNet(HasClassCounts):

    def get_dataloader(self, b: int, tr_va: TrVa, on_gpu: bool = False) -> DataLoader:
        raise NotImplementedError()

    def input_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    def output_shape(self, batch_size) -> torch.Size:
        return torch.Size([1000])

    def can_download(self):
        return True

    def is_downloaded(self):
        raise NotImplementedError()

    def download(self):
        raise ImagenetNotDownloadableError()

    def can_move_to_local(self):
        raise NotImplementedError()

    def move_to_local(self):
        raise NotImplementedError()

    def is_on_local(self):
        raise NotImplementedError()

    def class_counts(self, tr_va: TrVa) -> torch.Tensor:
        raise NotImplementedError()


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
    loaders = {"train_loader": train_loader, "val_loader": val_loader}

    return loaders


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


def get_dataloader(
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

    if tr_va == "tr":
        return train_loader
    return val_loader


if __name__ == "__main__":
    # extract("/home/anon/optexp/datasets/ImageNet")
    raise ImagenetNotDownloadableError()
