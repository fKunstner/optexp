import os
import torch
import numpy as np
from distutils.dir_util import copy_tree
from optexp.config import get_logger
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_heavy_tailed_imagenet(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):

    dataset_dir = str(save_path) + "/HeavyTailedImageNet"
    local_disk = os.getenv("SLURM_TMPDIR")
    if local_disk is not None:
        # bad and hack
        if not os.path.isfile(
            local_disk + "/HeavyTailedImageNet/HEAVY_TAILED_IMAGENET"
        ):
            os.system(
                f"tar -xzf  {dataset_dir}/HeavyTailedImageNet.tar.gz -C {local_disk}"
            )
            os.system(f"touch {local_disk}/HeavyTailedImageNet/HEAVY_TAILED_IMAGENET")

        local_disk += "/HeavyTailedImageNet"
    else:
        local_disk = dataset_dir

    traindir = os.path.join(local_disk, "train")
    valdir = os.path.join(local_disk, "val")

    if normalize:
        normalize_transform = transforms.Normalize(
            mean=[0.4984, 0.4675, 0.4102], std=[0.2192, 0.2147, 0.2155]
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


def load_decaying_imagenet(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):

    dataset_dir = str(save_path) + "/DecayingImageNet"
    local_disk = os.getenv("SLURM_TMPDIR")
    if local_disk is not None:
        local_disk += "/DecayingImageNet"
        # bad and hack
        if not os.path.isfile(local_disk + "/DECAYING_IMAGENET"):
            os.system(f"cp --recursive {dataset_dir} {local_disk}")
            os.system(f"touch {local_disk}/DECAYING_IMAGENET")
    else:
        local_disk = dataset_dir

    traindir = os.path.join(local_disk, "train")
    valdir = os.path.join(local_disk, "val")

    if normalize:
        normalize_transform = transforms.Normalize(
            mean=[0.4827, 0.4511, 0.3953], std=[0.2166, 0.2121, 0.2099]
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


def load_small_imagenet(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):

    dataset_dir = str(save_path) + "/SmallImageNet"
    local_disk = os.getenv("SLURM_TMPDIR")
    if local_disk is not None:
        local_disk += "/SmallImageNet"
        # bad and hack
        if not os.path.isfile(local_disk + "/SMALL_IMAGENET"):
            os.system(f"cp --recursive {dataset_dir} {local_disk}")
            os.system(f"touch {local_disk}/SMALL_IMAGENET")
    else:
        local_disk = dataset_dir

    traindir = os.path.join(local_disk, "train")
    valdir = os.path.join(local_disk, "val")

    if normalize:
        normalize_transform = transforms.Normalize(
            mean=[0.4840, 0.4531, 0.4013], std=[0.2180, 0.2138, 0.2126]
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



def load_imagenet(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):

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
