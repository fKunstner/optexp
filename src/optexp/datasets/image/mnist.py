from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from optexp import config
from optexp.datasets import Dataset
from optexp.datasets.dataset import TR_VA


@dataclass(frozen=True)
class MNIST(Dataset):

    normalize: bool = True
    flatten: bool = False

    def output_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    def should_download(self):
        raise NotImplementedError()

    def input_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    def load(
        self,
        b: int,
        tr_va: TR_VA,
        on_gpu: bool = True,
        num_workers: int = 8,
        shuffle: bool = True,
    ):
        if on_gpu:
            return self.load_tensor_dataset(shuffle)

        transform_list = [transforms.ToTensor()]
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=0.1307, std=0.3081))

        if self.flatten:
            transform_list.append(transforms.Lambda(torch.flatten))
        else:
            transform_list.append(transforms.Lambda(partial(torch.unsqueeze, dim=0)))

        train_dataset = torchvision.datasets.MNIST(
            root=str(config.get_dataset_directory()),
            train=True,
            transform=transforms.Compose(transform_list),
        )

        val_dataset = torchvision.datasets.MNIST(
            root=str(config.get_dataset_directory()),
            train=False,
            transform=transforms.Compose(transform_list),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=b,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=b,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        targets = train_dataset.targets.clone().detach()
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

    def load_tensor_dataset(self, shuffle):
        train_dataset = torchvision.datasets.MNIST(
            root=str(config.get_dataset_directory()),
            train=True,
            transform=transforms.ToTensor(),
        )

        val_dataset = torchvision.datasets.MNIST(
            root=str(config.get_dataset_directory()),
            train=False,
            transform=transforms.ToTensor(),
        )
        raw_train_data = train_dataset.data.div(255.0)
        raw_val_data = val_dataset.data.div(255.0)

        if self.normalize:
            raw_train_data = transforms.Normalize(mean=0.1307, std=0.3081)(
                raw_train_data
            )
            raw_val_data = transforms.Normalize(mean=0.1307, std=0.3081)(raw_val_data)
        if self.flatten:
            raw_train_data = torch.flatten(raw_train_data, start_dim=1)
            raw_val_data = torch.flatten(raw_val_data, start_dim=1)
        else:
            raw_train_data = raw_train_data.unsqueeze(1)
            raw_val_data = raw_val_data.unsqueeze(1)

        targets = train_dataset.targets.clone().detach()

        train_targets = train_dataset.targets.to(torch.int)
        val_targets = val_dataset.targets.to(torch.int)
        train_dataset = TensorDataset(
            raw_train_data.to(config.get_device()),
            train_targets.to(config.get_device()),
        )
        val_dataset = TensorDataset(
            raw_val_data.to(config.get_device()),
            val_targets.to(config.get_device()),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

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
        download_mnist(config.get_dataset_directory())


def download_mnist(save_path):
    for train in [True, False]:
        MNIST(save_path, download=True, train=train)


def transform1(x):
    return x.to(torch.float32).flatten()


def transform2(x):
    mean = torch.tensor(
        0.1307, dtype=torch.float32, device=x.device, requires_grad=False
    )
    std = torch.tensor(
        0.3081, dtype=torch.float32, device=x.device, requires_grad=False
    )
    return (x - mean) / std


def transform3(x):
    return torch.flatten(x)


def transform4(x):
    return x.unsqueeze(0)


def load_mnist(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):
    if normalize:
        mean = torch.tensor(
            0.1307, dtype=torch.float32, device=device, requires_grad=False
        )
        std = torch.tensor(
            0.3081, dtype=torch.float32, device=device, requires_grad=False
        )
    else:
        mean = torch.tensor(0, dtype=torch.float32, device=device)
        std = torch.tensor(1, dtype=torch.float32, device=device)

    import pdb

    if flatten:
        transform = transforms.Compose(
            [
                transforms.Lambda(transform1),
                transforms.Lambda(lambda x: print("a", flatten) or pdb.set_trace()),
                transforms.Lambda(transform2),
                transforms.Lambda(transform3),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Lambda(transform1),
                transforms.Lambda(lambda x: print("a", flatten) or pdb.set_trace()),
                transforms.Lambda(transform2),
                transforms.Lambda(transform4),
            ]
        )

    train_set = torchvision.datasets.MNIST(
        save_path,
        download=False,
        train=True,
        transform=transform,
    )

    val_set = torchvision.datasets.MNIST(
        save_path,
        download=False,
        train=False,
        transform=transform,
    )
    train_set.data = train_set.data.to(device)
    train_set.targets = train_set.targets.to(device)
    val_set.data = val_set.data.to(device)
    val_set.targets = val_set.targets.to(device)
    output_shape = np.array([train_set.targets.max().item() + 1])

    train_data_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_data_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    loaders = {"train_loader": train_data_loader, "val_loader": val_data_loader}
    features, _ = next(iter(train_data_loader))
    input_shape = np.array(list(features[0].shape))
    result = loaders, input_shape, output_shape, torch.bincount(train_set.targets)
    return result