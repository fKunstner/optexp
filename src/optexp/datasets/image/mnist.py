from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torchvision
from torchvision import transforms

from optexp import config
from optexp.datasets import Dataset
from optexp.datasets.dataset import TrVa


@dataclass(frozen=True)
class MNIST(Dataset):

    normalize: bool = True
    flatten: bool = False

    def output_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    def should_download(self):
        return True

    def should_move_to_local(self):
        return True

    def input_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    def load(
        self,
        b: int,
        tr_va: TrVa,
        on_gpu: bool = True,
        num_workers: int = 8,
        shuffle: bool = True,
    ):
        if on_gpu:
            return self.load_tensor_dataset(shuffle, b)

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

    def load_tensor_dataset(self, batch_size, shuffle):
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
        train_dataset = torch.utils.data.TensorDataset(
            raw_train_data.to(config.get_device()),
            train_targets.to(config.get_device()),
        )
        val_dataset = torch.utils.data.TensorDataset(
            raw_val_data.to(config.get_device()),
            val_targets.to(config.get_device()),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
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
        path = str(config.get_dataset_directory())
        for train in [True, False]:
            torchvision.datasets.MNIST(path, download=True, train=train)
