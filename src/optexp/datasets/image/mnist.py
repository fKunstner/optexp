from dataclasses import dataclass

import torch
import torchvision

from optexp import config
from optexp.datasets.dataset import ClassificationDataset, TrVa


@dataclass(frozen=True)
class MNIST(ClassificationDataset):

    def input_shape(self, batch_size) -> torch.Size:
        return torch.Size([batch_size, 1, 28, 28])

    def output_shape(self, batch_size) -> torch.Size:
        return torch.Size([10])

    def should_download(self):
        return True

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

    def should_move_to_local(self):
        return True

    def is_on_local(self):
        raise NotImplementedError()

    def move_to_local(self):
        raise NotImplementedError()

    def class_counts(self, tr_va: TrVa) -> torch.Tensor:
        return torch.bincount(self._get_dataset(tr_va).targets)

    def _get_dataset(self, tr_va: TrVa):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=0.1307, std=0.3081),
            ]
        )

        dataset = torchvision.datasets.MNIST(
            root=str(config.get_dataset_directory()),
            train=tr_va == "tr",
            transform=transform,
        )
        return dataset

    def get_dataloader(
        self,
        b: int,
        tr_va: TrVa,
        num_workers: int = 8,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self._get_dataset(tr_va),
            batch_size=b,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
