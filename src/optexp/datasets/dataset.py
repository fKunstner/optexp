from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch.types import Device
from torch.utils.data import DataLoader

from optexp.experiments.component import Component

TrVa = Literal["tr", "va"]


@dataclass(frozen=True)
class Dataset(Component):
    @abstractmethod
    def get_dataloader(
        self,
        b: int,
        tr_va: TrVa,
    ) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def input_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    @abstractmethod
    def output_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    @abstractmethod
    def get_num_samples(self, tr_va: TrVa) -> int:
        raise NotImplementedError()


class HasClassCounts:

    @abstractmethod
    def class_counts(self, tr_va: TrVa) -> torch.Tensor:
        raise NotImplementedError()


class MovableToLocalMixin:
    @abstractmethod
    def is_on_local(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def move_to_local(self):
        raise NotImplementedError()


class Downloadable:

    @abstractmethod
    def is_downloaded(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def download(self):
        raise NotImplementedError()


class AvailableAsTensor:
    @abstractmethod
    def get_tensor_dataloader(
        self, b: int, tr_va: TrVa, to_device: Optional[Device] = None
    ) -> torch.utils.data.DataLoader:
        raise NotImplementedError()


class DatasetNotDownloadableError(NotImplementedError):
    pass


def epoch_to_steps(epochs: int, dataset: Dataset, batch_size: int) -> int:
    return epochs * dataset.get_num_samples("tr") // batch_size
