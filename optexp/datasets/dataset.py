from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
from attrs import frozen
from torch.types import Device
from torch.utils.data import DataLoader

from optexp.component import Component

TrVa = Literal["tr", "va"]


@frozen
class Dataset(ABC, Component):
    """Abstract base class for datasets."""

    @abstractmethod
    def get_dataloader(
        self,
        b: int,
        tr_va: TrVa,
        num_workers: int,
    ) -> DataLoader:
        """Return a dataloader with batch size ``b`` for the training or validation dataset."""
        raise NotImplementedError()

    @abstractmethod
    def input_shape(self, batch_size: int) -> torch.Size:
        """The expected input shape of the data, including the batch size"""
        raise NotImplementedError()

    @abstractmethod
    def output_shape(self, batch_size: int) -> torch.Size:
        """The expected output shape of the data, the shape of the targets"""
        raise NotImplementedError()

    @abstractmethod
    def get_num_samples(self, tr_va: TrVa) -> int:
        """The number of samples in the training or validation sets."""
        raise NotImplementedError()


class HasClassCounts:
    """Extension for datasets that provide class frequencies."""

    @abstractmethod
    def class_counts(self, tr_va: TrVa) -> torch.Tensor:
        raise NotImplementedError()


class MovableToLocal:
    """Extension for large datasets that need to be moved to local storage on SLURM nodes."""

    @abstractmethod
    def is_on_local(self) -> bool:
        """Returns True if the dataset is already in local storage."""
        raise NotImplementedError()

    @abstractmethod
    def move_to_local(self):
        """Moves the dataset from the workspace to local storage."""
        raise NotImplementedError()


class Downloadable:
    """Extension for datasets that can be downloaded.

    For interaction with the ``prepare`` CLI command.
    """

    @abstractmethod
    def is_downloaded(self) -> bool:
        """Checks whether the dataset is already in the workspace."""
        raise NotImplementedError()

    @abstractmethod
    def download(self):
        """Downloads the dataset to the workspace."""
        raise NotImplementedError()


class InMemory:
    """Extension for small datasets that can be loaded directly into RAM."""

    @abstractmethod
    def get_in_memory_dataloader(
        self, b: int, tr_va: TrVa, num_workers: int, device: Optional[Device] = None
    ) -> torch.utils.data.DataLoader:
        """Returns a Dataloader with the dataset already loaded into RAM on the device."""
        raise NotImplementedError()
