from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader

TR_VA = Literal["tr", "va"]


@dataclass(frozen=True)
class Dataset:
    @abstractmethod
    def load(
        self,
        b: int,
        tr_va: TR_VA,
        on_gpu: bool = False,
    ) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def input_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    @abstractmethod
    def output_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    @abstractmethod
    def download(self):
        raise NotImplementedError()

    @abstractmethod
    def should_download(self):
        raise NotImplementedError()

    @abstractmethod
    def should_move_to_local(self):
        """Whether the dataset should be moved to the local machine when running on a
        cluster."""

        raise NotImplementedError()

    @abstractmethod
    def move_to_local(self):
        raise NotImplementedError()


@dataclass(frozen=True)
class ClassificationDataset(Dataset):

    @abstractmethod
    def class_frequencies(self) -> torch.Tensor:
        raise NotImplementedError()
