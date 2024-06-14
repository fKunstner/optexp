from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch

TrVa = Literal["tr", "va"]


@dataclass(frozen=True)
class Dataset:
    @abstractmethod
    def get_dataloader(
        self,
        b: int,
        tr_va: TrVa,
    ) -> torch.utils.data.DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def input_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    @abstractmethod
    def output_shape(self, batch_size) -> torch.Size:
        raise NotImplementedError()

    @abstractmethod
    def should_download(self):
        raise NotImplementedError()

    @abstractmethod
    def is_downloaded(self):
        raise NotImplementedError()

    @abstractmethod
    def download(self):
        raise NotImplementedError()

    @abstractmethod
    def should_move_to_local(self):
        """Whether the dataset should be moved to the local machine when running on a
        cluster."""

        raise NotImplementedError()

    @abstractmethod
    def move_to_local(self):
        raise NotImplementedError()

    @abstractmethod
    def is_on_local(self):
        raise NotImplementedError()


@dataclass(frozen=True)
class ClassificationDataset(Dataset):

    @abstractmethod
    def class_counts(self, tr_va: TrVa) -> torch.Tensor:
        raise NotImplementedError()


class DatasetNotDownloadableError(NotImplementedError):
    pass
