from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from optexp.config import get_logger
from optexp.datasets import get_dataset


@dataclass(frozen=True)
class Dataset:
    """
    Defining and loading the dataset.

    Attributes:
        name: The name of the dataset to load.
        batch_size: The batch size to use.

    """

    name: str
    batch_size: int

    @abstractmethod
    def load(
        self,
    ) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, torch.Tensor]:
        """
        Uses a helper function to load a dataset with PyTorch DataLoader class.

        Returns:
            First PyTorch DataLoader object corresponds to the train set and second is the validation set.
            First NumPy arrays represents the input shape (shape of the features) and the second
            is output shape (shape of the labels) which is useful for defining the model.
        """

    @abstractmethod
    def download(self):
        pass

    def should_download(self):
        return False
