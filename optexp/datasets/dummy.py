from typing import Optional

import torch
from torch.types import Device
from torch.utils.data import TensorDataset

from optexp.datasets import Dataset
from optexp.datasets.dataset import TrVa
from optexp.datasets.utils import make_dataloader


def make_dataset(n):
    x = torch.linspace(0, 1, n)
    y = x + 0.1 * torch.sin(10 * 2 * torch.pi * x)
    return x.reshape(-1, 1), y.reshape(-1, 1)


class DummyRegression(Dataset):
    """A dummy dataset for testing purposes.

    The data is generated by a sine function::

        y = x + 0.1 * sin(10 * 2 * pi * x)

    The inputs x form a linear grid from 0 to 1 (``linspace(0, 1, n_samples)``).

    Args:
        n_tr (int, optional): number of training samples. Defaults to 100.
        n_va (int, optional): number of validation samples. Defaults to 10.
    """

    n_tr: int = 100
    n_va: int = 10

    def get_num_samples(self, tr_va: TrVa) -> int:
        if tr_va == "tr":
            return self.n_tr
        if tr_va == "va":
            return self.n_va
        raise ValueError(f"Invalid tr_va: {tr_va}")

    def input_shape(self, batch_size) -> torch.Size:
        return torch.Size([batch_size, 1])

    def output_shape(self, batch_size) -> torch.Size:
        return torch.Size([batch_size, 1])

    @staticmethod
    def _get_dataset(tr_va: TrVa, to_device: Optional[Device] = None):
        if tr_va == "tr":
            data, targets = make_dataset(DummyRegression.n_tr)
        elif tr_va == "va":
            data, targets = make_dataset(DummyRegression.n_va)
        else:
            raise ValueError(f"Invalid tr_va: {tr_va}")

        if to_device is not None:
            data, targets = data.to(to_device), targets.to(to_device)
        return TensorDataset(data, targets)

    def get_dataloader(
        self, b: int, tr_va: TrVa, num_workers: int, to_device: Optional[Device] = None
    ) -> torch.utils.data.DataLoader:
        return make_dataloader(
            self._get_dataset(tr_va, to_device=to_device), b, num_workers
        )

    def get_tensor_dataloader(
        self, b: int, tr_va: TrVa, num_workers: int, to_device: Optional[Device] = None
    ) -> torch.utils.data.DataLoader:
        return self.get_dataloader(b, tr_va, num_workers, to_device=to_device)