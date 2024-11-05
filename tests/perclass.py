from typing import Optional

import torch
from torch.types import Device
from torch.utils.data import DataLoader, TensorDataset

from optexp.datasets import Dataset
from optexp.datasets.dataset import HasClassCounts, Split
from optexp.datasets.utils import make_dataloader
from optexp.metrics import CrossEntropy, PerClass

if __name__ == "__main__":
    perclassmetric = PerClass(CrossEntropy(), groups=2)

    class TestDataset(Dataset, HasClassCounts):

        n = 40
        d = 100
        c = 22

        def get_num_samples(self, split: Split) -> int:
            return self.n

        def class_counts(self, split: Split) -> torch.Tensor:
            return torch.bincount(self._get_dataset(split).tensors[1])

        def data_input_shape(self, batch_size) -> torch.Size:
            return torch.Size([batch_size, self.d])

        def model_output_shape(self, batch_size) -> torch.Size:
            return torch.Size([batch_size, self.c])

        def has_test_set(self) -> bool:
            return False

        def _get_dataset(self, split: Split):
            return TensorDataset(*self._get_data_matrices(split))

        def _get_data_matrices(self, split: Split, to_device: Optional[Device] = None):
            X = torch.randn(self.n, self.d)
            class_frequencies = ([10] * 2) + ([1] * 20)
            y = torch.zeros(self.n, dtype=torch.long)
            current_class = 0
            samples_for_that_class = 0

            for i in range(self.n):
                y[i] = current_class
                samples_for_that_class += 1
                if samples_for_that_class == class_frequencies[current_class]:
                    current_class += 1
                    samples_for_that_class = 0

            if to_device is not None:
                X, y = X.to(to_device), y.to(to_device)
            return X, y

        def get_dataloader(
            self, b: int, split: Split, num_workers: int
        ) -> torch.utils.data.DataLoader:
            return make_dataloader(self._get_dataset(split), b, num_workers)

    dataset = TestDataset()
    dataloader = dataset.get_dataloader(10, "tr", 0)

    class Mock(object):
        pass

    mock_expinfo = Mock()
    setattr(mock_expinfo, "exp", Mock())
    setattr(mock_expinfo.exp, "problem", Mock())
    setattr(mock_expinfo.exp.problem, "dataset", dataset)

    model = torch.nn.Linear(dataset.d, dataset.c)
    model.weight.data.fill_(0)
    model.bias.data.fill_(0)

    for inputs, labels in dataloader:
        print(perclassmetric(model(inputs), labels, mock_expinfo))
        break
