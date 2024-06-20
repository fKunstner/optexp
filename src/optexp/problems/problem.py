from dataclasses import dataclass
from typing import List, Type

import torch

from optexp.datasets import Dataset
from optexp.models import Model
from optexp.problems.metrics import Metric


@dataclass
class Problem:
    """Wrapper for a model and dataset defining a problem to optimize.

    Attributes:
        model: The model to optimize.
        dataset: The dataset to optimize the model on.
        batch_size: The effective batch size for each step.
        lossfunc: The loss function to use for optimization.
        metrics: The metrics to evaluate the model on.
    """

    model: Model
    dataset: Dataset
    batch_size: int
    lossfunc: Type[torch.nn.Module]
    metrics: List[Type[Metric]]

    def __post_init__(self):
        if self.dataset.get_num_samples("tr") % self.batch_size != 0:
            raise ValueError(
                "Batch size must divide the number of samples in the dataset."
            )
