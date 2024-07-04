from dataclasses import dataclass
from typing import List, Type

import torch

from optexp.components.component import Component
from optexp.components.datasets import Dataset
from optexp.components.metric import Metric
from optexp.components.model import Model


@dataclass(frozen=True)
class Problem(Component):
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
