from dataclasses import dataclass
from typing import List, Literal, Type

import torch

from optexp.datasets import Dataset
from optexp.models import Model
from optexp.problems.metrics import Metric


@dataclass
class Problem:
    """Wrapper for a model and dataset defining a problem to optimize.

    Attributes:
        model: The model that will be optimized.
        dataset: The dataset to use.
    """

    model: Model
    dataset: Dataset
    batch_size: int
    lossfunc: Type[torch.nn.Module]
    metrics: List[Type[Metric]]
    micro_batch_size: Literal["auto"] | int = "auto"
