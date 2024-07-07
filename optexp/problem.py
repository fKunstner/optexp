from dataclasses import dataclass
from typing import List, Type

import torch

from optexp.component import Component
from optexp.datasets.dataset import Dataset
from optexp.metrics.metric import Metric
from optexp.models.model import Model


@dataclass(frozen=True)
class Problem(Component):
    """Specify a problem.

    Args:
        model (Model): model to optimize.
        dataset (Dataset): dataset to fit the model to.
        batch_size (int): effective batch size.
           To use gradient accumulation, set the ``micro_batch_size``
           in :class:`optexp.hardwareconfig.HardwareConfig`.
        lossfunc (Type[torch.nn.Module]): loss function to use for optimization,
           given as the class of a subtype of `torch.nn.Module`
           (e.g. ``torch.nn.CrossEntropyLoss``, not `torch.nn.CrossEntropyLoss`).
        metrics (List[Metric]): metrics to evaluate.
    """

    model: Model
    dataset: Dataset
    batch_size: int
    lossfunc: Type[torch.nn.Module]
    metrics: List[Type[Metric]]
