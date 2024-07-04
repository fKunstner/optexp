from abc import ABC, abstractmethod
from typing import Tuple

import torch

from optexp.component import Component


class Metric(Component, ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class LossMetric(Metric):
    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
