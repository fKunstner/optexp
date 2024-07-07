from abc import ABC, abstractmethod
from typing import Tuple

import torch

from optexp.component import Component


class Metric(Component, ABC):
    """Abstract base class for metrics."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class LossLikeMetric(Metric):
    """Abstract base class for loss-like metrics, which take inputs and labels."""

    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ModelMetric(Metric):
    """Abstract base class for metrics that takes a model."""

    @abstractmethod
    def __call__(self, model: torch.nn.Module):
        raise NotImplementedError()
