from abc import ABC, abstractmethod
from typing import Tuple

import torch

from optexp.component import Component


class Metric(Component, ABC):
    """Abstract base class for metrics."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def smaller_better(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_scalar(self) -> bool:
        raise NotImplementedError


class LossLikeMetric(Metric):
    """Abstract base class for loss-like metrics, which take inputs and labels."""

    @abstractmethod
    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def unreduced_call(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class GraphMetric(Metric):
    """Abstract base class for metrics that take raw data inputs, outputs, and labels."""

    @abstractmethod
    def __call__(
        self, data, mask: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
