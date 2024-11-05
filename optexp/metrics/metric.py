from abc import ABC, abstractmethod
from typing import Tuple

import torch
from attr import frozen

from optexp.component import Component
from optexp.datastructures import ExpInfo


@frozen
class Metric(Component, ABC):
    """Abstract base class for metrics."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def smaller_is_better(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_scalar(self) -> bool:
        raise NotImplementedError


class LossLikeMetric(Metric, ABC):
    """Abstract base class for loss-like metrics, which take inputs and labels."""

    @abstractmethod
    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor, exp_info: ExpInfo
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def unreduced_call(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class GraphLossLikeMetric(Metric):
    """Abstract base class for metrics that take raw data inputs, outputs, and labels."""

    @abstractmethod
    def __call__(
        self,
        data,
        mask: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        exp_info: ExpInfo,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
