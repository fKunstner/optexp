from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Union

import torch
from torch.nn import Module, Parameter

from optexp.component import Component


@dataclass(frozen=True)
class Optimizer(Component, ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass


@dataclass(frozen=True)
class WeightDecayStrategy(Component):
    """Abstract base class for weight decay strategies."""

    def make_param_groups(
        self, model: Module, weight_decay: float
    ) -> List[Dict[str, Union[Iterable[Parameter], float]]]:
        raise NotImplementedError
