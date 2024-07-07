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


class Regularizable(ABC):
    """Abstract base class for regular"""

    @abstractmethod
    def regularizer_loss(self, model: torch.nn.Module) -> torch.Tensor:
        pass


@dataclass(frozen=True)
class WeightDecayStrategy(Component):
    """Abstract base class for weight decay strategies."""

    def make_param_groups(
        self, model: Module, weight_decay: float
    ) -> List[Dict[str, Union[Iterable[Parameter], float]]]:
        raise NotImplementedError

    def regularizer_loss(self, model: Module, weight_decay: float) -> torch.Tensor:
        loss = None
        for group in self.make_param_groups(model, weight_decay):
            if group["weight_decay"] != 0:  # type: ignore
                for p in group["params"]:  # type: ignore
                    if loss is None:
                        loss = group["weight_decay"] * p.norm(p=2) ** 2
                    else:
                        loss += group["weight_decay"] * p.norm(p=2) ** 2

        if loss is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        return loss  # type: ignore
