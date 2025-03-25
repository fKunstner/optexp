from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Union, Optional, Callable, overload

import torch
from attrs import frozen
from torch.nn import Module, Parameter

from optexp.component import Component


@frozen
class Optimizer(Component, ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass

    def plot_style(self):
        return {}


class MultiOptimizer(Optimizer):
    """A wrapper  for  multiple optimizers"""

    optimizers: List[Optimizer]

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:

        # todo some sort of validation that all parameters groups of the model are coveredd
        raise NotImplementedError()

    def validate_groups(self):
        raise NotImplementedError()

def create_multi_opt(optimizers: List[Optimizer]) -> torch.optim.Optimizer:

    class TorchMultiOptimizer(torch.optim.Optimizer):
        pass


class Regularizable(ABC):
    """Abstract base class for regular"""

    @abstractmethod
    def regularizer_loss(self, model: torch.nn.Module) -> torch.Tensor:
        pass


@frozen
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


class TorchMultiOptimizer(torch.optim.Optimizer):
    torch_optimizers: List[torch.optim.Optimizer]

    def __init__(self, torch_optimizers: List[torch.optim.Optimizer]):
        self.torch_optimizers= torch_optimizers
    @overload
    def step(self, closure: None = ...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        for opt in self.torch_optimizers:
            opt.step(closure=closure)