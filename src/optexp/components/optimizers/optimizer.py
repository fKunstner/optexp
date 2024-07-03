from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Union

import torch
from torch.nn import Module, Parameter

from optexp.components.component import Component, dataclass_component


@dataclass_component()
class Optimizer(Component, ABC):

    @abstractmethod
    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass


@dataclass_component()
class WeightDecayStrategy(Component):

    def make_param_groups(
        self, model: Module, weight_decay: float
    ) -> List[Dict[str, Union[Iterable[Parameter], float]]]:
        raise NotImplementedError
