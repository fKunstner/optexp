from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from optexp.experiments.component import Component


@dataclass(frozen=True)
class Optimizer(Component, ABC):

    @abstractmethod
    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass
