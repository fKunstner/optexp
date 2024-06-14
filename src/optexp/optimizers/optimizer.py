from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class Optimizer(ABC):

    @abstractmethod
    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass
